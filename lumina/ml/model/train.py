"""Training loop for the LightingTransformer model.

Multi-task training with genre-aware weighting:
  - Huber loss for color head (smooth L1, robust to outliers)
  - MSE loss for spatial head
  - BCE loss for strobe/blackout predictions in effect head
  - Temporal consistency loss to penalize frame-to-frame jitter

Features:
  - Automatic resume from latest checkpoint (``--resume``)
  - Graceful shutdown on Ctrl+C / SIGTERM / laptop lid close — saves
    a checkpoint before exiting so you can resume later
  - Checkpoints saved every epoch to ``data/models/checkpoints/``
  - Optional wandb logging when ``--wandb`` flag is passed

Usage:
    # Start fresh
    python -m lumina.ml.model.train

    # Resume from last checkpoint (re-run the same command)
    python -m lumina.ml.model.train --resume

    # Custom settings
    python -m lumina.ml.model.train --epochs 100 --batch-size 64 --resume
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from lumina.ml.model.architecture import (
    COLOR_HEAD_DIM,
    EFFECT_HEAD_DIM,
    SPATIAL_HEAD_DIM,
    LightingTransformer,
)
from lumina.ml.model.dataset import create_dataloaders

logger = logging.getLogger(__name__)

# ── Default paths ────────────────────────────────────────────────────

_DEFAULT_DATA_DIR = Path("data/features/aligned")
_DEFAULT_CHECKPOINT_DIR = Path("data/models/checkpoints")

# ── Loss weights ─────────────────────────────────────────────────────

W_COLOR = 1.0
W_SPATIAL = 0.5
W_STROBE = 1.0
W_BLACKOUT = 2.0  # Blackout is rarer, weight higher
W_TEMPORAL = 0.1


# ── Graceful shutdown ────────────────────────────────────────────────


class _ShutdownRequested(Exception):
    """Raised when a signal requests graceful shutdown."""


_shutdown_flag = False


def _signal_handler(signum: int, frame: object) -> None:
    """Handle SIGINT/SIGTERM by setting the shutdown flag.

    On first signal, sets a flag so the current epoch finishes and a
    checkpoint is saved.  On second signal, exits immediately.
    """
    global _shutdown_flag
    if _shutdown_flag:
        logger.warning("Second interrupt received — exiting immediately")
        sys.exit(1)
    sig_name = signal.Signals(signum).name
    logger.info("Received %s — finishing current epoch and saving checkpoint...", sig_name)
    _shutdown_flag = True


# ── Loss functions ───────────────────────────────────────────────────


def temporal_consistency_loss(predictions: torch.Tensor) -> torch.Tensor:
    """Penalize frame-to-frame jitter in model predictions.

    Computes mean squared differences between consecutive frames.
    This encourages smooth transitions and discourages rapid oscillation
    that doesn't correspond to musical changes.

    Args:
        predictions: Tensor of shape (batch, seq_len, features).

    Returns:
        Scalar loss tensor.
    """
    diffs = predictions[:, 1:] - predictions[:, :-1]
    return torch.mean(diffs**2)


def compute_loss(
    color_pred: torch.Tensor,
    spatial_pred: torch.Tensor,
    effect_pred: torch.Tensor,
    color_target: torch.Tensor,
    spatial_target: torch.Tensor,
    effect_target: torch.Tensor,
    confidence: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute multi-task loss with confidence weighting.

    Args:
        color_pred: Model color output (batch, seq, 6).
        spatial_pred: Model spatial output (batch, seq, 5).
        effect_pred: Model effect output (batch, seq, 3).
        color_target: Ground truth color (batch, seq, 6).
        spatial_target: Ground truth spatial (batch, seq, 5).
        effect_target: Ground truth effect (batch, seq, 3).
        confidence: Per-frame scene confidence (batch, seq).

    Returns:
        Tuple of (total_loss, loss_dict) where loss_dict has individual
        loss component values for logging.
    """
    # Expand confidence for broadcasting: (batch, seq) -> (batch, seq, 1).
    conf_weight = confidence.unsqueeze(-1)

    # Color: Huber loss (smooth L1).
    color_loss = F.huber_loss(color_pred, color_target, reduction="none")
    color_loss = (color_loss * conf_weight).mean()

    # Spatial: MSE loss.
    spatial_loss = F.mse_loss(spatial_pred, spatial_target, reduction="none")
    spatial_loss = (spatial_loss * conf_weight).mean()

    # Effect: split into strobe/blackout (BCE) and brightness_delta (MSE).
    # effect_target[:, :, 0] = is_strobe (binary)
    # effect_target[:, :, 1] = is_blackout (binary)
    # effect_target[:, :, 2] = brightness_delta_magnitude (continuous)
    strobe_loss = F.binary_cross_entropy(
        effect_pred[:, :, 0:1],
        effect_target[:, :, 0:1],
        reduction="none",
    )
    strobe_loss = (strobe_loss * conf_weight).mean()

    blackout_loss = F.binary_cross_entropy(
        effect_pred[:, :, 1:2],
        effect_target[:, :, 1:2],
        reduction="none",
    )
    blackout_loss = (blackout_loss * conf_weight).mean()

    delta_loss = F.mse_loss(
        effect_pred[:, :, 2:3],
        effect_target[:, :, 2:3],
        reduction="none",
    )
    delta_loss = (delta_loss * conf_weight).mean()

    # Temporal consistency on all outputs concatenated.
    all_pred = torch.cat([color_pred, spatial_pred, effect_pred], dim=-1)
    temp_loss = temporal_consistency_loss(all_pred)

    # Weighted sum.
    total = (
        W_COLOR * color_loss
        + W_SPATIAL * spatial_loss
        + W_STROBE * strobe_loss
        + W_BLACKOUT * blackout_loss
        + W_SPATIAL * delta_loss
        + W_TEMPORAL * temp_loss
    )

    loss_dict = {
        "loss/total": total.item(),
        "loss/color": color_loss.item(),
        "loss/spatial": spatial_loss.item(),
        "loss/strobe": strobe_loss.item(),
        "loss/blackout": blackout_loss.item(),
        "loss/delta": delta_loss.item(),
        "loss/temporal": temp_loss.item(),
    }

    return total, loss_dict


# ── Training loop ────────────────────────────────────────────────────


def train_one_epoch(
    model: LightingTransformer,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> dict[str, float]:
    """Train the model for one epoch.

    Args:
        model: The LightingTransformer model.
        loader: Training DataLoader.
        optimizer: Optimizer instance.
        device: Target device.
        epoch: Current epoch number (for logging).

    Returns:
        Dict of averaged loss components over the epoch.
    """
    model.train()
    epoch_losses: dict[str, list[float]] = {}
    num_batches = 0

    for batch in loader:
        # Check for graceful shutdown between batches.
        if _shutdown_flag:
            break

        music_features = batch["music_features"].to(device)
        genre_ids = batch["genre_ids"].to(device)
        segment_ids = batch["segment_ids"].to(device)
        color_targets = batch["color_targets"].to(device)
        spatial_targets = batch["spatial_targets"].to(device)
        effect_targets = batch["effect_targets"].to(device)
        confidence = batch["confidence"].to(device)

        # Forward pass.
        color_pred, spatial_pred, effect_pred = model(
            music_features, genre_ids, segment_ids
        )

        # Compute loss.
        loss, loss_dict = compute_loss(
            color_pred,
            spatial_pred,
            effect_pred,
            color_targets,
            spatial_targets,
            effect_targets,
            confidence,
        )

        # Backward pass.
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Accumulate losses.
        for key, value in loss_dict.items():
            epoch_losses.setdefault(key, []).append(value)
        num_batches += 1

    # Average losses.
    if not epoch_losses:
        return {}
    avg_losses = {key: sum(vals) / len(vals) for key, vals in epoch_losses.items()}
    logger.info(
        "Epoch %d train — loss: %.4f (color: %.4f, spatial: %.4f, "
        "strobe: %.4f, blackout: %.4f, temporal: %.4f)",
        epoch,
        avg_losses.get("loss/total", 0.0),
        avg_losses.get("loss/color", 0.0),
        avg_losses.get("loss/spatial", 0.0),
        avg_losses.get("loss/strobe", 0.0),
        avg_losses.get("loss/blackout", 0.0),
        avg_losses.get("loss/temporal", 0.0),
    )
    return avg_losses


@torch.no_grad()
def validate(
    model: LightingTransformer,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    epoch: int,
) -> dict[str, float]:
    """Validate the model on a held-out set.

    Args:
        model: The LightingTransformer model.
        loader: Validation DataLoader.
        device: Target device.
        epoch: Current epoch number (for logging).

    Returns:
        Dict of averaged loss components over the validation set.
    """
    model.eval()
    epoch_losses: dict[str, list[float]] = {}

    for batch in loader:
        music_features = batch["music_features"].to(device)
        genre_ids = batch["genre_ids"].to(device)
        segment_ids = batch["segment_ids"].to(device)
        color_targets = batch["color_targets"].to(device)
        spatial_targets = batch["spatial_targets"].to(device)
        effect_targets = batch["effect_targets"].to(device)
        confidence = batch["confidence"].to(device)

        color_pred, spatial_pred, effect_pred = model(
            music_features, genre_ids, segment_ids
        )

        _, loss_dict = compute_loss(
            color_pred,
            spatial_pred,
            effect_pred,
            color_targets,
            spatial_targets,
            effect_targets,
            confidence,
        )

        for key, value in loss_dict.items():
            epoch_losses.setdefault(key, []).append(value)

    avg_losses = {key: sum(vals) / len(vals) for key, vals in epoch_losses.items()}
    logger.info(
        "Epoch %d val   — loss: %.4f (color: %.4f, spatial: %.4f, "
        "strobe: %.4f, blackout: %.4f)",
        epoch,
        avg_losses.get("loss/total", 0.0),
        avg_losses.get("loss/color", 0.0),
        avg_losses.get("loss/spatial", 0.0),
        avg_losses.get("loss/strobe", 0.0),
        avg_losses.get("loss/blackout", 0.0),
    )
    return avg_losses


def save_checkpoint(
    model: LightingTransformer,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    val_loss: float,
    best_val_loss: float,
    checkpoint_dir: Path,
    is_best: bool = False,
) -> Path:
    """Save a training checkpoint with full state for resume.

    Args:
        model: The model to save.
        optimizer: Optimizer state to save.
        scheduler: LR scheduler state to save.
        epoch: Current epoch number.
        val_loss: Validation loss for this checkpoint.
        best_val_loss: Best validation loss seen so far.
        checkpoint_dir: Directory to save checkpoints.
        is_best: If True, also save as ``best_model.pt``.

    Returns:
        Path to the saved checkpoint file.
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "val_loss": val_loss,
        "best_val_loss": best_val_loss,
    }

    # Always save as "latest" for easy resume.
    latest_path = checkpoint_dir / "latest_checkpoint.pt"
    torch.save(state, latest_path)

    # Also save numbered checkpoint.
    path = checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pt"
    torch.save(state, path)
    logger.info("Saved checkpoint: %s (epoch %d)", path, epoch)

    if is_best:
        best_path = checkpoint_dir / "best_model.pt"
        torch.save(state, best_path)
        logger.info("Saved best model: %s (val_loss=%.4f)", best_path, val_loss)

    return path


def _find_latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    """Find the latest checkpoint to resume from.

    Args:
        checkpoint_dir: Directory containing checkpoint files.

    Returns:
        Path to latest checkpoint, or None if no checkpoints exist.
    """
    latest = checkpoint_dir / "latest_checkpoint.pt"
    if latest.exists():
        return latest
    return None


def train(
    data_dir: Path | None = None,
    checkpoint_dir: Path | None = None,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-2,
    num_workers: int = 4,
    use_wandb: bool = False,
    resume: bool = False,
    seed: int = 42,
) -> Path:
    """Full training pipeline with resume and graceful shutdown support.

    Saves a checkpoint after every epoch. On interrupt (Ctrl+C, SIGTERM,
    laptop lid close), finishes the current batch and saves before exiting.
    Use ``resume=True`` to pick up where you left off.

    Args:
        data_dir: Path to aligned Parquet training data.
        checkpoint_dir: Directory to save checkpoints.
        epochs: Total number of training epochs.
        batch_size: Batch size.
        learning_rate: AdamW learning rate.
        weight_decay: AdamW weight decay.
        num_workers: DataLoader workers.
        use_wandb: Enable wandb experiment tracking.
        resume: If True, resume from latest checkpoint in checkpoint_dir.
        seed: Random seed.

    Returns:
        Path to the best model checkpoint.
    """
    global _shutdown_flag
    _shutdown_flag = False

    if data_dir is None:
        data_dir = _DEFAULT_DATA_DIR
    if checkpoint_dir is None:
        checkpoint_dir = _DEFAULT_CHECKPOINT_DIR

    # Register signal handlers for graceful shutdown.
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # Reproducibility.
    torch.manual_seed(seed)

    # Device.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on device: %s", device)
    if device.type == "cuda":
        logger.info("GPU: %s", torch.cuda.get_device_name(0))

    # Data loaders.
    train_loader, val_loader, _ = create_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
    )

    if len(train_loader.dataset) == 0:  # type: ignore[arg-type]
        logger.error("No training data found in %s", data_dir)
        msg = f"No training data found in {data_dir}"
        raise RuntimeError(msg)

    # Model.
    model = LightingTransformer().to(device)
    param_count = model.count_parameters()
    logger.info("Model parameters: %d (~%.1fK)", param_count, param_count / 1000)

    # Optimizer.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    # Learning rate scheduler: cosine annealing.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=learning_rate * 0.01,
    )

    # Resume from checkpoint if requested.
    start_epoch = 1
    best_val_loss = float("inf")

    if resume:
        ckpt_path = _find_latest_checkpoint(checkpoint_dir)
        if ckpt_path is not None:
            logger.info("Resuming from checkpoint: %s", ckpt_path)
            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "scheduler_state_dict" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            best_val_loss = checkpoint.get("best_val_loss", checkpoint.get("val_loss", float("inf")))
            logger.info(
                "Resumed at epoch %d (best_val_loss=%.4f)",
                start_epoch,
                best_val_loss,
            )
        else:
            logger.info("No checkpoint found in %s — starting fresh", checkpoint_dir)

    if start_epoch > epochs:
        logger.info("Already completed %d/%d epochs — nothing to do", start_epoch - 1, epochs)
        return checkpoint_dir / "best_model.pt"

    logger.info(
        "Training epochs %d-%d (%d total, %d remaining)",
        start_epoch,
        epochs,
        epochs,
        epochs - start_epoch + 1,
    )

    # Optional wandb.
    wandb_run = None
    if use_wandb:
        try:
            import wandb

            wandb_run = wandb.init(
                project="lumina-lighting",
                config={
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "model_params": param_count,
                    "resumed_from_epoch": start_epoch - 1,
                },
                resume="allow",
            )
            logger.info("wandb initialized: %s", wandb_run.url)
        except ImportError:
            logger.warning("wandb not installed, skipping experiment tracking")
            use_wandb = False

    # Training loop.
    best_model_path = checkpoint_dir / "best_model.pt"

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.monotonic()

        train_losses = train_one_epoch(model, train_loader, optimizer, device, epoch)

        val_losses: dict[str, float] = {}
        if len(val_loader.dataset) > 0 and not _shutdown_flag:  # type: ignore[arg-type]
            val_losses = validate(model, val_loader, device, epoch)

        scheduler.step()

        elapsed = time.monotonic() - t0
        logger.info("Epoch %d completed in %.1fs", epoch, elapsed)

        # Track best model.
        val_loss = val_losses.get("loss/total", float("inf"))
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        # Save checkpoint every epoch for resume support.
        save_checkpoint(
            model, optimizer, scheduler, epoch, val_loss, best_val_loss,
            checkpoint_dir, is_best=is_best,
        )

        # wandb logging.
        if use_wandb and wandb_run is not None:
            import wandb

            log_data = {f"train/{k}": v for k, v in train_losses.items()}
            log_data.update({f"val/{k}": v for k, v in val_losses.items()})
            log_data["lr"] = scheduler.get_last_lr()[0]
            log_data["epoch_time_s"] = elapsed
            wandb.log(log_data, step=epoch)

        # Check for graceful shutdown.
        if _shutdown_flag:
            logger.info(
                "Shutdown requested — saved checkpoint at epoch %d. "
                "Re-run with --resume to continue.",
                epoch,
            )
            break

    if use_wandb and wandb_run is not None:
        import wandb

        wandb.finish()

    logger.info("Training complete. Best val loss: %.4f", best_val_loss)
    logger.info("Best model: %s", best_model_path)
    logger.info("Latest checkpoint: %s", checkpoint_dir / "latest_checkpoint.pt")
    return best_model_path


# ── CLI entry point ──────────────────────────────────────────────────


def main() -> None:
    """CLI entry point for training."""
    parser = argparse.ArgumentParser(
        description="Train LUMINA LightingTransformer",
        epilog=(
            "Resume example: python -m lumina.ml.model.train --resume\n"
            "Press Ctrl+C to gracefully stop — checkpoint is saved automatically."
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=_DEFAULT_DATA_DIR,
        help="Path to aligned Parquet training data (default: data/features/aligned)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=_DEFAULT_CHECKPOINT_DIR,
        help="Directory to save checkpoints (default: data/models/checkpoints)",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Total number of epochs (default: 50)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument("--wandb", action="store_true", help="Enable wandb experiment tracking")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from latest checkpoint in checkpoint-dir",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    train(
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        use_wandb=args.wandb,
        resume=args.resume,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
