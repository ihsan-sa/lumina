"""Training loop for the LightingTransformer model.

Multi-task training with genre-aware weighting:
  - Huber loss for color head (smooth L1, robust to outliers)
  - MSE loss for spatial head
  - BCE loss for strobe/blackout predictions in effect head
  - Temporal consistency loss to penalize frame-to-frame jitter

Optimizer: AdamW, lr=1e-4, batch_size=32, ~50 epochs.
Checkpoints saved to ``data/models/checkpoints/``.
Optional wandb logging when ``--wandb`` flag is passed.

Usage:
    python -m lumina.ml.model.train [--data-dir PATH] [--epochs 50] [--wandb]
"""

from __future__ import annotations

import argparse
import logging
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
    epoch: int,
    val_loss: float,
    checkpoint_dir: Path,
    is_best: bool = False,
) -> Path:
    """Save a training checkpoint.

    Args:
        model: The model to save.
        optimizer: Optimizer state to save.
        epoch: Current epoch number.
        val_loss: Validation loss for this checkpoint.
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
        "val_loss": val_loss,
    }

    path = checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pt"
    torch.save(state, path)
    logger.info("Saved checkpoint: %s", path)

    if is_best:
        best_path = checkpoint_dir / "best_model.pt"
        torch.save(state, best_path)
        logger.info("Saved best model: %s (val_loss=%.4f)", best_path, val_loss)

    return path


def train(
    data_dir: Path | None = None,
    checkpoint_dir: Path | None = None,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-2,
    num_workers: int = 4,
    use_wandb: bool = False,
    seed: int = 42,
) -> Path:
    """Full training pipeline.

    Args:
        data_dir: Path to aligned Parquet training data.
        checkpoint_dir: Directory to save checkpoints.
        epochs: Number of training epochs.
        batch_size: Batch size.
        learning_rate: AdamW learning rate.
        weight_decay: AdamW weight decay.
        num_workers: DataLoader workers.
        use_wandb: Enable wandb experiment tracking.
        seed: Random seed.

    Returns:
        Path to the best model checkpoint.
    """
    if data_dir is None:
        data_dir = _DEFAULT_DATA_DIR
    if checkpoint_dir is None:
        checkpoint_dir = _DEFAULT_CHECKPOINT_DIR

    # Reproducibility.
    torch.manual_seed(seed)

    # Device.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on device: %s", device)

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
                },
            )
            logger.info("wandb initialized: %s", wandb_run.url)
        except ImportError:
            logger.warning("wandb not installed, skipping experiment tracking")
            use_wandb = False

    # Training loop.
    best_val_loss = float("inf")
    best_model_path = checkpoint_dir / "best_model.pt"

    for epoch in range(1, epochs + 1):
        t0 = time.monotonic()

        train_losses = train_one_epoch(model, train_loader, optimizer, device, epoch)

        val_losses: dict[str, float] = {}
        if len(val_loader.dataset) > 0:  # type: ignore[arg-type]
            val_losses = validate(model, val_loader, device, epoch)

        scheduler.step()

        elapsed = time.monotonic() - t0
        logger.info("Epoch %d completed in %.1fs", epoch, elapsed)

        # Track best model.
        val_loss = val_losses.get("loss/total", float("inf"))
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        # Save checkpoint every 10 epochs and on best.
        if epoch % 10 == 0 or is_best or epoch == epochs:
            save_checkpoint(
                model, optimizer, epoch, val_loss, checkpoint_dir, is_best=is_best
            )

        # wandb logging.
        if use_wandb and wandb_run is not None:
            import wandb

            log_data = {f"train/{k}": v for k, v in train_losses.items()}
            log_data.update({f"val/{k}": v for k, v in val_losses.items()})
            log_data["lr"] = scheduler.get_last_lr()[0]
            log_data["epoch_time_s"] = elapsed
            wandb.log(log_data, step=epoch)

    if use_wandb and wandb_run is not None:
        import wandb

        wandb.finish()

    logger.info("Training complete. Best val loss: %.4f", best_val_loss)
    return best_model_path


# ── CLI entry point ──────────────────────────────────────────────────


def main() -> None:
    """CLI entry point for training."""
    parser = argparse.ArgumentParser(description="Train LUMINA LightingTransformer")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=_DEFAULT_DATA_DIR,
        help="Path to aligned Parquet training data",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=_DEFAULT_CHECKPOINT_DIR,
        help="Directory to save checkpoints",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
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
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
