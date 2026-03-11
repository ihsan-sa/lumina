"""PyTorch Dataset for loading aligned audio-lighting training pairs.

Reads Parquet files from ``data/features/aligned/`` where each file
contains one video's worth of time-aligned MusicState + VideoLightingFrame
features at 10fps.  Returns sequences of 40 frames (4-second context
window) suitable for the LightingTransformer.

Supports genre-stratified splits: 80% train, 10% val, 10% test.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from lumina.ml.model.architecture import (
    CONTEXT_WINDOW,
    GENRE_LABELS,
    SEGMENT_LABELS,
    genre_to_index,
    segment_to_index,
)

logger = logging.getLogger(__name__)

# Default data directory (relative to repo root).
_DEFAULT_DATA_DIR = Path("data/features/aligned")

# Column groups expected in Parquet files.
# MusicState float features (order must match architecture.NUM_MUSIC_FEATURES).
MUSIC_FEATURE_COLS: list[str] = [
    "energy",
    "beat_phase",
    "bar_phase",
    "spectral_centroid",
    "sub_bass_energy",
    "vocal_energy",
    "drop_probability",
    "is_beat",
    "is_downbeat",
    "energy_derivative",
    "bpm",
    "layer_count",
    "notes_per_beat",
    "note_pattern_phase",
    "headroom",
    "motif_repetition",
]

# Lighting target columns (color head targets).
COLOR_TARGET_COLS: list[str] = [
    "dominant_hue",
    "dominant_saturation",
    "secondary_hue",
    "color_diversity",
    "color_temperature",
    "overall_brightness",
]

# Spatial head targets.
SPATIAL_TARGET_COLS: list[str] = [
    "left_brightness",
    "center_brightness",
    "right_brightness",
    "spatial_symmetry",
    "brightness_variance",
]

# Effect head targets.
EFFECT_TARGET_COLS: list[str] = [
    "is_strobe",
    "is_blackout",
    "brightness_delta",
]

# Categorical columns.
GENRE_COL = "genre_profile"
SEGMENT_COL = "segment"


def _normalize_hue(hue: float) -> float:
    """Normalize hue from 0-360 to 0-1 range.

    Args:
        hue: Hue value in degrees (0-360).

    Returns:
        Normalized hue in 0-1 range.
    """
    return (hue % 360.0) / 360.0


class LightingDataset(Dataset[dict[str, torch.Tensor]]):
    """PyTorch Dataset that yields context-windowed training sequences.

    Each item is a dict containing:
      - ``music_features``: (CONTEXT_WINDOW, NUM_MUSIC_FEATURES) float tensor
      - ``genre_ids``: (CONTEXT_WINDOW,) long tensor
      - ``segment_ids``: (CONTEXT_WINDOW,) long tensor
      - ``color_targets``: (CONTEXT_WINDOW, 6) float tensor
      - ``spatial_targets``: (CONTEXT_WINDOW, 5) float tensor
      - ``effect_targets``: (CONTEXT_WINDOW, 3) float tensor
      - ``confidence``: (CONTEXT_WINDOW,) float tensor (scene confidence)

    Args:
        data_dir: Directory containing aligned Parquet files.
        split: One of "train", "val", "test".
        context_window: Number of frames per sequence.
        stride: Stride between consecutive sequences within a video.
        seed: Random seed for reproducible splits.
    """

    def __init__(
        self,
        data_dir: Path | None = None,
        split: str = "train",
        context_window: int = CONTEXT_WINDOW,
        stride: int = 10,
        seed: int = 42,
    ) -> None:
        if data_dir is None:
            data_dir = _DEFAULT_DATA_DIR

        self._data_dir = Path(data_dir)
        self._split = split
        self._context_window = context_window
        self._stride = stride

        # Discover all Parquet files.
        all_files = sorted(self._data_dir.glob("*.parquet"))
        if not all_files:
            logger.warning("No Parquet files found in %s", self._data_dir)

        # Genre-stratified split.
        split_files = self._stratified_split(all_files, split, seed)
        logger.info(
            "LightingDataset split=%s: %d/%d files",
            split,
            len(split_files),
            len(all_files),
        )

        # Load and index all sequences.
        self._sequences: list[tuple[int, int]] = []  # (video_idx, start_frame)
        self._dataframes: list[pd.DataFrame] = []

        for file_path in split_files:
            df = self._load_parquet(file_path)
            if df is None or len(df) < context_window:
                continue
            video_idx = len(self._dataframes)
            self._dataframes.append(df)

            # Create sliding window indices.
            num_sequences = (len(df) - context_window) // stride + 1
            for i in range(num_sequences):
                self._sequences.append((video_idx, i * stride))

        logger.info(
            "LightingDataset: %d sequences from %d videos",
            len(self._sequences),
            len(self._dataframes),
        )

    def _stratified_split(
        self,
        files: list[Path],
        split: str,
        seed: int,
    ) -> list[Path]:
        """Split files by genre to ensure balanced representation.

        Groups files by genre prefix (first directory component or filename
        pattern), then assigns 80/10/10 within each genre group.

        Args:
            files: All available Parquet file paths.
            split: One of "train", "val", "test".
            seed: Random seed.

        Returns:
            List of file paths for the requested split.
        """
        rng = np.random.default_rng(seed)

        # Group by genre: try to read genre from file metadata.
        genre_groups: dict[str, list[Path]] = {}
        for f in files:
            # Attempt to extract genre from parquet metadata or filename.
            genre = self._infer_genre(f)
            genre_groups.setdefault(genre, []).append(f)

        result: list[Path] = []
        for genre, group_files in sorted(genre_groups.items()):
            indices = rng.permutation(len(group_files))
            n = len(group_files)
            train_end = int(n * 0.8)
            val_end = train_end + int(n * 0.1)

            if split == "train":
                selected = indices[:train_end]
            elif split == "val":
                selected = indices[train_end:val_end]
            else:  # test
                selected = indices[val_end:]

            # Ensure at least one file per genre in each split if possible.
            if len(selected) == 0 and n > 0:
                selected = indices[:1]

            for idx in selected:
                result.append(group_files[idx])

        return result

    def _infer_genre(self, file_path: Path) -> str:
        """Infer genre label from filename or parent directory.

        Looks for known genre labels in the filename. Falls back to
        "unknown" if no genre is detected.

        Args:
            file_path: Path to a Parquet file.

        Returns:
            Genre label string.
        """
        name = file_path.stem.lower()
        for genre in GENRE_LABELS:
            if genre in name:
                return genre
        # Check parent directory name.
        parent = file_path.parent.name.lower()
        for genre in GENRE_LABELS:
            if genre in parent:
                return genre
        return "unknown"

    def _load_parquet(self, file_path: Path) -> pd.DataFrame | None:
        """Load and validate a single Parquet file.

        Ensures all required columns are present and applies normalization.

        Args:
            file_path: Path to the Parquet file.

        Returns:
            DataFrame with validated columns, or None on error.
        """
        try:
            df = pd.read_parquet(file_path)
        except Exception:
            logger.exception("Failed to load %s", file_path)
            return None

        # Check required columns exist.
        required = set(
            MUSIC_FEATURE_COLS
            + COLOR_TARGET_COLS
            + SPATIAL_TARGET_COLS
            + EFFECT_TARGET_COLS
            + [GENRE_COL, SEGMENT_COL]
        )
        missing = required - set(df.columns)
        if missing:
            logger.warning("Skipping %s — missing columns: %s", file_path, missing)
            return None

        # Normalize hue columns from 0-360 to 0-1.
        for col in ("dominant_hue", "secondary_hue"):
            if col in df.columns:
                df[col] = df[col].apply(_normalize_hue)

        # Normalize BPM to 0-1 range (assume 60-200 BPM range).
        if "bpm" in df.columns:
            df["bpm"] = (df["bpm"].clip(60.0, 200.0) - 60.0) / 140.0

        # Convert boolean columns to float.
        for col in ("is_beat", "is_downbeat", "is_strobe", "is_blackout"):
            if col in df.columns:
                df[col] = df[col].astype(float)

        # Fill NaN with 0.
        df = df.fillna(0.0)

        return df

    def __len__(self) -> int:
        return len(self._sequences)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get a single training sequence.

        Args:
            idx: Sequence index.

        Returns:
            Dict with tensors for model input and targets.
        """
        video_idx, start = self._sequences[idx]
        df = self._dataframes[video_idx]
        window = df.iloc[start : start + self._context_window]

        # Music features.
        music_features = torch.tensor(
            window[MUSIC_FEATURE_COLS].values, dtype=torch.float32
        )

        # Genre IDs — convert string labels to indices.
        genre_ids = torch.tensor(
            [genre_to_index(g) for g in window[GENRE_COL]],
            dtype=torch.long,
        )

        # Segment IDs — convert string labels to indices.
        segment_ids = torch.tensor(
            [segment_to_index(s) for s in window[SEGMENT_COL]],
            dtype=torch.long,
        )

        # Targets.
        color_targets = torch.tensor(
            window[COLOR_TARGET_COLS].values, dtype=torch.float32
        )
        spatial_targets = torch.tensor(
            window[SPATIAL_TARGET_COLS].values, dtype=torch.float32
        )
        effect_targets = torch.tensor(
            window[EFFECT_TARGET_COLS].values, dtype=torch.float32
        )

        # Scene confidence for weighting (if available).
        if "scene_confidence" in window.columns:
            confidence = torch.tensor(
                window["scene_confidence"].values, dtype=torch.float32
            )
        else:
            confidence = torch.ones(self._context_window, dtype=torch.float32)

        return {
            "music_features": music_features,
            "genre_ids": genre_ids,
            "segment_ids": segment_ids,
            "color_targets": color_targets,
            "spatial_targets": spatial_targets,
            "effect_targets": effect_targets,
            "confidence": confidence,
        }


def create_dataloaders(
    data_dir: Path | None = None,
    batch_size: int = 32,
    num_workers: int = 4,
    seed: int = 42,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create train, validation, and test DataLoaders.

    Args:
        data_dir: Directory containing aligned Parquet files.
        batch_size: Batch size for training.
        num_workers: Number of data loading workers.
        seed: Random seed for reproducible splits.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    train_ds = LightingDataset(data_dir, split="train", seed=seed)
    val_ds = LightingDataset(data_dir, split="val", seed=seed)
    test_ds = LightingDataset(data_dir, split="test", seed=seed)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
