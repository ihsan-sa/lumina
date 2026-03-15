"""Real-time inference wrapper for the LightingTransformer.

Maintains a sliding window of MusicState context and runs the model
at 10fps internally, upscaling to 60fps via linear interpolation for
smooth fixture output.

Usage:
    engine = LightingInferenceEngine.from_checkpoint("data/models/checkpoints/best_model.pt")
    intent = engine.predict(music_state)
"""

from __future__ import annotations

import logging
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from lumina.audio.models import MusicState
from lumina.ml.model.architecture import (
    ANALYSIS_FPS,
    CONTEXT_WINDOW,
    LightingIntent,
    LightingTransformer,
    genre_to_index,
    segment_to_index,
)

logger = logging.getLogger(__name__)

# Target output rate.
OUTPUT_FPS = 60

# Interval between model evaluations (10fps -> 100ms).
_MODEL_INTERVAL_S = 1.0 / ANALYSIS_FPS

# Ratio for upscaling 10fps -> 60fps.
_UPSAMPLE_RATIO = OUTPUT_FPS // ANALYSIS_FPS  # 6


def _extract_features(state: MusicState) -> list[float]:
    """Extract the ordered float feature vector from a MusicState.

    Feature order matches MUSIC_FEATURE_COLS in dataset.py.

    Args:
        state: Current MusicState frame.

    Returns:
        List of float features matching the expected column order.
    """
    return [
        state.energy,
        state.beat_phase,
        state.bar_phase,
        state.spectral_centroid,
        state.sub_bass_energy,
        state.vocal_energy,
        state.drop_probability,
        float(state.is_beat),
        float(state.is_downbeat),
        state.energy_derivative,
        # Normalize BPM to 0-1 (same as dataset normalization).
        max(0.0, min(1.0, (state.bpm - 60.0) / 140.0)),
    ]


def _dominant_genre(genre_weights: dict[str, float]) -> str:
    """Get the highest-weighted genre from genre_weights.

    Args:
        genre_weights: Dict mapping genre names to weights.

    Returns:
        Genre name with highest weight, or "generic" if empty.
    """
    if not genre_weights:
        return "generic"
    return max(genre_weights, key=genre_weights.get)  # type: ignore[arg-type]


def _raw_outputs_to_intent(
    color: np.ndarray,
    spatial: np.ndarray,
    effect: np.ndarray,
) -> LightingIntent:
    """Convert raw model outputs to a LightingIntent dataclass.

    Args:
        color: Array of shape (6,) — model color head output.
        spatial: Array of shape (5,) — model spatial head output.
        effect: Array of shape (3,) — model effect head output.

    Returns:
        LightingIntent instance.
    """
    # Color head: hue, saturation, secondary_hue, diversity, temperature, brightness.
    dominant_hue = float(color[0]) * 360.0  # Scale back to 0-360.
    dominant_sat = float(color[1])
    secondary_hue = float(color[2]) * 360.0
    color_diversity = float(color[3])
    overall_brightness = float(color[5])

    # Spatial head: left, center, right, symmetry, variance.
    left = float(spatial[0])
    center = float(spatial[1])
    right = float(spatial[2])
    symmetry = float(spatial[3])

    # Effect head: strobe_prob, blackout_prob, brightness_delta.
    strobe_prob = float(effect[0])
    blackout_prob = float(effect[1])

    return LightingIntent(
        dominant_color=(dominant_hue, dominant_sat, overall_brightness),
        secondary_color=(secondary_hue, dominant_sat * 0.8, overall_brightness * 0.7),
        overall_brightness=overall_brightness,
        color_diversity=color_diversity,
        spatial_distribution=(left, center, right),
        spatial_symmetry=symmetry,
        strobe_active=strobe_prob > 0.5,
        strobe_intensity=strobe_prob,
        blackout=blackout_prob > 0.5,
    )


class LightingInferenceEngine:
    """Real-time inference engine for the LightingTransformer.

    Maintains a sliding window of context frames and runs the model
    at 10fps internally.  Between model evaluations, the engine
    interpolates between the last two predictions to produce smooth
    60fps output.

    Args:
        model: Trained LightingTransformer model.
        device: Torch device to run inference on.
    """

    def __init__(
        self,
        model: LightingTransformer,
        device: torch.device | None = None,
    ) -> None:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._model = model.to(device)
        self._model.eval()
        self._device = device

        # Sliding context window.
        self._feature_buffer: deque[list[float]] = deque(maxlen=CONTEXT_WINDOW)
        self._genre_buffer: deque[int] = deque(maxlen=CONTEXT_WINDOW)
        self._segment_buffer: deque[int] = deque(maxlen=CONTEXT_WINDOW)

        # Interpolation state.
        self._prev_intent: LightingIntent | None = None
        self._curr_intent: LightingIntent | None = None
        self._last_model_time: float = 0.0
        self._frame_counter: int = 0

        # Raw output buffers for interpolation.
        self._prev_raw: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self._curr_raw: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path | str,
        device: torch.device | None = None,
    ) -> LightingInferenceEngine:
        """Load a trained model from a checkpoint file.

        Args:
            checkpoint_path: Path to a ``.pt`` checkpoint file.
            device: Target device. Auto-detects CUDA if available.

        Returns:
            Initialized LightingInferenceEngine ready for prediction.
        """
        checkpoint_path = Path(checkpoint_path)
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = LightingTransformer()
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(
            "Loaded model from %s (epoch %d, val_loss %.4f)",
            checkpoint_path,
            checkpoint.get("epoch", -1),
            checkpoint.get("val_loss", -1.0),
        )

        return cls(model=model, device=device)

    def predict(self, state: MusicState) -> LightingIntent:
        """Generate a LightingIntent prediction for the current frame.

        Called at 60fps by the lighting engine.  Internally runs the
        transformer at 10fps and interpolates between predictions for
        smooth output.

        Args:
            state: Current MusicState from the audio pipeline.

        Returns:
            LightingIntent for this frame.
        """
        self._frame_counter += 1
        now = time.monotonic()

        # Extract features and push to context buffers.
        features = _extract_features(state)
        genre_id = genre_to_index(_dominant_genre(state.genre_weights))
        segment_id = segment_to_index(state.segment)

        self._feature_buffer.append(features)
        self._genre_buffer.append(genre_id)
        self._segment_buffer.append(segment_id)

        # Run model at 10fps (every 6th frame at 60fps).
        should_run_model = (
            now - self._last_model_time >= _MODEL_INTERVAL_S
            or self._curr_intent is None
        )

        if should_run_model and len(self._feature_buffer) > 0:
            self._run_model()
            self._last_model_time = now
            self._frame_counter = 0

        # Interpolate between previous and current predictions.
        return self._interpolate()

    def _run_model(self) -> None:
        """Execute the transformer on the current context window."""
        seq_len = len(self._feature_buffer)

        # Pad to CONTEXT_WINDOW if we don't have enough history.
        features_list = list(self._feature_buffer)
        genre_list = list(self._genre_buffer)
        segment_list = list(self._segment_buffer)

        if seq_len < CONTEXT_WINDOW:
            pad_count = CONTEXT_WINDOW - seq_len
            features_list = [features_list[0]] * pad_count + features_list
            genre_list = [genre_list[0]] * pad_count + genre_list
            segment_list = [segment_list[0]] * pad_count + segment_list

        # Build tensors: (1, CONTEXT_WINDOW, ...).
        music_tensor = torch.tensor(
            [features_list], dtype=torch.float32, device=self._device
        )
        genre_tensor = torch.tensor(
            [genre_list], dtype=torch.long, device=self._device
        )
        segment_tensor = torch.tensor(
            [segment_list], dtype=torch.long, device=self._device
        )

        with torch.no_grad():
            color_out, spatial_out, effect_out = self._model(
                music_tensor, genre_tensor, segment_tensor
            )

        # Take the last frame's prediction (most recent context).
        color_np = color_out[0, -1].cpu().numpy()
        spatial_np = spatial_out[0, -1].cpu().numpy()
        effect_np = effect_out[0, -1].cpu().numpy()

        # Shift prediction buffers.
        self._prev_raw = self._curr_raw
        self._curr_raw = (color_np, spatial_np, effect_np)

        self._prev_intent = self._curr_intent
        self._curr_intent = _raw_outputs_to_intent(color_np, spatial_np, effect_np)

    def _interpolate(self) -> LightingIntent:
        """Interpolate between previous and current model predictions.

        Uses linear interpolation based on how many 60fps frames have
        elapsed since the last model evaluation.

        Returns:
            Interpolated LightingIntent.
        """
        if self._curr_intent is None:
            # No prediction yet — return a neutral default.
            return LightingIntent(
                dominant_color=(0.0, 0.0, 0.0),
                secondary_color=(0.0, 0.0, 0.0),
                overall_brightness=0.0,
                color_diversity=0.0,
                spatial_distribution=(0.0, 0.0, 0.0),
                spatial_symmetry=1.0,
                strobe_active=False,
                strobe_intensity=0.0,
                blackout=False,
            )

        if self._prev_raw is None or self._curr_raw is None:
            return self._curr_intent

        # Interpolation factor: 0.0 at model eval, 1.0 just before next eval.
        t = min(1.0, self._frame_counter / _UPSAMPLE_RATIO)

        prev_color, prev_spatial, prev_effect = self._prev_raw
        curr_color, curr_spatial, curr_effect = self._curr_raw

        # Linearly interpolate continuous values.
        color_interp = prev_color * (1.0 - t) + curr_color * t
        spatial_interp = prev_spatial * (1.0 - t) + curr_spatial * t
        effect_interp = prev_effect * (1.0 - t) + curr_effect * t

        return _raw_outputs_to_intent(color_interp, spatial_interp, effect_interp)

    @property
    def is_ready(self) -> bool:
        """Whether the engine has produced at least one prediction."""
        return self._curr_intent is not None

    def reset(self) -> None:
        """Clear all state buffers. Call when switching songs."""
        self._feature_buffer.clear()
        self._genre_buffer.clear()
        self._segment_buffer.clear()
        self._prev_intent = None
        self._curr_intent = None
        self._prev_raw = None
        self._curr_raw = None
        self._last_model_time = 0.0
        self._frame_counter = 0
