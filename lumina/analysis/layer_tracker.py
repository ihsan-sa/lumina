"""Layer tracking: count active sound layers per frame from demucs stems.

Detects how many instrument layers (drums, bass, vocals, other) are active
at each analysis frame. This drives fixture count decisions — sparse sections
with 1 stem use fewer fixtures, dense sections with all 4 stems use all.

Algorithm:
1. Compute per-stem RMS envelope with hop=512.
2. Stem is "active" if RMS > 15% of its own peak.
3. EMA smoothing prevents flicker from brief gaps.
4. Detect layer_change events when stems cross threshold.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from lumina.audio.source_separator import StemSet

logger = logging.getLogger(__name__)

# Analysis constants
_HOP_SIZE = 512
_ACTIVE_THRESHOLD = 0.15  # 15% of stem peak RMS
_EMA_ALPHA = 0.05  # Smoothing factor (lower = more smoothing)
_STEM_NAMES = ("drums", "bass", "vocals", "other")


@dataclass(slots=True)
class LayerFrame:
    """Per-frame layer activity snapshot.

    Args:
        active_count: Number of active stems (0-4).
        layer_mask: Per-stem activity level (0.0-1.0).
        layer_change: Event string when a stem crosses threshold,
            e.g. "add_drums", "drop_vocals", or None.
    """

    active_count: int
    layer_mask: dict[str, float]
    layer_change: str | None


def _rms_envelope(audio: np.ndarray, hop: int) -> np.ndarray:
    """Compute RMS envelope of an audio signal.

    Args:
        audio: Mono float32 audio.
        hop: Hop size in samples.

    Returns:
        RMS values, one per frame.
    """
    n_frames = len(audio) // hop
    if n_frames == 0:
        return np.array([0.0], dtype=np.float32)

    # Reshape into frames and compute RMS
    trimmed = audio[: n_frames * hop]
    frames = trimmed.reshape(n_frames, hop)
    return np.sqrt(np.mean(frames ** 2, axis=1)).astype(np.float32)


class LayerTracker:
    """Track active sound layers from demucs stems.

    Args:
        sr: Sample rate.
        hop: Hop size for RMS envelope.
    """

    def __init__(self, sr: int = 44100, hop: int = _HOP_SIZE) -> None:
        self._sr = sr
        self._hop = hop

    def analyze(self, stems: StemSet) -> list[LayerFrame]:
        """Analyze stems to produce per-frame layer activity.

        Args:
            stems: Demucs stem separation result.

        Returns:
            List of LayerFrame, one per analysis hop.
        """
        stem_arrays = {
            "drums": stems.drums,
            "bass": stems.bass,
            "vocals": stems.vocals,
            "other": stems.other,
        }

        # Compute RMS envelopes
        envelopes: dict[str, np.ndarray] = {}
        for name, audio in stem_arrays.items():
            envelopes[name] = _rms_envelope(audio, self._hop)

        # Align to shortest envelope
        n_frames = min(len(env) for env in envelopes.values())
        if n_frames == 0:
            return [LayerFrame(active_count=0, layer_mask={}, layer_change=None)]

        # Compute per-stem peak for threshold
        peaks: dict[str, float] = {}
        for name, env in envelopes.items():
            peak = float(np.max(env[:n_frames]))
            peaks[name] = max(peak, 1e-8)  # avoid division by zero

        # Build smoothed activity and detect layer changes
        frames: list[LayerFrame] = []
        smoothed: dict[str, float] = {name: 0.0 for name in _STEM_NAMES}
        prev_active: dict[str, bool] = {name: False for name in _STEM_NAMES}

        for i in range(n_frames):
            layer_mask: dict[str, float] = {}
            active_count = 0
            change: str | None = None

            for name in _STEM_NAMES:
                raw = float(envelopes[name][i]) / peaks[name]
                # EMA smoothing
                smoothed[name] = smoothed[name] * (1 - _EMA_ALPHA) + raw * _EMA_ALPHA
                layer_mask[name] = min(1.0, smoothed[name])

                is_active = smoothed[name] > _ACTIVE_THRESHOLD
                if is_active:
                    active_count += 1

                # Detect transitions
                if is_active and not prev_active[name]:
                    change = f"add_{name}"
                elif not is_active and prev_active[name]:
                    change = f"drop_{name}"
                prev_active[name] = is_active

            frames.append(LayerFrame(
                active_count=active_count,
                layer_mask=layer_mask,
                layer_change=change,
            ))

        return frames

    def resample_to_fps(
        self, frames: list[LayerFrame], target_n: int, fps: int
    ) -> list[LayerFrame]:
        """Resample layer frames to match the target frame count at fps.

        Args:
            frames: Layer frames from analyze().
            target_n: Target number of output frames.
            fps: Target frame rate.

        Returns:
            Resampled list of LayerFrame at the target frame count.
        """
        if not frames:
            return [
                LayerFrame(active_count=0, layer_mask={}, layer_change=None)
            ] * target_n

        n_src = len(frames)
        # Compute time ratio: analysis frames per second
        analysis_fps = self._sr / self._hop

        result: list[LayerFrame] = []
        for i in range(target_n):
            t = i / fps  # time in seconds
            src_idx = int(t * analysis_fps)
            src_idx = min(src_idx, n_src - 1)
            result.append(frames[src_idx])
        return result
