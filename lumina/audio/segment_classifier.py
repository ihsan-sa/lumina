"""Song segment classification: verse, chorus, drop, breakdown, intro, outro, bridge.

Classifies the current segment of a song using energy contours, spectral
features, onset patterns, and structural cues. Works in both streaming
mode (updating every few seconds) and offline mode (full-song analysis).

Classification approach (rule-based, Phase 1):

Each segment type has a characteristic feature signature:
- **Intro/Outro:** Low-to-moderate energy, sparse onsets, gradual change.
- **Verse:** Moderate energy, steady rhythm, vocals present, sub-bass present.
- **Chorus:** High energy, dense onsets, vocals present, bright spectrum.
- **Drop:** Highest energy spike, dense kicks/sub-bass, spectral brightness.
- **Breakdown:** Energy dip, sparse onsets, reduced sub-bass, atmospheric.
- **Bridge:** Moderate energy, different from verse (lower vocal or spectral shift).

The classifier maintains a rolling window of feature statistics and
computes similarity scores against each segment prototype. In offline
mode, it applies temporal smoothing to avoid rapid segment switching.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)

# Segment labels matching MusicState.segment values
SEGMENT_LABELS = ("intro", "verse", "chorus", "drop", "breakdown", "bridge", "outro")

# Minimum segment duration in seconds (prevents rapid flickering)
_MIN_SEGMENT_DURATION = 4.0


class SegmentType(Enum):
    """Song segment types."""

    INTRO = "intro"
    VERSE = "verse"
    CHORUS = "chorus"
    DROP = "drop"
    BREAKDOWN = "breakdown"
    BRIDGE = "bridge"
    OUTRO = "outro"


@dataclass(slots=True)
class SegmentFrame:
    """Segment classification for a single time frame.

    Args:
        segment: Current segment label string.
        confidence: 0.0-1.0 confidence in the classification.
        scores: Per-segment similarity scores (dict of label -> score).
    """

    segment: str
    confidence: float
    scores: dict[str, float]


# ── Segment prototypes ───────────────────────────────────────────────
# Each prototype defines the expected feature ranges for a segment type.
# Values are (center, tolerance) — closer to center = higher score.

_PROTOTYPES: dict[str, dict[str, tuple[float, float]]] = {
    "intro": {
        "energy": (0.15, 0.20),
        "energy_derivative": (0.01, 0.02),  # gently rising
        "spectral_centroid_norm": (0.25, 0.25),
        "onset_density": (0.10, 0.15),
        "vocal_energy": (0.05, 0.15),
        "sub_bass_energy": (0.10, 0.15),
    },
    "verse": {
        "energy": (0.45, 0.20),
        "energy_derivative": (0.0, 0.01),  # steady
        "spectral_centroid_norm": (0.35, 0.20),
        "onset_density": (0.35, 0.20),
        "vocal_energy": (0.55, 0.25),
        "sub_bass_energy": (0.35, 0.20),
    },
    "chorus": {
        "energy": (0.70, 0.20),
        "energy_derivative": (0.0, 0.02),
        "spectral_centroid_norm": (0.55, 0.20),
        "onset_density": (0.55, 0.20),
        "vocal_energy": (0.65, 0.25),
        "sub_bass_energy": (0.50, 0.20),
    },
    "drop": {
        "energy": (0.90, 0.15),
        "energy_derivative": (0.0, 0.03),
        "spectral_centroid_norm": (0.60, 0.25),
        "onset_density": (0.70, 0.25),
        "vocal_energy": (0.15, 0.25),  # vocals usually absent in drop
        "sub_bass_energy": (0.75, 0.20),
    },
    "breakdown": {
        "energy": (0.20, 0.15),
        "energy_derivative": (-0.01, 0.02),  # falling or low
        "spectral_centroid_norm": (0.30, 0.25),
        "onset_density": (0.10, 0.15),
        "vocal_energy": (0.20, 0.25),
        "sub_bass_energy": (0.10, 0.15),
    },
    "bridge": {
        "energy": (0.40, 0.20),
        "energy_derivative": (0.0, 0.02),
        "spectral_centroid_norm": (0.45, 0.25),
        "onset_density": (0.25, 0.20),
        "vocal_energy": (0.35, 0.25),
        "sub_bass_energy": (0.25, 0.20),
    },
    "outro": {
        "energy": (0.20, 0.25),
        "energy_derivative": (-0.01, 0.02),  # falling
        "spectral_centroid_norm": (0.25, 0.25),
        "onset_density": (0.10, 0.15),
        "vocal_energy": (0.10, 0.20),
        "sub_bass_energy": (0.15, 0.20),
    },
}


class SegmentClassifier:
    """Real-time song segment classifier.

    Classifies audio into segment types using a rolling window of
    audio features compared against segment prototypes. Enforces
    minimum segment duration to prevent rapid switching.

    Args:
        fps: Input frame rate.
        window_seconds: Rolling window size for feature averaging.
        min_segment_seconds: Minimum duration before allowing segment change.
        position_weight: How much track position (0-1) influences intro/outro.
    """

    def __init__(
        self,
        fps: int = 60,
        window_seconds: float = 4.0,
        min_segment_seconds: float = _MIN_SEGMENT_DURATION,
        position_weight: float = 0.3,
    ) -> None:
        self._fps = fps
        self._window_size = max(1, int(fps * window_seconds))
        self._min_segment_frames = int(fps * min_segment_seconds)
        self._position_weight = position_weight

        # Spectral centroid normalization constant (Hz → 0-1 range)
        # 10 kHz maps to ~1.0
        self._centroid_norm_hz = 10000.0

        self.reset()

    def reset(self) -> None:
        """Reset all internal state."""
        self._energy_history: deque[float] = deque(maxlen=self._window_size)
        self._deriv_history: deque[float] = deque(maxlen=self._window_size)
        self._centroid_history: deque[float] = deque(maxlen=self._window_size)
        self._onset_history: deque[float] = deque(maxlen=self._window_size)
        self._vocal_history: deque[float] = deque(maxlen=self._window_size)
        self._bass_history: deque[float] = deque(maxlen=self._window_size)

        self._current_segment = "verse"
        self._frames_in_segment = 0
        self._total_frames = 0
        self._track_duration_frames: int | None = None

    def set_track_duration(self, duration_seconds: float) -> None:
        """Set total track duration for position-aware classification.

        Knowing the track length helps detect intro/outro via position.

        Args:
            duration_seconds: Total track duration in seconds.
        """
        self._track_duration_frames = int(duration_seconds * self._fps)

    # ── Public API ────────────────────────────────────────────────

    def process_frame(
        self,
        energy: float,
        energy_derivative: float,
        spectral_centroid: float,
        sub_bass_energy: float,
        vocal_energy: float,
        has_onset: bool,
    ) -> SegmentFrame:
        """Process a single frame and return segment classification.

        Args:
            energy: 0.0-1.0 overall energy.
            energy_derivative: Rate of energy change.
            spectral_centroid: Brightness in Hz.
            sub_bass_energy: 0.0-1.0 sub-bass level.
            vocal_energy: 0.0-1.0 vocal presence.
            has_onset: Whether an onset was detected this frame.

        Returns:
            SegmentFrame with current segment and confidence.
        """
        # Store in history
        self._energy_history.append(energy)
        self._deriv_history.append(energy_derivative)
        centroid_norm = min(1.0, spectral_centroid / self._centroid_norm_hz)
        self._centroid_history.append(centroid_norm)
        self._onset_history.append(1.0 if has_onset else 0.0)
        self._vocal_history.append(vocal_energy)
        self._bass_history.append(sub_bass_energy)

        self._total_frames += 1
        self._frames_in_segment += 1

        # Compute windowed features
        features = self._compute_window_features()

        # Score each segment type
        scores = self._score_segments(features)

        # Apply position bias for intro/outro
        scores = self._apply_position_bias(scores)

        # Select best segment (with minimum duration enforcement)
        best_segment = max(scores, key=scores.get)  # type: ignore[arg-type]
        best_score = scores[best_segment]

        if self._frames_in_segment >= self._min_segment_frames:
            if best_segment != self._current_segment:
                self._current_segment = best_segment
                self._frames_in_segment = 0

        # Confidence is the margin between best and second-best
        sorted_scores = sorted(scores.values(), reverse=True)
        confidence = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else 1.0
        confidence = max(0.0, min(1.0, confidence * 3.0))  # scale up for readability

        return SegmentFrame(
            segment=self._current_segment,
            confidence=confidence,
            scores=scores,
        )

    def classify_offline(
        self,
        energies: list[float],
        energy_derivatives: list[float],
        spectral_centroids: list[float],
        sub_bass_energies: list[float],
        vocal_energies: list[float],
        has_onsets: list[bool],
    ) -> list[SegmentFrame]:
        """Classify segments for an entire track (offline mode).

        Uses forward-backward smoothing for better segment boundaries.

        Args:
            energies: Energy per frame.
            energy_derivatives: Derivative per frame.
            spectral_centroids: Centroid per frame.
            sub_bass_energies: Sub-bass per frame.
            vocal_energies: Vocal energy per frame.
            has_onsets: Onset flags per frame.

        Returns:
            List of SegmentFrame, one per input frame.
        """
        self.reset()
        n = len(energies)
        if n == 0:
            return []

        self.set_track_duration(n / self._fps)

        # Forward pass: classify each frame
        raw_frames: list[SegmentFrame] = []
        for i in range(n):
            frame = self.process_frame(
                energy=energies[i],
                energy_derivative=energy_derivatives[i],
                spectral_centroid=spectral_centroids[i],
                sub_bass_energy=sub_bass_energies[i],
                vocal_energy=vocal_energies[i],
                has_onset=has_onsets[i],
            )
            raw_frames.append(frame)

        # Smooth: enforce minimum segment length by majority vote
        smoothed = self._smooth_segments(raw_frames)
        return smoothed

    # ── Internal ──────────────────────────────────────────────────

    def _compute_window_features(self) -> dict[str, float]:
        """Compute averaged features over the rolling window.

        Returns:
            Feature dictionary matching prototype keys.
        """
        return {
            "energy": float(np.mean(self._energy_history)) if self._energy_history else 0.0,
            "energy_derivative": (
                float(np.mean(self._deriv_history)) if self._deriv_history else 0.0
            ),
            "spectral_centroid_norm": (
                float(np.mean(self._centroid_history)) if self._centroid_history else 0.0
            ),
            "onset_density": (
                float(np.mean(self._onset_history)) if self._onset_history else 0.0
            ),
            "vocal_energy": (
                float(np.mean(self._vocal_history)) if self._vocal_history else 0.0
            ),
            "sub_bass_energy": (
                float(np.mean(self._bass_history)) if self._bass_history else 0.0
            ),
        }

    def _score_segments(self, features: dict[str, float]) -> dict[str, float]:
        """Score each segment type against its prototype.

        Uses Gaussian similarity: score = exp(-(x - center)^2 / (2 * tol^2)).

        Args:
            features: Current windowed features.

        Returns:
            Dict of segment label -> similarity score.
        """
        scores: dict[str, float] = {}

        for label, prototype in _PROTOTYPES.items():
            total_score = 0.0
            n_features = 0

            for feat_name, (center, tolerance) in prototype.items():
                value = features.get(feat_name, 0.0)
                # Gaussian similarity
                diff = value - center
                score = float(np.exp(-(diff**2) / (2 * tolerance**2)))
                total_score += score
                n_features += 1

            scores[label] = total_score / n_features if n_features > 0 else 0.0

        return scores

    def _apply_position_bias(self, scores: dict[str, float]) -> dict[str, float]:
        """Bias intro/outro scores based on track position.

        Early positions boost intro, late positions boost outro.

        Args:
            scores: Raw segment scores.

        Returns:
            Position-adjusted scores.
        """
        if self._track_duration_frames is None or self._track_duration_frames <= 0:
            return scores

        position = self._total_frames / self._track_duration_frames
        position = max(0.0, min(1.0, position))

        adjusted = dict(scores)
        w = self._position_weight

        # Intro bias: strong at start, fades by 15%
        if position < 0.15:
            intro_boost = (1.0 - position / 0.15) * w
            adjusted["intro"] = adjusted.get("intro", 0.0) + intro_boost

        # Outro bias: strong at end, starts at 85%
        if position > 0.85:
            outro_boost = ((position - 0.85) / 0.15) * w
            adjusted["outro"] = adjusted.get("outro", 0.0) + outro_boost

        return adjusted

    @staticmethod
    def _smooth_segments(frames: list[SegmentFrame]) -> list[SegmentFrame]:
        """Apply majority-vote smoothing to segment labels.

        Short segments (< min window) are absorbed into neighbors.

        Args:
            frames: Raw classified frames.

        Returns:
            Smoothed frames with more stable segment boundaries.
        """
        if len(frames) <= 1:
            return frames

        # Find segment runs
        runs: list[tuple[str, int, int]] = []  # (label, start_idx, length)
        current_label = frames[0].segment
        run_start = 0

        for i in range(1, len(frames)):
            if frames[i].segment != current_label:
                runs.append((current_label, run_start, i - run_start))
                current_label = frames[i].segment
                run_start = i
        runs.append((current_label, run_start, len(frames) - run_start))

        # Absorb very short runs (< 60 frames ≈ 1 second) into neighbors
        min_run = 60
        if len(runs) >= 3:
            merged = [runs[0]]
            for i in range(1, len(runs) - 1):
                label, start, length = runs[i]
                if length < min_run:
                    # Replace with previous run's label
                    merged.append((merged[-1][0], start, length))
                else:
                    merged.append(runs[i])
            merged.append(runs[-1])
            runs = merged

        # Rebuild frames with smoothed labels
        result: list[SegmentFrame] = []
        for label, start, length in runs:
            for j in range(length):
                orig = frames[start + j]
                result.append(SegmentFrame(
                    segment=label,
                    confidence=orig.confidence,
                    scores=orig.scores,
                ))

        return result
