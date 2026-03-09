"""Song segment classification: verse, chorus, drop, breakdown, intro, outro, bridge.

Classifies the current segment of a song using energy dynamics, spectral
features, onset patterns, and structural cues. Works in both streaming
mode (updating every few seconds) and offline mode (full-song analysis).

Classification approach (rule-based, Phase 1):

Uses a decision-tree driven by **energy dynamics** — the key insight is
that drops, breakdowns, and builds are defined by *changes* in energy,
not just absolute levels.  A drop is a sharp energy spike; a breakdown
is a sharp energy dip.  The classifier uses both short-term (0.5s) and
medium-term (2s) feature windows to detect these transitions.

Decision tree (evaluated in priority order):
1. **Position-based intro/outro:** First/last 10% of track at low energy.
2. **Drop:** Instantaneous energy > 0.6 AND (short derivative > 0.08
   OR sustained high energy with heavy sub-bass).
3. **Breakdown:** Short energy average < 0.25, sparse onsets, falling.
4. **Chorus:** High energy (> 0.5) with strong vocals (> 0.35).
5. **Bridge:** Low-moderate energy, low vocals, mid-track.
6. **Verse:** Default for everything else.
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

# Window sizes in seconds
_SHORT_WINDOW_S = 0.5  # For detecting transitions (derivatives)
_MEDIUM_WINDOW_S = 2.0  # For sustained-state detection

# ── Thresholds ────────────────────────────────────────────────────────

# Drop detection
_DROP_ENERGY_MIN = 0.55  # Minimum instantaneous energy for drop
_DROP_DERIV_THRESHOLD = 0.08  # Short-term derivative spike for onset
_DROP_SUSTAINED_ENERGY = 0.60  # Sustained energy for ongoing drop
_DROP_SUSTAINED_BASS = 0.35  # Sub-bass for ongoing drop

# Breakdown detection
_BREAKDOWN_ENERGY_MAX = 0.28  # Max short energy for breakdown
_BREAKDOWN_ONSET_MAX = 0.20  # Max onset density for breakdown

# Chorus detection
_CHORUS_ENERGY_MIN = 0.45  # Min medium energy for chorus
_CHORUS_VOCAL_MIN = 0.35  # Min vocal energy for chorus

# Bridge detection
_BRIDGE_ENERGY_MAX = 0.42  # Max medium energy for bridge
_BRIDGE_VOCAL_MAX = 0.30  # Max vocal for bridge

# Intro/outro position thresholds
_INTRO_POSITION = 0.10  # First 10%
_OUTRO_POSITION = 0.90  # Last 10%
_INTRO_OUTRO_ENERGY_MAX = 0.35  # Low energy required


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


class SegmentClassifier:
    """Real-time song segment classifier using energy dynamics.

    Uses short-term (0.5s) and medium-term (2s) feature windows to
    detect both sudden transitions (drops, breakdowns) and sustained
    states (verse, chorus).  Enforces minimum segment duration to
    prevent rapid switching.

    Args:
        fps: Input frame rate.
        min_segment_seconds: Minimum duration before allowing segment change.
        position_weight: How much track position influences intro/outro.
    """

    def __init__(
        self,
        fps: int = 60,
        min_segment_seconds: float = _MIN_SEGMENT_DURATION,
        position_weight: float = 0.3,
        **_kwargs: object,
    ) -> None:
        self._fps = fps
        self._short_size = max(1, int(fps * _SHORT_WINDOW_S))
        self._medium_size = max(1, int(fps * _MEDIUM_WINDOW_S))
        self._min_segment_frames = int(fps * min_segment_seconds)
        self._position_weight = position_weight
        self._centroid_norm_hz = 10000.0

        self.reset()

    def reset(self) -> None:
        """Reset all internal state."""
        self._energy_short: deque[float] = deque(maxlen=self._short_size)
        self._energy_medium: deque[float] = deque(maxlen=self._medium_size)
        self._onset_medium: deque[float] = deque(maxlen=self._medium_size)
        self._vocal_medium: deque[float] = deque(maxlen=self._medium_size)
        self._bass_medium: deque[float] = deque(maxlen=self._medium_size)
        self._centroid_medium: deque[float] = deque(maxlen=self._medium_size)

        # For short-term derivative: two half-windows
        self._energy_prev_half: deque[float] = deque(maxlen=self._short_size)
        self._energy_curr_half: deque[float] = deque(maxlen=self._short_size)

        self._current_segment = "verse"
        self._frames_in_segment = 0
        self._total_frames = 0
        self._track_duration_frames: int | None = None

    def set_track_duration(self, duration_seconds: float) -> None:
        """Set total track duration for position-aware classification.

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
            energy_derivative: Rate of energy change (from energy tracker).
            spectral_centroid: Brightness in Hz.
            sub_bass_energy: 0.0-1.0 sub-bass level.
            vocal_energy: 0.0-1.0 vocal presence.
            has_onset: Whether an onset was detected this frame.

        Returns:
            SegmentFrame with current segment and confidence.
        """
        # Rotate derivative half-windows (prev ← curr, curr ← new)
        if len(self._energy_curr_half) >= self._short_size:
            self._energy_prev_half.append(self._energy_curr_half[0])
        self._energy_curr_half.append(energy)

        # Store in history windows
        self._energy_short.append(energy)
        self._energy_medium.append(energy)
        self._onset_medium.append(1.0 if has_onset else 0.0)
        self._vocal_medium.append(vocal_energy)
        self._bass_medium.append(sub_bass_energy)
        centroid_norm = min(1.0, spectral_centroid / self._centroid_norm_hz)
        self._centroid_medium.append(centroid_norm)

        self._total_frames += 1
        self._frames_in_segment += 1

        # Compute multi-scale features
        energy_instant = energy
        energy_short = float(np.mean(self._energy_short))
        energy_medium = float(np.mean(self._energy_medium))
        short_deriv = self._compute_short_derivative()
        onset_density = float(np.mean(self._onset_medium))
        vocal_avg = float(np.mean(self._vocal_medium))
        bass_avg = float(np.mean(self._bass_medium))

        # Track position (0-1)
        position = self._get_position()

        # ── Decision tree ─────────────────────────────────────────
        candidate, confidence = self._decide(
            energy_instant=energy_instant,
            energy_short=energy_short,
            energy_medium=energy_medium,
            short_deriv=short_deriv,
            onset_density=onset_density,
            vocal_avg=vocal_avg,
            bass_avg=bass_avg,
            position=position,
        )

        # Build scores dict for diagnostic visibility
        scores = self._build_scores(
            energy_instant,
            energy_short,
            energy_medium,
            short_deriv,
            onset_density,
            vocal_avg,
            bass_avg,
            position,
        )

        # Enforce minimum segment duration
        if (
            self._frames_in_segment >= self._min_segment_frames
            and candidate != self._current_segment
        ):
            self._current_segment = candidate
            self._frames_in_segment = 0

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

    def _compute_short_derivative(self) -> float:
        """Short-term energy derivative: mean(curr_half) - mean(prev_half).

        Returns:
            Positive = energy rising, negative = falling.
        """
        if len(self._energy_prev_half) < 2 or len(self._energy_curr_half) < 2:
            return 0.0
        curr = float(np.mean(self._energy_curr_half))
        prev = float(np.mean(self._energy_prev_half))
        return curr - prev

    def _get_position(self) -> float:
        """Track position as 0.0-1.0."""
        if self._track_duration_frames and self._track_duration_frames > 0:
            return max(0.0, min(1.0, self._total_frames / self._track_duration_frames))
        return 0.5  # unknown position → mid-track (no intro/outro bias)

    def _decide(
        self,
        energy_instant: float,
        energy_short: float,
        energy_medium: float,
        short_deriv: float,
        onset_density: float,
        vocal_avg: float,
        bass_avg: float,
        position: float,
    ) -> tuple[str, float]:
        """Decision tree for segment classification.

        Returns:
            Tuple of (segment_label, confidence).
        """
        # 1. Intro: early position + low energy
        if position < _INTRO_POSITION and energy_medium < _INTRO_OUTRO_ENERGY_MAX:
            return "intro", 0.8

        # 2. Outro: late position + low energy
        if position > _OUTRO_POSITION and energy_medium < _INTRO_OUTRO_ENERGY_MAX:
            return "outro", 0.8

        # 3. Drop: sharp energy spike OR sustained high energy with bass
        if energy_instant > _DROP_ENERGY_MIN and short_deriv > _DROP_DERIV_THRESHOLD:
            return "drop", min(1.0, short_deriv / 0.2 + 0.5)
        if (
            energy_short > _DROP_SUSTAINED_ENERGY
            and bass_avg > _DROP_SUSTAINED_BASS
            and vocal_avg < _CHORUS_VOCAL_MIN
        ):
            return "drop", 0.7

        # 4. Breakdown: low energy, sparse onsets
        if energy_short < _BREAKDOWN_ENERGY_MAX and onset_density < _BREAKDOWN_ONSET_MAX:
            return "breakdown", 0.7
        if short_deriv < -0.08 and energy_short < 0.35:
            return "breakdown", 0.6

        # 5. Chorus: high energy + vocals
        if energy_medium > _CHORUS_ENERGY_MIN and vocal_avg > _CHORUS_VOCAL_MIN:
            return "chorus", 0.6

        # 6. Bridge: low-moderate energy, low vocals, mid-track
        if (
            energy_medium < _BRIDGE_ENERGY_MAX
            and vocal_avg < _BRIDGE_VOCAL_MAX
            and 0.2 < position < 0.85
        ):
            return "bridge", 0.5

        # 7. Default: verse
        return "verse", 0.5

    @staticmethod
    def _build_scores(
        energy_instant: float,
        energy_short: float,
        energy_medium: float,
        short_deriv: float,
        onset_density: float,
        vocal_avg: float,
        bass_avg: float,
        position: float,
    ) -> dict[str, float]:
        """Build diagnostic score dict for visibility into the decision.

        Returns:
            Approximate score per segment (not used for classification).
        """
        scores: dict[str, float] = {}

        # Drop score: energy + derivative
        drop_e = max(0.0, (energy_instant - 0.4) / 0.6)
        drop_d = max(0.0, short_deriv / 0.15)
        drop_b = max(0.0, (bass_avg - 0.2) / 0.6)
        scores["drop"] = min(1.0, 0.4 * drop_e + 0.4 * drop_d + 0.2 * drop_b)

        # Breakdown score
        bd_e = max(0.0, 1.0 - energy_short / 0.35)
        bd_o = max(0.0, 1.0 - onset_density / 0.25)
        scores["breakdown"] = min(1.0, 0.6 * bd_e + 0.4 * bd_o)

        # Chorus score
        ch_e = max(0.0, (energy_medium - 0.3) / 0.5)
        ch_v = max(0.0, (vocal_avg - 0.2) / 0.5)
        scores["chorus"] = min(1.0, 0.5 * ch_e + 0.5 * ch_v)

        # Verse score: moderate energy, not extreme
        v_e = max(0.0, 1.0 - abs(energy_medium - 0.45) / 0.3)
        scores["verse"] = min(1.0, v_e)

        # Bridge score
        br_e = max(0.0, 1.0 - abs(energy_medium - 0.35) / 0.3)
        br_v = max(0.0, 1.0 - vocal_avg / 0.5)
        scores["bridge"] = min(1.0, 0.5 * br_e + 0.5 * br_v)

        # Intro/outro: position-based
        scores["intro"] = max(0.0, 1.0 - position / 0.15) if position < 0.15 else 0.0
        scores["outro"] = max(0.0, (position - 0.85) / 0.15) if position > 0.85 else 0.0

        return scores

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

        # Absorb very short runs (< 120 frames = 2 seconds) into neighbors
        min_run = 120
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
                result.append(
                    SegmentFrame(
                        segment=label,
                        confidence=orig.confidence,
                        scores=orig.scores,
                    )
                )

        return result
