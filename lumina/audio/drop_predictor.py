"""Drop prediction via tension analysis with 1-4 bar look-ahead.

Predicts the probability of a musical "drop" (high-energy moment after a
build-up) in the next 1-4 bars. Uses a rule-based tension model that
tracks multiple indicators common across EDM, trap, and hip-hop:

**Tension indicators (signs of an incoming drop):**
- Rising energy over several bars (build-up)
- Snare/percussion rolls (accelerating onset density)
- High-frequency "riser" content (rising spectral centroid)
- Vocal absence or reduction (breakdowns often strip vocals before drop)
- Sub-bass reduction (low-end often cuts before drop for contrast)

**Release indicators (drop already happened or not building):**
- Sudden energy spike (drop just landed)
- High sub-bass energy (drop is playing)
- Stable or falling energy (no build-up in progress)

The predictor operates on a rolling window of recent frames to build
a multi-bar context for its prediction.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# Window sizes (in frames at fps rate)
_DEFAULT_HISTORY_BARS = 8  # How many bars of history to consider
_DEFAULT_BEATS_PER_BAR = 4


@dataclass(slots=True)
class DropFrame:
    """Drop prediction for a single time frame.

    Args:
        drop_probability: 0.0-1.0 probability of a drop in the next 1-4 bars.
        tension: 0.0-1.0 current musical tension level.
        rising_energy: True if energy has been consistently rising.
        onset_density: Onset events per second in the recent window.
    """

    drop_probability: float
    tension: float
    rising_energy: bool
    onset_density: float


@dataclass(slots=True)
class _FeatureSnapshot:
    """Internal snapshot of audio features at a single frame.

    Stores the features needed for tension analysis, captured from
    external audio analysis modules.
    """

    energy: float
    energy_derivative: float
    spectral_centroid: float
    sub_bass_energy: float
    vocal_energy: float
    has_onset: bool


class DropPredictor:
    """Musical drop predictor using tension analysis.

    Consumes features from the energy tracker, onset detector, and vocal
    detector to estimate the probability of a drop in the next 1-4 bars.
    Operates frame-by-frame, maintaining a rolling history window.

    Args:
        fps: Input frame rate.
        bpm: Initial tempo estimate (updated via update_bpm).
        beats_per_bar: Time signature numerator.
        history_bars: Number of bars of history to analyze.
        smoothing: EMA smoothing for the output probability.
    """

    def __init__(
        self,
        fps: int = 60,
        bpm: float = 128.0,
        beats_per_bar: int = _DEFAULT_BEATS_PER_BAR,
        history_bars: int = _DEFAULT_HISTORY_BARS,
        smoothing: float = 0.8,
    ) -> None:
        self._fps = fps
        self._bpm = bpm
        self._beats_per_bar = beats_per_bar
        self._history_bars = history_bars
        self._smoothing = smoothing

        self._history_size = self._compute_history_size()
        self._history: deque[_FeatureSnapshot] = deque(maxlen=self._history_size)

        self.reset()

    def _compute_history_size(self) -> int:
        """Compute history buffer size in frames from bars and BPM."""
        bar_duration = (60.0 / self._bpm) * self._beats_per_bar
        return max(1, int(bar_duration * self._history_bars * self._fps))

    def reset(self) -> None:
        """Reset all internal state."""
        self._history.clear()
        self._prev_probability = 0.0

    def update_bpm(self, bpm: float) -> None:
        """Update the tempo estimate.

        Args:
            bpm: New BPM value. Adjusts the history window size.
        """
        self._bpm = max(1.0, bpm)
        self._history_size = self._compute_history_size()
        # Resize history deque
        new_history: deque[_FeatureSnapshot] = deque(
            self._history, maxlen=self._history_size
        )
        self._history = new_history

    # ── Public API ────────────────────────────────────────────────

    def process_frame(
        self,
        energy: float,
        energy_derivative: float,
        spectral_centroid: float,
        sub_bass_energy: float,
        vocal_energy: float,
        has_onset: bool,
    ) -> DropFrame:
        """Process a single frame of features and return drop prediction.

        This method is called once per frame (at fps rate) with features
        from other audio analysis modules.

        Args:
            energy: 0.0-1.0 overall energy from EnergyTracker.
            energy_derivative: Energy rate of change.
            spectral_centroid: Spectral brightness in Hz.
            sub_bass_energy: 0.0-1.0 sub-bass energy.
            vocal_energy: 0.0-1.0 vocal presence.
            has_onset: Whether an onset was detected this frame.

        Returns:
            DropFrame with drop probability and tension metrics.
        """
        snapshot = _FeatureSnapshot(
            energy=energy,
            energy_derivative=energy_derivative,
            spectral_centroid=spectral_centroid,
            sub_bass_energy=sub_bass_energy,
            vocal_energy=vocal_energy,
            has_onset=has_onset,
        )
        self._history.append(snapshot)

        tension = self._compute_tension()
        rising = self._is_energy_rising()
        onset_density = self._onset_density()

        # Map tension to drop probability with nonlinear curve
        # Low tension → very low probability
        # High tension → high probability (exponential ramp)
        raw_prob = tension**2  # Quadratic: suppresses low tension, amplifies high
        raw_prob = min(1.0, raw_prob)

        # Smooth the output
        probability = (
            self._smoothing * self._prev_probability
            + (1 - self._smoothing) * raw_prob
        )
        self._prev_probability = probability

        return DropFrame(
            drop_probability=probability,
            tension=tension,
            rising_energy=rising,
            onset_density=onset_density,
        )

    def process_features(
        self,
        energies: list[float],
        energy_derivatives: list[float],
        spectral_centroids: list[float],
        sub_bass_energies: list[float],
        vocal_energies: list[float],
        has_onsets: list[bool],
    ) -> list[DropFrame]:
        """Process multiple frames of features at once.

        Convenience method for batch processing. Each list must have the
        same length.

        Args:
            energies: Energy values per frame.
            energy_derivatives: Energy derivative per frame.
            spectral_centroids: Spectral centroid per frame.
            sub_bass_energies: Sub-bass energy per frame.
            vocal_energies: Vocal energy per frame.
            has_onsets: Onset flags per frame.

        Returns:
            List of DropFrame, one per input frame.
        """
        n = len(energies)
        return [
            self.process_frame(
                energy=energies[i],
                energy_derivative=energy_derivatives[i],
                spectral_centroid=spectral_centroids[i],
                sub_bass_energy=sub_bass_energies[i],
                vocal_energy=vocal_energies[i],
                has_onset=has_onsets[i],
            )
            for i in range(n)
        ]

    # ── Tension computation ───────────────────────────────────────

    def _compute_tension(self) -> float:
        """Compute overall musical tension from history.

        Combines multiple indicators into a single tension value.
        Each indicator is weighted based on its reliability as a
        drop predictor.

        Returns:
            0.0-1.0 tension level.
        """
        if len(self._history) < 2:
            return 0.0

        scores: list[tuple[float, float]] = []  # (score, weight)

        # 1. Rising energy trend (strongest indicator)
        energy_trend = self._energy_trend_score()
        scores.append((energy_trend, 0.35))

        # 2. Onset density acceleration (snare rolls, build-ups)
        density_score = self._onset_acceleration_score()
        scores.append((density_score, 0.20))

        # 3. Rising spectral centroid (risers)
        centroid_score = self._centroid_rise_score()
        scores.append((centroid_score, 0.15))

        # 4. Vocal reduction (breakdowns often strip vocals)
        vocal_score = self._vocal_reduction_score()
        scores.append((vocal_score, 0.15))

        # 5. Sub-bass reduction (bass often drops out before the drop)
        bass_score = self._sub_bass_reduction_score()
        scores.append((bass_score, 0.15))

        # Weighted sum
        total_weight = sum(w for _, w in scores)
        if total_weight < 1e-10:
            return 0.0

        tension = sum(s * w for s, w in scores) / total_weight
        return max(0.0, min(1.0, tension))

    def _energy_trend_score(self) -> float:
        """Score how consistently energy has been rising.

        Returns:
            0.0-1.0 score. 1.0 = energy has been rising consistently
            over the full history window.
        """
        if len(self._history) < 4:
            return 0.0

        # Divide history into segments and check if each is higher than previous
        n = len(self._history)
        segment_size = max(1, n // 4)

        segment_means: list[float] = []
        for i in range(0, n, segment_size):
            segment = list(self._history)[i : i + segment_size]
            if segment:
                segment_means.append(np.mean([s.energy for s in segment]))

        if len(segment_means) < 2:
            return 0.0

        # Count rising transitions
        rising = sum(
            1 for i in range(1, len(segment_means))
            if segment_means[i] > segment_means[i - 1]
        )
        total = len(segment_means) - 1

        return rising / total if total > 0 else 0.0

    def _onset_acceleration_score(self) -> float:
        """Score onset density acceleration (build-up rolls).

        Returns:
            0.0-1.0 score. 1.0 = onset density increasing strongly.
        """
        if len(self._history) < 4:
            return 0.0

        history_list = list(self._history)
        n = len(history_list)
        half = n // 2

        # Compare onset density: first half vs second half
        first_onsets = sum(1 for s in history_list[:half] if s.has_onset)
        second_onsets = sum(1 for s in history_list[half:] if s.has_onset)

        first_density = first_onsets / max(1, half)
        second_density = second_onsets / max(1, n - half)

        if first_density < 1e-6:
            return min(1.0, second_density * 10)  # any onsets from nothing = tension

        ratio = second_density / first_density
        # ratio > 1 means accelerating
        return max(0.0, min(1.0, (ratio - 1.0) * 2.0))

    def _centroid_rise_score(self) -> float:
        """Score rising spectral centroid (riser effects).

        Returns:
            0.0-1.0 score. 1.0 = spectral centroid rising consistently.
        """
        if len(self._history) < 4:
            return 0.0

        history_list = list(self._history)
        n = len(history_list)
        half = n // 2

        first_centroid = np.mean([s.spectral_centroid for s in history_list[:half]])
        second_centroid = np.mean([s.spectral_centroid for s in history_list[half:]])

        if first_centroid < 1e-6:
            return 0.0

        rise_ratio = (second_centroid - first_centroid) / first_centroid
        return max(0.0, min(1.0, rise_ratio * 3.0))

    def _vocal_reduction_score(self) -> float:
        """Score vocal energy reduction (breakdown indicator).

        Returns:
            0.0-1.0 score. 1.0 = vocals dropped significantly.
        """
        if len(self._history) < 4:
            return 0.0

        history_list = list(self._history)
        n = len(history_list)
        quarter = max(1, n // 4)

        # Compare early vocals to recent vocals
        early_vocal = np.mean([s.vocal_energy for s in history_list[:quarter]])
        recent_vocal = np.mean([s.vocal_energy for s in history_list[-quarter:]])

        if early_vocal < 0.05:
            return 0.0  # No vocals to begin with

        reduction = (early_vocal - recent_vocal) / early_vocal
        return max(0.0, min(1.0, reduction * 2.0))

    def _sub_bass_reduction_score(self) -> float:
        """Score sub-bass energy reduction (pre-drop bass cut).

        Returns:
            0.0-1.0 score. 1.0 = sub-bass dropped significantly.
        """
        if len(self._history) < 4:
            return 0.0

        history_list = list(self._history)
        n = len(history_list)
        quarter = max(1, n // 4)

        early_bass = np.mean([s.sub_bass_energy for s in history_list[:quarter]])
        recent_bass = np.mean([s.sub_bass_energy for s in history_list[-quarter:]])

        if early_bass < 0.05:
            return 0.0

        reduction = (early_bass - recent_bass) / early_bass
        return max(0.0, min(1.0, reduction * 2.0))

    # ── Helper metrics ────────────────────────────────────────────

    def _is_energy_rising(self) -> bool:
        """Check if energy is currently trending upward.

        Returns:
            True if energy has been rising over the recent window.
        """
        if len(self._history) < 4:
            return False

        recent = list(self._history)[-self._fps:]  # last ~1 second
        if len(recent) < 4:
            return False

        # Check if positive derivatives outnumber negative
        positive = sum(1 for s in recent if s.energy_derivative > 0.001)
        return positive > len(recent) * 0.6

    def _onset_density(self) -> float:
        """Compute recent onset density (onsets per second).

        Returns:
            Onset density in events per second.
        """
        if len(self._history) == 0:
            return 0.0

        # Use last 1 second of history
        window_frames = min(self._fps, len(self._history))
        recent = list(self._history)[-window_frames:]
        onset_count = sum(1 for s in recent if s.has_onset)

        duration = window_frames / self._fps
        return onset_count / duration if duration > 0 else 0.0
