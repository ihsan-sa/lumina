"""Tests for the LUMINA drop prediction module.

Tests verify tension analysis, drop probability estimation, and
the individual tension indicators (energy trend, onset acceleration,
spectral rise, vocal/bass reduction).
"""

from __future__ import annotations

import numpy as np
import pytest

from lumina.audio.drop_predictor import DropFrame, DropPredictor


# ── Helpers ───────────────────────────────────────────────────────────


def feed_steady_frames(
    predictor: DropPredictor,
    n_frames: int,
    energy: float = 0.5,
    spectral_centroid: float = 1000.0,
    sub_bass_energy: float = 0.3,
    vocal_energy: float = 0.3,
    onset_every: int = 0,
) -> list[DropFrame]:
    """Feed N frames with constant features."""
    results: list[DropFrame] = []
    for i in range(n_frames):
        has_onset = onset_every > 0 and i % onset_every == 0
        results.append(predictor.process_frame(
            energy=energy,
            energy_derivative=0.0,
            spectral_centroid=spectral_centroid,
            sub_bass_energy=sub_bass_energy,
            vocal_energy=vocal_energy,
            has_onset=has_onset,
        ))
    return results


def feed_buildup_frames(
    predictor: DropPredictor,
    n_frames: int,
    start_energy: float = 0.2,
    end_energy: float = 0.9,
    start_centroid: float = 500.0,
    end_centroid: float = 5000.0,
    vocal_start: float = 0.5,
    vocal_end: float = 0.05,
    bass_start: float = 0.4,
    bass_end: float = 0.05,
    onset_acceleration: bool = True,
) -> list[DropFrame]:
    """Feed frames simulating a build-up towards a drop."""
    results: list[DropFrame] = []
    for i in range(n_frames):
        t = i / max(1, n_frames - 1)  # 0.0 to 1.0

        energy = start_energy + t * (end_energy - start_energy)
        centroid = start_centroid + t * (end_centroid - start_centroid)
        vocal = vocal_start + t * (vocal_end - vocal_start)
        bass = bass_start + t * (bass_end - bass_start)
        deriv = (end_energy - start_energy) / n_frames

        # Onset density increases during build-up
        if onset_acceleration:
            # Onset every N frames, decreasing N (= more frequent)
            onset_interval = max(1, int(20 * (1 - t * 0.8)))
            has_onset = i % onset_interval == 0
        else:
            has_onset = False

        results.append(predictor.process_frame(
            energy=energy,
            energy_derivative=deriv,
            spectral_centroid=centroid,
            sub_bass_energy=bass,
            vocal_energy=vocal,
            has_onset=has_onset,
        ))
    return results


# ── DropFrame dataclass ──────────────────────────────────────────────


class TestDropFrame:
    def test_fields(self) -> None:
        frame = DropFrame(drop_probability=0.7, tension=0.8, rising_energy=True,
                          onset_density=4.0)
        assert frame.drop_probability == 0.7
        assert frame.tension == 0.8
        assert frame.rising_energy is True
        assert frame.onset_density == 4.0

    def test_equality(self) -> None:
        a = DropFrame(drop_probability=0.5, tension=0.6, rising_energy=False,
                      onset_density=2.0)
        b = DropFrame(drop_probability=0.5, tension=0.6, rising_energy=False,
                      onset_density=2.0)
        assert a == b


# ── Basic prediction behavior ────────────────────────────────────────


class TestBasicPrediction:
    def _make_predictor(self, **kwargs: float | int) -> DropPredictor:
        return DropPredictor(fps=60, bpm=128.0, **kwargs)

    def test_steady_low_tension(self) -> None:
        """Constant low energy should produce low tension/probability."""
        pred = self._make_predictor(smoothing=0.3)
        frames = feed_steady_frames(pred, 300, energy=0.3)
        last = frames[-1]
        assert last.tension < 0.3
        assert last.drop_probability < 0.2

    def test_buildup_increases_tension(self) -> None:
        """A build-up pattern should increase tension over time."""
        pred = self._make_predictor(smoothing=0.3)
        frames = feed_buildup_frames(pred, 600)
        early_tension = frames[100].tension
        late_tension = frames[-1].tension
        assert late_tension > early_tension

    def test_buildup_increases_probability(self) -> None:
        """A build-up should increase drop probability."""
        pred = self._make_predictor(smoothing=0.3)
        frames = feed_buildup_frames(pred, 600)
        # After a full build-up, probability should be elevated
        assert frames[-1].drop_probability > 0.1

    def test_probability_in_range(self) -> None:
        """Drop probability should always be in [0, 1]."""
        pred = self._make_predictor()
        frames = feed_buildup_frames(pred, 300)
        for f in frames:
            assert 0.0 <= f.drop_probability <= 1.0
            assert 0.0 <= f.tension <= 1.0

    def test_no_history_zero_tension(self) -> None:
        """With insufficient history, tension should be zero."""
        pred = self._make_predictor()
        frame = pred.process_frame(
            energy=0.5, energy_derivative=0.1, spectral_centroid=1000.0,
            sub_bass_energy=0.3, vocal_energy=0.3, has_onset=False,
        )
        assert frame.tension == 0.0


# ── Individual tension indicators ────────────────────────────────────


class TestTensionIndicators:
    def _make_predictor(self, **kwargs: float | int) -> DropPredictor:
        return DropPredictor(fps=60, bpm=128.0, **kwargs)

    def test_rising_energy_detected(self) -> None:
        """Rising energy frames should set rising_energy=True."""
        pred = self._make_predictor()
        # Feed rising energy frames
        for i in range(120):
            t = i / 120
            pred.process_frame(
                energy=0.2 + 0.6 * t,
                energy_derivative=0.005,
                spectral_centroid=1000.0,
                sub_bass_energy=0.3,
                vocal_energy=0.3,
                has_onset=False,
            )
        frame = pred.process_frame(
            energy=0.8, energy_derivative=0.005,
            spectral_centroid=1000.0, sub_bass_energy=0.3,
            vocal_energy=0.3, has_onset=False,
        )
        assert frame.rising_energy is True

    def test_falling_energy_not_rising(self) -> None:
        """Falling energy should not set rising_energy."""
        pred = self._make_predictor()
        for i in range(120):
            t = i / 120
            pred.process_frame(
                energy=0.8 - 0.6 * t,
                energy_derivative=-0.005,
                spectral_centroid=1000.0,
                sub_bass_energy=0.3,
                vocal_energy=0.3,
                has_onset=False,
            )
        frame = pred.process_frame(
            energy=0.2, energy_derivative=-0.005,
            spectral_centroid=1000.0, sub_bass_energy=0.3,
            vocal_energy=0.3, has_onset=False,
        )
        assert frame.rising_energy is False

    def test_onset_density_computed(self) -> None:
        """Onset density should reflect onset frequency."""
        pred = self._make_predictor()
        # Feed 60 frames (1 second) with onset every 6th frame = 10 onsets/sec
        for i in range(60):
            pred.process_frame(
                energy=0.5, energy_derivative=0.0,
                spectral_centroid=1000.0, sub_bass_energy=0.3,
                vocal_energy=0.3, has_onset=(i % 6 == 0),
            )
        frame = pred.process_frame(
            energy=0.5, energy_derivative=0.0,
            spectral_centroid=1000.0, sub_bass_energy=0.3,
            vocal_energy=0.3, has_onset=True,
        )
        assert frame.onset_density > 5.0

    def test_no_onsets_zero_density(self) -> None:
        """No onsets should produce zero onset density."""
        pred = self._make_predictor()
        frames = feed_steady_frames(pred, 120, onset_every=0)
        assert frames[-1].onset_density == 0.0


# ── Build-up vs steady state ────────────────────────────────────────


class TestBuildupDetection:
    def _make_predictor(self, **kwargs: float | int) -> DropPredictor:
        return DropPredictor(fps=60, bpm=128.0, **kwargs)

    def test_buildup_higher_than_steady(self) -> None:
        """Build-up should produce higher tension than steady state."""
        pred_buildup = self._make_predictor(smoothing=0.3)
        pred_steady = self._make_predictor(smoothing=0.3)

        buildup_frames = feed_buildup_frames(pred_buildup, 400)
        steady_frames = feed_steady_frames(pred_steady, 400, energy=0.5)

        assert buildup_frames[-1].tension > steady_frames[-1].tension

    def test_full_buildup_pattern(self) -> None:
        """A complete build-up (rising energy + centroid + onset accel + vocal/bass cut)
        should produce high tension."""
        pred = self._make_predictor(smoothing=0.3)
        frames = feed_buildup_frames(
            pred, 600,
            start_energy=0.2, end_energy=0.85,
            start_centroid=500.0, end_centroid=6000.0,
            vocal_start=0.6, vocal_end=0.05,
            bass_start=0.5, bass_end=0.05,
            onset_acceleration=True,
        )
        # Tension should be elevated by end of build-up
        assert frames[-1].tension > 0.3


# ── BPM update ───────────────────────────────────────────────────────


class TestBpmUpdate:
    def test_update_bpm_changes_history_size(self) -> None:
        """Updating BPM should resize the history window."""
        pred = DropPredictor(fps=60, bpm=120.0)
        initial_size = pred._history_size

        pred.update_bpm(180.0)
        new_size = pred._history_size

        # Faster BPM = shorter bars = smaller history
        assert new_size < initial_size

    def test_update_bpm_preserves_history(self) -> None:
        """BPM update should preserve existing history data."""
        pred = DropPredictor(fps=60, bpm=120.0)
        feed_steady_frames(pred, 60)
        history_len = len(pred._history)

        pred.update_bpm(130.0)
        # History should be preserved (or truncated to new max, not cleared)
        assert len(pred._history) > 0
        assert len(pred._history) <= pred._history_size


# ── Reset ─────────────────────────────────────────────────────────────


class TestReset:
    def test_reset_clears_state(self) -> None:
        """reset() should clear history and probability."""
        pred = DropPredictor()
        feed_buildup_frames(pred, 120)
        assert len(pred._history) > 0

        pred.reset()
        assert len(pred._history) == 0
        assert pred._prev_probability == 0.0

    def test_prediction_after_reset(self) -> None:
        """Prediction after reset should start from zero tension."""
        pred = DropPredictor(smoothing=0.3)
        feed_buildup_frames(pred, 120)

        pred.reset()
        frame = pred.process_frame(
            energy=0.5, energy_derivative=0.0,
            spectral_centroid=1000.0, sub_bass_energy=0.3,
            vocal_energy=0.3, has_onset=False,
        )
        assert frame.tension == 0.0
        assert frame.drop_probability == 0.0


# ── Batch processing ─────────────────────────────────────────────────


class TestBatchProcessing:
    def test_process_features_length(self) -> None:
        """process_features should return one frame per input."""
        pred = DropPredictor()
        n = 30
        frames = pred.process_features(
            energies=[0.5] * n,
            energy_derivatives=[0.0] * n,
            spectral_centroids=[1000.0] * n,
            sub_bass_energies=[0.3] * n,
            vocal_energies=[0.3] * n,
            has_onsets=[False] * n,
        )
        assert len(frames) == n

    def test_process_features_matches_sequential(self) -> None:
        """Batch should produce same results as sequential calls."""
        pred1 = DropPredictor(smoothing=0.5)
        pred2 = DropPredictor(smoothing=0.5)

        n = 20
        energies = [0.1 + 0.04 * i for i in range(n)]
        derivs = [0.04] * n
        centroids = [1000.0] * n
        basses = [0.3] * n
        vocals = [0.3] * n
        onsets = [i % 5 == 0 for i in range(n)]

        batch = pred1.process_features(
            energies, derivs, centroids, basses, vocals, onsets,
        )
        sequential = [
            pred2.process_frame(energies[i], derivs[i], centroids[i],
                                basses[i], vocals[i], onsets[i])
            for i in range(n)
        ]

        for b, s in zip(batch, sequential):
            assert b.drop_probability == pytest.approx(s.drop_probability)
            assert b.tension == pytest.approx(s.tension)
