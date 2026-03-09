"""Tests for the LUMINA energy tracking module.

Tests verify energy envelope computation, derivative tracking,
spectral centroid, and sub-bass energy extraction.
"""

from __future__ import annotations

import numpy as np
import pytest

from lumina.audio.energy_tracker import EnergyFrame, EnergyTracker


# ── Helpers ───────────────────────────────────────────────────────────


def make_sine(freq: float, duration: float, sr: int = 44100, amp: float = 0.5) -> np.ndarray:
    """Generate a sine wave."""
    t = np.arange(int(sr * duration), dtype=np.float32) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def make_silence(duration: float, sr: int = 44100) -> np.ndarray:
    """Generate silence."""
    return np.zeros(int(sr * duration), dtype=np.float32)


def make_ramp(duration: float, sr: int = 44100) -> np.ndarray:
    """Generate audio with linearly increasing amplitude."""
    n = int(sr * duration)
    t = np.arange(n, dtype=np.float32) / sr
    envelope = np.linspace(0.0, 1.0, n, dtype=np.float32)
    return (envelope * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


# ── EnergyFrame dataclass ────────────────────────────────────────────


class TestEnergyFrame:
    def test_fields(self) -> None:
        frame = EnergyFrame(energy=0.5, energy_derivative=0.1, spectral_centroid=1000.0,
                            sub_bass_energy=0.2)
        assert frame.energy == 0.5
        assert frame.energy_derivative == 0.1
        assert frame.spectral_centroid == 1000.0
        assert frame.sub_bass_energy == 0.2

    def test_equality(self) -> None:
        a = EnergyFrame(energy=0.5, energy_derivative=0.1, spectral_centroid=500.0,
                        sub_bass_energy=0.0)
        b = EnergyFrame(energy=0.5, energy_derivative=0.1, spectral_centroid=500.0,
                        sub_bass_energy=0.0)
        assert a == b


# ── Energy envelope ──────────────────────────────────────────────────


class TestEnergyEnvelope:
    def _make_tracker(self, **kwargs: float | int) -> EnergyTracker:
        return EnergyTracker(sr=44100, fps=60, **kwargs)

    def test_silence_gives_zero_energy(self) -> None:
        """Silent audio should produce near-zero energy."""
        tracker = self._make_tracker()
        audio = make_silence(1.0)
        frames = tracker.process_chunk(audio)
        assert len(frames) > 0
        for f in frames:
            assert f.energy < 0.01

    def test_loud_signal_gives_high_energy(self) -> None:
        """A loud sine wave should produce high energy."""
        tracker = self._make_tracker(smoothing=0.0)
        audio = make_sine(440.0, 1.0, amp=0.9)
        frames = tracker.process_chunk(audio)
        # After a few frames the energy should be high
        assert frames[-1].energy > 0.5

    def test_energy_in_range(self) -> None:
        """Energy should always be in [0, 1]."""
        tracker = self._make_tracker()
        audio = make_sine(440.0, 2.0, amp=1.0)
        frames = tracker.process_chunk(audio)
        for f in frames:
            assert 0.0 <= f.energy <= 1.0

    def test_louder_signal_higher_energy(self) -> None:
        """A louder signal should produce higher energy than a quieter one."""
        tracker = self._make_tracker(smoothing=0.0)

        # Feed loud first to establish a high peak, then quiet
        loud = make_sine(440.0, 0.5, amp=0.9)
        quiet = make_sine(440.0, 0.5, amp=0.1)

        loud_frames = tracker.process_chunk(loud)
        quiet_frames = tracker.process_chunk(quiet)

        # Loud frames should have higher energy than quiet frames
        assert loud_frames[-1].energy > quiet_frames[-1].energy

    def test_smoothing_reduces_jitter(self) -> None:
        """Higher smoothing should produce less frame-to-frame variation."""
        # Create audio with abrupt changes
        audio = np.concatenate([
            make_sine(440.0, 0.5, amp=0.8),
            make_silence(0.5),
            make_sine(440.0, 0.5, amp=0.8),
        ])

        smooth = self._make_tracker(smoothing=0.9)
        rough = self._make_tracker(smoothing=0.1)

        smooth_frames = smooth.process_chunk(audio)
        rough_frames = rough.process_chunk(audio)

        # Smooth version should have lower max derivative magnitude
        smooth_diffs = [abs(smooth_frames[i].energy - smooth_frames[i - 1].energy)
                        for i in range(1, len(smooth_frames))]
        rough_diffs = [abs(rough_frames[i].energy - rough_frames[i - 1].energy)
                       for i in range(1, len(rough_frames))]

        assert max(smooth_diffs) < max(rough_diffs)


# ── Energy derivative ────────────────────────────────────────────────


class TestEnergyDerivative:
    def _make_tracker(self, **kwargs: float | int) -> EnergyTracker:
        return EnergyTracker(sr=44100, fps=60, **kwargs)

    def test_rising_energy_positive_derivative(self) -> None:
        """When energy increases, derivative should be positive."""
        tracker = self._make_tracker(smoothing=0.1, derivative_smoothing=0.1)
        audio = make_ramp(2.0)
        frames = tracker.process_chunk(audio)
        # In the rising portion, derivative should be positive
        rising_derivs = [f.energy_derivative for f in frames[10:]]
        assert sum(1 for d in rising_derivs if d > 0) > len(rising_derivs) * 0.5

    def test_steady_signal_near_zero_derivative(self) -> None:
        """A constant signal should have near-zero derivative once stabilized."""
        tracker = self._make_tracker(smoothing=0.3, derivative_smoothing=0.3)
        audio = make_sine(440.0, 3.0, amp=0.5)
        frames = tracker.process_chunk(audio)
        # After stabilization (last quarter), derivative should be near zero
        late_derivs = [abs(f.energy_derivative) for f in frames[len(frames) * 3 // 4:]]
        assert np.mean(late_derivs) < 0.01

    def test_falling_energy_negative_derivative(self) -> None:
        """When energy drops, derivative should go negative."""
        tracker = self._make_tracker(smoothing=0.1, derivative_smoothing=0.1)
        # Loud then silent
        audio = np.concatenate([
            make_sine(440.0, 1.0, amp=0.9),
            make_silence(1.0),
        ])
        frames = tracker.process_chunk(audio)
        # Find frames around the transition point
        mid = len(frames) // 2
        after_drop = frames[mid + 5: mid + 20]
        neg_derivs = [f.energy_derivative for f in after_drop if f.energy_derivative < 0]
        assert len(neg_derivs) > 0


# ── Spectral centroid ────────────────────────────────────────────────


class TestSpectralCentroid:
    def _make_tracker(self, **kwargs: float | int) -> EnergyTracker:
        return EnergyTracker(sr=44100, fps=60, **kwargs)

    def test_silence_zero_centroid(self) -> None:
        """Silence should have zero spectral centroid."""
        tracker = self._make_tracker()
        audio = make_silence(0.5)
        frames = tracker.process_chunk(audio)
        for f in frames:
            assert f.spectral_centroid == 0.0

    def test_low_freq_lower_centroid(self) -> None:
        """A low-frequency tone should have lower centroid than a high one."""
        tracker_low = self._make_tracker()
        tracker_high = self._make_tracker()

        low = make_sine(200.0, 1.0)
        high = make_sine(4000.0, 1.0)

        low_frames = tracker_low.process_chunk(low)
        high_frames = tracker_high.process_chunk(high)

        avg_low = np.mean([f.spectral_centroid for f in low_frames])
        avg_high = np.mean([f.spectral_centroid for f in high_frames])

        assert avg_low < avg_high

    def test_centroid_near_fundamental(self) -> None:
        """For a pure sine, centroid should be near the fundamental frequency."""
        tracker = self._make_tracker()
        audio = make_sine(1000.0, 1.0)
        frames = tracker.process_chunk(audio)
        avg_centroid = np.mean([f.spectral_centroid for f in frames])
        # Pure sine centroid should be close to 1000 Hz
        assert avg_centroid == pytest.approx(1000.0, abs=100.0)


# ── Sub-bass energy ──────────────────────────────────────────────────


class TestSubBassEnergy:
    def _make_tracker(self, **kwargs: float | int) -> EnergyTracker:
        return EnergyTracker(sr=44100, fps=60, **kwargs)

    def test_sub_bass_tone_high_energy(self) -> None:
        """A 50 Hz tone should produce high sub-bass energy."""
        tracker = self._make_tracker()
        audio = make_sine(50.0, 1.0, amp=0.8)
        frames = tracker.process_chunk(audio)
        avg_sub = np.mean([f.sub_bass_energy for f in frames])
        assert avg_sub > 0.1

    def test_high_freq_low_sub_bass(self) -> None:
        """A 4 kHz tone should produce very low sub-bass energy."""
        tracker = self._make_tracker()
        audio = make_sine(4000.0, 1.0)
        frames = tracker.process_chunk(audio)
        avg_sub = np.mean([f.sub_bass_energy for f in frames])
        assert avg_sub < 0.01

    def test_sub_bass_in_range(self) -> None:
        """Sub-bass energy should be in [0, 1]."""
        tracker = self._make_tracker()
        audio = make_sine(50.0, 1.0, amp=1.0)
        frames = tracker.process_chunk(audio)
        for f in frames:
            assert 0.0 <= f.sub_bass_energy <= 1.0


# ── Offline analysis ─────────────────────────────────────────────────


class TestOfflineAnalysis:
    def _make_tracker(self, **kwargs: float | int) -> EnergyTracker:
        return EnergyTracker(sr=44100, fps=60, **kwargs)

    def test_offline_returns_correct_frame_count(self) -> None:
        """Offline should return fps * duration frames."""
        tracker = self._make_tracker()
        audio = make_sine(440.0, 2.0)
        frames = tracker.analyze_offline(audio)
        expected = max(1, len(audio) // (44100 // 60))
        assert len(frames) == expected

    def test_offline_empty_audio(self) -> None:
        """Empty audio should return empty list."""
        tracker = self._make_tracker()
        frames = tracker.analyze_offline(np.array([], dtype=np.float32))
        assert frames == []

    def test_offline_energy_profile(self) -> None:
        """Offline analysis of loud-then-quiet should show energy drop."""
        tracker = self._make_tracker(smoothing=0.3)
        audio = np.concatenate([
            make_sine(440.0, 1.0, amp=0.9),
            make_silence(1.0),
        ])
        frames = tracker.analyze_offline(audio)
        # First half should have higher average energy than second half
        mid = len(frames) // 2
        first_half_avg = np.mean([f.energy for f in frames[:mid]])
        second_half_avg = np.mean([f.energy for f in frames[mid:]])
        assert first_half_avg > second_half_avg


# ── Bass stem analysis ───────────────────────────────────────────────


class TestBassStemAnalysis:
    def _make_tracker(self, **kwargs: float | int) -> EnergyTracker:
        return EnergyTracker(sr=44100, fps=60, **kwargs)

    def test_bass_stem_replaces_sub_bass(self) -> None:
        """Sub-bass energy should come from bass stem, not FFT bands."""
        tracker = self._make_tracker()
        # Full mix: high-frequency sine (no sub-bass content)
        audio = make_sine(4000.0, 1.0, amp=0.8)
        # Bass stem: low-frequency sine (strong bass)
        bass = make_sine(50.0, 1.0, amp=0.8)

        frames = tracker.analyze_offline_with_bass_stem(audio, bass)
        assert len(frames) > 0
        # Sub-bass should reflect the bass stem, not the high-freq mix
        avg_sub = np.mean([f.sub_bass_energy for f in frames[5:]])
        assert avg_sub > 0.3

    def test_silent_bass_gives_zero_sub_bass(self) -> None:
        """Silent bass stem should give zero sub-bass energy."""
        tracker = self._make_tracker()
        audio = make_sine(440.0, 1.0, amp=0.5)
        bass = make_silence(1.0)

        frames = tracker.analyze_offline_with_bass_stem(audio, bass)
        for f in frames:
            assert f.sub_bass_energy < 0.01

    def test_energy_unchanged_with_bass_stem(self) -> None:
        """Overall energy should still come from full mix, not bass stem."""
        tracker = self._make_tracker(smoothing=0.3)
        audio = make_sine(440.0, 1.0, amp=0.8)
        bass = make_silence(1.0)

        frames_with_stem = tracker.analyze_offline_with_bass_stem(audio, bass)
        tracker.reset()
        frames_without = tracker.analyze_offline(audio)

        # Energy values should be identical (bass stem only affects sub_bass)
        for f1, f2 in zip(frames_with_stem, frames_without):
            assert f1.energy == pytest.approx(f2.energy, abs=0.001)

    def test_bass_stem_normalized(self) -> None:
        """Bass stem sub-bass should be in [0, 1]."""
        tracker = self._make_tracker()
        audio = make_sine(440.0, 2.0, amp=1.0)
        bass = make_sine(60.0, 2.0, amp=1.0)

        frames = tracker.analyze_offline_with_bass_stem(audio, bass)
        for f in frames:
            assert 0.0 <= f.sub_bass_energy <= 1.0


# ── Reset ─────────────────────────────────────────────────────────────


class TestReset:
    def test_reset_clears_state(self) -> None:
        """reset() should return tracker to initial state."""
        tracker = EnergyTracker(sr=44100, fps=60)
        audio = make_sine(440.0, 1.0, amp=0.9)
        tracker.process_chunk(audio)

        # Energy should be nonzero
        assert tracker._prev_energy > 0

        tracker.reset()
        assert tracker._prev_energy == 0.0
        assert tracker._prev_derivative == 0.0

    def test_process_after_reset(self) -> None:
        """Processing after reset should work as if fresh."""
        tracker = EnergyTracker(sr=44100, fps=60, smoothing=0.0)
        audio = make_sine(440.0, 0.5, amp=0.5)

        frames1 = tracker.process_chunk(audio)
        tracker.reset()
        frames2 = tracker.process_chunk(audio)

        # The results should be identical (same starting conditions)
        assert len(frames1) == len(frames2)
        for f1, f2 in zip(frames1, frames2):
            assert f1.energy == pytest.approx(f2.energy, abs=0.01)


# ── Streaming consistency ────────────────────────────────────────────


class TestStreamingChunks:
    def test_empty_chunk(self) -> None:
        """Empty chunk should return empty list."""
        tracker = EnergyTracker()
        frames = tracker.process_chunk(np.array([], dtype=np.float32))
        assert frames == []

    def test_chunk_frame_count(self) -> None:
        """Frame count should match chunk duration × fps."""
        tracker = EnergyTracker(sr=44100, fps=60)
        hop = 44100 // 60
        # Exactly 1 second = 60 frames
        chunk = make_sine(440.0, 1.0)
        frames = tracker.process_chunk(chunk)
        expected = len(chunk) // hop
        assert len(frames) == expected
