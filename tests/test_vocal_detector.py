"""Tests for the LUMINA vocal detection module.

Tests verify vocal energy detection, harmonic ratio computation,
and language-agnostic voice presence detection using synthetic audio.
"""

from __future__ import annotations

import numpy as np
import pytest

from lumina.audio.vocal_detector import VocalDetector, VocalFrame


# ── Helpers ───────────────────────────────────────────────────────────


def make_sine(freq: float, duration: float, sr: int = 44100, amp: float = 0.5) -> np.ndarray:
    """Generate a pure sine wave."""
    t = np.arange(int(sr * duration), dtype=np.float32) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def make_silence(duration: float, sr: int = 44100) -> np.ndarray:
    """Generate silence."""
    return np.zeros(int(sr * duration), dtype=np.float32)


def make_noise(duration: float, sr: int = 44100, amp: float = 0.5) -> np.ndarray:
    """Generate white noise (no harmonic structure)."""
    n = int(sr * duration)
    rng = np.random.default_rng(42)
    return (amp * rng.standard_normal(n)).astype(np.float32)


def make_vocal_like(
    f0: float,
    duration: float,
    sr: int = 44100,
    amp: float = 0.5,
    n_harmonics: int = 8,
) -> np.ndarray:
    """Generate a vocal-like harmonic signal.

    Simulates a voiced vocal with fundamental and harmonics, similar to
    the harmonic series of a human voice. Language-agnostic — just a
    pitched harmonic signal in the vocal range.

    Args:
        f0: Fundamental frequency in Hz (e.g., 200 for male, 400 for female).
        duration: Duration in seconds.
        sr: Sample rate.
        amp: Amplitude.
        n_harmonics: Number of harmonics to include.

    Returns:
        Harmonic audio signal.
    """
    n = int(sr * duration)
    t = np.arange(n, dtype=np.float32) / sr
    signal = np.zeros(n, dtype=np.float32)

    for h in range(1, n_harmonics + 1):
        # Harmonics decay as 1/h (natural voice spectral slope)
        harmonic_amp = amp / h
        freq = f0 * h
        if freq > sr / 2:
            break
        signal += harmonic_amp * np.sin(2 * np.pi * freq * t).astype(np.float32)

    # Normalize to amp
    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = signal * amp / peak

    return signal


def make_drum_like(sr: int = 44100) -> np.ndarray:
    """Generate a percussive, non-harmonic signal."""
    n = int(sr * 1.0)
    t = np.arange(n, dtype=np.float32) / sr
    # Low frequency thump + noise burst, not harmonic
    envelope = np.exp(-t * 20)
    rng = np.random.default_rng(42)
    noise = rng.standard_normal(n).astype(np.float32)
    thump = np.sin(2 * np.pi * 60 * t).astype(np.float32)
    return (0.7 * envelope * (0.5 * thump + 0.5 * noise)).astype(np.float32)


# ── VocalFrame dataclass ─────────────────────────────────────────────


class TestVocalFrame:
    def test_fields(self) -> None:
        frame = VocalFrame(vocal_energy=0.6, is_vocal=True, harmonic_ratio=0.8)
        assert frame.vocal_energy == 0.6
        assert frame.is_vocal is True
        assert frame.harmonic_ratio == 0.8

    def test_equality(self) -> None:
        a = VocalFrame(vocal_energy=0.5, is_vocal=True, harmonic_ratio=0.7)
        b = VocalFrame(vocal_energy=0.5, is_vocal=True, harmonic_ratio=0.7)
        assert a == b


# ── Vocal detection basics ───────────────────────────────────────────


class TestVocalDetection:
    def _make_detector(self, **kwargs: float | int) -> VocalDetector:
        return VocalDetector(sr=44100, fps=60, **kwargs)

    def test_silence_no_vocal(self) -> None:
        """Silence should produce no vocal detection."""
        det = self._make_detector()
        audio = make_silence(1.0)
        frames = det.process_chunk(audio)
        assert len(frames) > 0
        for f in frames:
            assert f.is_vocal is False
            assert f.vocal_energy < 0.01

    def test_vocal_signal_detected(self) -> None:
        """A vocal-like harmonic signal should be detected."""
        det = self._make_detector(smoothing=0.3, threshold=0.1)
        audio = make_vocal_like(f0=200.0, duration=2.0)
        frames = det.process_chunk(audio)
        # After warmup, should detect vocal presence
        late_frames = frames[len(frames) // 2:]
        vocal_count = sum(1 for f in late_frames if f.is_vocal)
        assert vocal_count > len(late_frames) * 0.5

    def test_vocal_energy_in_range(self) -> None:
        """Vocal energy should always be in [0, 1]."""
        det = self._make_detector()
        audio = make_vocal_like(f0=300.0, duration=1.0)
        frames = det.process_chunk(audio)
        for f in frames:
            assert 0.0 <= f.vocal_energy <= 1.0

    def test_noise_low_vocal_energy(self) -> None:
        """White noise should have low harmonic ratio."""
        det = self._make_detector(smoothing=0.1)
        audio = make_noise(1.0)
        frames = det.process_chunk(audio)
        avg_harmonic = np.mean([f.harmonic_ratio for f in frames])
        # Noise has no harmonic structure
        assert avg_harmonic < 0.5

    def test_empty_chunk(self) -> None:
        """Empty audio should return empty list."""
        det = self._make_detector()
        frames = det.process_chunk(np.array([], dtype=np.float32))
        assert frames == []

    def test_frame_count(self) -> None:
        """Should return correct number of frames."""
        det = self._make_detector()
        hop = 44100 // 60
        audio = make_silence(1.0)
        frames = det.process_chunk(audio)
        expected = len(audio) // hop
        assert len(frames) == expected


# ── Harmonic ratio ───────────────────────────────────────────────────


class TestHarmonicRatio:
    def _make_detector(self, **kwargs: float | int) -> VocalDetector:
        return VocalDetector(sr=44100, fps=60, **kwargs)

    def test_pure_tone_high_harmonicity(self) -> None:
        """A pure tone in the vocal range should have high harmonic ratio."""
        det = self._make_detector()
        # 300 Hz = well within vocal fundamental range
        audio = make_sine(300.0, 1.0)
        frames = det.process_chunk(audio)
        avg_harmonic = np.mean([f.harmonic_ratio for f in frames])
        assert avg_harmonic > 0.3

    def test_harmonic_series_high_harmonicity(self) -> None:
        """Harmonic series (vocal-like) should have high harmonic ratio."""
        det = self._make_detector()
        audio = make_vocal_like(f0=200.0, duration=1.0, n_harmonics=6)
        frames = det.process_chunk(audio)
        avg_harmonic = np.mean([f.harmonic_ratio for f in frames])
        assert avg_harmonic > 0.3

    def test_harmonic_ratio_in_range(self) -> None:
        """Harmonic ratio should be in [0, 1]."""
        det = self._make_detector()
        for audio_fn in [
            lambda: make_silence(0.5),
            lambda: make_noise(0.5),
            lambda: make_vocal_like(200.0, 0.5),
            lambda: make_sine(1000.0, 0.5),
        ]:
            det.reset()
            frames = det.process_chunk(audio_fn())
            for f in frames:
                assert 0.0 <= f.harmonic_ratio <= 1.0


# ── Vocal vs non-vocal discrimination ────────────────────────────────


class TestVocalDiscrimination:
    def _make_detector(self, **kwargs: float | int) -> VocalDetector:
        return VocalDetector(sr=44100, fps=60, **kwargs)

    def test_vocal_higher_than_drums(self) -> None:
        """Vocal-like audio should have higher vocal energy than drums."""
        det_vocal = self._make_detector(smoothing=0.3)
        det_drum = self._make_detector(smoothing=0.3)

        vocal = make_vocal_like(f0=250.0, duration=2.0)
        drum = make_drum_like()

        vocal_frames = det_vocal.process_chunk(vocal)
        drum_frames = det_drum.process_chunk(drum)

        avg_vocal = np.mean([f.vocal_energy for f in vocal_frames[len(vocal_frames) // 2:]])
        avg_drum = np.mean([f.vocal_energy for f in drum_frames[len(drum_frames) // 2:]])

        assert avg_vocal > avg_drum

    def test_high_freq_sine_low_vocal(self) -> None:
        """A very high frequency sine (above vocal range) should have lower vocal energy."""
        det_high = self._make_detector(smoothing=0.3)
        det_vocal = self._make_detector(smoothing=0.3)
        # 10 kHz pure sine — above vocal range, low vocal band energy
        high = make_sine(10000.0, 2.0)
        vocal = make_vocal_like(f0=250.0, duration=2.0)

        high_frames = det_high.process_chunk(high)
        vocal_frames = det_vocal.process_chunk(vocal)

        # The vocal-like signal should have higher vocal energy
        avg_high = np.mean([f.vocal_energy for f in high_frames[len(high_frames) // 2:]])
        avg_vocal = np.mean([f.vocal_energy for f in vocal_frames[len(vocal_frames) // 2:]])
        assert avg_vocal > avg_high

    def test_low_freq_sine_moderate_harmonicity(self) -> None:
        """A sine in the vocal fundamental range should show harmonicity."""
        det = self._make_detector()
        audio = make_sine(200.0, 1.0)
        frames = det.process_chunk(audio)
        avg_harmonic = np.mean([f.harmonic_ratio for f in frames])
        assert avg_harmonic > 0.2

    def test_different_vocal_pitches(self) -> None:
        """Should detect vocals across different pitch ranges (male/female)."""
        for f0 in [120.0, 200.0, 350.0, 500.0]:
            det = self._make_detector(smoothing=0.3, threshold=0.1)
            audio = make_vocal_like(f0=f0, duration=2.0, n_harmonics=6)
            frames = det.process_chunk(audio)
            late = frames[len(frames) // 2:]
            avg_energy = np.mean([f.vocal_energy for f in late])
            assert avg_energy > 0.1, f"Failed for f0={f0}Hz, avg_energy={avg_energy}"


# ── Reset and offline ────────────────────────────────────────────────


class TestResetAndOffline:
    def test_reset_clears_state(self) -> None:
        """reset() should clear tracking state."""
        det = VocalDetector()
        audio = make_vocal_like(200.0, 1.0)
        det.process_chunk(audio)
        assert det._prev_vocal_energy > 0

        det.reset()
        assert det._prev_vocal_energy == 0.0

    def test_offline_returns_frames(self) -> None:
        """Offline analysis should return correct number of frames."""
        det = VocalDetector(sr=44100, fps=60)
        audio = make_vocal_like(200.0, 2.0)
        frames = det.analyze_offline(audio)
        expected = max(1, len(audio) // (44100 // 60))
        assert len(frames) == expected

    def test_offline_detects_vocal(self) -> None:
        """Offline should detect vocals in harmonic audio."""
        det = VocalDetector(sr=44100, fps=60, smoothing=0.3, threshold=0.1)
        audio = make_vocal_like(f0=250.0, duration=3.0)
        frames = det.analyze_offline(audio)
        late = frames[len(frames) // 2:]
        vocal_ratio = sum(1 for f in late if f.is_vocal) / len(late)
        assert vocal_ratio > 0.5


# ── Streaming consistency ────────────────────────────────────────────


class TestStreamingConsistency:
    def test_multiple_chunks(self) -> None:
        """Should maintain state across multiple chunks."""
        det = VocalDetector(sr=44100, fps=60, smoothing=0.3)
        vocal = make_vocal_like(200.0, 1.0)
        silence = make_silence(1.0)

        vocal_frames = det.process_chunk(vocal)
        silence_frames = det.process_chunk(silence)

        # Vocal energy should decrease during silence
        last_vocal_energy = vocal_frames[-1].vocal_energy
        last_silence_energy = silence_frames[-1].vocal_energy
        assert last_silence_energy < last_vocal_energy
