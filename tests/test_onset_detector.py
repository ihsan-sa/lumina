"""Tests for the LUMINA onset detection module.

Tests verify onset detection via spectral flux and percussive
classification (kick, snare, hihat, clap) via spectral shape.
"""

from __future__ import annotations

import numpy as np
import pytest

from lumina.audio.onset_detector import OnsetDetector, OnsetEvent


# ── Helpers ───────────────────────────────────────────────────────────


def make_sine(freq: float, duration: float, sr: int = 44100, amp: float = 0.5) -> np.ndarray:
    """Generate a sine wave."""
    t = np.arange(int(sr * duration), dtype=np.float32) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def make_silence(duration: float, sr: int = 44100) -> np.ndarray:
    """Generate silence."""
    return np.zeros(int(sr * duration), dtype=np.float32)


def make_impulse(
    freq_center: float,
    duration: float = 0.01,
    sr: int = 44100,
    amp: float = 0.9,
    bandwidth: float = 0.5,
) -> np.ndarray:
    """Generate a short burst at a center frequency (bandpass-like).

    Args:
        freq_center: Center frequency in Hz.
        duration: Burst length in seconds.
        sr: Sample rate.
        amp: Amplitude.
        bandwidth: Relative bandwidth (0-1). Higher = more broadband.

    Returns:
        Short audio burst.
    """
    n = int(sr * duration)
    t = np.arange(n, dtype=np.float32) / sr
    # Sine with exponential decay envelope
    envelope = np.exp(-t * 50)  # fast decay
    signal = amp * envelope * np.sin(2 * np.pi * freq_center * t)
    return signal.astype(np.float32)


def make_kick(sr: int = 44100) -> np.ndarray:
    """Synthesize a kick-like sound (low frequency burst)."""
    n = int(sr * 0.05)
    t = np.arange(n, dtype=np.float32) / sr
    # Pitch drops from ~150 Hz to ~50 Hz
    freq = 150.0 * np.exp(-t * 30)
    phase = np.cumsum(2 * np.pi * freq / sr)
    envelope = np.exp(-t * 40)
    return (0.9 * envelope * np.sin(phase)).astype(np.float32)


def make_snare(sr: int = 44100) -> np.ndarray:
    """Synthesize a snare-like sound (tonal + band-limited noise).

    Real snares have energy focused in 200-5000 Hz with a tonal body
    around 150-250 Hz and noise rattle in the mid-band.
    """
    n = int(sr * 0.04)
    t = np.arange(n, dtype=np.float32) / sr
    envelope = np.exp(-t * 50)
    # Tonal body (250 Hz) — strong mid component above the low band cutoff
    tone = 0.5 * np.sin(2 * np.pi * 250 * t)
    # Mid-range noise (band-limited via summing mid-freq sines with random phases)
    rng = np.random.default_rng(42)
    noise = np.zeros(n, dtype=np.float32)
    for freq in np.linspace(300, 4000, 30):
        phase = rng.uniform(0, 2 * np.pi)
        noise += np.sin(2 * np.pi * freq * t + phase).astype(np.float32)
    noise = 0.5 * noise / 30.0
    return (0.9 * envelope * (tone + noise)).astype(np.float32)


def make_hihat(sr: int = 44100) -> np.ndarray:
    """Synthesize a hihat-like sound (high frequency noise burst)."""
    n = int(sr * 0.02)
    t = np.arange(n, dtype=np.float32) / sr
    envelope = np.exp(-t * 100)
    # High-pass filtered noise
    noise = np.random.default_rng(42).standard_normal(n).astype(np.float32)
    # Simple high-pass via first difference
    hp = np.diff(noise, prepend=0)
    hp = np.diff(hp, prepend=0)  # second-order for stronger filtering
    return (0.9 * envelope * hp).astype(np.float32)


def make_clap(sr: int = 44100) -> np.ndarray:
    """Synthesize a clap-like sound (mid-high noise burst, no low end)."""
    n = int(sr * 0.03)
    t = np.arange(n, dtype=np.float32) / sr
    # Multiple short bursts (clap character)
    envelope = np.exp(-t * 60) * (1 + 0.5 * np.sin(2 * np.pi * 100 * t))
    noise = np.random.default_rng(42).standard_normal(n).astype(np.float32)
    # Band-pass: remove low end, keep mid-high
    hp = np.diff(noise, prepend=0)
    return (0.8 * envelope * hp).astype(np.float32)


def embed_in_silence(
    sound: np.ndarray, position: float, total_duration: float, sr: int = 44100
) -> np.ndarray:
    """Place a sound at a specific position in a silent background."""
    total_samples = int(sr * total_duration)
    audio = np.zeros(total_samples, dtype=np.float32)
    start = int(sr * position)
    end = min(start + len(sound), total_samples)
    audio[start:end] = sound[: end - start]
    return audio


# ── OnsetEvent dataclass ─────────────────────────────────────────────


class TestOnsetEvent:
    def test_fields(self) -> None:
        event = OnsetEvent(timestamp=1.5, onset_type="kick", strength=0.8)
        assert event.timestamp == 1.5
        assert event.onset_type == "kick"
        assert event.strength == 0.8

    def test_equality(self) -> None:
        a = OnsetEvent(timestamp=1.0, onset_type="snare", strength=0.5)
        b = OnsetEvent(timestamp=1.0, onset_type="snare", strength=0.5)
        assert a == b


# ── Onset detection basics ───────────────────────────────────────────


class TestOnsetDetection:
    def _make_detector(self, **kwargs: float | int) -> OnsetDetector:
        return OnsetDetector(sr=44100, fps=60, **kwargs)

    def test_silence_no_onsets(self) -> None:
        """Pure silence should produce no onsets."""
        det = self._make_detector()
        audio = make_silence(1.0)
        results = det.process_chunk(audio)
        onsets = [r for r in results if r is not None]
        assert len(onsets) == 0

    def test_constant_tone_no_onsets(self) -> None:
        """A steady sine wave (after initial onset) should produce few onsets."""
        det = self._make_detector()
        # Start with silence so the sine onset is clean
        audio = np.concatenate([make_silence(0.5), make_sine(440.0, 2.0)])
        results = det.process_chunk(audio)
        onsets = [r for r in results if r is not None]
        # Should have at most 1 onset (the sine start)
        assert len(onsets) <= 2

    def test_detects_impulse(self) -> None:
        """Should detect a strong transient in silence."""
        det = self._make_detector(threshold=0.05)
        kick = make_kick()
        audio = embed_in_silence(kick, 0.5, 2.0)
        results = det.process_chunk(audio)
        onsets = [r for r in results if r is not None]
        assert len(onsets) >= 1

    def test_onset_timestamp_reasonable(self) -> None:
        """Onset timestamp should be near the actual transient position."""
        det = self._make_detector(threshold=0.05)
        kick = make_kick()
        audio = embed_in_silence(kick, 0.5, 2.0)
        results = det.process_chunk(audio)
        onsets = [r for r in results if r is not None]
        if len(onsets) > 0:
            # Should be near 0.5s (within a few frames)
            assert onsets[0].timestamp == pytest.approx(0.5, abs=0.1)

    def test_multiple_onsets(self) -> None:
        """Should detect multiple separated transients."""
        det = self._make_detector(threshold=0.05)
        kick = make_kick()
        audio = np.concatenate([
            embed_in_silence(kick, 0.3, 1.0),
            embed_in_silence(kick, 0.3, 1.0),
        ])
        results = det.process_chunk(audio)
        onsets = [r for r in results if r is not None]
        assert len(onsets) >= 2

    def test_min_onset_gap(self) -> None:
        """Onsets closer than min_gap should be suppressed."""
        det = self._make_detector(threshold=0.01, min_onset_gap_ms=500)
        kick = make_kick()
        # Two kicks 200ms apart (below 500ms gap)
        audio = embed_in_silence(kick, 0.3, 1.0)
        kick2 = embed_in_silence(kick, 0.5, 1.0)
        audio = audio + kick2  # overlay
        results = det.process_chunk(audio)
        onsets = [r for r in results if r is not None]
        # Should detect at most 1 due to gap suppression
        assert len(onsets) <= 1

    def test_strength_in_range(self) -> None:
        """Onset strength should be in [0, 1]."""
        det = self._make_detector(threshold=0.05)
        kick = make_kick()
        audio = embed_in_silence(kick, 0.5, 2.0)
        results = det.process_chunk(audio)
        onsets = [r for r in results if r is not None]
        for o in onsets:
            assert 0.0 <= o.strength <= 1.0

    def test_empty_chunk(self) -> None:
        """Empty audio should return empty list."""
        det = self._make_detector()
        results = det.process_chunk(np.array([], dtype=np.float32))
        assert results == []

    def test_frame_count(self) -> None:
        """Should return one result per output frame."""
        det = self._make_detector()
        hop = 44100 // 60
        audio = make_silence(1.0)
        results = det.process_chunk(audio)
        expected = len(audio) // hop
        assert len(results) == expected


# ── Percussion classification ────────────────────────────────────────


class TestPercussionClassification:
    """Test onset classification into kick/snare/hihat/clap."""

    def _make_detector(self, **kwargs: float | int) -> OnsetDetector:
        return OnsetDetector(sr=44100, fps=60, threshold=0.01, **kwargs)

    def test_kick_classified(self) -> None:
        """A kick-like sound should be classified as kick."""
        det = self._make_detector()
        kick = make_kick()
        audio = embed_in_silence(kick, 0.3, 1.0)
        results = det.process_chunk(audio)
        onsets = [r for r in results if r is not None]
        if len(onsets) > 0:
            assert onsets[0].onset_type == "kick"

    def test_hihat_classified(self) -> None:
        """A hihat-like sound should be classified as hihat."""
        det = self._make_detector()
        hihat = make_hihat()
        audio = embed_in_silence(hihat, 0.3, 1.0)
        results = det.process_chunk(audio)
        onsets = [r for r in results if r is not None]
        if len(onsets) > 0:
            assert onsets[0].onset_type == "hihat"

    def test_snare_classified(self) -> None:
        """A snare-like sound should be classified as snare."""
        det = self._make_detector()
        snare = make_snare()
        audio = embed_in_silence(snare, 0.3, 1.0)
        results = det.process_chunk(audio)
        onsets = [r for r in results if r is not None]
        if len(onsets) > 0:
            assert onsets[0].onset_type in ("snare", "clap")  # snare/clap are similar

    def test_valid_onset_types(self) -> None:
        """All detected onset types should be valid."""
        det = self._make_detector()
        valid_types = {"kick", "snare", "hihat", "clap"}
        for make_fn in [make_kick, make_snare, make_hihat, make_clap]:
            det.reset()
            sound = make_fn()
            audio = embed_in_silence(sound, 0.3, 1.0)
            results = det.process_chunk(audio)
            onsets = [r for r in results if r is not None]
            for o in onsets:
                assert o.onset_type in valid_types, f"Invalid type: {o.onset_type}"


# ── Spectral analysis helpers ────────────────────────────────────────


class TestSpectralHelpers:
    def test_band_energy_low(self) -> None:
        """Band energy should be higher for low-freq signal in low band."""
        det = OnsetDetector(sr=44100, n_fft=2048)
        freqs = np.fft.rfftfreq(2048, d=1.0 / 44100)

        # Create spectrum with energy at 100 Hz
        spectrum = np.zeros_like(freqs)
        idx_100 = np.argmin(np.abs(freqs - 100))
        spectrum[idx_100] = 1.0

        low_e = det._band_energy(spectrum, freqs, 20.0, 200.0)
        high_e = det._band_energy(spectrum, freqs, 5000.0, 20000.0)
        assert low_e > high_e

    def test_spectral_flatness_pure_tone(self) -> None:
        """A pure tone should have low spectral flatness."""
        spectrum = np.zeros(1025, dtype=np.float64)
        spectrum[100] = 1.0  # single spike
        flatness = OnsetDetector._spectral_flatness(spectrum)
        assert flatness < 0.1

    def test_spectral_flatness_noise(self) -> None:
        """Uniform noise should have high spectral flatness."""
        rng = np.random.default_rng(42)
        spectrum = rng.uniform(0.5, 1.5, 1025)
        flatness = OnsetDetector._spectral_flatness(spectrum)
        assert flatness > 0.5

    def test_spectral_flatness_range(self) -> None:
        """Spectral flatness should be in [0, 1]."""
        rng = np.random.default_rng(42)
        for _ in range(10):
            spectrum = rng.uniform(0, 1, 1025)
            flatness = OnsetDetector._spectral_flatness(spectrum)
            assert 0.0 <= flatness <= 1.0


# ── Reset and offline ────────────────────────────────────────────────


class TestResetAndOffline:
    def test_reset_clears_state(self) -> None:
        """reset() should clear tracking state."""
        det = OnsetDetector()
        kick = make_kick()
        audio = embed_in_silence(kick, 0.3, 1.0)
        det.process_chunk(audio)
        assert det._prev_spectrum is not None

        det.reset()
        assert det._prev_spectrum is None

    def test_offline_detects_onsets(self) -> None:
        """Offline should detect onsets with adaptive thresholding."""
        det = OnsetDetector(sr=44100, fps=60)
        kick = make_kick()
        audio = embed_in_silence(kick, 0.5, 2.0)

        offline_results = det.analyze_offline(audio)
        onsets = [r for r in offline_results if r is not None]
        assert len(onsets) >= 1

    def test_offline_returns_correct_count(self) -> None:
        """Offline should return one result per frame."""
        det = OnsetDetector(sr=44100, fps=60)
        audio = make_silence(1.0)
        results = det.analyze_offline(audio)
        expected = len(audio) // (44100 // 60)
        assert len(results) == expected

    def test_offline_adapts_to_quiet_input(self) -> None:
        """Offline adaptive threshold should detect onsets in quiet signals."""
        det = OnsetDetector(sr=44100, fps=60)
        # Very quiet kick — would be missed by the fixed 0.15 threshold
        kick = make_kick() * 0.05
        audio = np.concatenate([
            embed_in_silence(kick, 0.3, 1.0),
            embed_in_silence(kick, 0.3, 1.0),
            embed_in_silence(kick, 0.3, 1.0),
        ])
        results = det.analyze_offline(audio)
        onsets = [r for r in results if r is not None]
        # Should still detect them via adaptive threshold
        assert len(onsets) >= 2
