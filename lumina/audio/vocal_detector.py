"""Language-agnostic vocal presence and energy detection.

Detects vocal presence in audio using spectral features characteristic
of human voice: harmonic structure in the 80-1100 Hz fundamental range,
formant energy in 300-3500 Hz, and harmonic-to-noise ratio.

This module operates on the full mix by default. For higher accuracy,
pre-separate vocals using demucs (see source_separator.py) and feed
the isolated vocal track. The detection is language-agnostic — it works
on signal-level features, not linguistic content.

Design choice: We use spectral harmonicity rather than ML models to
keep Phase 1 lightweight and avoid language bias. The harmonic ratio
(energy in harmonic partials vs total energy) is the primary indicator,
combined with spectral flux patterns typical of vocal onsets.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# Vocal frequency ranges (Hz)
_VOCAL_FUNDAMENTAL_LOW = 80.0    # Lowest male fundamental
_VOCAL_FUNDAMENTAL_HIGH = 1100.0  # Highest female fundamental
_FORMANT_LOW = 300.0
_FORMANT_HIGH = 3500.0
_VOCAL_PRESENCE_BAND = (300.0, 4000.0)  # Overall vocal presence band

# Smoothing defaults
_DEFAULT_SMOOTHING = 0.7


@dataclass(slots=True)
class VocalFrame:
    """Vocal analysis for a single time frame.

    Args:
        vocal_energy: 0.0-1.0 vocal presence level. High values indicate
            strong vocal content regardless of language.
        is_vocal: True if vocal presence exceeds detection threshold.
        harmonic_ratio: 0.0-1.0 ratio of harmonic to total energy in the
            vocal band. Higher = more likely voiced content.
    """

    vocal_energy: float
    is_vocal: bool
    harmonic_ratio: float


class VocalDetector:
    """Language-agnostic vocal presence detector.

    Detects vocals using spectral harmonicity in the vocal frequency
    range. Works on full mix audio or pre-separated vocal tracks.

    The detector computes:
    1. Energy in the vocal presence band (300-4000 Hz) relative to total.
    2. Harmonic ratio via autocorrelation in the vocal fundamental range.
    3. Smoothed vocal energy combining both indicators.

    Args:
        sr: Sample rate in Hz.
        fps: Output frame rate.
        n_fft: FFT size for spectral analysis.
        smoothing: EMA smoothing for vocal energy (0=none, 1=frozen).
        threshold: Minimum vocal_energy to set is_vocal=True.
    """

    def __init__(
        self,
        sr: int = 44100,
        fps: int = 60,
        n_fft: int = 2048,
        smoothing: float = _DEFAULT_SMOOTHING,
        threshold: float = 0.15,
    ) -> None:
        self._sr = sr
        self._fps = fps
        self._n_fft = n_fft
        self._smoothing = smoothing
        self._threshold = threshold
        self._hop_length = sr // fps

        # Pre-compute frequency bins
        self._freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)

        # Autocorrelation lag range for vocal pitch detection
        # Lags corresponding to 80-1100 Hz fundamentals
        self._min_lag = max(1, int(sr / _VOCAL_FUNDAMENTAL_HIGH))
        self._max_lag = min(n_fft // 2, int(sr / _VOCAL_FUNDAMENTAL_LOW))

        self.reset()

    def reset(self) -> None:
        """Reset all internal state."""
        self._prev_vocal_energy = 0.0
        self._peak_vocal = 1e-6

    # ── Public API ────────────────────────────────────────────────

    def process_chunk(self, chunk: np.ndarray) -> list[VocalFrame]:
        """Process a chunk of audio and return vocal analysis per frame.

        Args:
            chunk: Mono float32 audio samples, normalized to [-1, 1].

        Returns:
            List of VocalFrame, one per output frame.
        """
        if len(chunk) == 0:
            return []

        num_frames = max(1, len(chunk) // self._hop_length)
        frames: list[VocalFrame] = []

        for i in range(num_frames):
            start = i * self._hop_length
            end = min(start + self._hop_length, len(chunk))
            window = chunk[start:end]
            frames.append(self._analyze_window(window))

        return frames

    def analyze_offline(self, audio: np.ndarray) -> list[VocalFrame]:
        """Analyze a complete audio signal.

        Args:
            audio: Mono float32 audio, normalized to [-1, 1].

        Returns:
            List of VocalFrame at fps rate.
        """
        self.reset()
        return self.process_chunk(audio)

    # ── Internal ──────────────────────────────────────────────────

    def _analyze_window(self, window: np.ndarray) -> VocalFrame:
        """Compute vocal features for a single window.

        Args:
            window: Audio samples for this frame.

        Returns:
            VocalFrame with vocal energy and harmonicity.
        """
        if len(window) == 0 or np.max(np.abs(window)) < 1e-8:
            self._prev_vocal_energy = self._smoothing * self._prev_vocal_energy
            return VocalFrame(
                vocal_energy=self._prev_vocal_energy,
                is_vocal=False,
                harmonic_ratio=0.0,
            )

        # Spectral analysis
        vocal_band_ratio = self._vocal_band_energy(window)

        # Harmonicity via autocorrelation
        harmonic_ratio = self._harmonic_ratio(window)

        # Combine indicators: vocal band presence x harmonicity
        # Both must be present for confident vocal detection
        raw_vocal = vocal_band_ratio * (0.4 + 0.6 * harmonic_ratio)
        raw_vocal = min(1.0, raw_vocal)

        # Adaptive normalization
        if raw_vocal > self._peak_vocal:
            self._peak_vocal = raw_vocal
        else:
            self._peak_vocal *= 0.9998

        normalized = raw_vocal / self._peak_vocal if self._peak_vocal > 1e-6 else 0.0
        normalized = min(1.0, normalized)

        # Smooth
        vocal_energy = (
            self._smoothing * self._prev_vocal_energy
            + (1 - self._smoothing) * normalized
        )
        self._prev_vocal_energy = vocal_energy

        return VocalFrame(
            vocal_energy=vocal_energy,
            is_vocal=vocal_energy >= self._threshold,
            harmonic_ratio=harmonic_ratio,
        )

    def _vocal_band_energy(self, window: np.ndarray) -> float:
        """Compute ratio of energy in the vocal presence band to total.

        Args:
            window: Audio samples.

        Returns:
            0.0-1.0 ratio of vocal band energy to total energy.
        """
        windowed = window * np.hanning(len(window)).astype(window.dtype)
        if len(windowed) < self._n_fft:
            padded = np.zeros(self._n_fft, dtype=window.dtype)
            padded[: len(windowed)] = windowed
        else:
            padded = windowed[: self._n_fft]

        spectrum = np.abs(np.fft.rfft(padded)) ** 2  # power spectrum
        total = float(np.sum(spectrum))
        if total < 1e-20:
            return 0.0

        vocal_mask = (
            (self._freqs >= _VOCAL_PRESENCE_BAND[0])
            & (self._freqs <= _VOCAL_PRESENCE_BAND[1])
        )
        vocal_energy = float(np.sum(spectrum[vocal_mask]))

        return vocal_energy / total

    def _harmonic_ratio(self, window: np.ndarray) -> float:
        """Estimate harmonicity via normalized autocorrelation.

        Voiced speech/singing produces periodic signals with strong
        autocorrelation peaks at the fundamental period. Unvoiced
        sounds (noise, consonants) lack such periodicity.

        Args:
            window: Audio samples.

        Returns:
            0.0-1.0 harmonic ratio. Higher = more harmonic (voiced).
        """
        if len(window) < self._max_lag + 1:
            return 0.0

        # Normalize
        sig = window - np.mean(window)
        norm = float(np.sum(sig**2))
        if norm < 1e-20:
            return 0.0

        # Compute autocorrelation only for vocal pitch lags
        max_lag = min(self._max_lag, len(sig) - 1)
        if max_lag <= self._min_lag:
            return 0.0

        # Efficient autocorrelation via FFT
        n = len(sig)
        fft_size = 1
        while fft_size < 2 * n:
            fft_size *= 2
        sig_fft = np.fft.rfft(sig, n=fft_size)
        autocorr = np.fft.irfft(sig_fft * np.conj(sig_fft))[:n]

        # Normalized autocorrelation in vocal lag range
        lags = autocorr[self._min_lag : max_lag + 1] / norm

        if len(lags) == 0:
            return 0.0

        peak = float(np.max(lags))
        return max(0.0, min(1.0, peak))
