"""Energy envelope tracking with derivative computation.

Tracks overall energy, energy rate-of-change (derivative), spectral
centroid (brightness), and sub-bass energy from audio. Designed for both
streaming (chunk-by-chunk) and offline analysis.

The energy envelope uses RMS in overlapping windows, smoothed via
exponential moving average. The derivative is computed as the first
difference of the smoothed energy, also smoothed to reduce jitter.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# Frequency band boundaries (Hz)
_SUB_BASS_LOW = 20.0
_SUB_BASS_HIGH = 80.0


@dataclass(slots=True)
class EnergyFrame:
    """Energy analysis for a single time frame.

    Args:
        energy: 0.0-1.0 overall energy level (RMS-based).
        energy_derivative: Rate of change; positive = rising, negative = falling.
        spectral_centroid: Brightness indicator in Hz.
        sub_bass_energy: 0.0-1.0 energy in the 20-80 Hz band.
    """

    energy: float
    energy_derivative: float
    spectral_centroid: float
    sub_bass_energy: float


class EnergyTracker:
    """Real-time energy envelope and spectral feature tracker.

    Computes RMS energy, its smoothed derivative, spectral centroid, and
    sub-bass energy from audio. Supports streaming and offline modes.

    Energy values are normalized to [0, 1] using an adaptive peak tracker
    that decays slowly, so quiet sections register as low energy even if
    the absolute RMS is nonzero.

    Args:
        sr: Sample rate in Hz.
        fps: Output frame rate (frames per second).
        smoothing: EMA smoothing factor for energy (0=no smoothing, 1=frozen).
        derivative_smoothing: EMA smoothing factor for the derivative.
        n_fft: FFT size for spectral analysis.
    """

    def __init__(
        self,
        sr: int = 44100,
        fps: int = 60,
        smoothing: float = 0.6,
        derivative_smoothing: float = 0.4,
        n_fft: int = 2048,
    ) -> None:
        self._sr = sr
        self._fps = fps
        self._smoothing = smoothing
        self._derivative_smoothing = derivative_smoothing
        self._n_fft = n_fft
        self._hop_length = sr // fps

        self.reset()

    def reset(self) -> None:
        """Reset all internal state for a fresh analysis."""
        self._prev_energy = 0.0
        self._prev_derivative = 0.0
        self._peak_energy = 1e-6  # adaptive peak for normalization (avoid /0)
        self._peak_decay = 0.9995  # slow decay per frame

    # ── Public API ────────────────────────────────────────────────

    def process_chunk(self, chunk: np.ndarray) -> list[EnergyFrame]:
        """Process a chunk of audio and return energy frames at fps rate.

        Args:
            chunk: Mono float32 audio samples, normalized to [-1, 1].

        Returns:
            List of EnergyFrame, one per output frame spanning the chunk.
        """
        if len(chunk) == 0:
            return []

        num_frames = max(1, len(chunk) // self._hop_length)
        frames: list[EnergyFrame] = []

        for i in range(num_frames):
            start = i * self._hop_length
            end = min(start + self._hop_length, len(chunk))
            window = chunk[start:end]

            frame = self._analyze_window(window)
            frames.append(frame)

        return frames

    def analyze_offline(self, audio: np.ndarray) -> list[EnergyFrame]:
        """Analyze a complete audio signal.

        Processes the entire signal at once, which allows better
        normalization since the global peak is known.

        Args:
            audio: Mono float32 audio, normalized to [-1, 1].

        Returns:
            List of EnergyFrame at fps rate for the entire signal.
        """
        self.reset()

        if len(audio) == 0:
            return []

        num_frames = max(1, len(audio) // self._hop_length)

        # Pre-compute all raw RMS values for global normalization
        raw_rms = np.zeros(num_frames, dtype=np.float64)
        for i in range(num_frames):
            start = i * self._hop_length
            end = min(start + self._hop_length, len(audio))
            window = audio[start:end]
            raw_rms[i] = float(np.sqrt(np.mean(window**2)))

        global_peak = float(np.max(raw_rms)) if len(raw_rms) > 0 else 1e-6
        if global_peak < 1e-6:
            global_peak = 1e-6
        self._peak_energy = global_peak

        # Now analyze frame by frame with the known peak
        frames: list[EnergyFrame] = []
        for i in range(num_frames):
            start = i * self._hop_length
            end = min(start + self._hop_length, len(audio))
            window = audio[start:end]
            frame = self._analyze_window(window)
            frames.append(frame)

        return frames

    def analyze_offline_with_bass_stem(
        self,
        audio: np.ndarray,
        bass_stem: np.ndarray,
    ) -> list[EnergyFrame]:
        """Analyze audio with sub-bass computed from an isolated bass stem.

        Uses the full mix for energy, derivative, and spectral centroid,
        but replaces the sub-bass energy with RMS from the demucs-isolated
        bass stem (already band-limited, no FFT filtering needed).

        Args:
            audio: Mono float32 full-mix audio.
            bass_stem: Mono float32 isolated bass stem (same length as audio).

        Returns:
            List of EnergyFrame at fps rate.
        """
        # Run standard analysis on the full mix
        frames = self.analyze_offline(audio)

        if len(frames) == 0:
            return frames

        num_frames = len(frames)

        # Compute per-frame bass RMS from the isolated stem
        bass_rms = np.zeros(num_frames, dtype=np.float64)
        for i in range(num_frames):
            start = i * self._hop_length
            end = min(start + self._hop_length, len(bass_stem))
            if end > start:
                window = bass_stem[start:end]
                bass_rms[i] = float(np.sqrt(np.mean(window**2)))

        # Normalize using global peak of the bass stem
        peak = float(np.max(bass_rms)) if len(bass_rms) > 0 else 1e-6
        if peak < 1e-6:
            peak = 1e-6

        # Replace sub_bass_energy in each frame
        for i in range(num_frames):
            normalized = min(1.0, bass_rms[i] / peak)
            frames[i] = EnergyFrame(
                energy=frames[i].energy,
                energy_derivative=frames[i].energy_derivative,
                spectral_centroid=frames[i].spectral_centroid,
                sub_bass_energy=normalized,
            )

        return frames

    # ── Internal ──────────────────────────────────────────────────

    def _analyze_window(self, window: np.ndarray) -> EnergyFrame:
        """Compute energy features for a single analysis window.

        Args:
            window: Audio samples for this frame.

        Returns:
            EnergyFrame with smoothed energy, derivative, and spectral features.
        """
        # RMS energy
        rms = float(np.sqrt(np.mean(window**2))) if len(window) > 0 else 0.0

        # Update adaptive peak
        if rms > self._peak_energy:
            self._peak_energy = rms
        else:
            self._peak_energy *= self._peak_decay

        # Normalize to [0, 1]
        raw_energy = rms / self._peak_energy if self._peak_energy > 1e-6 else 0.0
        raw_energy = min(1.0, raw_energy)

        # Smooth energy via EMA
        energy = self._smoothing * self._prev_energy + (1 - self._smoothing) * raw_energy

        # Derivative (smoothed)
        raw_derivative = energy - self._prev_energy
        derivative = (
            self._derivative_smoothing * self._prev_derivative
            + (1 - self._derivative_smoothing) * raw_derivative
        )

        self._prev_energy = energy
        self._prev_derivative = derivative

        # Spectral centroid
        centroid = self._compute_spectral_centroid(window)

        # Sub-bass energy
        sub_bass = self._compute_sub_bass_energy(window)

        return EnergyFrame(
            energy=energy,
            energy_derivative=derivative,
            spectral_centroid=centroid,
            sub_bass_energy=sub_bass,
        )

    def _compute_spectral_centroid(self, window: np.ndarray) -> float:
        """Compute spectral centroid (brightness) in Hz.

        Args:
            window: Audio samples.

        Returns:
            Spectral centroid in Hz, or 0.0 for silence.
        """
        # Apply Hann window to data, then zero-pad to n_fft
        windowed = window * np.hanning(len(window)).astype(window.dtype)
        if len(windowed) < self._n_fft:
            padded = np.zeros(self._n_fft, dtype=window.dtype)
            padded[: len(windowed)] = windowed
        else:
            padded = windowed[: self._n_fft]

        spectrum = np.abs(np.fft.rfft(padded))
        freqs = np.fft.rfftfreq(self._n_fft, d=1.0 / self._sr)

        total = float(np.sum(spectrum))
        if total < 1e-10:
            return 0.0

        return float(np.sum(freqs * spectrum) / total)

    def _compute_sub_bass_energy(self, window: np.ndarray) -> float:
        """Compute normalized energy in the sub-bass band (20-80 Hz).

        Args:
            window: Audio samples.

        Returns:
            0.0-1.0 sub-bass energy relative to total energy.
        """
        if len(window) < self._n_fft:
            padded = np.zeros(self._n_fft, dtype=window.dtype)
            padded[: len(window)] = window
        else:
            padded = window[: self._n_fft]

        spectrum = np.abs(np.fft.rfft(padded))
        freqs = np.fft.rfftfreq(self._n_fft, d=1.0 / self._sr)

        total_energy = float(np.sum(spectrum**2))
        if total_energy < 1e-10:
            return 0.0

        sub_mask = (freqs >= _SUB_BASS_LOW) & (freqs <= _SUB_BASS_HIGH)
        sub_energy = float(np.sum(spectrum[sub_mask] ** 2))

        return min(1.0, sub_energy / total_energy)
