"""Onset detection with percussive instrument classification.

Detects transient onsets in audio and classifies them as kick, snare,
hihat, or clap based on spectral shape. Uses band-limited energy ratios
for classification — no ML models required.

Classification heuristic (based on spectral energy distribution):
- **Kick:** Dominant energy below 200 Hz, low spectral centroid.
- **Snare:** Broadband energy with strong mid-band (200-5000 Hz) and
  noise-like character (high spectral flatness).
- **Hihat:** Dominant energy above 5000 Hz, high spectral centroid.
- **Clap:** Similar to snare but with less low-end, mid-high focus
  (1000-8000 Hz), and typically higher spectral flatness.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# Onset detection constants
_DEFAULT_THRESHOLD = 0.15  # Minimum spectral flux for onset
_MIN_ONSET_GAP_MS = 30.0  # Minimum gap between onsets (ms)

# Classification frequency bands (Hz)
_LOW_BAND = (20.0, 200.0)
_MID_BAND = (200.0, 5000.0)
_HIGH_BAND = (5000.0, 20000.0)
_MID_HIGH_BAND = (1000.0, 8000.0)

# Classification thresholds
_KICK_LOW_RATIO = 0.45  # Minimum low-band ratio for kick
_HIHAT_HIGH_RATIO = 0.4  # Minimum high-band ratio for hihat
_SNARE_FLATNESS = 0.3  # Minimum spectral flatness for snare/clap
_CLAP_MID_HIGH_RATIO = 0.5  # Minimum mid-high ratio for clap (vs snare)


@dataclass(slots=True)
class OnsetEvent:
    """A detected onset with instrument classification.

    Args:
        timestamp: Time of onset in seconds (relative to chunk start).
        onset_type: Classification: "kick", "snare", "hihat", or "clap".
        strength: 0.0-1.0 onset strength (spectral flux magnitude).
    """

    timestamp: float
    onset_type: str
    strength: float


class OnsetDetector:
    """Real-time onset detector with percussive classification.

    Uses spectral flux for onset detection and frequency band analysis
    for drum/percussion classification. Operates frame-by-frame in
    streaming mode.

    Args:
        sr: Sample rate in Hz.
        fps: Output frame rate (how often to check for onsets).
        n_fft: FFT size for spectral analysis.
        threshold: Minimum spectral flux to trigger an onset.
        min_onset_gap_ms: Minimum gap between consecutive onsets in ms.
    """

    def __init__(
        self,
        sr: int = 44100,
        fps: int = 60,
        n_fft: int = 2048,
        threshold: float = _DEFAULT_THRESHOLD,
        min_onset_gap_ms: float = _MIN_ONSET_GAP_MS,
    ) -> None:
        self._sr = sr
        self._fps = fps
        self._n_fft = n_fft
        self._threshold = threshold
        self._min_onset_gap_samples = int(sr * min_onset_gap_ms / 1000.0)
        self._hop_length = sr // fps

        self.reset()

    def reset(self) -> None:
        """Reset all internal state."""
        self._prev_spectrum: np.ndarray | None = None
        self._samples_since_onset = self._min_onset_gap_samples  # allow onset on first frame
        self._total_samples = 0

    # ── Public API ────────────────────────────────────────────────

    def process_chunk(self, chunk: np.ndarray) -> list[OnsetEvent | None]:
        """Process a chunk of audio and return onset info per frame.

        Args:
            chunk: Mono float32 audio samples, normalized to [-1, 1].

        Returns:
            List of (OnsetEvent or None), one per output frame. None means
            no onset was detected in that frame.
        """
        if len(chunk) == 0:
            return []

        num_frames = max(1, len(chunk) // self._hop_length)
        chunk_start_time = self._total_samples / self._sr
        results: list[OnsetEvent | None] = []

        for i in range(num_frames):
            start = i * self._hop_length
            end = min(start + self._hop_length, len(chunk))
            window = chunk[start:end]

            frame_time = chunk_start_time + start / self._sr
            onset = self._analyze_frame(window, frame_time)
            results.append(onset)

        self._total_samples += len(chunk)
        return results

    def analyze_offline(self, audio: np.ndarray) -> list[OnsetEvent | None]:
        """Analyze a complete audio signal with adaptive thresholding.

        Uses a two-pass approach: first computes all spectral flux values,
        then sets the detection threshold relative to the signal's own
        flux distribution. This adapts to different input types (full mix
        vs isolated drum stems from demucs).

        Args:
            audio: Mono float32 audio, normalized to [-1, 1].

        Returns:
            List of (OnsetEvent or None) at fps rate.
        """
        self.reset()

        if len(audio) == 0:
            return []

        num_frames = max(1, len(audio) // self._hop_length)

        # Pass 1: compute all spectra and flux values
        spectra: list[np.ndarray] = []
        flux_values = np.zeros(num_frames)

        for i in range(num_frames):
            start = i * self._hop_length
            end = min(start + self._hop_length, len(audio))
            window = audio[start:end]
            spectrum = self._compute_spectrum(window)
            spectra.append(spectrum)

            if i > 0:
                flux_values[i] = self._spectral_flux_pair(spectra[i - 1], spectrum)

        # Adaptive threshold: use the flux distribution
        # Only consider positive (non-zero) flux values
        positive_flux = flux_values[flux_values > 1e-10]

        if len(positive_flux) > 0:
            # Threshold = median + 0.5 * (90th percentile - median)
            # This adapts to the signal level while staying selective
            median_flux = float(np.median(positive_flux))
            p90_flux = float(np.percentile(positive_flux, 90))
            adaptive_threshold = median_flux + 0.5 * (p90_flux - median_flux)
            # Floor: never go below 1% of the 90th percentile
            adaptive_threshold = max(adaptive_threshold, p90_flux * 0.01)
        else:
            adaptive_threshold = self._threshold

        # Pass 2: detect onsets using the adaptive threshold
        results: list[OnsetEvent | None] = []
        samples_since_onset = self._min_onset_gap_samples

        for i in range(num_frames):
            frame_time = i * self._hop_length / self._sr
            samples_since_onset += self._hop_length

            if (
                flux_values[i] >= adaptive_threshold
                and samples_since_onset >= self._min_onset_gap_samples
            ):
                onset_type = self._classify_onset(spectra[i])
                strength = min(1.0, flux_values[i] / (adaptive_threshold * 5))
                results.append(
                    OnsetEvent(
                        timestamp=frame_time,
                        onset_type=onset_type,
                        strength=strength,
                    )
                )
                samples_since_onset = 0
            else:
                results.append(None)

        return results

    @staticmethod
    def _spectral_flux_pair(prev: np.ndarray, curr: np.ndarray) -> float:
        """Compute half-wave rectified spectral flux between two spectra.

        Args:
            prev: Previous frame magnitude spectrum.
            curr: Current frame magnitude spectrum.

        Returns:
            Non-negative spectral flux value.
        """
        min_len = min(len(prev), len(curr))
        diff = curr[:min_len] - prev[:min_len]
        positive_diff = np.maximum(diff, 0.0)
        return float(np.mean(positive_diff))

    # ── Internal ──────────────────────────────────────────────────

    def _analyze_frame(self, window: np.ndarray, frame_time: float) -> OnsetEvent | None:
        """Detect and classify onset in a single frame.

        Args:
            window: Audio samples for this frame.
            frame_time: Absolute time of this frame in seconds.

        Returns:
            OnsetEvent if onset detected, None otherwise.
        """
        spectrum = self._compute_spectrum(window)

        # Spectral flux (half-wave rectified difference)
        flux = self._spectral_flux(spectrum)

        self._samples_since_onset += len(window)

        if (
            flux >= self._threshold
            and self._samples_since_onset >= self._min_onset_gap_samples
        ):
            onset_type = self._classify_onset(spectrum)
            strength = min(1.0, flux / (self._threshold * 5))

            self._samples_since_onset = 0
            self._prev_spectrum = spectrum

            return OnsetEvent(
                timestamp=frame_time,
                onset_type=onset_type,
                strength=strength,
            )

        self._prev_spectrum = spectrum
        return None

    def _compute_spectrum(self, window: np.ndarray) -> np.ndarray:
        """Compute magnitude spectrum with windowing.

        Args:
            window: Audio samples.

        Returns:
            Magnitude spectrum array.
        """
        windowed = window * np.hanning(len(window)).astype(window.dtype)
        if len(windowed) < self._n_fft:
            padded = np.zeros(self._n_fft, dtype=window.dtype)
            padded[: len(windowed)] = windowed
        else:
            padded = windowed[: self._n_fft]

        return np.abs(np.fft.rfft(padded))

    def _spectral_flux(self, spectrum: np.ndarray) -> float:
        """Compute half-wave rectified spectral flux.

        Measures the increase in spectral energy compared to the
        previous frame. Only positive changes count (half-wave
        rectification), which focuses on onsets rather than offsets.

        Args:
            spectrum: Current magnitude spectrum.

        Returns:
            Non-negative spectral flux value.
        """
        if self._prev_spectrum is None:
            return 0.0

        # Ensure same length
        min_len = min(len(spectrum), len(self._prev_spectrum))
        diff = spectrum[:min_len] - self._prev_spectrum[:min_len]

        # Half-wave rectify (only positive changes)
        positive_diff = np.maximum(diff, 0.0)
        return float(np.mean(positive_diff))

    def _classify_onset(self, spectrum: np.ndarray) -> str:
        """Classify an onset as kick, snare, hihat, or clap.

        Uses the spectral shape of the onset frame to determine
        the most likely percussion instrument.

        Args:
            spectrum: Magnitude spectrum at onset.

        Returns:
            One of "kick", "snare", "hihat", "clap".
        """
        freqs = np.fft.rfftfreq(self._n_fft, d=1.0 / self._sr)

        # Band energies
        low_energy = self._band_energy(spectrum, freqs, *_LOW_BAND)
        mid_energy = self._band_energy(spectrum, freqs, *_MID_BAND)
        high_energy = self._band_energy(spectrum, freqs, *_HIGH_BAND)
        mid_high_energy = self._band_energy(spectrum, freqs, *_MID_HIGH_BAND)

        total_energy = low_energy + mid_energy + high_energy
        if total_energy < 1e-10:
            return "kick"  # default for silence edge case

        low_ratio = low_energy / total_energy
        high_ratio = high_energy / total_energy
        mid_high_ratio = mid_high_energy / total_energy

        # Spectral flatness (geometric mean / arithmetic mean)
        flatness = self._spectral_flatness(spectrum)

        # Classification decision tree
        if low_ratio >= _KICK_LOW_RATIO:
            return "kick"

        if high_ratio >= _HIHAT_HIGH_RATIO:
            return "hihat"

        # Snare vs clap: both are broadband with some noise character
        if flatness >= _SNARE_FLATNESS:
            if mid_high_ratio >= _CLAP_MID_HIGH_RATIO and low_ratio < 0.15:
                return "clap"
            return "snare"

        # Default: if nothing else matches strongly, use centroid
        centroid = float(np.sum(freqs * spectrum) / np.sum(spectrum)) if np.sum(spectrum) > 0 else 0
        if centroid > 5000:
            return "hihat"
        if centroid < 500:
            return "kick"
        return "snare"

    @staticmethod
    def _band_energy(
        spectrum: np.ndarray,
        freqs: np.ndarray,
        low_hz: float,
        high_hz: float,
    ) -> float:
        """Compute energy in a frequency band.

        Args:
            spectrum: Magnitude spectrum.
            freqs: Frequency values for each bin.
            low_hz: Lower band edge in Hz.
            high_hz: Upper band edge in Hz.

        Returns:
            Sum of squared magnitudes in the band.
        """
        mask = (freqs >= low_hz) & (freqs <= high_hz)
        return float(np.sum(spectrum[mask] ** 2))

    @staticmethod
    def _spectral_flatness(spectrum: np.ndarray) -> float:
        """Compute spectral flatness (Wiener entropy).

        Ratio of geometric mean to arithmetic mean of the spectrum.
        1.0 = white noise, 0.0 = pure tone.

        Args:
            spectrum: Magnitude spectrum.

        Returns:
            Spectral flatness in [0, 1].
        """
        # Filter out zeros to avoid log(0)
        nonzero = spectrum[spectrum > 0]
        if len(nonzero) < 2:
            return 0.0

        log_mean = float(np.mean(np.log(nonzero + 1e-20)))
        geometric_mean = np.exp(log_mean)
        arithmetic_mean = float(np.mean(nonzero))

        if arithmetic_mean < 1e-20:
            return 0.0

        return min(1.0, geometric_mean / arithmetic_mean)
