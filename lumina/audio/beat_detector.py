"""Beat detection module using madmom RNNs and DBN tracking.

Provides both streaming (chunk-by-chunk) and offline (whole-file) beat
detection, producing BeatInfo frames that populate MusicState fields.

Streaming mode buffers audio and periodically runs madmom's RNN beat
processor on a sliding window, then interpolates beat phase between
analysis passes. Offline mode uses the downbeat-aware processor for
bar-aligned results.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# Lazy-loaded madmom imports (heavyweight NN weight loading)
_beat_nn_cache: object | None = None
_downbeat_nn_cache: object | None = None


def _get_beat_nn() -> object:
    """Get or create the cached RNNBeatProcessor."""
    global _beat_nn_cache  # noqa: PLW0603
    if _beat_nn_cache is None:
        from madmom.features.beats import RNNBeatProcessor

        _beat_nn_cache = RNNBeatProcessor()
    return _beat_nn_cache


def _get_downbeat_nn() -> object:
    """Get or create the cached RNNDownBeatProcessor."""
    global _downbeat_nn_cache  # noqa: PLW0603
    if _downbeat_nn_cache is None:
        from madmom.features.downbeats import RNNDownBeatProcessor

        _downbeat_nn_cache = RNNDownBeatProcessor()
    return _downbeat_nn_cache


@dataclass(slots=True)
class BeatInfo:
    """Beat tracking state for a single time frame.

    Args:
        bpm: Current tempo estimate in beats per minute.
        beat_phase: 0.0-1.0 position within the current beat.
        bar_phase: 0.0-1.0 position within the current bar.
        is_beat: True on the frame closest to a beat onset.
        is_downbeat: True on the frame closest to a bar downbeat.
    """

    bpm: float
    beat_phase: float
    bar_phase: float
    is_beat: bool
    is_downbeat: bool


class BeatDetector:
    """Real-time beat and downbeat detector.

    Uses madmom's RNN beat processor for activation extraction and DBN
    (Dynamic Bayesian Network) for beat tracking. Supports two modes:

    - **Streaming** (``process_chunk``): Feed audio chunks incrementally.
      Analysis runs periodically on a sliding buffer; phase is interpolated
      between analysis passes.
    - **Offline** (``analyze_offline``): Process a complete audio signal at
      once with downbeat-aware tracking for proper bar alignment.

    Args:
        sr: Sample rate in Hz.
        fps: Output frame rate (activation frames per second).
        beats_per_bar: Time signature numerator (default 4 for 4/4).
        min_bpm: Minimum tempo for DBN tracking.
        max_bpm: Maximum tempo for DBN tracking.
    """

    def __init__(
        self,
        sr: int = 44100,
        fps: int = 100,
        beats_per_bar: int = 4,
        min_bpm: float = 55.0,
        max_bpm: float = 215.0,
    ) -> None:
        self._sr = sr
        self._fps = fps
        self._beats_per_bar = beats_per_bar
        self._min_bpm = min_bpm
        self._max_bpm = max_bpm
        self._hop_length = sr // fps

        # Buffer configuration
        self._max_buffer_seconds = 10.0
        self._min_analysis_seconds = 3.0
        self._analysis_interval_seconds = 1.0
        self._bpm_smoothing = 0.3

        self.reset()

    def reset(self) -> None:
        """Reset all internal state for a fresh analysis."""
        self._buffer = np.array([], dtype=np.float32)
        self._samples_since_analysis = 0
        self._total_samples = 0

        # Tracking state
        self._bpm = 120.0
        self._phase_ref_time = 0.0
        self._base_beat_number = 0
        self._prev_abs_beat = 0
        self._tracking_started = False
        self._has_beats = False

    # ── Public API ────────────────────────────────────────────────

    def process_chunk(self, chunk: np.ndarray) -> list[BeatInfo]:
        """Process a chunk of audio and return beat info at fps rate.

        Buffers audio internally and runs madmom analysis when enough
        new audio has accumulated. Between analyses, beat phase is
        interpolated from the current BPM estimate.

        Args:
            chunk: Mono float32 audio samples, normalized to [-1, 1].

        Returns:
            List of BeatInfo, one per output frame spanning the chunk.
        """
        chunk_start_time = self._total_samples / self._sr

        # Buffer management
        self._buffer = np.append(self._buffer, chunk)
        self._samples_since_analysis += len(chunk)
        self._total_samples += len(chunk)

        max_samples = int(self._sr * self._max_buffer_seconds)
        if len(self._buffer) > max_samples:
            self._buffer = self._buffer[-max_samples:]

        # Run analysis when enough new audio has arrived
        min_samples = int(self._sr * self._min_analysis_seconds)
        interval_samples = int(self._sr * self._analysis_interval_seconds)
        if (
            len(self._buffer) >= min_samples
            and self._samples_since_analysis >= interval_samples
        ):
            self._run_streaming_analysis()
            self._samples_since_analysis = 0

        # Generate output frames
        chunk_duration = len(chunk) / self._sr
        num_frames = max(1, round(chunk_duration * self._fps))

        return [
            self._get_frame_info(chunk_start_time + i / self._fps)
            for i in range(num_frames)
        ]

    def analyze_offline(self, audio: np.ndarray) -> list[BeatInfo]:
        """Analyze a complete audio signal with downbeat tracking.

        Uses madmom's downbeat-aware processor for proper bar alignment.
        More accurate than streaming since it sees the full signal.

        Args:
            audio: Mono float32 audio, normalized to [-1, 1].

        Returns:
            List of BeatInfo at fps rate for the entire signal.
        """
        self.reset()
        duration = len(audio) / self._sr
        num_frames = max(1, round(duration * self._fps))

        from madmom.audio.signal import Signal
        from madmom.features.downbeats import DBNDownBeatTrackingProcessor

        sig = Signal(audio, sample_rate=self._sr)
        downbeat_nn = _get_downbeat_nn()
        activations = downbeat_nn(sig)

        dbn = DBNDownBeatTrackingProcessor(
            beats_per_bar=[self._beats_per_bar],
            fps=self._fps,
            min_bpm=self._min_bpm,
            max_bpm=self._max_bpm,
        )
        results = dbn(activations)

        if len(results) == 0:
            return [
                BeatInfo(bpm=self._bpm, beat_phase=0.0, bar_phase=0.0,
                         is_beat=False, is_downbeat=False)
            ] * num_frames

        beat_times: np.ndarray = results[:, 0]
        beat_positions = results[:, 1].astype(int)

        bpm = self._compute_bpm(beat_times)

        # Find first downbeat for bar alignment
        first_db_idx: int | None = None
        for i, pos in enumerate(beat_positions):
            if pos == 1:
                first_db_idx = i
                break

        self._update_tracking(beat_times, bpm=bpm, first_downbeat_idx=first_db_idx)

        # Build frame → (is_beat, is_downbeat) map from ground truth
        beat_frame_map: dict[int, tuple[bool, bool]] = {}
        for t_val, pos in zip(beat_times, beat_positions):
            frame = round(float(t_val) * self._fps)
            if 0 <= frame < num_frames:
                beat_frame_map[frame] = (True, pos == 1)

        frame_results: list[BeatInfo] = []
        for f in range(num_frames):
            t = f / self._fps
            info = self._get_frame_info(t)
            # Override is_beat/is_downbeat with madmom ground truth
            if f in beat_frame_map:
                info.is_beat = True
                info.is_downbeat = beat_frame_map[f][1]
            else:
                info.is_beat = False
                info.is_downbeat = False
            frame_results.append(info)

        return frame_results

    # ── Internal: tracking state ──────────────────────────────────

    def _update_tracking(
        self,
        beat_times: np.ndarray,
        *,
        bpm: float | None = None,
        first_downbeat_idx: int | None = None,
    ) -> None:
        """Update internal tracking state from detected beats.

        Can be called directly for testing with known beat positions.

        Args:
            beat_times: Sorted array of absolute beat timestamps.
            bpm: BPM override. If None, computed from beat intervals.
            first_downbeat_idx: Index of first downbeat in beat_times
                (for bar alignment in offline mode).
        """
        if len(beat_times) == 0:
            return

        new_bpm = bpm if bpm is not None else self._compute_bpm(beat_times)

        if self._has_beats:
            self._bpm = (
                self._bpm_smoothing * self._bpm
                + (1 - self._bpm_smoothing) * new_bpm
            )
        else:
            self._bpm = new_bpm

        last_beat = float(beat_times[-1])

        if self._has_beats:
            # Streaming: maintain continuous beat numbering
            beat_period = 60.0 / self._bpm
            elapsed = last_beat - self._phase_ref_time
            beats_elapsed = round(elapsed / beat_period)
            self._base_beat_number += beats_elapsed
        elif first_downbeat_idx is not None:
            # Offline with downbeat info: align so first downbeat = beat 0
            self._base_beat_number = len(beat_times) - 1 - first_downbeat_idx
        else:
            # First time, no downbeat info: assume first beat is downbeat
            self._base_beat_number = len(beat_times) - 1

        self._phase_ref_time = last_beat
        self._has_beats = True

    def _get_frame_info(self, timestamp: float) -> BeatInfo:
        """Compute BeatInfo at a specific timestamp via phase interpolation.

        Args:
            timestamp: Absolute time in seconds.

        Returns:
            Interpolated BeatInfo for this timestamp.
        """
        if not self._has_beats:
            return BeatInfo(
                bpm=self._bpm, beat_phase=0.0, bar_phase=0.0,
                is_beat=False, is_downbeat=False,
            )

        beat_period = 60.0 / self._bpm
        elapsed = timestamp - self._phase_ref_time
        beats_since_ref = elapsed / beat_period

        # Beat phase (Python % is always non-negative for positive divisor)
        beat_phase = beats_since_ref % 1.0

        # Absolute beat number
        abs_beat = self._base_beat_number + math.floor(beats_since_ref)

        # is_beat via edge detection (skip first frame to avoid false positive)
        if not self._tracking_started:
            is_beat = False
            self._tracking_started = True
        else:
            is_beat = abs_beat > self._prev_abs_beat
        self._prev_abs_beat = abs_beat

        # Bar position
        beat_in_bar = abs_beat % self._beats_per_bar
        bar_phase = (beat_in_bar + beat_phase) / self._beats_per_bar
        is_downbeat = is_beat and (beat_in_bar == 0)

        return BeatInfo(
            bpm=self._bpm,
            beat_phase=beat_phase,
            bar_phase=bar_phase,
            is_beat=is_beat,
            is_downbeat=is_downbeat,
        )

    # ── Internal: analysis ────────────────────────────────────────

    def _compute_bpm(self, beat_times: np.ndarray) -> float:
        """Estimate BPM from beat timestamps.

        Args:
            beat_times: Sorted array of beat timestamps in seconds.

        Returns:
            Estimated BPM, or current BPM if estimation fails.
        """
        if len(beat_times) < 2:
            return self._bpm

        ibis = np.diff(beat_times)
        min_ibi = 60.0 / self._max_bpm
        max_ibi = 60.0 / self._min_bpm
        valid = ibis[(ibis >= min_ibi) & (ibis <= max_ibi)]

        if len(valid) == 0:
            return self._bpm

        return 60.0 / float(np.median(valid))

    def _run_streaming_analysis(self) -> None:
        """Run madmom beat detection on the current buffer."""
        from madmom.audio.signal import Signal
        from madmom.features.beats import DBNBeatTrackingProcessor

        buffer_start_time = (self._total_samples - len(self._buffer)) / self._sr

        sig = Signal(self._buffer, sample_rate=self._sr)
        beat_nn = _get_beat_nn()
        activations = beat_nn(sig)

        dbn = DBNBeatTrackingProcessor(
            fps=self._fps,
            min_bpm=self._min_bpm,
            max_bpm=self._max_bpm,
        )
        beats = dbn(activations)

        if len(beats) > 0:
            abs_beats = beats + buffer_start_time
            self._update_tracking(abs_beats)
            logger.debug(
                "Streaming analysis: %d beats, BPM=%.1f",
                len(beats), self._bpm,
            )
