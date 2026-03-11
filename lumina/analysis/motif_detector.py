"""Motif detection: identify repeating musical patterns at bar and note level.

Two levels of pattern awareness:

**Level A — Macro motifs (bar-level repetition):**
Detect repeating multi-bar phrases using chromagram+MFCC self-similarity.
Purpose: "This 4-bar section sounds like that earlier 4-bar section"
→ use the same visual treatment.

**Level B — Micro patterns (note-level sequences):**
Detect repeating note sequences within the melodic stem to enable
"each note = different light."
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from lumina.audio.beat_detector import BeatInfo

logger = logging.getLogger(__name__)

# Macro motif detection constants
_SIMILARITY_THRESHOLD = 0.82  # Cosine similarity for "same motif"
_MIN_BARS_PER_MOTIF = 2
_MAX_BARS_PER_MOTIF = 4

# Micro pattern detection constants
_ONSET_THRESHOLD_RATIO = 0.3  # Relative to peak spectral flux
_IOI_CV_THRESHOLD = 0.30  # Max coefficient of variation for "regular"
_MIN_NOTES_PER_BEAT = 2
_MAX_NOTES_PER_BEAT = 8
_PATTERN_WINDOW_BEATS = 4  # Look-back window in beats


@dataclass(slots=True)
class MotifSegment:
    """A contiguous region assigned to a macro motif.

    Args:
        start_time: Start time in seconds.
        end_time: End time in seconds.
        motif_id: Unique ID for this motif group.
        repetition: Which occurrence of this motif (0-indexed).
        similarity: Cosine similarity to the motif prototype.
    """

    start_time: float
    end_time: float
    motif_id: int
    repetition: int
    similarity: float


@dataclass
class MotifTimeline:
    """Complete motif analysis for a song.

    Args:
        segments: Bar-level motif segments.
        n_motifs: Total number of distinct motifs found.
    """

    segments: list[MotifSegment] = field(default_factory=list)
    n_motifs: int = 0


@dataclass(slots=True)
class NotePattern:
    """Per-frame note-level pattern info.

    Args:
        notes_per_beat: Number of regular notes per beat (0 = no pattern).
        pattern_phase: 0.0-1.0 position in the note cycle.
        is_regular: Whether the pattern is detected as regular.
    """

    notes_per_beat: int
    pattern_phase: float
    is_regular: bool


class MotifDetector:
    """Detect repeating musical patterns at macro and micro levels.

    Args:
        sr: Sample rate.
        fps: Target output frame rate.
    """

    def __init__(self, sr: int = 44100, fps: int = 60) -> None:
        self._sr = sr
        self._fps = fps

    def detect_macro_motifs(
        self,
        audio: np.ndarray,
        beat_results: list[BeatInfo],
    ) -> MotifTimeline:
        """Detect repeating bar-level phrases via self-similarity.

        Args:
            audio: Mono float32 audio.
            beat_results: Beat tracking results.

        Returns:
            MotifTimeline with detected motif segments.
        """
        import librosa  # type: ignore[import-untyped]

        frame_interval = 1.0 / self._fps
        duration = len(audio) / self._sr

        # Find bar boundaries from downbeats
        bar_times: list[float] = []
        for i, b in enumerate(beat_results):
            if b.is_downbeat:
                bar_times.append(i * frame_interval)
        bar_times.append(duration)

        if len(bar_times) < _MIN_BARS_PER_MOTIF + 1:
            logger.info("Too few bars (%d) for motif detection", len(bar_times) - 1)
            return MotifTimeline()

        # Compute chroma + MFCC features per bar
        chroma = librosa.feature.chroma_cqt(y=audio, sr=self._sr, hop_length=512)
        mfcc = librosa.feature.mfcc(y=audio, sr=self._sr, n_mfcc=13, hop_length=512)

        hop_sr = self._sr / 512  # frames per second in chroma/mfcc

        bar_features: list[np.ndarray] = []
        for i in range(len(bar_times) - 1):
            start_frame = int(bar_times[i] * hop_sr)
            end_frame = int(bar_times[i + 1] * hop_sr)
            end_frame = max(start_frame + 1, min(end_frame, chroma.shape[1]))
            start_frame = min(start_frame, end_frame - 1)

            # Concatenate mean chroma + mean MFCC for this bar
            chroma_mean = np.mean(chroma[:, start_frame:end_frame], axis=1)
            mfcc_mean = np.mean(mfcc[:, start_frame:end_frame], axis=1)
            bar_feat = np.concatenate([chroma_mean, mfcc_mean])

            # Normalize
            norm = np.linalg.norm(bar_feat)
            if norm > 1e-8:
                bar_feat = bar_feat / norm
            bar_features.append(bar_feat)

        n_bars = len(bar_features)

        # Compare multi-bar windows via cosine similarity
        # Try 2-bar and 4-bar windows
        motif_segments: list[MotifSegment] = []
        motif_prototypes: list[np.ndarray] = []
        motif_id_counter = 0

        for window_size in [_MAX_BARS_PER_MOTIF, _MIN_BARS_PER_MOTIF]:
            if n_bars < window_size * 2:
                continue

            # Build window feature vectors
            window_features: list[np.ndarray] = []
            window_starts: list[int] = []
            for i in range(0, n_bars - window_size + 1, window_size):
                feat = np.mean(bar_features[i : i + window_size], axis=0)
                norm = np.linalg.norm(feat)
                if norm > 1e-8:
                    feat = feat / norm
                window_features.append(feat)
                window_starts.append(i)

            # Find clusters of similar windows
            assigned: set[int] = set()
            for i in range(len(window_features)):
                if i in assigned:
                    continue

                cluster = [i]
                assigned.add(i)

                for j in range(i + 1, len(window_features)):
                    if j in assigned:
                        continue
                    sim = float(np.dot(window_features[i], window_features[j]))
                    if sim >= _SIMILARITY_THRESHOLD:
                        cluster.append(j)
                        assigned.add(j)

                # Only create motif if it repeats (cluster size > 1)
                if len(cluster) > 1:
                    mid = motif_id_counter
                    motif_id_counter += 1
                    motif_prototypes.append(window_features[cluster[0]])

                    for rep, idx in enumerate(sorted(cluster)):
                        bar_start = window_starts[idx]
                        bar_end = min(bar_start + window_size, n_bars)
                        sim = float(np.dot(
                            window_features[idx], window_features[cluster[0]]
                        ))
                        motif_segments.append(MotifSegment(
                            start_time=bar_times[bar_start],
                            end_time=bar_times[bar_end],
                            motif_id=mid,
                            repetition=rep,
                            similarity=sim,
                        ))

        # Sort by start time
        motif_segments.sort(key=lambda s: s.start_time)
        logger.info(
            "Detected %d motif segments (%d distinct motifs)",
            len(motif_segments),
            motif_id_counter,
        )

        return MotifTimeline(
            segments=motif_segments,
            n_motifs=motif_id_counter,
        )

    def detect_micro_patterns(
        self,
        other_stem: np.ndarray,
        beat_results: list[BeatInfo],
    ) -> list[NotePattern]:
        """Detect repeating note-level sequences in the melodic stem.

        Args:
            other_stem: Mono float32 "other" stem from demucs.
            beat_results: Beat tracking results.

        Returns:
            List of NotePattern, one per output frame at fps.
        """
        frame_interval = 1.0 / self._fps
        n_output = len(beat_results)
        duration = n_output * frame_interval

        # Compute spectral flux for onset detection on "other" stem
        hop = 512
        n_fft = 2048
        n_frames = 1 + (len(other_stem) - n_fft) // hop
        if n_frames < 2:
            return [NotePattern(0, 0.0, False)] * n_output

        # STFT magnitude
        spec = np.abs(np.array([
            np.fft.rfft(other_stem[i * hop : i * hop + n_fft] * np.hanning(n_fft))
            for i in range(n_frames)
        ]))

        # Spectral flux (half-wave rectified difference)
        flux = np.zeros(n_frames, dtype=np.float32)
        for i in range(1, n_frames):
            diff = spec[i] - spec[i - 1]
            flux[i] = float(np.sum(np.maximum(diff, 0)))

        flux_peak = float(np.max(flux)) if np.max(flux) > 0 else 1.0
        onset_threshold = flux_peak * _ONSET_THRESHOLD_RATIO

        # Find onset times
        spec_fps = self._sr / hop
        onset_times: list[float] = []
        min_gap = 0.03  # 30ms minimum gap
        last_onset = -1.0

        for i in range(1, n_frames - 1):
            if (flux[i] > onset_threshold
                and flux[i] > flux[i - 1]
                and flux[i] > flux[i + 1]):
                t = i / spec_fps
                if t - last_onset > min_gap:
                    onset_times.append(t)
                    last_onset = t

        # For each output frame, look at recent onsets and detect regular patterns
        results: list[NotePattern] = []
        beat_times: list[float] = []
        for i, b in enumerate(beat_results):
            if b.is_beat:
                beat_times.append(i * frame_interval)

        # Estimate beat duration
        if len(beat_times) >= 2:
            beat_dur = float(np.median(np.diff(beat_times)))
        else:
            beat_dur = 0.5  # fallback 120 BPM

        window_duration = beat_dur * _PATTERN_WINDOW_BEATS

        for i in range(n_output):
            t = i * frame_interval

            # Get onsets in the look-back window
            window_start = t - window_duration
            recent_onsets = [
                ot for ot in onset_times
                if window_start <= ot <= t
            ]

            if len(recent_onsets) < 3:
                results.append(NotePattern(0, 0.0, False))
                continue

            # Compute inter-onset intervals
            iois = np.diff(recent_onsets)
            if len(iois) == 0:
                results.append(NotePattern(0, 0.0, False))
                continue

            mean_ioi = float(np.mean(iois))
            std_ioi = float(np.std(iois))

            # Coefficient of variation
            cv = std_ioi / mean_ioi if mean_ioi > 1e-8 else 999.0

            if cv < _IOI_CV_THRESHOLD and mean_ioi > 0.01:
                # Regular pattern detected
                notes_per_beat = max(
                    _MIN_NOTES_PER_BEAT,
                    min(_MAX_NOTES_PER_BEAT, round(beat_dur / mean_ioi)),
                )

                # Compute phase within the note cycle
                if recent_onsets:
                    time_since_last = t - recent_onsets[-1]
                    phase = (time_since_last / mean_ioi) % 1.0
                else:
                    phase = 0.0

                results.append(NotePattern(
                    notes_per_beat=notes_per_beat,
                    pattern_phase=phase,
                    is_regular=True,
                ))
            else:
                results.append(NotePattern(0, 0.0, False))

        return results
