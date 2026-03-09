"""Offline structural analysis for song segmentation.

Replaces the frame-by-frame SegmentClassifier in file mode with a
global structural analysis that produces stable, musically meaningful
sections lasting 8-30 seconds.

Algorithm:
1. Build self-similarity matrix from MFCCs at reduced frame rate (~10fps).
2. Compute novelty function via checkerboard kernel convolution.
3. Peak-pick section boundaries with minimum section duration.
4. Extract per-section features from pre-computed analyzer results.
5. Cluster sections by feature similarity (repeated sections get same label).
6. Label clusters via energy contrast with predecessors.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from lumina.audio.beat_detector import BeatInfo
from lumina.audio.energy_tracker import EnergyFrame
from lumina.audio.onset_detector import OnsetEvent
from lumina.audio.segment_classifier import SegmentFrame
from lumina.audio.source_separator import StemSet
from lumina.audio.vocal_detector import VocalFrame

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Section:
    """A structural section of a song.

    Args:
        start_time: Section start in seconds.
        end_time: Section end in seconds.
        segment_type: One of SEGMENT_LABELS.
        confidence: 0.0-1.0 classification confidence.
        features: Aggregate feature values for this section.
    """

    start_time: float
    end_time: float
    segment_type: str
    confidence: float
    features: dict[str, float]


@dataclass(slots=True)
class StructuralMap:
    """Complete structural analysis of a song.

    Args:
        sections: Ordered list of non-overlapping sections.
        duration: Total song duration in seconds.
    """

    sections: list[Section]
    duration: float


class StructuralAnalyzer:
    """Offline song structure analyzer using self-similarity.

    Args:
        sr: Sample rate in Hz.
        fps: Output frame rate.
        min_section_duration: Minimum section length in seconds.
    """

    def __init__(
        self,
        sr: int = 44100,
        fps: int = 60,
        min_section_duration: float = 4.0,
    ) -> None:
        self._sr = sr
        self._fps = fps
        self._min_section_duration = min_section_duration
        self._hop_length = sr // fps

    def analyze(
        self,
        audio: np.ndarray,
        stems: StemSet,
        beat_results: list[BeatInfo],
        energy_results: list[EnergyFrame],
        onset_results: list[OnsetEvent | None],
        vocal_results: list[VocalFrame],
    ) -> StructuralMap:
        """Analyze song structure from audio and pre-computed features.

        Args:
            audio: Full mix mono float32.
            stems: Demucs-separated stems.
            beat_results: Beat tracking frames.
            energy_results: Energy analysis frames.
            onset_results: Onset detection frames.
            vocal_results: Vocal detection frames.

        Returns:
            StructuralMap with labeled sections.
        """
        duration = len(audio) / self._sr
        n_frames = min(
            len(beat_results),
            len(energy_results),
            len(onset_results),
            len(vocal_results),
        )

        if n_frames < self._fps * 2:
            # Too short for structural analysis — return single section
            return StructuralMap(
                sections=[
                    Section(
                        start_time=0.0,
                        end_time=duration,
                        segment_type="verse",
                        confidence=0.5,
                        features={},
                    )
                ],
                duration=duration,
            )

        # Step 1: Self-similarity matrix from MFCCs
        boundaries = self._detect_boundaries(audio, duration)

        # Ensure boundaries include 0 and duration
        if len(boundaries) == 0 or boundaries[0] > 0.1:
            boundaries = [0.0, *boundaries]
        if boundaries[-1] < duration - 0.1:
            boundaries.append(duration)

        # Step 2: Extract per-section features
        sections = self._extract_section_features(
            boundaries,
            energy_results,
            onset_results,
            vocal_results,
            n_frames,
        )

        # Step 3: Cluster similar sections
        sections = self._cluster_sections(sections)

        # Step 4: Label sections via contrast
        sections = self._label_sections(sections, duration)

        # Step 5: Merge short sections
        sections = self._merge_short_sections(sections)

        logger.info(
            "Structural analysis: %d sections in %.1fs",
            len(sections),
            duration,
        )
        for sec in sections:
            logger.info(
                "  [%.1f - %.1f] %s (conf=%.2f)",
                sec.start_time,
                sec.end_time,
                sec.segment_type,
                sec.confidence,
            )

        return StructuralMap(sections=sections, duration=duration)

    def map_to_frames(
        self,
        structural_map: StructuralMap,
        num_frames: int,
        fps: int,
    ) -> list[SegmentFrame]:
        """Convert a StructuralMap to per-frame SegmentFrames.

        Args:
            structural_map: Analyzed song structure.
            num_frames: Total number of output frames.
            fps: Output frame rate.

        Returns:
            List of SegmentFrame, one per frame.
        """
        frame_interval = 1.0 / fps
        result: list[SegmentFrame] = []

        for i in range(num_frames):
            t = i * frame_interval
            section = self._find_section(structural_map.sections, t)
            result.append(
                SegmentFrame(
                    segment=section.segment_type,
                    confidence=section.confidence,
                    scores=section.features,
                )
            )

        return result

    # ── Boundary detection ────────────────────────────────────────

    def _detect_boundaries(
        self,
        audio: np.ndarray,
        duration: float,
    ) -> list[float]:
        """Detect section boundaries using self-similarity novelty.

        Args:
            audio: Mono float32 audio.
            duration: Song duration in seconds.

        Returns:
            Sorted list of boundary timestamps in seconds.
        """
        import librosa  # type: ignore[import-untyped]

        # Subsample to ~10fps to keep matrix manageable
        subsample_fps = 10
        hop = self._sr // subsample_fps
        n_mfcc = 20

        # Compute MFCCs
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self._sr,
            n_mfcc=n_mfcc,
            hop_length=hop,
        )

        # Build recurrence matrix (self-similarity)
        rec = librosa.segment.recurrence_matrix(
            mfcc,
            width=int(subsample_fps * self._min_section_duration),
            mode="affinity",
            sym=True,
        )

        # Novelty function via checkerboard kernel
        novelty = librosa.segment.novelty(rec)

        # Peak-pick boundaries
        from scipy.signal import find_peaks

        min_distance = int(subsample_fps * self._min_section_duration)
        # Adaptive threshold: peaks above mean + 0.5 * std
        threshold = float(np.mean(novelty) + 0.5 * np.std(novelty))
        peaks, _ = find_peaks(
            novelty,
            distance=max(1, min_distance),
            height=max(threshold, 0.01),
        )

        # Convert peak indices to timestamps
        boundaries = [float(p) / subsample_fps for p in peaks]

        # Filter boundaries within valid range
        boundaries = [b for b in boundaries if 0.5 < b < duration - 0.5]

        return sorted(boundaries)

    # ── Feature extraction ────────────────────────────────────────

    def _extract_section_features(
        self,
        boundaries: list[float],
        energy_results: list[EnergyFrame],
        onset_results: list[OnsetEvent | None],
        vocal_results: list[VocalFrame],
        n_frames: int,
    ) -> list[Section]:
        """Compute aggregate features for each section.

        Args:
            boundaries: Sorted boundary timestamps.
            energy_results: Energy frames.
            onset_results: Onset frames.
            vocal_results: Vocal frames.
            n_frames: Total frame count.

        Returns:
            List of Section with features but no labels yet.
        """
        sections: list[Section] = []

        for i in range(len(boundaries) - 1):
            start_t = boundaries[i]
            end_t = boundaries[i + 1]

            start_frame = max(0, int(start_t * self._fps))
            end_frame = min(n_frames, int(end_t * self._fps))

            if end_frame <= start_frame:
                continue

            # Gather frames for this section
            energies = [energy_results[f].energy for f in range(start_frame, end_frame)]
            derivs = [energy_results[f].energy_derivative for f in range(start_frame, end_frame)]
            centroids = [
                energy_results[f].spectral_centroid for f in range(start_frame, end_frame)
            ]
            sub_bass = [energy_results[f].sub_bass_energy for f in range(start_frame, end_frame)]
            onsets = [onset_results[f] is not None for f in range(start_frame, end_frame)]
            vocals = [vocal_results[f].vocal_energy for f in range(start_frame, end_frame)]

            features = {
                "mean_energy": float(np.mean(energies)),
                "energy_variance": float(np.var(energies)),
                "onset_density": float(np.mean([1.0 if o else 0.0 for o in onsets])),
                "vocal_presence": float(np.mean(vocals)),
                "sub_bass_energy": float(np.mean(sub_bass)),
                "spectral_centroid": float(np.mean(centroids)),
                "energy_derivative_mean": float(np.mean(derivs)),
            }

            sections.append(
                Section(
                    start_time=start_t,
                    end_time=end_t,
                    segment_type="verse",  # placeholder
                    confidence=0.5,
                    features=features,
                )
            )

        return sections

    # ── Clustering ────────────────────────────────────────────────

    def _cluster_sections(self, sections: list[Section]) -> list[Section]:
        """Cluster similar sections so repeated parts get the same label.

        Uses simple agglomerative merging based on feature distance.
        Sections in the same cluster will later get the same segment label.

        Args:
            sections: Sections with computed features.

        Returns:
            Same sections, now with cluster IDs stored in features["_cluster"].
        """
        n = len(sections)
        if n <= 1:
            for sec in sections:
                sec.features["_cluster"] = 0.0
            return sections

        # Build feature matrix
        feat_keys = [
            "mean_energy",
            "energy_variance",
            "onset_density",
            "vocal_presence",
            "sub_bass_energy",
        ]
        matrix = np.zeros((n, len(feat_keys)))
        for i, sec in enumerate(sections):
            for j, key in enumerate(feat_keys):
                matrix[i, j] = sec.features.get(key, 0.0)

        # Normalize features to [0, 1]
        mins = matrix.min(axis=0)
        maxs = matrix.max(axis=0)
        ranges = maxs - mins
        ranges[ranges < 1e-10] = 1.0
        normalized = (matrix - mins) / ranges

        # Simple agglomerative clustering: merge closest pairs
        # Start with each section as its own cluster
        labels = list(range(n))
        max_k = min(7, n)

        # Compute pairwise distances
        distances: list[tuple[float, int, int]] = []
        for i in range(n):
            for j in range(i + 1, n):
                dist = float(np.linalg.norm(normalized[i] - normalized[j]))
                distances.append((dist, i, j))
        distances.sort()

        # Merge until we reach target k
        n_clusters = n
        for _dist, i, j in distances:
            if n_clusters <= max_k:
                break
            li, lj = labels[i], labels[j]
            if li != lj:
                # Merge: relabel all lj -> li
                for k in range(n):
                    if labels[k] == lj:
                        labels[k] = li
                n_clusters -= 1

        for i, sec in enumerate(sections):
            sec.features["_cluster"] = float(labels[i])

        return sections

    # ── Labeling ──────────────────────────────────────────────────

    def _label_sections(
        self,
        sections: list[Section],
        duration: float,
    ) -> list[Section]:
        """Label sections using energy contrast with predecessors.

        Args:
            sections: Sections with features and cluster IDs.
            duration: Total song duration.

        Returns:
            Sections with segment_type set.
        """
        if not sections:
            return sections

        # Label each section individually first
        for i, sec in enumerate(sections):
            position = sec.start_time / duration if duration > 0 else 0.5
            energy = sec.features.get("mean_energy", 0.5)
            onset_density = sec.features.get("onset_density", 0.3)
            vocal = sec.features.get("vocal_presence", 0.3)

            # Compute contrast with predecessor
            delta_energy = 0.0
            if i > 0:
                prev_energy = sections[i - 1].features.get("mean_energy", 0.5)
                delta_energy = energy - prev_energy

            # Decision tree (priority order)
            label = "verse"
            confidence = 0.5

            # Intro: first section, low energy, early position
            if position < 0.15 and energy < 0.35:
                label = "intro"
                confidence = 0.8
            # Outro: last section, low/falling energy
            elif position > 0.85 and energy < 0.40:
                label = "outro"
                confidence = 0.8
            # Drop: large positive energy jump + high onsets + low vocals
            elif delta_energy > 0.12 and onset_density > 0.15 and vocal < 0.40:
                label = "drop"
                confidence = min(1.0, 0.5 + delta_energy * 3)
            # Also drop: sustained high energy with heavy bass
            elif (
                energy > 0.60
                and sec.features.get("sub_bass_energy", 0) > 0.35
                and vocal < 0.35
            ):
                label = "drop"
                confidence = 0.7
            # Breakdown: large negative energy drop + sparse onsets
            elif delta_energy < -0.10 and onset_density < 0.20:
                label = "breakdown"
                confidence = 0.7
            # Also breakdown: very low energy in general
            elif energy < 0.25 and onset_density < 0.15:
                label = "breakdown"
                confidence = 0.6
            # Chorus: high energy + strong vocals
            elif energy > 0.45 and vocal > 0.35:
                label = "chorus"
                confidence = 0.6
            # Bridge: moderate energy, low vocals, mid-track
            elif energy < 0.42 and vocal < 0.30 and 0.2 < position < 0.85:
                label = "bridge"
                confidence = 0.5

            sec.segment_type = label
            sec.confidence = confidence

        # Propagate labels within clusters: majority vote per cluster
        clusters: dict[int, list[int]] = {}
        for i, sec in enumerate(sections):
            cid = int(sec.features.get("_cluster", i))
            clusters.setdefault(cid, []).append(i)

        for _cid, indices in clusters.items():
            if len(indices) <= 1:
                continue
            # Count labels in this cluster
            label_counts: dict[str, int] = {}
            for idx in indices:
                lbl = sections[idx].segment_type
                label_counts[lbl] = label_counts.get(lbl, 0) + 1
            # Majority label
            majority = max(label_counts, key=label_counts.get)  # type: ignore[arg-type]
            for idx in indices:
                sections[idx].segment_type = majority

        # Clean up internal cluster marker
        for sec in sections:
            sec.features.pop("_cluster", None)

        return sections

    # ── Merge short sections ──────────────────────────────────────

    def _merge_short_sections(self, sections: list[Section]) -> list[Section]:
        """Merge sections shorter than min_section_duration into neighbors.

        Args:
            sections: Labeled sections.

        Returns:
            Sections with short ones absorbed.
        """
        if len(sections) <= 1:
            return sections

        merged: list[Section] = [sections[0]]

        for i in range(1, len(sections)):
            sec = sections[i]
            sec_duration = sec.end_time - sec.start_time

            if sec_duration < self._min_section_duration and merged:
                # Absorb into previous section
                merged[-1] = Section(
                    start_time=merged[-1].start_time,
                    end_time=sec.end_time,
                    segment_type=merged[-1].segment_type,
                    confidence=merged[-1].confidence,
                    features=merged[-1].features,
                )
            else:
                merged.append(sec)

        return merged

    # ── Helpers ────────────────────────────────────────────────────

    @staticmethod
    def _find_section(sections: list[Section], timestamp: float) -> Section:
        """Find which section contains the given timestamp.

        Args:
            sections: Sorted, non-overlapping sections.
            timestamp: Time in seconds.

        Returns:
            Section containing the timestamp (or last section if past end).
        """
        for sec in sections:
            if sec.start_time <= timestamp < sec.end_time:
                return sec
        # Past end or between sections — return last
        return sections[-1] if sections else Section(
            start_time=0.0,
            end_time=0.0,
            segment_type="verse",
            confidence=0.5,
            features={},
        )
