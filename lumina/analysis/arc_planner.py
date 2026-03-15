"""Arc planner: pre-compute relative importance of each section for headroom.

Assigns a headroom budget (0.0-1.0) to each frame based on where it falls
in the song's overall energy arc. Sparse intros get low headroom (~0.3),
climax sections get 1.0. This prevents "using up" all visual intensity
during a quiet section, leaving nothing extra for the peak.

Algorithm:
1. Per section: significance = mean_energy * sqrt(mean_layer_count).
2. Rank sections by significance across the whole song.
3. Normalize to [0.15, 1.0] using percentile mapping.
4. Smooth transitions between sections (1-2 second ramp).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from lumina.analysis.layer_tracker import LayerFrame
from lumina.audio.energy_tracker import EnergyFrame
from lumina.audio.structural_analyzer import StructuralMap

logger = logging.getLogger(__name__)

_HEADROOM_FLOOR = 0.15  # Never fully invisible
_RAMP_SECONDS = 1.5  # Transition smoothing duration


@dataclass(slots=True)
class ArcFrame:
    """Per-frame headroom value.

    Args:
        headroom: 0.0-1.0 intensity budget for this moment.
        section_significance: Raw significance score of the current section.
    """

    headroom: float
    section_significance: float


class ArcPlanner:
    """Pre-compute headroom budgets from energy, layers, and structure.

    Args:
        fps: Target frame rate for output.
    """

    def __init__(self, fps: int = 60) -> None:
        self._fps = fps

    def plan(
        self,
        energy_results: list[EnergyFrame],
        layer_frames: list[LayerFrame],
        structural_map: StructuralMap,
    ) -> list[ArcFrame]:
        """Compute per-frame headroom from song-level analysis.

        Args:
            energy_results: Energy frames from EnergyTracker.
            layer_frames: Layer frames (resampled to fps).
            structural_map: Structural sections from StructuralAnalyzer.

        Returns:
            List of ArcFrame, one per frame (length = len(energy_results)).
        """
        n = len(energy_results)
        if n == 0:
            return []

        sections = structural_map.sections
        if not sections:
            return [ArcFrame(headroom=1.0, section_significance=1.0)] * n

        # Compute per-section significance
        section_scores: list[float] = []
        for sec in sections:
            start_frame = int(sec.start_time * self._fps)
            end_frame = int(sec.end_time * self._fps)
            start_frame = max(0, min(start_frame, n - 1))
            end_frame = max(start_frame + 1, min(end_frame, n))

            # Mean energy in this section
            energies = [energy_results[i].energy for i in range(start_frame, end_frame)]
            mean_energy = sum(energies) / len(energies) if energies else 0.0

            # Mean layer count (use layer_frames if available)
            if layer_frames:
                n_layers = min(len(layer_frames), end_frame)
                layer_counts = [
                    layer_frames[i].active_count
                    for i in range(min(start_frame, n_layers), min(end_frame, n_layers))
                ]
                mean_layers = sum(layer_counts) / len(layer_counts) if layer_counts else 1.0
            else:
                mean_layers = 1.0

            significance = mean_energy * (mean_layers ** 0.5)

            # Boost drop/chorus sections by 20% (they're structurally important)
            if sec.segment_type in ("drop", "chorus"):
                significance *= 1.2

            section_scores.append(significance)

        # Percentile-based normalization
        if len(section_scores) > 1:
            sorted_scores = sorted(section_scores)
            min_score = sorted_scores[0]
            max_score = sorted_scores[-1]
            score_range = max_score - min_score

            if score_range > 1e-8:
                normalized = [
                    _HEADROOM_FLOOR + (1.0 - _HEADROOM_FLOOR) * (s - min_score) / score_range
                    for s in section_scores
                ]
            else:
                normalized = [1.0] * len(section_scores)
        else:
            normalized = [1.0]

        # Map to per-frame headroom (raw, before smoothing)
        raw_headroom = np.ones(n, dtype=np.float32)
        section_sig = np.ones(n, dtype=np.float32)

        for sec_idx, sec in enumerate(sections):
            start_frame = max(0, int(sec.start_time * self._fps))
            end_frame = min(n, int(sec.end_time * self._fps))
            raw_headroom[start_frame:end_frame] = normalized[sec_idx]
            section_sig[start_frame:end_frame] = section_scores[sec_idx]

        # Smooth transitions with EMA
        ramp_frames = int(_RAMP_SECONDS * self._fps)
        if ramp_frames > 1:
            alpha = 2.0 / (ramp_frames + 1)
            smoothed = np.copy(raw_headroom)
            for i in range(1, n):
                smoothed[i] = smoothed[i - 1] * (1 - alpha) + raw_headroom[i] * alpha
            raw_headroom = smoothed

        return [
            ArcFrame(
                headroom=float(np.clip(raw_headroom[i], _HEADROOM_FLOOR, 1.0)),
                section_significance=float(section_sig[i]),
            )
            for i in range(n)
        ]
