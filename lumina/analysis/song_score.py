"""Song score: aggregate layer, motif, and arc data into per-frame ScoreFrames.

Thin aggregator that combines LayerTracker + MotifDetector + ArcPlanner
outputs into a unified ScoreFrame consumed by the lighting engine. Also
generates motif-to-pattern and motif-to-color assignments by consulting
the active genre profile's preferences.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from lumina.analysis.arc_planner import ArcFrame
from lumina.analysis.layer_tracker import LayerFrame
from lumina.analysis.motif_detector import MotifTimeline, NotePattern

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ScoreFrame:
    """Per-frame combined score from all new analysis layers.

    Args:
        layer_count: Number of active stems (0-4).
        layer_mask: Per-stem activity level.
        motif_id: Which macro motif is playing (None if none).
        motif_repetition: How many times this motif has been heard before.
        notes_per_beat: Number of regular notes per beat (0 = no pattern).
        note_pattern_phase: Position in note cycle (0.0-1.0).
        headroom: Intensity budget (0.0-1.0).
    """

    layer_count: int
    layer_mask: dict[str, float]
    motif_id: int | None
    motif_repetition: int
    notes_per_beat: int
    note_pattern_phase: float
    headroom: float


@dataclass
class MotifAssignment:
    """Visual assignment for a detected motif.

    Args:
        pattern_name: Lighting pattern to use for this motif.
        color_index: Index into the profile's color palette.
    """

    pattern_name: str
    color_index: int


class SongScore:
    """Build per-frame ScoreFrames from analysis layers.

    Args:
        fps: Target frame rate.
    """

    def __init__(self, fps: int = 60) -> None:
        self._fps = fps
        self.motif_assignments: dict[int, MotifAssignment] = {}

    def build(
        self,
        layer_frames: list[LayerFrame],
        note_patterns: list[NotePattern],
        arc_frames: list[ArcFrame],
        motif_timeline: MotifTimeline,
        n_frames: int,
        pattern_preferences: list[str] | None = None,
    ) -> list[ScoreFrame]:
        """Combine all analysis layers into ScoreFrames.

        Args:
            layer_frames: Layer activity per frame (resampled to fps).
            note_patterns: Note-level patterns per frame.
            arc_frames: Headroom per frame.
            motif_timeline: Macro motif segments.
            n_frames: Total number of output frames.
            pattern_preferences: Ordered list of pattern names for motif
                assignment. If None, motifs still get IDs but no
                pattern assignment.

        Returns:
            List of ScoreFrame, length = n_frames.
        """
        # Assign patterns to motifs round-robin from preferences
        if pattern_preferences and motif_timeline.n_motifs > 0:
            for mid in range(motif_timeline.n_motifs):
                pat_idx = mid % len(pattern_preferences)
                self.motif_assignments[mid] = MotifAssignment(
                    pattern_name=pattern_preferences[pat_idx],
                    color_index=mid,
                )

        # Pre-compute motif lookup: for each frame, which motif is active?
        motif_at_frame: list[tuple[int | None, int]] = [(None, 0)] * n_frames

        for seg in motif_timeline.segments:
            start_f = max(0, int(seg.start_time * self._fps))
            end_f = min(n_frames, int(seg.end_time * self._fps))
            for f in range(start_f, end_f):
                motif_at_frame[f] = (seg.motif_id, seg.repetition)

        # Build score frames
        frames: list[ScoreFrame] = []
        for i in range(n_frames):
            # Layer data
            if i < len(layer_frames):
                lf = layer_frames[i]
                layer_count = lf.active_count
                layer_mask = dict(lf.layer_mask)
            else:
                layer_count = 0
                layer_mask = {}

            # Note pattern data
            if i < len(note_patterns):
                np_frame = note_patterns[i]
                notes_per_beat = np_frame.notes_per_beat
                note_phase = np_frame.pattern_phase
            else:
                notes_per_beat = 0
                note_phase = 0.0

            # Arc data
            headroom = arc_frames[i].headroom if i < len(arc_frames) else 1.0

            # Motif data
            motif_id, motif_rep = motif_at_frame[i]

            frames.append(ScoreFrame(
                layer_count=layer_count,
                layer_mask=layer_mask,
                motif_id=motif_id,
                motif_repetition=motif_rep,
                notes_per_beat=notes_per_beat,
                note_pattern_phase=note_phase,
                headroom=headroom,
            ))

        logger.info(
            "SongScore built: %d frames, %d motifs, %d motif assignments",
            n_frames,
            motif_timeline.n_motifs,
            len(self.motif_assignments),
        )
        return frames
