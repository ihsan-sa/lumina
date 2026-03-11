"""Tests for the SongScore aggregator."""

from __future__ import annotations

import pytest

from lumina.analysis.arc_planner import ArcFrame
from lumina.analysis.layer_tracker import LayerFrame
from lumina.analysis.motif_detector import MotifSegment, MotifTimeline, NotePattern
from lumina.analysis.song_score import MotifAssignment, ScoreFrame, SongScore


def _make_layer_frames(n: int, count: int = 2) -> list[LayerFrame]:
    return [
        LayerFrame(
            active_count=count,
            layer_mask={"drums": 0.8, "bass": 0.5, "vocals": 0.0, "other": 0.0},
            layer_change=None,
        )
        for _ in range(n)
    ]


def _make_note_patterns(n: int, regular: bool = False) -> list[NotePattern]:
    return [
        NotePattern(
            notes_per_beat=4 if regular else 0,
            pattern_phase=0.0,
            is_regular=regular,
        )
        for _ in range(n)
    ]


def _make_arc_frames(n: int, headroom: float = 0.7) -> list[ArcFrame]:
    return [
        ArcFrame(headroom=headroom, section_significance=0.5)
        for _ in range(n)
    ]


class TestSongScore:
    """Tests for SongScore aggregation."""

    def test_basic_build(self) -> None:
        n = 120
        score = SongScore(fps=60)
        frames = score.build(
            _make_layer_frames(n),
            _make_note_patterns(n),
            _make_arc_frames(n),
            MotifTimeline(),
            n_frames=n,
        )
        assert len(frames) == n
        assert all(isinstance(f, ScoreFrame) for f in frames)

    def test_layer_data_propagated(self) -> None:
        n = 60
        score = SongScore(fps=60)
        frames = score.build(
            _make_layer_frames(n, count=3),
            _make_note_patterns(n),
            _make_arc_frames(n),
            MotifTimeline(),
            n_frames=n,
        )
        for f in frames:
            assert f.layer_count == 3

    def test_headroom_propagated(self) -> None:
        n = 60
        score = SongScore(fps=60)
        frames = score.build(
            _make_layer_frames(n),
            _make_note_patterns(n),
            _make_arc_frames(n, headroom=0.4),
            MotifTimeline(),
            n_frames=n,
        )
        for f in frames:
            assert f.headroom == pytest.approx(0.4)

    def test_motif_mapped_to_frames(self) -> None:
        n = 120  # 2 seconds at 60fps
        timeline = MotifTimeline(
            segments=[
                MotifSegment(start_time=0.0, end_time=1.0,
                             motif_id=0, repetition=0, similarity=1.0),
                MotifSegment(start_time=1.0, end_time=2.0,
                             motif_id=0, repetition=1, similarity=0.9),
            ],
            n_motifs=1,
        )
        score = SongScore(fps=60)
        frames = score.build(
            _make_layer_frames(n),
            _make_note_patterns(n),
            _make_arc_frames(n),
            timeline,
            n_frames=n,
            pattern_preferences=["chase_lr", "alternate"],
        )
        # First second: motif 0, rep 0
        assert frames[30].motif_id == 0
        assert frames[30].motif_repetition == 0
        # Second second: motif 0, rep 1
        assert frames[90].motif_id == 0
        assert frames[90].motif_repetition == 1

    def test_motif_assignment_round_robin(self) -> None:
        timeline = MotifTimeline(
            segments=[
                MotifSegment(0, 1, motif_id=0, repetition=0, similarity=1.0),
                MotifSegment(1, 2, motif_id=1, repetition=0, similarity=1.0),
                MotifSegment(2, 3, motif_id=2, repetition=0, similarity=1.0),
            ],
            n_motifs=3,
        )
        score = SongScore(fps=60)
        score.build(
            _make_layer_frames(180),
            _make_note_patterns(180),
            _make_arc_frames(180),
            timeline,
            n_frames=180,
            pattern_preferences=["chase_lr", "alternate"],
        )
        # 3 motifs, 2 patterns → round-robin
        assert score.motif_assignments[0].pattern_name == "chase_lr"
        assert score.motif_assignments[1].pattern_name == "alternate"
        assert score.motif_assignments[2].pattern_name == "chase_lr"

    def test_no_motifs_no_crash(self) -> None:
        n = 60
        score = SongScore(fps=60)
        frames = score.build(
            _make_layer_frames(n),
            _make_note_patterns(n),
            _make_arc_frames(n),
            MotifTimeline(),
            n_frames=n,
        )
        for f in frames:
            assert f.motif_id is None
            assert f.motif_repetition == 0

    def test_note_pattern_propagated(self) -> None:
        n = 60
        score = SongScore(fps=60)
        frames = score.build(
            _make_layer_frames(n),
            _make_note_patterns(n, regular=True),
            _make_arc_frames(n),
            MotifTimeline(),
            n_frames=n,
        )
        for f in frames:
            assert f.notes_per_beat == 4
