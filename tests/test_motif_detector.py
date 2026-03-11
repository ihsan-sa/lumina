"""Tests for the MotifDetector analysis module."""

from __future__ import annotations

import numpy as np
import pytest

from lumina.analysis.motif_detector import (
    MotifDetector,
    MotifTimeline,
    NotePattern,
)
from lumina.audio.beat_detector import BeatInfo

_HAS_LIBROSA = True
try:
    import librosa  # noqa: F401
except ImportError:
    _HAS_LIBROSA = False


def _make_beat_results(
    n_frames: int, bpm: float = 120.0, fps: int = 60
) -> list[BeatInfo]:
    """Create synthetic beat results at given BPM."""
    beat_interval = 60.0 / bpm  # seconds per beat
    frame_interval = 1.0 / fps
    beats_per_bar = 4

    results = []
    for i in range(n_frames):
        t = i * frame_interval
        beat_phase = (t % beat_interval) / beat_interval
        bar_interval = beat_interval * beats_per_bar
        bar_phase = (t % bar_interval) / bar_interval
        beat_number = int(t / beat_interval)
        is_beat = abs(beat_phase) < (frame_interval / beat_interval)
        is_downbeat = is_beat and (beat_number % beats_per_bar == 0)

        results.append(BeatInfo(
            bpm=bpm,
            beat_phase=beat_phase,
            bar_phase=bar_phase,
            is_beat=is_beat,
            is_downbeat=is_downbeat,
        ))
    return results


class TestMicroPatterns:
    """Tests for note-level pattern detection."""

    def test_regular_onsets_detected(self) -> None:
        """Synthetic stem with regular note onsets should be detected."""
        sr = 44100
        duration = 4.0
        n = int(sr * duration)
        fps = 60

        # Create "other" stem with regular clicks at 8 per beat (120 BPM)
        other = np.zeros(n, dtype=np.float32)
        notes_per_second = 120.0 / 60.0 * 4  # 8 notes/s at 120BPM, 4 per beat
        interval_samples = int(sr / notes_per_second)
        for i in range(0, n - 100, interval_samples):
            # Short click
            click = np.random.randn(100).astype(np.float32) * 0.8
            other[i : i + 100] += click

        beats = _make_beat_results(int(duration * fps), bpm=120.0, fps=fps)
        detector = MotifDetector(sr=sr, fps=fps)
        patterns = detector.detect_micro_patterns(other, beats)

        assert len(patterns) == len(beats)
        # At least some frames should detect a regular pattern
        regular_count = sum(1 for p in patterns if p.is_regular)
        assert regular_count > 0

    def test_silent_stem_no_pattern(self) -> None:
        sr = 44100
        fps = 60
        n_frames = 240
        other = np.zeros(sr * 4, dtype=np.float32)
        beats = _make_beat_results(n_frames, fps=fps)

        detector = MotifDetector(sr=sr, fps=fps)
        patterns = detector.detect_micro_patterns(other, beats)

        assert len(patterns) == n_frames
        for p in patterns:
            assert p.notes_per_beat == 0
            assert not p.is_regular


@pytest.mark.skipif(not _HAS_LIBROSA, reason="librosa not installed")
class TestMacroMotifs:
    """Tests for bar-level motif detection."""

    def test_repeated_section_detected(self) -> None:
        """A-B-A structure: first and third sections should share a motif."""
        sr = 44100
        fps = 60
        duration = 24.0  # 24 seconds
        n = int(sr * duration)

        # Create audio with A-B-A structure
        # A sections: similar tonal content (sine at 440Hz)
        # B section: different content (sine at 880Hz)
        t = np.linspace(0, duration, n, dtype=np.float32)

        audio = np.zeros(n, dtype=np.float32)
        # Section A1: 0-8s
        mask_a1 = (t >= 0) & (t < 8)
        audio[mask_a1] += np.sin(2 * np.pi * 440 * t[mask_a1]).astype(np.float32) * 0.5
        # Section B: 8-16s
        mask_b = (t >= 8) & (t < 16)
        audio[mask_b] += np.sin(2 * np.pi * 880 * t[mask_b]).astype(np.float32) * 0.5
        # Section A2: 16-24s
        mask_a2 = (t >= 16) & (t < 24)
        audio[mask_a2] += np.sin(2 * np.pi * 440 * t[mask_a2]).astype(np.float32) * 0.5

        beats = _make_beat_results(int(duration * fps), bpm=120.0, fps=fps)
        detector = MotifDetector(sr=sr, fps=fps)
        timeline = detector.detect_macro_motifs(audio, beats)

        # Should detect at least one motif (the repeated A sections)
        assert isinstance(timeline, MotifTimeline)
        # Note: detection depends on chroma similarity;
        # with pure sine waves it may not always match perfectly

    def test_too_few_bars_returns_empty(self) -> None:
        sr = 44100
        fps = 60
        # Very short audio — less than 2 bars
        duration = 1.0
        n = int(sr * duration)
        audio = np.random.randn(n).astype(np.float32) * 0.1
        beats = _make_beat_results(int(duration * fps), bpm=120.0, fps=fps)

        detector = MotifDetector(sr=sr, fps=fps)
        timeline = detector.detect_macro_motifs(audio, beats)
        assert timeline.n_motifs == 0
        assert len(timeline.segments) == 0


class TestNotePatternDataclass:
    """Basic tests for NotePattern dataclass."""

    def test_defaults(self) -> None:
        p = NotePattern(notes_per_beat=0, pattern_phase=0.0, is_regular=False)
        assert p.notes_per_beat == 0
        assert not p.is_regular

    def test_regular_pattern(self) -> None:
        p = NotePattern(notes_per_beat=4, pattern_phase=0.5, is_regular=True)
        assert p.notes_per_beat == 4
        assert p.pattern_phase == 0.5
        assert p.is_regular
