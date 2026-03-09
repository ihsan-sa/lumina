"""Tests for the LUMINA structural analysis module.

Tests verify section boundary detection, feature extraction,
labeling logic, and frame mapping without requiring librosa's
full analysis (mocked where needed).
"""

from __future__ import annotations

import numpy as np
import pytest

from lumina.audio.beat_detector import BeatInfo
from lumina.audio.energy_tracker import EnergyFrame
from lumina.audio.onset_detector import OnsetEvent
from lumina.audio.segment_classifier import SegmentFrame, SEGMENT_LABELS
from lumina.audio.source_separator import StemSet
from lumina.audio.structural_analyzer import (
    Section,
    StructuralAnalyzer,
    StructuralMap,
)
from lumina.audio.vocal_detector import VocalFrame


# ── Helpers ───────────────────────────────────────────────────────────


def _make_stems(n: int, sr: int = 44100) -> StemSet:
    """Create a dummy StemSet of given sample count."""
    return StemSet(
        drums=np.zeros(n, dtype=np.float32),
        bass=np.zeros(n, dtype=np.float32),
        vocals=np.zeros(n, dtype=np.float32),
        other=np.zeros(n, dtype=np.float32),
        sample_rate=sr,
    )


def _make_beat_results(n: int) -> list[BeatInfo]:
    return [
        BeatInfo(bpm=120.0, beat_phase=0.0, bar_phase=0.0, is_beat=False, is_downbeat=False)
        for _ in range(n)
    ]


def _make_energy_results(
    n: int,
    energies: list[float] | None = None,
) -> list[EnergyFrame]:
    if energies is not None:
        return [
            EnergyFrame(energy=e, energy_derivative=0.0, spectral_centroid=2000.0, sub_bass_energy=0.3)
            for e in energies
        ]
    return [
        EnergyFrame(energy=0.5, energy_derivative=0.0, spectral_centroid=2000.0, sub_bass_energy=0.3)
        for _ in range(n)
    ]


def _make_onset_results(n: int) -> list[OnsetEvent | None]:
    return [None] * n


def _make_vocal_results(
    n: int,
    vocal_energies: list[float] | None = None,
) -> list[VocalFrame]:
    if vocal_energies is not None:
        return [
            VocalFrame(vocal_energy=v, is_vocal=v > 0.15, harmonic_ratio=0.5)
            for v in vocal_energies
        ]
    return [
        VocalFrame(vocal_energy=0.3, is_vocal=True, harmonic_ratio=0.5)
        for _ in range(n)
    ]


# ── Section dataclass ────────────────────────────────────────────────


class TestSection:
    def test_fields(self) -> None:
        sec = Section(
            start_time=0.0,
            end_time=30.0,
            segment_type="verse",
            confidence=0.7,
            features={"mean_energy": 0.5},
        )
        assert sec.start_time == 0.0
        assert sec.end_time == 30.0
        assert sec.segment_type == "verse"
        assert sec.confidence == 0.7

    def test_valid_segment_types(self) -> None:
        for label in SEGMENT_LABELS:
            sec = Section(
                start_time=0.0, end_time=10.0,
                segment_type=label, confidence=0.5, features={},
            )
            assert sec.segment_type == label


# ── StructuralMap ────────────────────────────────────────────────────


class TestStructuralMap:
    def test_basic(self) -> None:
        sections = [
            Section(0.0, 30.0, "intro", 0.8, {}),
            Section(30.0, 60.0, "verse", 0.6, {}),
        ]
        sm = StructuralMap(sections=sections, duration=60.0)
        assert len(sm.sections) == 2
        assert sm.duration == 60.0


# ── map_to_frames ────────────────────────────────────────────────────


class TestMapToFrames:
    def test_correct_frame_count(self) -> None:
        """map_to_frames should return exactly num_frames frames."""
        analyzer = StructuralAnalyzer(fps=60)
        sm = StructuralMap(
            sections=[Section(0.0, 10.0, "verse", 0.6, {"mean_energy": 0.5})],
            duration=10.0,
        )
        frames = analyzer.map_to_frames(sm, num_frames=600, fps=60)
        assert len(frames) == 600

    def test_frame_labels_match_sections(self) -> None:
        """Frames should get the label of their containing section."""
        analyzer = StructuralAnalyzer(fps=60)
        sm = StructuralMap(
            sections=[
                Section(0.0, 5.0, "intro", 0.8, {}),
                Section(5.0, 10.0, "verse", 0.6, {}),
            ],
            duration=10.0,
        )
        frames = analyzer.map_to_frames(sm, num_frames=600, fps=60)
        # Frame at t=2.0 (frame 120) should be intro
        assert frames[120].segment == "intro"
        # Frame at t=7.0 (frame 420) should be verse
        assert frames[420].segment == "verse"

    def test_all_frames_have_valid_labels(self) -> None:
        """Every frame should have a valid segment label."""
        analyzer = StructuralAnalyzer(fps=60)
        sm = StructuralMap(
            sections=[
                Section(0.0, 5.0, "intro", 0.8, {}),
                Section(5.0, 15.0, "chorus", 0.7, {}),
                Section(15.0, 20.0, "outro", 0.6, {}),
            ],
            duration=20.0,
        )
        frames = analyzer.map_to_frames(sm, num_frames=1200, fps=60)
        for f in frames:
            assert f.segment in SEGMENT_LABELS


# ── Labeling logic ───────────────────────────────────────────────────


class TestLabelingLogic:
    def _make_analyzer(self) -> StructuralAnalyzer:
        return StructuralAnalyzer(sr=44100, fps=60, min_section_duration=4.0)

    def test_intro_labeled(self) -> None:
        """Early low-energy section should be labeled intro."""
        analyzer = self._make_analyzer()
        sections = [
            Section(0.0, 10.0, "verse", 0.5, {
                "mean_energy": 0.2, "onset_density": 0.1,
                "vocal_presence": 0.1, "sub_bass_energy": 0.1,
                "energy_variance": 0.01,
            }),
            Section(10.0, 60.0, "verse", 0.5, {
                "mean_energy": 0.6, "onset_density": 0.3,
                "vocal_presence": 0.5, "sub_bass_energy": 0.3,
                "energy_variance": 0.05,
            }),
        ]
        labeled = analyzer._label_sections(sections, duration=60.0)
        assert labeled[0].segment_type == "intro"

    def test_outro_labeled(self) -> None:
        """Late low-energy section should be labeled outro."""
        analyzer = self._make_analyzer()
        sections = [
            Section(0.0, 52.0, "verse", 0.5, {
                "mean_energy": 0.6, "onset_density": 0.3,
                "vocal_presence": 0.5, "sub_bass_energy": 0.3,
                "energy_variance": 0.05,
            }),
            Section(52.0, 60.0, "verse", 0.5, {
                "mean_energy": 0.2, "onset_density": 0.1,
                "vocal_presence": 0.1, "sub_bass_energy": 0.1,
                "energy_variance": 0.01,
            }),
        ]
        labeled = analyzer._label_sections(sections, duration=60.0)
        assert labeled[-1].segment_type == "outro"

    def test_drop_on_energy_jump(self) -> None:
        """Section after large energy increase should be labeled drop."""
        analyzer = self._make_analyzer()
        sections = [
            Section(0.0, 20.0, "verse", 0.5, {
                "mean_energy": 0.3, "onset_density": 0.1,
                "vocal_presence": 0.2, "sub_bass_energy": 0.2,
                "energy_variance": 0.02,
            }),
            Section(20.0, 40.0, "verse", 0.5, {
                "mean_energy": 0.75, "onset_density": 0.4,
                "vocal_presence": 0.15, "sub_bass_energy": 0.5,
                "energy_variance": 0.05,
            }),
        ]
        labeled = analyzer._label_sections(sections, duration=60.0)
        assert labeled[1].segment_type == "drop"

    def test_breakdown_on_energy_drop(self) -> None:
        """Section after large energy decrease should be labeled breakdown."""
        analyzer = self._make_analyzer()
        sections = [
            Section(10.0, 30.0, "verse", 0.5, {
                "mean_energy": 0.7, "onset_density": 0.4,
                "vocal_presence": 0.3, "sub_bass_energy": 0.4,
                "energy_variance": 0.05,
            }),
            Section(30.0, 50.0, "verse", 0.5, {
                "mean_energy": 0.2, "onset_density": 0.05,
                "vocal_presence": 0.1, "sub_bass_energy": 0.1,
                "energy_variance": 0.01,
            }),
        ]
        labeled = analyzer._label_sections(sections, duration=60.0)
        assert labeled[1].segment_type == "breakdown"


# ── Merge short sections ─────────────────────────────────────────────


class TestMergeShortSections:
    def test_short_sections_absorbed(self) -> None:
        """Sections shorter than min_section_duration are absorbed."""
        analyzer = StructuralAnalyzer(min_section_duration=4.0)
        sections = [
            Section(0.0, 10.0, "verse", 0.6, {}),
            Section(10.0, 12.0, "drop", 0.5, {}),  # 2s — too short
            Section(12.0, 30.0, "chorus", 0.7, {}),
        ]
        merged = analyzer._merge_short_sections(sections)
        assert len(merged) == 2
        assert merged[0].end_time == 12.0  # Absorbed the short section
        assert merged[1].segment_type == "chorus"

    def test_long_sections_kept(self) -> None:
        """Sections longer than min_section_duration are kept."""
        analyzer = StructuralAnalyzer(min_section_duration=4.0)
        sections = [
            Section(0.0, 10.0, "verse", 0.6, {}),
            Section(10.0, 25.0, "chorus", 0.7, {}),
            Section(25.0, 40.0, "verse", 0.6, {}),
        ]
        merged = analyzer._merge_short_sections(sections)
        assert len(merged) == 3


# ── Short audio fallback ─────────────────────────────────────────────


class TestShortAudioFallback:
    def test_very_short_audio_single_section(self) -> None:
        """Audio shorter than 2s should return a single verse section."""
        analyzer = StructuralAnalyzer(sr=44100, fps=60)
        audio = np.zeros(44100, dtype=np.float32)  # 1 second
        stems = _make_stems(44100)
        n = 60  # 1 second of frames
        result = analyzer.analyze(
            audio,
            stems,
            _make_beat_results(n),
            _make_energy_results(n),
            _make_onset_results(n),
            _make_vocal_results(n),
        )
        assert len(result.sections) == 1
        assert result.sections[0].segment_type == "verse"
