"""Tests for LUMINA app entry point — MusicState assembly, CLI, and stem routing."""

from __future__ import annotations

from pathlib import Path

from lumina.app import AppConfig, _assemble_music_state, parse_args
from lumina.audio.beat_detector import BeatInfo
from lumina.audio.drop_predictor import DropFrame
from lumina.audio.energy_tracker import EnergyFrame
from lumina.audio.genre_classifier import GenreFrame
from lumina.audio.models import MusicState
from lumina.audio.onset_detector import OnsetEvent
from lumina.audio.segment_classifier import SegmentFrame
from lumina.audio.source_separator import StemSet
from lumina.audio.vocal_detector import VocalFrame

# ── Fixtures ────────────────────────────────────────────────────


def _make_beat(
    bpm: float = 128.0,
    beat_phase: float = 0.5,
    bar_phase: float = 0.25,
    is_beat: bool = False,
    is_downbeat: bool = False,
) -> BeatInfo:
    return BeatInfo(
        bpm=bpm,
        beat_phase=beat_phase,
        bar_phase=bar_phase,
        is_beat=is_beat,
        is_downbeat=is_downbeat,
    )


def _make_energy(
    energy: float = 0.6,
    energy_derivative: float = 0.05,
    spectral_centroid: float = 2000.0,
    sub_bass_energy: float = 0.3,
) -> EnergyFrame:
    return EnergyFrame(
        energy=energy,
        energy_derivative=energy_derivative,
        spectral_centroid=spectral_centroid,
        sub_bass_energy=sub_bass_energy,
    )


def _make_onset(
    onset_type: str = "kick",
    timestamp: float = 0.0,
    strength: float = 0.8,
) -> OnsetEvent:
    return OnsetEvent(timestamp=timestamp, onset_type=onset_type, strength=strength)


def _make_vocal(
    vocal_energy: float = 0.5,
    is_vocal: bool = True,
    harmonic_ratio: float = 0.7,
) -> VocalFrame:
    return VocalFrame(
        vocal_energy=vocal_energy,
        is_vocal=is_vocal,
        harmonic_ratio=harmonic_ratio,
    )


def _make_segment(
    segment: str = "chorus",
    confidence: float = 0.8,
) -> SegmentFrame:
    return SegmentFrame(segment=segment, confidence=confidence, scores={})


def _make_genre(
    genre_weights: dict[str, float] | None = None,
) -> GenreFrame:
    weights = genre_weights or {"festival_edm": 0.6, "uk_bass": 0.4}
    return GenreFrame(
        family="electronic",
        family_weights={"electronic": 0.8, "hiphop_rap": 0.2},
        genre_weights=weights,
    )


def _make_drop(
    drop_probability: float = 0.3,
    tension: float = 0.4,
) -> DropFrame:
    return DropFrame(
        drop_probability=drop_probability,
        tension=tension,
        rising_energy=True,
        onset_density=4.0,
    )


# ── TestMusicStateAssembly ──────────────────────────────────────


class TestMusicStateAssembly:
    """Tests for _assemble_music_state."""

    def test_all_fields_mapped(self) -> None:
        """All 15 MusicState fields are correctly populated."""
        state = _assemble_music_state(
            timestamp=5.0,
            beat=_make_beat(
                bpm=140.0,
                beat_phase=0.7,
                bar_phase=0.3,
                is_beat=True,
                is_downbeat=False,
            ),
            energy=_make_energy(
                energy=0.9,
                energy_derivative=0.15,
                spectral_centroid=3000.0,
                sub_bass_energy=0.5,
            ),
            onset=_make_onset(onset_type="snare"),
            vocal=_make_vocal(vocal_energy=0.8),
            segment=_make_segment(segment="drop"),
            genre=_make_genre(genre_weights={"rage_trap": 0.7, "psych_rnb": 0.3}),
            drop=_make_drop(drop_probability=0.6),
        )

        assert isinstance(state, MusicState)
        assert state.timestamp == 5.0
        assert state.bpm == 140.0
        assert state.beat_phase == 0.7
        assert state.bar_phase == 0.3
        assert state.is_beat is True
        assert state.is_downbeat is False
        assert state.energy == 0.9
        assert state.energy_derivative == 0.15
        assert state.segment == "drop"
        assert state.genre_weights == {"rage_trap": 0.7, "psych_rnb": 0.3}
        assert state.vocal_energy == 0.8
        assert state.spectral_centroid == 3000.0
        assert state.sub_bass_energy == 0.5
        assert state.onset_type == "snare"
        assert state.drop_probability == 0.6

    def test_onset_none(self) -> None:
        """When onset is None, onset_type is None."""
        state = _assemble_music_state(
            timestamp=0.0,
            beat=_make_beat(),
            energy=_make_energy(),
            onset=None,
            vocal=_make_vocal(),
            segment=_make_segment(),
            genre=_make_genre(),
            drop=_make_drop(),
        )
        assert state.onset_type is None

    def test_onset_with_type(self) -> None:
        """Onset event type is correctly propagated."""
        for onset_type in ("kick", "snare", "hihat", "clap"):
            state = _assemble_music_state(
                timestamp=0.0,
                beat=_make_beat(),
                energy=_make_energy(),
                onset=_make_onset(onset_type=onset_type),
                vocal=_make_vocal(),
                segment=_make_segment(),
                genre=_make_genre(),
                drop=_make_drop(),
            )
            assert state.onset_type == onset_type


# ── TestAppConfig ───────────────────────────────────────────────


class TestAppConfig:
    """Tests for AppConfig and CLI argument parsing."""

    def test_defaults(self) -> None:
        """Default config values."""
        config = AppConfig()
        assert config.mode == "file"
        assert config.file_path is None
        assert config.host == "0.0.0.0"
        assert config.port == 8765
        assert config.fps == 60
        assert config.sr == 44100
        assert config.udp_target is None

    def test_parse_file_mode(self) -> None:
        """Parse --mode file --file song.mp3."""
        config = parse_args(["--mode", "file", "--file", "song.mp3"])
        assert config.mode == "file"
        assert config.file_path == Path("song.mp3")

    def test_parse_live_mode(self) -> None:
        """Parse --mode live."""
        config = parse_args(["--mode", "live"])
        assert config.mode == "live"

    def test_parse_all_options(self) -> None:
        """Parse all CLI options."""
        config = parse_args(
            [
                "--mode",
                "file",
                "--file",
                "track.flac",
                "--host",
                "127.0.0.1",
                "--port",
                "9000",
                "--fps",
                "30",
                "--sr",
                "22050",
                "--udp-target",
                "192.168.1.50:5150",
            ]
        )
        assert config.host == "127.0.0.1"
        assert config.port == 9000
        assert config.fps == 30
        assert config.sr == 22050
        assert config.udp_target == "192.168.1.50:5150"


# ── TestStemRouting ────────────────────────────────────────────


class TestStemRouting:
    """Tests verifying that the file mode pipeline imports are wired correctly."""

    def test_source_separator_importable(self) -> None:
        """SourceSeparator should be importable from app."""
        from lumina.app import SourceSeparator

        assert SourceSeparator is not None

    def test_structural_analyzer_importable(self) -> None:
        """StructuralAnalyzer should be importable from app."""
        from lumina.app import StructuralAnalyzer

        assert StructuralAnalyzer is not None

    def test_stemset_has_expected_fields(self) -> None:
        """StemSet should have drums, bass, vocals, other fields."""
        import numpy as np

        n = 1000
        stems = StemSet(
            drums=np.zeros(n, dtype=np.float32),
            bass=np.zeros(n, dtype=np.float32),
            vocals=np.zeros(n, dtype=np.float32),
            other=np.zeros(n, dtype=np.float32),
            sample_rate=44100,
        )
        assert hasattr(stems, "drums")
        assert hasattr(stems, "bass")
        assert hasattr(stems, "vocals")
        assert hasattr(stems, "other")
        assert stems.sample_rate == 44100

    def test_energy_tracker_has_bass_stem_method(self) -> None:
        """EnergyTracker should have analyze_offline_with_bass_stem."""
        from lumina.audio.energy_tracker import EnergyTracker

        tracker = EnergyTracker()
        assert hasattr(tracker, "analyze_offline_with_bass_stem")

    def test_genre_classifier_has_classify_file(self) -> None:
        """GenreClassifier should have classify_file method."""
        from lumina.audio.genre_classifier import GenreClassifier

        clf = GenreClassifier()
        assert hasattr(clf, "classify_file")
