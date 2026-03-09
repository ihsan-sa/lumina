"""Tests for lumina.lighting.engine."""

from __future__ import annotations

from lumina.audio.models import MusicState
from lumina.lighting.engine import LightingEngine
from lumina.lighting.fixture_map import FixtureMap


def _state(**kwargs: object) -> MusicState:
    defaults: dict[str, object] = {"energy": 0.5, "segment": "verse", "bpm": 140.0}
    defaults.update(kwargs)
    return MusicState(**defaults)  # type: ignore[arg-type]


class TestEngineInit:
    def test_default_fixture_map(self) -> None:
        engine = LightingEngine()
        assert len(engine.fixture_map) == 15

    def test_custom_fixture_map(self) -> None:
        fm = FixtureMap(fixtures=[])
        engine = LightingEngine(fixture_map=fm)
        assert len(engine.fixture_map) == 0

    def test_rage_trap_registered(self) -> None:
        engine = LightingEngine()
        assert "rage_trap" in engine.profile_names


class TestProfileSelection:
    def test_selects_rage_trap_by_weight(self) -> None:
        engine = LightingEngine()
        state = _state(genre_weights={"rage_trap": 0.8, "psych_rnb": 0.2})
        cmds = engine.generate(state)
        assert len(cmds) == 15

    def test_fallback_when_no_weights(self) -> None:
        engine = LightingEngine()
        state = _state(genre_weights={})
        cmds = engine.generate(state)
        assert len(cmds) == 15

    def test_fallback_when_unknown_profile(self) -> None:
        engine = LightingEngine()
        state = _state(genre_weights={"unknown_profile": 1.0})
        cmds = engine.generate(state)
        assert len(cmds) == 15

    def test_selects_highest_registered_weight(self) -> None:
        engine = LightingEngine()
        # psych_rnb not registered yet, so rage_trap should win
        state = _state(genre_weights={"psych_rnb": 0.6, "rage_trap": 0.4})
        cmds = engine.generate(state)
        assert len(cmds) == 15


class TestOutputStructure:
    def test_one_command_per_fixture(self) -> None:
        engine = LightingEngine()
        state = _state(genre_weights={"rage_trap": 1.0})
        cmds = engine.generate(state)
        ids = {c.fixture_id for c in cmds}
        assert ids == set(range(1, 16))

    def test_all_fixture_ids_present(self) -> None:
        engine = LightingEngine()
        fm = engine.fixture_map
        state = _state(genre_weights={"rage_trap": 1.0}, is_beat=True)
        cmds = engine.generate(state)
        assert {c.fixture_id for c in cmds} == set(fm.ids)

    def test_commands_valid_ranges(self) -> None:
        engine = LightingEngine()
        for seg in ("verse", "chorus", "drop", "breakdown", "intro"):
            state = _state(
                genre_weights={"rage_trap": 1.0},
                segment=seg,
                energy=0.9,
                is_beat=True,
                is_downbeat=True,
            )
            cmds = engine.generate(state)
            for c in cmds:
                assert 0 <= c.red <= 255
                assert 0 <= c.green <= 255
                assert 0 <= c.blue <= 255
                assert 0 <= c.white <= 255
                assert 0 <= c.strobe_rate <= 255
                assert 0 <= c.strobe_intensity <= 255
                assert 0 <= c.special <= 255


class TestConsistency:
    def test_deterministic_output(self) -> None:
        engine = LightingEngine()
        state = _state(
            genre_weights={"rage_trap": 1.0},
            segment="drop",
            energy=0.9,
            is_downbeat=True,
            is_beat=True,
            timestamp=1.0,
        )
        cmds1 = engine.generate(state)
        cmds2 = engine.generate(state)
        for c1, c2 in zip(cmds1, cmds2, strict=True):
            assert c1 == c2
