"""Tests for spatial behavior of lighting profiles.

Verifies that each profile produces different pattern selections for
different segments, uses fixture count escalation, and handles LED bars
and laser correctly.
"""

from __future__ import annotations

from lumina.audio.models import MusicState
from lumina.control.protocol import FixtureCommand
from lumina.lighting.engine import LightingEngine
from lumina.lighting.fixture_map import FixtureMap, FixtureType


def _state(**kwargs: object) -> MusicState:
    defaults: dict[str, object] = {
        "energy": 0.5,
        "segment": "verse",
        "bpm": 128.0,
        "beat_phase": 0.25,
        "bar_phase": 0.5,
        "timestamp": 10.0,
        "vocal_energy": 0.3,
    }
    defaults.update(kwargs)
    return MusicState(**defaults)  # type: ignore[arg-type]


def _generate(profile: str, **kwargs: object) -> list[FixtureCommand]:
    engine = LightingEngine()
    state = _state(genre_weights={profile: 1.0}, **kwargs)
    return engine.generate(state)


def _active_count(cmds: list[FixtureCommand], fixture_type: FixtureType) -> int:
    """Count how many fixtures of a type have non-zero output."""
    fm = FixtureMap()
    type_ids = {f.fixture_id for f in fm.by_type(fixture_type)}
    return sum(
        1 for c in cmds
        if c.fixture_id in type_ids
        and (c.red + c.green + c.blue + c.white + c.strobe_rate + c.strobe_intensity + c.special > 0)
    )


def _laser_special(cmds: list[FixtureCommand]) -> int:
    """Get the laser fixture's special byte."""
    fm = FixtureMap()
    laser_ids = {f.fixture_id for f in fm.by_type(FixtureType.LASER)}
    for c in cmds:
        if c.fixture_id in laser_ids:
            return c.special
    return -1


# ─── Rage Trap ────────────────────────────────────────────────────


class TestRageTrapPatterns:
    def test_15_fixtures_all_segments(self) -> None:
        for seg in ["verse", "chorus", "drop", "breakdown", "intro", "outro"]:
            cmds = _generate("rage_trap", segment=seg)
            assert len(cmds) == 15, f"segment={seg}: got {len(cmds)}"

    def test_drop_all_fixtures_active(self) -> None:
        cmds = _generate(
            "rage_trap", segment="drop", energy=0.9,
            is_beat=True, beat_phase=0.25,
        )
        active = _active_count(cmds, FixtureType.PAR)
        assert active == 8  # All pars active during drop on-beat

    def test_breakdown_few_pars(self) -> None:
        cmds = _generate(
            "rage_trap", segment="breakdown", energy=0.2,
        )
        active = _active_count(cmds, FixtureType.PAR)
        assert active <= 4

    def test_laser_active_during_drop(self) -> None:
        cmds = _generate(
            "rage_trap", segment="drop", energy=0.9,
            is_beat=True, beat_phase=0.25,
        )
        assert _laser_special(cmds) > 0

    def test_laser_off_during_breakdown(self) -> None:
        cmds = _generate("rage_trap", segment="breakdown", energy=0.2)
        assert _laser_special(cmds) == 0

    def test_verse_vs_drop_different(self) -> None:
        verse = _generate("rage_trap", segment="verse", energy=0.5)
        drop = _generate("rage_trap", segment="drop", energy=0.9, is_beat=True)
        # Drop should be much brighter overall
        verse_total = sum(c.red + c.green + c.blue for c in verse)
        drop_total = sum(c.red + c.green + c.blue for c in drop)
        assert drop_total > verse_total


# ─── Festival EDM ────────────────────────────────────────────────


class TestFestivalEdmPatterns:
    def test_15_fixtures_all_segments(self) -> None:
        for seg in ["verse", "chorus", "drop", "breakdown", "intro", "outro"]:
            cmds = _generate("festival_edm", segment=seg)
            assert len(cmds) == 15, f"segment={seg}: got {len(cmds)}"

    def test_drop_led_bars_active(self) -> None:
        cmds = _generate(
            "festival_edm", segment="drop", energy=0.9,
            is_beat=True, beat_phase=0.25,
        )
        active = _active_count(cmds, FixtureType.LED_BAR)
        assert active == 2  # Both LED bars active during drop

    def test_breakdown_minimal(self) -> None:
        cmds = _generate("festival_edm", segment="breakdown", energy=0.2)
        active_pars = _active_count(cmds, FixtureType.PAR)
        active_strobes = _active_count(cmds, FixtureType.STROBE)
        assert active_pars <= 2  # Only 1 breathing par
        assert active_strobes == 0

    def test_laser_active_during_drop(self) -> None:
        cmds = _generate(
            "festival_edm", segment="drop", energy=0.9, is_beat=True,
        )
        assert _laser_special(cmds) > 0

    def test_groove_uses_alternating(self) -> None:
        """Groove should have different colors on adjacent fixtures."""
        cmds = _generate("festival_edm", segment="verse", energy=0.6, is_beat=True)
        fm = FixtureMap()
        pars = fm.by_type(FixtureType.PAR)
        par_cmds = [c for c in cmds if c.fixture_id in {f.fixture_id for f in pars}]
        active = [c for c in par_cmds if c.red + c.green + c.blue > 0]
        # With alternate pattern, adjacent fixtures should differ
        if len(active) >= 2:
            colors = [(c.red, c.green, c.blue) for c in active]
            assert len(set(colors)) > 1  # Not all identical


# ─── Psych R&B ────────────────────────────────────────────────────


class TestPsychRnbPatterns:
    def test_15_fixtures_all_segments(self) -> None:
        for seg in ["verse", "chorus", "drop", "breakdown", "intro", "outro"]:
            cmds = _generate("psych_rnb", segment=seg)
            assert len(cmds) == 15, f"segment={seg}: got {len(cmds)}"

    def test_laser_always_off(self) -> None:
        for seg in ["verse", "chorus", "drop", "breakdown"]:
            cmds = _generate("psych_rnb", segment=seg, energy=0.9, is_beat=True)
            assert _laser_special(cmds) == 0, f"Laser should be off in {seg}"

    def test_no_strobe_burst_ever(self) -> None:
        """Psych R&B should never use max strobe rates."""
        for seg in ["verse", "chorus", "drop", "breakdown"]:
            cmds = _generate("psych_rnb", segment=seg, energy=0.9, is_beat=True)
            for c in cmds:
                assert c.strobe_rate < 200, f"Strobe too aggressive in {seg}"

    def test_verse_has_drift(self) -> None:
        """Verse should have per-fixture phase offset (not all same brightness)."""
        cmds = _generate("psych_rnb", segment="verse", energy=0.5)
        fm = FixtureMap()
        pars = fm.by_type(FixtureType.PAR)
        par_cmds = {c.fixture_id: c for c in cmds if c.fixture_id in {f.fixture_id for f in pars}}
        active = [c for c in par_cmds.values() if c.red + c.green + c.blue > 0]
        if len(active) >= 3:
            brightnesses = [c.red + c.green + c.blue for c in active]
            # With phase offset, fixtures should have varying brightness
            assert max(brightnesses) - min(brightnesses) > 0 or len(active) < 3

    def test_drop_led_bars_high(self) -> None:
        cmds = _generate("psych_rnb", segment="drop", energy=0.9)
        active = _active_count(cmds, FixtureType.LED_BAR)
        assert active == 2


# ─── Generic ──────────────────────────────────────────────────────


class TestGenericPatterns:
    def test_15_fixtures_all_segments(self) -> None:
        for seg in ["verse", "chorus", "drop", "breakdown", "intro", "outro"]:
            cmds = _generate("generic", segment=seg)
            assert len(cmds) == 15, f"segment={seg}: got {len(cmds)}"

    def test_safe_ranges_all_segments(self) -> None:
        for seg in ["verse", "chorus", "drop", "breakdown", "intro", "outro"]:
            for energy in [0.2, 0.5, 0.9]:
                cmds = _generate(
                    "generic", segment=seg, energy=energy,
                    is_beat=True, is_downbeat=True,
                )
                for c in cmds:
                    assert 0 <= c.red <= 255
                    assert 0 <= c.green <= 255
                    assert 0 <= c.blue <= 255
                    assert 0 <= c.white <= 255
                    assert 0 <= c.strobe_rate <= 255
                    assert 0 <= c.strobe_intensity <= 255
                    assert 0 <= c.special <= 255

    def test_led_bar_follows_pars(self) -> None:
        """LED bars should have non-zero output when pars do."""
        cmds = _generate("generic", segment="verse", energy=0.5)
        par_active = _active_count(cmds, FixtureType.PAR) > 0
        bar_active = _active_count(cmds, FixtureType.LED_BAR) > 0
        assert par_active == bar_active

    def test_laser_during_drop_only(self) -> None:
        verse_laser = _laser_special(_generate("generic", segment="verse"))
        drop_laser = _laser_special(_generate("generic", segment="drop", energy=0.9))
        assert verse_laser == 0
        assert drop_laser > 0

    def test_breakdown_few_fixtures(self) -> None:
        cmds = _generate("generic", segment="breakdown", energy=0.2)
        active = _active_count(cmds, FixtureType.PAR)
        assert active <= 4

    def test_drop_more_active_than_verse(self) -> None:
        verse = _generate("generic", segment="verse", energy=0.3)
        drop = _generate("generic", segment="drop", energy=0.9)
        verse_active = _active_count(verse, FixtureType.PAR)
        drop_active = _active_count(drop, FixtureType.PAR)
        assert drop_active >= verse_active


# ─── Cross-profile ────────────────────────────────────────────────


class TestCrossProfile:
    def test_all_profiles_deterministic(self) -> None:
        engine = LightingEngine()
        for profile in ["rage_trap", "psych_rnb", "festival_edm", "generic"]:
            state = _state(
                genre_weights={profile: 1.0},
                segment="drop", energy=0.9,
                is_beat=True, timestamp=5.0,
            )
            c1 = engine.generate(state)
            c2 = engine.generate(state)
            for a, b in zip(c1, c2, strict=True):
                assert a == b, f"{profile}: non-deterministic"

    def test_fixture_escalation_low_vs_high(self) -> None:
        """At low energy, fewer fixtures active than at high energy."""
        for profile in ["rage_trap", "festival_edm", "generic"]:
            low = _generate(profile, segment="verse", energy=0.2)
            high = _generate(profile, segment="verse", energy=0.9)
            low_active = _active_count(low, FixtureType.PAR)
            high_active = _active_count(high, FixtureType.PAR)
            assert high_active >= low_active, f"{profile}: escalation failed"
