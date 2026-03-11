"""Tests for lumina.lighting.profiles.french_hard."""

from __future__ import annotations

from lumina.audio.models import MusicState
from lumina.lighting.fixture_map import FixtureMap, FixtureType
from lumina.lighting.profiles.french_hard import FrenchHardProfile


def _profile() -> FrenchHardProfile:
    return FrenchHardProfile(FixtureMap())


def _state(**kwargs: object) -> MusicState:
    defaults: dict[str, object] = {
        "energy": 0.5,
        "segment": "verse",
        "bpm": 140.0,
    }
    defaults.update(kwargs)
    return MusicState(**defaults)  # type: ignore[arg-type]


def _par_ids() -> set[int]:
    fm = FixtureMap()
    return {f.fixture_id for f in fm.by_type(FixtureType.PAR)}


def _strobe_ids() -> set[int]:
    fm = FixtureMap()
    return {f.fixture_id for f in fm.by_type(FixtureType.STROBE)}


def _led_bar_ids() -> set[int]:
    fm = FixtureMap()
    return {f.fixture_id for f in fm.by_type(FixtureType.LED_BAR)}


class TestOutputStructure:
    """Every generate() call returns exactly 15 commands."""

    def test_always_fifteen_commands(self) -> None:
        p = _profile()
        for seg in ("verse", "chorus", "drop", "breakdown", "bridge", "intro", "outro"):
            cmds = p.generate(_state(segment=seg))
            assert len(cmds) == 15, f"Segment '{seg}' produced {len(cmds)} commands"

    def test_fixture_ids_complete(self) -> None:
        p = _profile()
        cmds = p.generate(_state())
        ids = {c.fixture_id for c in cmds}
        assert ids == set(range(1, 16))

    def test_sorted_by_fixture_id(self) -> None:
        p = _profile()
        cmds = p.generate(_state(segment="chorus"))
        ids = [c.fixture_id for c in cmds]
        assert ids == sorted(ids)


class TestColdPalette:
    """Verify cold colors: blue >= red in most par commands."""

    def test_verse_pars_are_cold(self) -> None:
        p = _profile()
        cmds = p.generate(_state(segment="verse", energy=0.6, is_beat=True))
        cmd_map = {c.fixture_id: c for c in cmds}
        par_cmds = [cmd_map[pid] for pid in _par_ids()]
        lit_pars = [c for c in par_cmds if c.red > 0 or c.blue > 0]
        if lit_pars:
            for c in lit_pars:
                assert c.blue >= c.red, (
                    f"Fixture {c.fixture_id}: blue={c.blue} < red={c.red} -- not cold"
                )

    def test_chorus_pars_are_cold(self) -> None:
        p = _profile()
        cmds = p.generate(_state(segment="chorus", energy=0.7, beat_phase=0.1))
        cmd_map = {c.fixture_id: c for c in cmds}
        par_cmds = [cmd_map[pid] for pid in _par_ids()]
        lit_pars = [c for c in par_cmds if c.red > 0 or c.blue > 0]
        if lit_pars:
            for c in lit_pars:
                assert c.blue >= c.red, (
                    f"Fixture {c.fixture_id}: blue={c.blue} < red={c.red} -- not cold"
                )

    def test_breakdown_par_is_steel_blue(self) -> None:
        p = _profile()
        cmds = p.generate(_state(segment="breakdown"))
        cmd_map = {c.fixture_id: c for c in cmds}
        par_cmds = [cmd_map[pid] for pid in _par_ids()]
        lit_pars = [c for c in par_cmds if c.blue > 0]
        assert len(lit_pars) >= 1, "At least one par should be lit in breakdown"
        for c in lit_pars:
            assert c.blue > c.red, "Breakdown spotlight should be steel blue"


class TestDropBlinder:
    """First frame of drop should have high-intensity white output."""

    def test_blinder_first_frame(self) -> None:
        p = _profile()
        # First call with drop segment triggers blinder
        cmds = p.generate(_state(segment="drop", energy=0.9))
        cmd_map = {c.fixture_id: c for c in cmds}
        # All pars should have high white output
        for pid in _par_ids():
            c = cmd_map[pid]
            assert c.white > 200, (
                f"Par {pid} white={c.white} -- expected blinder (>200)"
            )

    def test_blinder_has_strobe_active(self) -> None:
        p = _profile()
        cmds = p.generate(_state(segment="drop", energy=0.9))
        cmd_map = {c.fixture_id: c for c in cmds}
        any_strobe = any(
            cmd_map[sid].strobe_rate > 0 for sid in _strobe_ids()
        )
        assert any_strobe, "Strobes should fire during drop blinder"

    def test_post_blinder_has_chase(self) -> None:
        """After blinder frames, drop should use chase_mirror pattern."""
        p = _profile()
        # Run 7 frames to exhaust blinder phase (6 frames)
        for _ in range(7):
            cmds = p.generate(_state(segment="drop", energy=0.8, bar_phase=0.3))
        cmd_map = {c.fixture_id: c for c in cmds}
        # After blinder, pars should show varying intensities (chase pattern)
        par_vals = [cmd_map[pid].blue for pid in _par_ids()]
        # Not all identical (chase creates variation)
        assert len(set(par_vals)) > 1, "Post-blinder drop should have chase variation"


class TestBreakdownMinimal:
    """Breakdown has max 1-2 lit pars."""

    def test_few_lit_pars(self) -> None:
        p = _profile()
        cmds = p.generate(_state(segment="breakdown", energy=0.2))
        cmd_map = {c.fixture_id: c for c in cmds}
        lit_count = sum(
            1 for pid in _par_ids()
            if cmd_map[pid].red > 0 or cmd_map[pid].green > 0 or cmd_map[pid].blue > 0
        )
        assert lit_count <= 2, f"Breakdown should have max 1-2 lit pars, got {lit_count}"

    def test_strobes_off(self) -> None:
        p = _profile()
        cmds = p.generate(_state(segment="breakdown"))
        cmd_map = {c.fixture_id: c for c in cmds}
        for sid in _strobe_ids():
            c = cmd_map[sid]
            assert c.strobe_rate == 0, f"Strobe {sid} should be off in breakdown"
            assert c.red == 0 and c.green == 0 and c.blue == 0, (
                f"Strobe {sid} should be black in breakdown"
            )

    def test_led_bars_off(self) -> None:
        p = _profile()
        cmds = p.generate(_state(segment="breakdown"))
        cmd_map = {c.fixture_id: c for c in cmds}
        for bid in _led_bar_ids():
            c = cmd_map[bid]
            assert c.red == 0 and c.green == 0 and c.blue == 0, (
                f"LED bar {bid} should be off in breakdown"
            )


class TestChorusAlternate:
    """Chorus has alternating pattern on pars."""

    def test_even_odd_differ(self) -> None:
        """Even and odd pars should have different colors (alternating)."""
        p = _profile()
        # beat_phase < 0.5: even = color, odd = black (or vice versa)
        cmds = p.generate(_state(segment="chorus", energy=0.7, beat_phase=0.2))
        cmd_map = {c.fixture_id: c for c in cmds}
        par_list = sorted(_par_ids())
        # Group by even/odd index
        even_whites = [cmd_map[par_list[i]].white for i in range(0, len(par_list), 2)]
        odd_whites = [cmd_map[par_list[i]].white for i in range(1, len(par_list), 2)]
        # In alternate pattern, one group should be bright and the other dark
        even_avg = sum(even_whites) / max(len(even_whites), 1)
        odd_avg = sum(odd_whites) / max(len(odd_whites), 1)
        assert abs(even_avg - odd_avg) > 50, (
            f"Even avg={even_avg:.0f}, odd avg={odd_avg:.0f} -- "
            "should differ (alternating pattern)"
        )

    def test_alternation_swaps_on_beat_half(self) -> None:
        """Pattern should swap at beat_phase 0.5."""
        p = _profile()
        cmds_a = p.generate(_state(segment="chorus", energy=0.7, beat_phase=0.2))
        cmds_b = p.generate(_state(segment="chorus", energy=0.7, beat_phase=0.7))
        map_a = {c.fixture_id: c for c in cmds_a}
        map_b = {c.fixture_id: c for c in cmds_b}
        par_list = sorted(_par_ids())
        # First par should swap between the two phases
        first_par = par_list[0]
        # The white values should differ between the two beat phases
        assert map_a[first_par].white != map_b[first_par].white, (
            "First par should swap brightness between beat_phase 0.2 and 0.7"
        )

    def test_chorus_all_pars_active(self) -> None:
        """Chorus should use all pars (wall of light)."""
        p = _profile()
        cmds = p.generate(_state(segment="chorus", energy=0.7, beat_phase=0.1))
        cmd_map = {c.fixture_id: c for c in cmds}
        active = sum(
            1 for pid in _par_ids()
            if cmd_map[pid].red > 0 or cmd_map[pid].blue > 0 or cmd_map[pid].white > 0
        )
        # In alternate, half are on and half are off each beat half
        assert active >= len(_par_ids()) // 2, (
            f"Chorus should have at least half the pars active, got {active}"
        )
