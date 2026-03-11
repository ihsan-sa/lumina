"""Tests for lumina.lighting.profiles.french_melodic."""

from __future__ import annotations

from lumina.audio.models import MusicState
from lumina.lighting.fixture_map import FixtureMap, FixtureType
from lumina.lighting.profiles.french_melodic import FrenchMelodicProfile


def _profile() -> FrenchMelodicProfile:
    return FrenchMelodicProfile(FixtureMap())


def _state(**kwargs: object) -> MusicState:
    defaults: dict[str, object] = {
        "energy": 0.5,
        "segment": "verse",
        "bpm": 130.0,
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


def _laser_ids() -> set[int]:
    fm = FixtureMap()
    return {f.fixture_id for f in fm.by_type(FixtureType.LASER)}


class TestOutputStructure:
    """Every generate() call returns exactly 15 commands with all IDs."""

    def test_always_fifteen_commands(self) -> None:
        p = _profile()
        for seg in ("verse", "chorus", "drop", "breakdown", "bridge", "intro", "outro"):
            cmds = p.generate(_state(segment=seg))
            assert len(cmds) == 15, f"Segment '{seg}' produced {len(cmds)} commands"

    def test_fixture_ids_complete(self) -> None:
        p = _profile()
        for seg in ("verse", "chorus", "drop", "breakdown", "intro", "outro"):
            cmds = p.generate(_state(segment=seg))
            ids = {c.fixture_id for c in cmds}
            assert ids == set(range(1, 16)), (
                f"Segment '{seg}' missing fixture IDs: {set(range(1, 16)) - ids}"
            )


class TestWarmPalette:
    """Verify warm palette: no blue > red in par commands during verse/chorus."""

    def test_verse_no_cold_blue(self) -> None:
        p = _profile()
        cmds = p.generate(_state(segment="verse", energy=0.6, bar_phase=0.3))
        cmd_map = {c.fixture_id: c for c in cmds}
        for pid in _par_ids():
            c = cmd_map[pid]
            if c.red > 0 or c.green > 0 or c.blue > 0:
                assert c.blue <= c.red, (
                    f"Par {pid} has cold blue (R={c.red}, B={c.blue})"
                )

    def test_chorus_no_cold_blue(self) -> None:
        p = _profile()
        cmds = p.generate(_state(segment="chorus", energy=0.7, bar_phase=0.5))
        cmd_map = {c.fixture_id: c for c in cmds}
        for pid in _par_ids():
            c = cmd_map[pid]
            if c.red > 0 or c.green > 0 or c.blue > 0:
                assert c.blue <= c.red, (
                    f"Par {pid} has cold blue (R={c.red}, B={c.blue})"
                )

    def test_drop_warm_hues_only(self) -> None:
        p = _profile()
        # After the initial burst (frame_count > 12), rainbow_roll uses hue 0.0-0.15
        # Generate twice to get past the burst
        p.generate(_state(segment="drop", energy=0.8, timestamp=0.0))
        for i in range(15):
            p.generate(_state(segment="drop", energy=0.8, timestamp=0.01 * (i + 1)))
        cmds = p.generate(_state(segment="drop", energy=0.8, timestamp=1.0, bar_phase=0.5))
        cmd_map = {c.fixture_id: c for c in cmds}
        for pid in _par_ids():
            c = cmd_map[pid]
            if c.red > 0:
                # Warm hues: red should dominate, blue should be minimal
                assert c.blue <= c.red, (
                    f"Par {pid} drop color not warm (R={c.red}, B={c.blue})"
                )

    def test_laser_always_off(self) -> None:
        p = _profile()
        for seg in ("verse", "chorus", "drop", "breakdown", "intro"):
            cmds = p.generate(_state(segment=seg))
            cmd_map = {c.fixture_id: c for c in cmds}
            for lid in _laser_ids():
                assert cmd_map[lid].special == 0, (
                    f"Laser should be off during {seg}"
                )


class TestHiHatResponse:
    """Hi-hat onsets should affect LED bar output."""

    def test_hihat_bumps_led_bars(self) -> None:
        p = _profile()
        # Generate without hi-hat
        cmds_no_hh = p.generate(_state(
            segment="verse", energy=0.5, onset_type=None, timestamp=1.0,
        ))
        map_no_hh = {c.fixture_id: c for c in cmds_no_hh}

        # Generate with hi-hat (resets bump tracker)
        p2 = _profile()
        cmds_hh = p2.generate(_state(
            segment="verse", energy=0.5, onset_type="hihat", timestamp=1.0,
        ))
        map_hh = {c.fixture_id: c for c in cmds_hh}

        # LED bars should have different output (hi-hat triggers bump)
        bar_ids = _led_bar_ids()
        hh_brightness = sum(map_hh[bid].red + map_hh[bid].green for bid in bar_ids)
        no_hh_brightness = sum(map_no_hh[bid].red + map_no_hh[bid].green for bid in bar_ids)
        # Hi-hat bump should increase brightness (or at minimum be non-zero)
        assert hh_brightness >= no_hh_brightness, (
            "LED bars should be at least as bright with hi-hat bump"
        )

    def test_hihat_then_decay(self) -> None:
        p = _profile()
        # Frame 1: hi-hat onset
        p.generate(_state(
            segment="verse", energy=0.5, onset_type="hihat", timestamp=1.0,
        ))
        # Frame 2: right after — bump should still be active
        cmds_soon = p.generate(_state(
            segment="verse", energy=0.5, onset_type=None, timestamp=1.02,
        ))
        # Frame 3: much later — bump should have decayed
        cmds_late = p.generate(_state(
            segment="verse", energy=0.5, onset_type=None, timestamp=2.0,
        ))
        bar_ids = _led_bar_ids()
        soon_brightness = sum(
            cmds_soon[i].red + cmds_soon[i].green
            for i, c in enumerate(cmds_soon) if c.fixture_id in bar_ids
        )
        late_brightness = sum(
            cmds_late[i].red + cmds_late[i].green
            for i, c in enumerate(cmds_late) if c.fixture_id in bar_ids
        )
        # After decay, brightness should be lower
        assert soon_brightness >= late_brightness, (
            "LED bar brightness should decay after hi-hat bump"
        )


class TestVerseChase:
    """Verse with moderate energy should have non-zero par output."""

    def test_verse_pars_active(self) -> None:
        p = _profile()
        cmds = p.generate(_state(
            segment="verse", energy=0.5, bar_phase=0.3, timestamp=1.0,
        ))
        cmd_map = {c.fixture_id: c for c in cmds}
        lit = [pid for pid in _par_ids() if cmd_map[pid].red > 0]
        assert len(lit) >= 1, "At least one par should be lit during verse"

    def test_verse_kick_bright(self) -> None:
        p = _profile()
        cmds = p.generate(_state(
            segment="verse", energy=0.7, onset_type="kick", is_beat=True, timestamp=1.0,
        ))
        cmd_map = {c.fixture_id: c for c in cmds}
        bright_pars = [pid for pid in _par_ids() if cmd_map[pid].red > 100]
        assert len(bright_pars) >= 1, "At least one par should be bright on kick"

    def test_verse_strobes_off(self) -> None:
        p = _profile()
        cmds = p.generate(_state(segment="verse", energy=0.5))
        cmd_map = {c.fixture_id: c for c in cmds}
        for sid in _strobe_ids():
            assert cmd_map[sid].strobe_rate == 0, "Strobes should be off during verse"


class TestBreakdownMinimal:
    """Breakdown should have most fixtures off."""

    def test_breakdown_strobes_off(self) -> None:
        p = _profile()
        cmds = p.generate(_state(segment="breakdown", energy=0.2))
        cmd_map = {c.fixture_id: c for c in cmds}
        for sid in _strobe_ids():
            assert cmd_map[sid].strobe_rate == 0
            assert cmd_map[sid].red == 0

    def test_breakdown_led_bars_off(self) -> None:
        p = _profile()
        cmds = p.generate(_state(segment="breakdown", energy=0.2))
        cmd_map = {c.fixture_id: c for c in cmds}
        for bid in _led_bar_ids():
            assert cmd_map[bid].red == 0
            assert cmd_map[bid].green == 0

    def test_breakdown_most_pars_off(self) -> None:
        p = _profile()
        cmds = p.generate(_state(segment="breakdown", energy=0.2))
        cmd_map = {c.fixture_id: c for c in cmds}
        dark_pars = [pid for pid in _par_ids() if cmd_map[pid].red == 0 and cmd_map[pid].green == 0]
        # At most 3 pars should be lit (candle flicker), so at least 5 should be dark
        assert len(dark_pars) >= 5, (
            f"Expected at least 5 dark pars in breakdown, got {len(dark_pars)}"
        )

    def test_breakdown_laser_off(self) -> None:
        p = _profile()
        cmds = p.generate(_state(segment="breakdown", energy=0.2))
        cmd_map = {c.fixture_id: c for c in cmds}
        for lid in _laser_ids():
            assert cmd_map[lid].special == 0


class TestChorusStrobe:
    """Chorus strobe behavior: gentle amber on downbeats only."""

    def test_chorus_downbeat_strobes_fire(self) -> None:
        p = _profile()
        cmds = p.generate(_state(
            segment="chorus", energy=0.7, is_downbeat=True, is_beat=True,
        ))
        cmd_map = {c.fixture_id: c for c in cmds}
        for sid in _strobe_ids():
            assert cmd_map[sid].strobe_rate == 100
            assert cmd_map[sid].strobe_intensity == 120

    def test_chorus_non_downbeat_strobes_off(self) -> None:
        p = _profile()
        cmds = p.generate(_state(
            segment="chorus", energy=0.7, is_downbeat=False, is_beat=False,
        ))
        cmd_map = {c.fixture_id: c for c in cmds}
        for sid in _strobe_ids():
            assert cmd_map[sid].strobe_rate == 0


class TestDropBurst:
    """Drop entry triggers initial strobe burst."""

    def test_drop_initial_burst(self) -> None:
        p = _profile()
        cmds = p.generate(_state(segment="drop", energy=0.9, timestamp=0.0))
        cmd_map = {c.fixture_id: c for c in cmds}
        for sid in _strobe_ids():
            assert cmd_map[sid].strobe_rate > 0 or cmd_map[sid].red > 0, (
                "Strobes should fire during drop initial burst"
            )
