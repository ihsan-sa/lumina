"""Tests for lumina.lighting.profiles.euro_alt."""

from __future__ import annotations

from lumina.audio.models import MusicState
from lumina.lighting.fixture_map import FixtureMap, FixtureType
from lumina.lighting.profiles.euro_alt import EuroAltProfile


def _profile() -> EuroAltProfile:
    return EuroAltProfile(FixtureMap())


def _state(**kwargs: object) -> MusicState:
    defaults: dict[str, object] = {
        "energy": 0.5,
        "segment": "verse",
        "bpm": 120.0,
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


class TestNoStrobes:
    """Euro alt NEVER uses strobes in any segment."""

    def test_no_strobes_any_segment(self) -> None:
        p = _profile()
        for seg in ("verse", "chorus", "drop", "breakdown", "bridge", "intro", "outro"):
            for energy in (0.1, 0.5, 0.9):
                cmds = p.generate(_state(segment=seg, energy=energy))
                cmd_map = {c.fixture_id: c for c in cmds}
                for sid in _strobe_ids():
                    cmd = cmd_map[sid]
                    assert cmd.strobe_rate == 0, (
                        f"Strobe {sid} has rate {cmd.strobe_rate} "
                        f"in segment '{seg}' at energy {energy}"
                    )
                    assert cmd.strobe_intensity == 0, (
                        f"Strobe {sid} has intensity {cmd.strobe_intensity} "
                        f"in segment '{seg}' at energy {energy}"
                    )

    def test_no_strobes_on_beats(self) -> None:
        p = _profile()
        cmds = p.generate(_state(
            segment="chorus", energy=0.9, is_beat=True, is_downbeat=True,
        ))
        cmd_map = {c.fixture_id: c for c in cmds}
        for sid in _strobe_ids():
            assert cmd_map[sid].strobe_rate == 0


class TestVerseMinimal:
    """Verse has only 1 lit par (spotlight isolate)."""

    def test_single_par_lit(self) -> None:
        p = _profile()
        cmds = p.generate(_state(segment="verse", energy=0.3))
        cmd_map = {c.fixture_id: c for c in cmds}

        lit_pars = [
            pid for pid in _par_ids()
            if cmd_map[pid].white > 0 or cmd_map[pid].red > 0
        ]
        assert len(lit_pars) == 1, (
            f"Expected 1 lit par in verse, got {len(lit_pars)}: {lit_pars}"
        )

    def test_verse_uses_white_channel(self) -> None:
        p = _profile()
        cmds = p.generate(_state(segment="verse", energy=0.5))
        cmd_map = {c.fixture_id: c for c in cmds}

        lit_pars = [
            pid for pid in _par_ids()
            if cmd_map[pid].white > 0
        ]
        assert len(lit_pars) >= 1, "Verse spotlight should use white channel"


class TestDropAllPars:
    """Drop has all pars with non-zero output."""

    def test_all_pars_active(self) -> None:
        p = _profile()
        cmds = p.generate(_state(segment="drop", energy=0.8, bar_phase=0.9))
        cmd_map = {c.fixture_id: c for c in cmds}

        for pid in _par_ids():
            cmd = cmd_map[pid]
            total = cmd.red + cmd.green + cmd.blue + cmd.white
            assert total > 0, (
                f"Par {pid} should be active during drop, "
                f"got RGBW=({cmd.red},{cmd.green},{cmd.blue},{cmd.white})"
            )

    def test_drop_led_bars_active(self) -> None:
        p = _profile()
        cmds = p.generate(_state(segment="drop", energy=0.8))
        cmd_map = {c.fixture_id: c for c in cmds}

        for bid in _led_bar_ids():
            cmd = cmd_map[bid]
            assert cmd.special > 0, (
                f"LED bar {bid} should be active during drop"
            )

    def test_drop_moderate_intensity(self) -> None:
        """Drop should not exceed ~70% intensity (gallery restraint)."""
        p = _profile()
        cmds = p.generate(_state(segment="drop", energy=1.0, bar_phase=1.0))
        cmd_map = {c.fixture_id: c for c in cmds}

        for pid in _par_ids():
            cmd = cmd_map[pid]
            # White channel should be <= ~70% of 255 (roughly 178) with some tolerance
            assert cmd.white <= 200, (
                f"Par {pid} white={cmd.white} exceeds moderate intensity cap"
            )


class TestBreakdownBlackout:
    """Breakdown has minimal light (mostly zeros)."""

    def test_all_pars_dark(self) -> None:
        p = _profile()
        cmds = p.generate(_state(segment="breakdown", energy=0.2))
        cmd_map = {c.fixture_id: c for c in cmds}

        for pid in _par_ids():
            cmd = cmd_map[pid]
            total = cmd.red + cmd.green + cmd.blue + cmd.white
            assert total == 0, (
                f"Par {pid} should be dark during breakdown, "
                f"got RGBW=({cmd.red},{cmd.green},{cmd.blue},{cmd.white})"
            )

    def test_only_one_led_bar_lit(self) -> None:
        p = _profile()
        cmds = p.generate(_state(segment="breakdown", energy=0.2))
        cmd_map = {c.fixture_id: c for c in cmds}

        lit_bars = [
            bid for bid in _led_bar_ids()
            if cmd_map[bid].special > 0 or cmd_map[bid].white > 0
        ]
        assert len(lit_bars) <= 1, (
            f"Expected at most 1 LED bar lit in breakdown, got {len(lit_bars)}"
        )

    def test_strobes_dark(self) -> None:
        p = _profile()
        cmds = p.generate(_state(segment="breakdown"))
        cmd_map = {c.fixture_id: c for c in cmds}

        for sid in _strobe_ids():
            cmd = cmd_map[sid]
            total = cmd.red + cmd.green + cmd.blue + cmd.white + cmd.strobe_rate
            assert total == 0, f"Strobe {sid} should be fully dark in breakdown"
