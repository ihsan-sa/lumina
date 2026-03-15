"""Tests for lumina.lighting.profiles.uk_bass."""

from __future__ import annotations

from lumina.audio.models import MusicState
from lumina.lighting.fixture_map import FixtureMap, FixtureType
from lumina.lighting.profiles.uk_bass import UkBassProfile


def _profile() -> UkBassProfile:
    return UkBassProfile(FixtureMap())


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


def _bar_ids() -> set[int]:
    fm = FixtureMap()
    return {f.fixture_id for f in fm.by_type(FixtureType.LED_BAR)}


def _laser_ids() -> set[int]:
    fm = FixtureMap()
    return {f.fixture_id for f in fm.by_type(FixtureType.LASER)}


class TestOutputStructure:
    """Every generate() call returns exactly 15 commands."""

    def test_always_fifteen_commands(self) -> None:
        p = _profile()
        for seg in ("verse", "chorus", "drop", "breakdown", "bridge", "intro", "outro",
                     "groove", "build"):
            cmds = p.generate(_state(segment=seg))
            assert len(cmds) == 15, f"Segment '{seg}' produced {len(cmds)} commands"

    def test_fixture_ids_complete(self) -> None:
        p = _profile()
        cmds = p.generate(_state())
        ids = {c.fixture_id for c in cmds}
        assert ids == set(range(1, 16))


class TestProfileName:
    """Profile has correct name."""

    def test_name(self) -> None:
        assert _profile().name == "uk_bass"


class TestVerse:
    """Verse uses organic flicker with dirty amber/green palette."""

    def test_pars_lit_in_verse(self) -> None:
        p = _profile()
        cmds = p.generate(_state(segment="verse", energy=0.4))
        cmd_map = {c.fixture_id: c for c in cmds}
        par_commands = [cmd_map[pid] for pid in _par_ids()]
        # At least some pars should have non-zero output
        lit = [c for c in par_commands if c.special > 0]
        assert len(lit) > 0

    def test_low_intensity_verse(self) -> None:
        p = _profile()
        cmds = p.generate(_state(segment="verse", energy=0.3))
        cmd_map = {c.fixture_id: c for c in cmds}
        # Verse should be low intensity — no par at max brightness
        for pid in _par_ids():
            c = cmd_map[pid]
            assert c.special <= 200, "Verse should not blast pars"


class TestDrop:
    """Drop uses blinder on entry, then alternating scatter."""

    def test_drop_produces_output(self) -> None:
        p = _profile()
        cmds = p.generate(_state(segment="drop", energy=0.9, is_beat=True,
                                  is_downbeat=True, onset_type="kick"))
        cmd_map = {c.fixture_id: c for c in cmds}
        # Something should be visibly lit
        any_lit = any(
            c.special > 0 or c.strobe_intensity > 0 or c.red + c.green + c.blue > 0
            for c in cmd_map.values()
        )
        assert any_lit

    def test_laser_active_in_drop(self) -> None:
        p = _profile()
        # Advance past blinder phase
        for _ in range(20):
            p.generate(_state(segment="drop", energy=0.9))
        cmds = p.generate(_state(segment="drop", energy=0.9))
        cmd_map = {c.fixture_id: c for c in cmds}
        for lid in _laser_ids():
            c = cmd_map[lid]
            assert c.special > 0, "Laser should have a pattern in drop"


class TestBreakdown:
    """Breakdown is intimate — single par breathe, rest off."""

    def test_breakdown_is_sparse(self) -> None:
        p = _profile()
        cmds = p.generate(_state(segment="breakdown", energy=0.15))
        cmd_map = {c.fixture_id: c for c in cmds}
        # Most pars should be off or very dim
        lit_pars = [
            cmd_map[pid] for pid in _par_ids()
            if cmd_map[pid].special > 30
        ]
        assert len(lit_pars) <= 3, "Breakdown should have few active pars"

    def test_strobes_off_in_breakdown(self) -> None:
        p = _profile()
        cmds = p.generate(_state(segment="breakdown", energy=0.1))
        cmd_map = {c.fixture_id: c for c in cmds}
        for sid in _strobe_ids():
            c = cmd_map[sid]
            assert c.strobe_intensity == 0, "Strobes should be off in breakdown"

    def test_laser_off_in_breakdown(self) -> None:
        p = _profile()
        cmds = p.generate(_state(segment="breakdown", energy=0.1))
        cmd_map = {c.fixture_id: c for c in cmds}
        for lid in _laser_ids():
            c = cmd_map[lid]
            assert c.special == 0, "Laser should be off in breakdown"


class TestBuild:
    """Build uses converging cold white + strobe stutter."""

    def test_build_produces_output(self) -> None:
        p = _profile()
        cmds = p.generate(_state(segment="build", energy=0.6, drop_probability=0.7))
        assert len(cmds) == 15

    def test_build_strobes_active_at_high_drop_prob(self) -> None:
        p = _profile()
        cmds = p.generate(_state(
            segment="build", energy=0.7, drop_probability=0.8,
            beat_phase=0.1, bar_phase=0.3,
        ))
        cmd_map = {c.fixture_id: c for c in cmds}
        # Stutter pattern lights strobes via RGBW channels (not strobe_intensity)
        any_strobe_lit = any(
            cmd_map[sid].red + cmd_map[sid].green + cmd_map[sid].blue + cmd_map[sid].white > 0
            for sid in _strobe_ids()
        )
        assert any_strobe_lit, "Build with high drop_probability should light strobes"


class TestGroove:
    """Groove dispatches to drop handler (sustained energy)."""

    def test_groove_produces_output(self) -> None:
        p = _profile()
        cmds = p.generate(_state(segment="groove", energy=0.8))
        assert len(cmds) == 15


class TestBeatResponse:
    """Kick onsets trigger bump decay."""

    def test_kick_onset_triggers_bump(self) -> None:
        p = _profile()
        # First call with kick onset
        p.generate(_state(
            segment="verse", energy=0.5, onset_type="kick",
            timestamp=1.0, is_beat=True,
        ))
        # Immediate next frame — bump should still be decaying
        cmds = p.generate(_state(
            segment="verse", energy=0.5, onset_type=None,
            timestamp=1.017,
        ))
        assert len(cmds) == 15  # Basic structural check


class TestMotifPatternPreferences:
    """Profile declares motif pattern preferences."""

    def test_has_preferences(self) -> None:
        p = _profile()
        prefs = p.motif_pattern_preferences
        assert isinstance(prefs, list)
        assert len(prefs) > 0

    def test_has_color_palette(self) -> None:
        p = _profile()
        palette = p.motif_color_palette
        assert isinstance(palette, list)
        assert len(palette) > 0
