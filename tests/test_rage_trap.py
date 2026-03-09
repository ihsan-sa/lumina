"""Tests for lumina.lighting.profiles.rage_trap."""

from __future__ import annotations

from lumina.audio.models import MusicState
from lumina.lighting.fixture_map import FixtureMap, FixtureType
from lumina.lighting.profiles.rage_trap import RageTrapProfile


def _profile() -> RageTrapProfile:
    return RageTrapProfile(FixtureMap())


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


def _uv_ids() -> set[int]:
    fm = FixtureMap()
    return {f.fixture_id for f in fm.by_type(FixtureType.UV)}


class TestOutputStructure:
    """Every generate() call returns exactly 8 commands."""

    def test_always_eight_commands(self) -> None:
        p = _profile()
        for seg in ("verse", "chorus", "drop", "breakdown", "bridge", "intro", "outro"):
            cmds = p.generate(_state(segment=seg))
            assert len(cmds) == 8, f"Segment '{seg}' produced {len(cmds)} commands"

    def test_fixture_ids_complete(self) -> None:
        p = _profile()
        cmds = p.generate(_state())
        ids = {c.fixture_id for c in cmds}
        assert ids == set(range(1, 9))


class TestPreDropBlackout:
    """High drop_probability → total blackout."""

    def test_all_black_when_drop_imminent(self) -> None:
        p = _profile()
        cmds = p.generate(_state(drop_probability=0.9, segment="verse"))
        for c in cmds:
            assert c.red == 0
            assert c.green == 0
            assert c.blue == 0
            assert c.strobe_rate == 0


class TestDropExplosion:
    """Drop segment at high energy → strobes + red."""

    def test_drop_downbeat_strobes_fire(self) -> None:
        p = _profile()
        cmds = p.generate(
            _state(segment="drop", energy=0.9, is_downbeat=True, is_beat=True)
        )
        cmd_map = {c.fixture_id: c for c in cmds}
        for sid in _strobe_ids():
            assert cmd_map[sid].strobe_rate > 0
            assert cmd_map[sid].strobe_intensity > 0

    def test_drop_pars_red_on_kick(self) -> None:
        p = _profile()
        cmds = p.generate(
            _state(segment="drop", energy=0.9, onset_type="kick", is_beat=True)
        )
        cmd_map = {c.fixture_id: c for c in cmds}
        for pid in _par_ids():
            assert cmd_map[pid].red == 255
            assert cmd_map[pid].green == 0

    def test_drop_between_beats_dark(self) -> None:
        p = _profile()
        cmds = p.generate(
            _state(segment="drop", energy=0.9, is_beat=False, is_downbeat=False)
        )
        cmd_map = {c.fixture_id: c for c in cmds}
        for pid in _par_ids():
            assert cmd_map[pid].red == 0
        for sid in _strobe_ids():
            assert cmd_map[sid].strobe_rate == 0

    def test_drop_uv_off(self) -> None:
        p = _profile()
        cmds = p.generate(_state(segment="drop", energy=0.9, is_downbeat=True))
        cmd_map = {c.fixture_id: c for c in cmds}
        for uid in _uv_ids():
            assert cmd_map[uid].special == 0


class TestBreakdownBreathe:
    """Breakdown segment → slow red breathing, no strobes."""

    def test_breakdown_no_strobes(self) -> None:
        p = _profile()
        cmds = p.generate(_state(segment="breakdown", bar_phase=0.5))
        cmd_map = {c.fixture_id: c for c in cmds}
        for sid in _strobe_ids():
            assert cmd_map[sid].strobe_rate == 0

    def test_breakdown_pars_nonzero(self) -> None:
        p = _profile()
        # At bar_phase=0.75, sine should be above minimum
        cmds = p.generate(_state(segment="breakdown", bar_phase=0.75))
        cmd_map = {c.fixture_id: c for c in cmds}
        any_red = any(cmd_map[pid].red > 0 for pid in _par_ids())
        assert any_red, "Pars should have some red during breakdown"

    def test_breakdown_red_only(self) -> None:
        p = _profile()
        cmds = p.generate(_state(segment="breakdown", bar_phase=0.5))
        cmd_map = {c.fixture_id: c for c in cmds}
        for pid in _par_ids():
            assert cmd_map[pid].green == 0
            assert cmd_map[pid].blue == 0


class TestVocalCalm:
    """High vocal energy in verse → intimate corner isolation."""

    def test_vocal_calm_most_pars_dark(self) -> None:
        p = _profile()
        cmds = p.generate(_state(vocal_energy=0.8, segment="verse"))
        cmd_map = {c.fixture_id: c for c in cmds}
        lit = [pid for pid in _par_ids() if cmd_map[pid].red > 0]
        dark = [pid for pid in _par_ids() if cmd_map[pid].red == 0]
        assert len(lit) <= 2, "At most 2 pars should be lit during vocal calm"
        assert len(dark) >= 2, "At least 2 pars should be dark"

    def test_vocal_calm_no_strobes(self) -> None:
        p = _profile()
        cmds = p.generate(_state(vocal_energy=0.8, segment="verse"))
        cmd_map = {c.fixture_id: c for c in cmds}
        for sid in _strobe_ids():
            assert cmd_map[sid].strobe_rate == 0


class TestVerseReactive:
    """Standard verse with beat-reactive red."""

    def test_kick_all_pars_red(self) -> None:
        p = _profile()
        cmds = p.generate(
            _state(segment="verse", onset_type="kick", is_beat=True)
        )
        cmd_map = {c.fixture_id: c for c in cmds}
        for pid in _par_ids():
            assert cmd_map[pid].red == 255

    def test_between_beats_dark(self) -> None:
        p = _profile()
        cmds = p.generate(
            _state(segment="verse", is_beat=False, onset_type=None)
        )
        cmd_map = {c.fixture_id: c for c in cmds}
        for pid in _par_ids():
            assert cmd_map[pid].red == 0

    def test_verse_uv_on(self) -> None:
        p = _profile()
        cmds = p.generate(_state(segment="verse", is_beat=False, onset_type=None))
        cmd_map = {c.fixture_id: c for c in cmds}
        for uid in _uv_ids():
            assert cmd_map[uid].special > 0


class TestAdlibScatter:
    """Clap onset → single random par flash."""

    def test_adlib_single_par_lit(self) -> None:
        p = _profile()
        cmds = p.generate(
            _state(segment="verse", onset_type="clap", timestamp=1.5)
        )
        cmd_map = {c.fixture_id: c for c in cmds}
        lit = [pid for pid in _par_ids() if cmd_map[pid].red > 0 or cmd_map[pid].white > 0]
        assert len(lit) == 1, f"Expected 1 lit par for ad-lib scatter, got {len(lit)}"

    def test_adlib_deterministic(self) -> None:
        p = _profile()
        s = _state(segment="verse", onset_type="clap", timestamp=1.5)
        cmds1 = p.generate(s)
        cmds2 = p.generate(s)
        for c1, c2 in zip(cmds1, cmds2, strict=True):
            assert c1.fixture_id == c2.fixture_id
            assert c1.red == c2.red
            assert c1.white == c2.white


class TestRedWhiteOnly:
    """Rage trap uses ONLY red and white — no other colors."""

    def test_no_green_or_blue_during_drop(self) -> None:
        p = _profile()
        for is_beat in (True, False):
            cmds = p.generate(
                _state(
                    segment="drop",
                    energy=0.9,
                    is_beat=is_beat,
                    is_downbeat=is_beat,
                    onset_type="kick" if is_beat else None,
                )
            )
            for c in cmds:
                # Strobes get white (RGB=255), pars get red (G=0, B=0)
                if c.fixture_id in _strobe_ids():
                    # Strobe white is OK (R=G=B=255)
                    pass
                elif c.fixture_id in _par_ids():
                    assert c.green == 0, f"Fixture {c.fixture_id} has green={c.green}"
                    assert c.blue == 0, f"Fixture {c.fixture_id} has blue={c.blue}"


class TestIntroOutro:
    """Intro/outro → minimal single par."""

    def test_intro_minimal(self) -> None:
        p = _profile()
        cmds = p.generate(_state(segment="intro"))
        cmd_map = {c.fixture_id: c for c in cmds}
        lit = [pid for pid in _par_ids() if cmd_map[pid].red > 0]
        assert len(lit) <= 1
