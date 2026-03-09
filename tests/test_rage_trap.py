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


class TestPreDropBuild:
    """High drop_probability → accelerating strobe build-up."""

    def test_strobes_fire_when_drop_imminent(self) -> None:
        p = _profile()
        # Pre-drop uses stutter pattern: strobes are fully on or fully off
        # depending on beat subdivision. Use beat_phase=0.0 to hit the "on"
        # half of the stutter cycle. Stutter sets RGB color (not strobe_rate).
        cmds = p.generate(_state(drop_probability=0.9, segment="verse", beat_phase=0.0))
        cmd_map = {c.fixture_id: c for c in cmds}
        any_active = any(
            cmd_map[sid].strobe_rate > 0 or cmd_map[sid].red > 0
            for sid in _strobe_ids()
        )
        assert any_active, "Strobes should be active during pre-drop stutter"

    def test_pars_have_red_when_drop_imminent(self) -> None:
        p = _profile()
        cmds = p.generate(_state(drop_probability=0.9, segment="verse"))
        cmd_map = {c.fixture_id: c for c in cmds}
        any_red = any(cmd_map[pid].red > 0 for pid in _par_ids())
        assert any_red, "Pars should pulse red during pre-drop"


class TestDropExplosion:
    """Drop segment at high energy → strobes + red."""

    def test_drop_downbeat_strobes_fire(self) -> None:
        p = _profile()
        cmds = p.generate(_state(segment="drop", energy=0.9, is_downbeat=True, is_beat=True))
        cmd_map = {c.fixture_id: c for c in cmds}
        for sid in _strobe_ids():
            assert cmd_map[sid].strobe_rate > 0
            assert cmd_map[sid].strobe_intensity > 0

    def test_drop_pars_red_on_kick(self) -> None:
        p = _profile()
        cmds = p.generate(_state(segment="drop", energy=0.9, onset_type="kick", is_beat=True))
        cmd_map = {c.fixture_id: c for c in cmds}
        for pid in _par_ids():
            assert cmd_map[pid].red == 255
            assert cmd_map[pid].green == 0

    def test_drop_between_beats_dark(self) -> None:
        p = _profile()
        # beat_phase=0.7 puts us in the "off-beat" half (>= 0.5 = blackout)
        cmds = p.generate(
            _state(
                segment="drop",
                energy=0.9,
                is_beat=False,
                is_downbeat=False,
                beat_phase=0.7,
            )
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
        # With 8 pars, spotlight_isolate lights 1 par at full + others at dim_others=0.05
        # So most pars will have very dim red. Check that at most 3 are "meaningfully" lit
        lit = [pid for pid in _par_ids() if cmd_map[pid].red > 10]
        dark = [pid for pid in _par_ids() if cmd_map[pid].red <= 10]
        assert len(lit) <= 3, f"At most 3 pars should be brightly lit during vocal calm, got {len(lit)}"
        assert len(dark) >= 5, f"At least 5 pars should be dark/dim, got {len(dark)}"

    def test_vocal_calm_no_strobes(self) -> None:
        p = _profile()
        cmds = p.generate(_state(vocal_energy=0.8, segment="verse"))
        cmd_map = {c.fixture_id: c for c in cmds}
        for sid in _strobe_ids():
            assert cmd_map[sid].strobe_rate == 0


class TestVerseReactive:
    """Standard verse with beat-reactive red."""

    def test_kick_active_pars_red(self) -> None:
        p = _profile()
        # Set timestamp far enough and high energy for all pars to be active
        cmds = p.generate(_state(
            segment="verse", onset_type="kick", is_beat=True,
            timestamp=60.0, energy=0.9,
        ))
        cmd_map = {c.fixture_id: c for c in cmds}
        any_red = any(cmd_map[pid].red > 200 for pid in _par_ids())
        assert any_red, "At least one par should be strongly red on kick"

    def test_between_beats_near_dark(self) -> None:
        p = _profile()
        cmds = p.generate(_state(segment="verse", is_beat=False, onset_type=None))
        cmd_map = {c.fixture_id: c for c in cmds}
        for pid in _par_ids():
            # Near-darkness: red should be very low (0-10 range)
            assert cmd_map[pid].red <= 10, (
                f"Par {pid} should be near-dark, got red={cmd_map[pid].red}"
            )

    def test_verse_uv_on(self) -> None:
        p = _profile()
        cmds = p.generate(_state(segment="verse", is_beat=False, onset_type=None))
        cmd_map = {c.fixture_id: c for c in cmds}
        for uid in _uv_ids():
            assert cmd_map[uid].special > 0


class TestAdlibScatter:
    """Clap onset → single random par flash."""

    def test_adlib_scatter_pars_lit(self) -> None:
        p = _profile()
        # Ad-lib scatter uses random_scatter with density=0.4 on 8 pars
        # so multiple pars may light up (deterministic based on timestamp)
        cmds = p.generate(_state(segment="verse", onset_type="clap", timestamp=1.5))
        cmd_map = {c.fixture_id: c for c in cmds}
        lit = [pid for pid in _par_ids() if cmd_map[pid].red > 0 or cmd_map[pid].white > 0]
        assert 1 <= len(lit) <= 6, f"Expected 1-6 lit pars for ad-lib scatter, got {len(lit)}"

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
