"""Tests for lumina.lighting.profiles.base."""

from __future__ import annotations

from lumina.audio.models import MusicState
from lumina.control.protocol import FixtureCommand
from lumina.lighting.fixture_map import FixtureMap, FixtureType
from lumina.lighting.profiles.base import (
    BLACK,
    RED,
    WHITE,
    BaseProfile,
    Color,
    color_from_hsv,
    ease_in_out,
    lerp_color,
    sine_pulse,
    triangle_wave,
)


# ─── Concrete test profile ──────────────────────────────────────────


class _TestProfile(BaseProfile):
    """Minimal concrete profile for testing base helpers."""

    name = "test"

    def generate(self, state: MusicState) -> list[FixtureCommand]:
        return self._blackout()


def _make_profile() -> _TestProfile:
    return _TestProfile(FixtureMap())


def _make_state(**kwargs: object) -> MusicState:
    return MusicState(**kwargs)  # type: ignore[arg-type]


# ─── Color tests ─────────────────────────────────────────────────────


class TestColor:
    def test_black_to_bytes(self) -> None:
        assert BLACK.to_bytes() == (0, 0, 0, 0)

    def test_white_to_bytes(self) -> None:
        assert WHITE.to_bytes() == (255, 255, 255, 255)

    def test_scaled(self) -> None:
        c = Color(1.0, 0.5, 0.0, 0.0).scaled(0.5)
        assert abs(c.r - 0.5) < 1e-6
        assert abs(c.g - 0.25) < 1e-6

    def test_lerp_color(self) -> None:
        c = lerp_color(BLACK, WHITE, 0.5)
        assert abs(c.r - 0.5) < 1e-6
        assert abs(c.w - 0.5) < 1e-6

    def test_lerp_clamps(self) -> None:
        c = lerp_color(BLACK, WHITE, 1.5)
        assert abs(c.r - 1.0) < 1e-6

    def test_color_from_hsv_red(self) -> None:
        c = color_from_hsv(0.0, 1.0, 1.0)
        assert abs(c.r - 1.0) < 1e-6
        assert c.g < 0.01
        assert c.b < 0.01


# ─── Intensity curves ───────────────────────────────────────────────


class TestIntensityCurves:
    def test_sine_pulse_range(self) -> None:
        for i in range(11):
            v = sine_pulse(i / 10.0)
            assert 0.0 <= v <= 1.0

    def test_triangle_wave_endpoints(self) -> None:
        assert abs(triangle_wave(0.0)) < 1e-6
        assert abs(triangle_wave(0.5) - 1.0) < 1e-6
        assert abs(triangle_wave(1.0)) < 1e-6

    def test_ease_in_out_endpoints(self) -> None:
        assert abs(ease_in_out(0.0)) < 1e-6
        assert abs(ease_in_out(1.0) - 1.0) < 1e-6
        assert 0.4 < ease_in_out(0.5) < 0.6


# ─── BaseProfile command building ───────────────────────────────────


class TestBlackout:
    def test_blackout_all_zero(self) -> None:
        p = _make_profile()
        cmds = p._blackout()
        assert len(cmds) == 15
        for c in cmds:
            assert c.red == 0
            assert c.green == 0
            assert c.blue == 0


class TestAllColor:
    def test_all_color_sets_pars(self) -> None:
        p = _make_profile()
        cmds = p._all_color(RED, intensity=1.0)
        fm = FixtureMap()
        par_ids = {f.fixture_id for f in fm.by_type(FixtureType.PAR)}
        for c in cmds:
            if c.fixture_id in par_ids:
                assert c.red == 255
                assert c.green == 0
                assert c.blue == 0


class TestChase:
    def test_chase_returns_correct_ids(self) -> None:
        p = _make_profile()
        fm = FixtureMap()
        pars = fm.by_type(FixtureType.PAR)
        result = p._chase(pars, 0.0, RED, width=0.3)
        assert set(result.keys()) == {f.fixture_id for f in pars}


class TestSweep:
    def test_sweep_x_covers_all(self) -> None:
        p = _make_profile()
        result = p._sweep_x(0.5, RED)
        assert len(result) == 15

    def test_sweep_y_covers_all(self) -> None:
        p = _make_profile()
        result = p._sweep_y(0.5, RED)
        assert len(result) == 15


class TestAlternating:
    def test_alternating_two_colors(self) -> None:
        p = _make_profile()
        fm = FixtureMap()
        pars = fm.by_type(FixtureType.PAR)
        result = p._alternating(pars, RED, BLACK, phase=0.0)
        # Even-index pars should be red
        assert result[pars[0].fixture_id].red == 255
        assert result[pars[1].fixture_id].red == 0


class TestCornerIsolation:
    def test_only_one_corner_lit(self) -> None:
        p = _make_profile()
        cmds = p._corner_isolation("front_left", RED)
        assert len(cmds) == 15
        # FRONT_LEFT role is now strobe ID 9; strobes route color to RGB
        lit = [c for c in cmds if c.red > 0]
        assert len(lit) == 1
        assert lit[0].fixture_id == 9


class TestMergeCommands:
    def test_merge_overrides(self) -> None:
        p = _make_profile()
        base = p._blackout()
        override = {1: FixtureCommand(fixture_id=1, red=255)}
        merged = p._merge_commands(override, base=base)
        assert len(merged) == 15
        assert merged[0].red == 255
        assert merged[1].red == 0


class TestStrobeOnBeat:
    def test_downbeat_max(self) -> None:
        p = _make_profile()
        state = _make_state(is_downbeat=True, is_beat=True)
        rate, intensity = p._strobe_on_beat(state)
        assert rate == 255
        assert intensity == 255

    def test_no_beat_silent(self) -> None:
        p = _make_profile()
        state = _make_state(is_downbeat=False, is_beat=False)
        rate, intensity = p._strobe_on_beat(state)
        assert rate == 0
        assert intensity == 0


# ─── Extended MusicState helpers ─────────────────────────────────────


class TestApplyHeadroom:
    def test_headroom_scales_rgb(self) -> None:
        p = _make_profile()
        p._store_headroom(_make_state(headroom=0.5))
        cmds = [FixtureCommand(fixture_id=1, red=200, green=100, blue=50, white=80)]
        result = p._apply_headroom(cmds)
        assert result[0].red == 100
        assert result[0].green == 50
        assert result[0].blue == 25
        assert result[0].white == 40

    def test_headroom_one_passthrough(self) -> None:
        p = _make_profile()
        p._store_headroom(_make_state(headroom=1.0))
        cmds = [FixtureCommand(fixture_id=1, red=200, green=100, blue=50)]
        result = p._apply_headroom(cmds)
        assert result[0].red == 200
        assert result[0].green == 100

    def test_headroom_preserves_strobe(self) -> None:
        """Strobe rate and intensity should NOT be scaled by headroom."""
        p = _make_profile()
        p._store_headroom(_make_state(headroom=0.5))
        cmds = [FixtureCommand(fixture_id=1, red=200, strobe_rate=180, strobe_intensity=220)]
        result = p._apply_headroom(cmds)
        assert result[0].strobe_rate == 180
        assert result[0].strobe_intensity == 220

    def test_headroom_preserves_special(self) -> None:
        p = _make_profile()
        p._store_headroom(_make_state(headroom=0.5))
        cmds = [FixtureCommand(fixture_id=1, red=200, special=128)]
        result = p._apply_headroom(cmds)
        assert result[0].special == 128


class TestActiveFixtureCount:
    def test_layer_count_zero_returns_all(self) -> None:
        p = _make_profile()
        state = _make_state(layer_count=0, energy=0.5)
        from lumina.lighting.fixture_map import FixtureMap, FixtureType
        fm = FixtureMap()
        total_pars = len(fm.by_type(FixtureType.PAR))
        assert p._active_fixture_count(state) == total_pars

    def test_layer_count_one_returns_few(self) -> None:
        p = _make_profile()
        state = _make_state(layer_count=1, energy=0.3)
        from lumina.lighting.fixture_map import FixtureMap, FixtureType
        fm = FixtureMap()
        total_pars = len(fm.by_type(FixtureType.PAR))
        count = p._active_fixture_count(state, total_pars)
        assert count < total_pars

    def test_layer_count_four_returns_all(self) -> None:
        p = _make_profile()
        state = _make_state(layer_count=4, energy=0.5)
        from lumina.lighting.fixture_map import FixtureMap, FixtureType
        fm = FixtureMap()
        total_pars = len(fm.by_type(FixtureType.PAR))
        count = p._active_fixture_count(state, total_pars)
        assert count == total_pars
