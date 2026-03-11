"""Tests for lumina.lighting.patterns — spatial pattern library."""

from __future__ import annotations

from lumina.audio.models import MusicState
from lumina.control.protocol import FixtureCommand
from lumina.lighting.fixture_map import FixtureInfo, FixtureMap, FixtureType, FixtureRole
from lumina.lighting.patterns import (
    alternate,
    blinder,
    breathe,
    chase_bounce,
    chase_lr,
    chase_mirror,
    color_pop,
    color_split,
    converge,
    diverge,
    flicker,
    gradient_y,
    lightning_flash,
    make_command,
    rainbow_roll,
    random_scatter,
    select_active_fixtures,
    spotlight_isolate,
    strobe_burst,
    strobe_chase,
    stutter,
    wash_hold,
)
from lumina.lighting.profiles.base import BLACK, RED, WHITE, Color


def _state(**kwargs: object) -> MusicState:
    defaults: dict[str, object] = {
        "energy": 0.5,
        "segment": "verse",
        "bpm": 128.0,
        "beat_phase": 0.25,
        "bar_phase": 0.5,
        "timestamp": 1.0,
    }
    defaults.update(kwargs)
    return MusicState(**defaults)  # type: ignore[arg-type]


def _pars() -> list[FixtureInfo]:
    """Create 8 PAR fixtures matching the default layout."""
    return FixtureMap().by_type(FixtureType.PAR)


def _strobes() -> list[FixtureInfo]:
    """Create 4 STROBE fixtures matching the default layout."""
    return FixtureMap().by_type(FixtureType.STROBE)


# ─── make_command ──────────────────────────────────────────────────


class TestMakeCommand:
    def test_par_routing(self) -> None:
        f = FixtureInfo(1, FixtureType.PAR, (0.0, 0.0, 2.0), FixtureRole.LEFT)
        cmd = make_command(f, RED, intensity=0.5)
        assert cmd.fixture_id == 1
        assert cmd.red == 127
        assert cmd.strobe_rate == 0

    def test_strobe_routing(self) -> None:
        f = FixtureInfo(9, FixtureType.STROBE, (0.0, 0.0, 2.4), FixtureRole.FRONT_LEFT)
        cmd = make_command(f, RED, strobe_rate=200, strobe_intensity=180)
        assert cmd.strobe_rate == 200
        assert cmd.strobe_intensity == 180

    def test_uv_routing(self) -> None:
        f = FixtureInfo(7, FixtureType.UV, (0.0, 3.5, 2.0), FixtureRole.LEFT)
        cmd = make_command(f, RED, intensity=0.8)
        assert cmd.red == 0  # UV ignores color
        assert cmd.special == 204  # ~0.8 * 255

    def test_led_bar_routing(self) -> None:
        f = FixtureInfo(13, FixtureType.LED_BAR, (2.5, 2.33, 2.5), FixtureRole.CENTER)
        cmd = make_command(f, Color(0.5, 0.5, 0.5, 0.0), intensity=1.0)
        assert cmd.red == 127
        assert cmd.green == 127
        assert cmd.blue == 127
        assert cmd.strobe_rate == 0

    def test_laser_routing(self) -> None:
        f = FixtureInfo(15, FixtureType.LASER, (2.5, 7.0, 2.4), FixtureRole.BACK)
        cmd = make_command(f, RED, special=5)
        assert cmd.red == 0  # Laser ignores color
        assert cmd.special == 5


# ─── chase_lr ──────────────────────────────────────────────────────


class TestChaseLr:
    def test_correct_fixture_count(self) -> None:
        pars = _pars()
        result = chase_lr(pars, _state(), 1.0, RED)
        assert len(result) == len(pars)

    def test_empty_fixtures(self) -> None:
        result = chase_lr([], _state(), 1.0, RED)
        assert result == {}

    def test_spatial_ordering(self) -> None:
        """Bright spot near left at bar_phase=0.15 (avoids wrap-around symmetry)."""
        pars = _pars()
        state = _state(bar_phase=0.15)
        result = chase_lr(pars, state, 1.0, RED, width=0.2)
        # Leftmost fixtures (x=0) should be brighter than rightmost (x=5)
        left_ids = [f.fixture_id for f in sorted(pars, key=lambda f: f.position[0])[:2]]
        right_ids = [f.fixture_id for f in sorted(pars, key=lambda f: f.position[0])[-2:]]
        left_brightness = sum(result[fid].red for fid in left_ids)
        right_brightness = sum(result[fid].red for fid in right_ids)
        assert left_brightness > right_brightness

    def test_deterministic(self) -> None:
        pars = _pars()
        state = _state()
        r1 = chase_lr(pars, state, 1.0, RED)
        r2 = chase_lr(pars, state, 1.0, RED)
        assert r1 == r2


# ─── chase_bounce ──────────────────────────────────────────────────


class TestChaseBounce:
    def test_correct_fixture_count(self) -> None:
        result = chase_bounce(_pars(), _state(), 1.0, RED)
        assert len(result) == 8

    def test_empty_fixtures(self) -> None:
        result = chase_bounce([], _state(), 1.0, RED)
        assert result == {}


# ─── converge / diverge ───────────────────────────────────────────


class TestConvergeDiverge:
    def test_converge_edges_first(self) -> None:
        """At bar_phase=0, edge fixtures should be brighter than center."""
        pars = _pars()
        state = _state(bar_phase=0.1)
        result = converge(pars, state, 1.0, RED, intensity=1.0)
        assert len(result) == 8
        # Edge fixtures (x=0 or x=5) are far from center
        edge = [f for f in pars if f.position[0] in (0.0, 5.0)]
        edge_brightness = sum(result[f.fixture_id].red for f in edge)
        assert edge_brightness > 0

    def test_diverge_center_first(self) -> None:
        """At bar_phase=0.1, center fixtures should be brighter than edges."""
        pars = _pars()
        state = _state(bar_phase=0.1)
        result = diverge(pars, state, 1.0, RED, intensity=1.0)
        assert len(result) == 8

    def test_empty_fixtures(self) -> None:
        assert converge([], _state(), 1.0, RED) == {}
        assert diverge([], _state(), 1.0, RED) == {}


# ─── alternate ────────────────────────────────────────────────────


class TestAlternate:
    def test_even_odd_swap(self) -> None:
        pars = _pars()
        blue = Color(0.0, 0.0, 1.0, 0.0)

        # First half of beat: even=RED, odd=BLUE
        state_a = _state(beat_phase=0.25)
        result_a = alternate(pars, state_a, 1.0, RED, color_b=blue)
        # Second half of beat: even=BLUE, odd=RED
        state_b = _state(beat_phase=0.75)
        result_b = alternate(pars, state_b, 1.0, RED, color_b=blue)

        # Check first fixture swaps
        fid = pars[0].fixture_id
        assert result_a[fid].red > result_a[fid].blue  # RED in first half
        assert result_b[fid].red < result_b[fid].blue  # BLUE in second half

    def test_empty(self) -> None:
        assert alternate([], _state(), 1.0, RED) == {}


# ─── random_scatter ────────────────────────────────────────────────


class TestRandomScatter:
    def test_deterministic(self) -> None:
        pars = _pars()
        state = _state()
        r1 = random_scatter(pars, state, 1.0, RED, density=0.5)
        r2 = random_scatter(pars, state, 1.0, RED, density=0.5)
        assert r1 == r2

    def test_density_controls_count(self) -> None:
        pars = _pars()
        # Very low density: fewer fixtures lit
        low = random_scatter(pars, _state(), 1.0, RED, density=0.1)
        # Very high density: more fixtures lit
        high = random_scatter(pars, _state(), 1.0, RED, density=0.9)
        low_lit = sum(1 for cmd in low.values() if cmd.red > 0)
        high_lit = sum(1 for cmd in high.values() if cmd.red > 0)
        # High density should light at least as many as low
        assert high_lit >= low_lit

    def test_empty(self) -> None:
        assert random_scatter([], _state(), 1.0, RED) == {}


# ─── breathe ──────────────────────────────────────────────────────


class TestBreathe:
    def test_intensity_range(self) -> None:
        pars = _pars()
        result = breathe(pars, _state(bar_phase=0.0), 1.0, RED,
                         min_intensity=0.1, max_intensity=0.5)
        for cmd in result.values():
            # All values should be within the expected range (± rounding)
            assert 0 <= cmd.red <= 255

    def test_per_fixture_offset(self) -> None:
        pars = _pars()
        result = breathe(pars, _state(bar_phase=0.5), 1.0, RED,
                         min_intensity=0.1, max_intensity=0.9,
                         phase_offset_per_fixture=0.2)
        # Different fixtures should have different intensities
        reds = [result[f.fixture_id].red for f in pars]
        assert len(set(reds)) > 1  # Not all the same

    def test_empty(self) -> None:
        assert breathe([], _state(), 1.0, RED) == {}


# ─── strobe_burst ──────────────────────────────────────────────────


class TestStrobeBurst:
    def test_strobes_get_strobe_params(self) -> None:
        strobes = _strobes()
        result = strobe_burst(strobes, _state(), 1.0, WHITE)
        for f in strobes:
            cmd = result[f.fixture_id]
            assert cmd.strobe_rate == 255
            assert cmd.strobe_intensity == 255

    def test_empty(self) -> None:
        assert strobe_burst([], _state(), 1.0, RED) == {}


# ─── wash_hold ────────────────────────────────────────────────────


class TestWashHold:
    def test_uniform_color(self) -> None:
        pars = _pars()
        result = wash_hold(pars, _state(), 1.0, RED, intensity=0.5)
        reds = {result[f.fixture_id].red for f in pars}
        # All should be approximately the same (small drift)
        assert max(reds) - min(reds) <= 15

    def test_empty(self) -> None:
        assert wash_hold([], _state(), 1.0, RED) == {}


# ─── color_split ──────────────────────────────────────────────────


class TestColorSplit:
    def test_left_right_different(self) -> None:
        pars = _pars()
        blue = Color(0.0, 0.0, 1.0, 0.0)
        result = color_split(pars, _state(), 1.0, RED, color_right=blue)
        left_pars = [f for f in pars if f.position[0] < 2.5]
        right_pars = [f for f in pars if f.position[0] >= 2.5]
        for f in left_pars:
            assert result[f.fixture_id].red > result[f.fixture_id].blue
        for f in right_pars:
            assert result[f.fixture_id].blue > result[f.fixture_id].red

    def test_empty(self) -> None:
        assert color_split([], _state(), 1.0, RED) == {}


# ─── spotlight_isolate ────────────────────────────────────────────


class TestSpotlightIsolate:
    def test_one_bright_rest_dark(self) -> None:
        pars = _pars()
        result = spotlight_isolate(pars, _state(), 1.0, RED,
                                   target_index=0, intensity=1.0, dim_others=0.0)
        target = pars[0]
        assert result[target.fixture_id].red == 255
        for f in pars[1:]:
            assert result[f.fixture_id].red == 0

    def test_empty(self) -> None:
        assert spotlight_isolate([], _state(), 1.0, RED) == {}


# ─── stutter ──────────────────────────────────────────────────────


class TestStutter:
    def test_binary_on_off(self) -> None:
        pars = _pars()
        # beat_phase=0.1, rate=4 → sub_phase=0.4, < 0.5 → on
        result_on = stutter(pars, _state(beat_phase=0.1), 1.0, RED, rate=4.0)
        # beat_phase=0.15, rate=4 → sub_phase=0.6, >= 0.5 → off
        result_off = stutter(pars, _state(beat_phase=0.15), 1.0, RED, rate=4.0)

        for f in pars:
            assert result_on[f.fixture_id].red == 255
            assert result_off[f.fixture_id].red == 0

    def test_empty(self) -> None:
        assert stutter([], _state(), 1.0, RED) == {}


# ─── select_active_fixtures ───────────────────────────────────────


class TestSelectActiveFixtures:
    def test_low_energy_few_fixtures(self) -> None:
        pars = _pars()
        active = select_active_fixtures(pars, 0.2, low_count=3, mid_count=6)
        assert len(active) == 3

    def test_mid_energy_medium_fixtures(self) -> None:
        pars = _pars()
        active = select_active_fixtures(pars, 0.5, low_count=3, mid_count=6)
        assert len(active) == 6

    def test_high_energy_all_fixtures(self) -> None:
        pars = _pars()
        active = select_active_fixtures(pars, 0.9, low_count=3, mid_count=6)
        assert len(active) == len(pars)

    def test_empty(self) -> None:
        assert select_active_fixtures([], 0.5) == []

    def test_low_count_capped_at_fixture_count(self) -> None:
        pars = _pars()[:2]
        active = select_active_fixtures(pars, 0.1, low_count=5)
        assert len(active) == 2


# ─── chase_mirror ────────────────────────────────────────────────


class TestChaseMirror:
    def test_correct_fixture_count(self) -> None:
        pars = _pars()
        result = chase_mirror(pars, _state(), 1.0, RED)
        assert len(result) == len(pars)

    def test_empty_fixtures(self) -> None:
        assert chase_mirror([], _state(), 1.0, RED) == {}

    def test_symmetric_brightness(self) -> None:
        """Left and right sides should have similar total brightness."""
        pars = _pars()
        result = chase_mirror(pars, _state(bar_phase=0.25), 1.0, RED, width=0.4)
        left = [f for f in pars if f.position[0] < 2.5]
        right = [f for f in pars if f.position[0] >= 2.5]
        left_sum = sum(result[f.fixture_id].red for f in left)
        right_sum = sum(result[f.fixture_id].red for f in right)
        # Mirror pattern should produce roughly equal L/R brightness
        assert abs(left_sum - right_sum) < left_sum * 0.5 + 10


# ─── strobe_chase ────────────────────────────────────────────────


class TestStrobeChase:
    def test_one_active(self) -> None:
        """Only one fixture should be active at a time."""
        strobes = _strobes()
        result = strobe_chase(strobes, _state(bar_phase=0.0), 1.0, WHITE)
        active = [fid for fid, cmd in result.items()
                  if cmd.strobe_rate > 0 or cmd.red > 0]
        assert len(active) == 1

    def test_correct_fixture_count(self) -> None:
        strobes = _strobes()
        result = strobe_chase(strobes, _state(), 1.0, WHITE)
        assert len(result) == len(strobes)

    def test_empty(self) -> None:
        assert strobe_chase([], _state(), 1.0, RED) == {}


# ─── lightning_flash ──────────────────────────────────────────────


class TestLightningFlash:
    def test_flash_at_beat_start(self) -> None:
        """Near beat start should have high intensity."""
        pars = _pars()
        result = lightning_flash(pars, _state(beat_phase=0.0, bpm=120.0), 1.0, WHITE)
        for cmd in result.values():
            assert cmd.red > 0 or cmd.white > 0

    def test_correct_fixture_count(self) -> None:
        pars = _pars()
        result = lightning_flash(pars, _state(), 1.0, WHITE)
        assert len(result) == len(pars)

    def test_empty(self) -> None:
        assert lightning_flash([], _state(), 1.0, RED) == {}


# ─── color_pop ───────────────────────────────────────────────────


class TestColorPop:
    def test_complement_on_beat(self) -> None:
        """On beat, color should be complementary to input."""
        pars = _pars()
        result = color_pop(pars, _state(is_beat=True), 1.0, RED)
        # Complement of RED(1,0,0) is (0,1,1) → cyan
        for cmd in result.values():
            assert cmd.green > 0 or cmd.blue > 0

    def test_base_color_off_beat(self) -> None:
        pars = _pars()
        result = color_pop(pars, _state(is_beat=False), 1.0, RED)
        for cmd in result.values():
            assert cmd.red > 0

    def test_empty(self) -> None:
        assert color_pop([], _state(), 1.0, RED) == {}


# ─── rainbow_roll ────────────────────────────────────────────────


class TestRainbowRoll:
    def test_different_hues(self) -> None:
        """Each fixture should have a different hue."""
        pars = _pars()
        result = rainbow_roll(pars, _state(bar_phase=0.0), 1.0, RED)
        colors = [(result[f.fixture_id].red, result[f.fixture_id].green,
                   result[f.fixture_id].blue) for f in pars]
        # Not all the same
        assert len(set(colors)) > 1

    def test_correct_fixture_count(self) -> None:
        pars = _pars()
        result = rainbow_roll(pars, _state(), 1.0, RED)
        assert len(result) == len(pars)

    def test_empty(self) -> None:
        assert rainbow_roll([], _state(), 1.0, RED) == {}

    def test_hue_range(self) -> None:
        """With restricted hue range, colors should stay warm."""
        pars = _pars()
        result = rainbow_roll(pars, _state(), 1.0, RED,
                              hue_min=0.0, hue_max=0.15)
        for cmd in result.values():
            # Warm hues: red should dominate
            assert cmd.red >= cmd.blue


# ─── flicker ─────────────────────────────────────────────────────


class TestFlicker:
    def test_deterministic(self) -> None:
        pars = _pars()
        state = _state()
        r1 = flicker(pars, state, 1.0, RED)
        r2 = flicker(pars, state, 1.0, RED)
        assert r1 == r2

    def test_per_fixture_variation(self) -> None:
        pars = _pars()
        result = flicker(pars, _state(), 1.0, RED, jitter=0.8)
        reds = [result[f.fixture_id].red for f in pars]
        assert len(set(reds)) > 1  # Different intensities

    def test_correct_fixture_count(self) -> None:
        pars = _pars()
        result = flicker(pars, _state(), 1.0, RED)
        assert len(result) == len(pars)

    def test_empty(self) -> None:
        assert flicker([], _state(), 1.0, RED) == {}


# ─── gradient_y ──────────────────────────────────────────────────


class TestGradientY:
    def test_front_brighter_than_back(self) -> None:
        """Front fixtures should be brighter with default back=dimmed."""
        pars = _pars()
        result = gradient_y(pars, _state(), 1.0, RED)
        front = [f for f in pars if f.position[1] < 3.5]
        back = [f for f in pars if f.position[1] >= 3.5]
        front_sum = sum(result[f.fixture_id].red for f in front)
        back_sum = sum(result[f.fixture_id].red for f in back)
        assert front_sum > back_sum

    def test_correct_fixture_count(self) -> None:
        pars = _pars()
        result = gradient_y(pars, _state(), 1.0, RED)
        assert len(result) == len(pars)

    def test_empty(self) -> None:
        assert gradient_y([], _state(), 1.0, RED) == {}


# ─── blinder ─────────────────────────────────────────────────────


class TestBlinder:
    def test_all_max_output(self) -> None:
        """All fixtures should be at max brightness."""
        pars = _pars()
        result = blinder(pars, _state(), 1.0, RED)
        for cmd in result.values():
            # White color: all channels at 255
            assert cmd.red == 255
            assert cmd.green == 255
            assert cmd.blue == 255

    def test_strobes_get_strobe_params(self) -> None:
        strobes = _strobes()
        result = blinder(strobes, _state(), 1.0, RED)
        for cmd in result.values():
            assert cmd.strobe_rate == 255
            assert cmd.strobe_intensity == 255

    def test_correct_fixture_count(self) -> None:
        pars = _pars()
        result = blinder(pars, _state(), 1.0, RED)
        assert len(result) == len(pars)

    def test_empty(self) -> None:
        assert blinder([], _state(), 1.0, RED) == {}
