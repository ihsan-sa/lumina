"""Tests for lumina.lighting.transitions — TransitionEngine and easing functions.

Covers easing functions, TransitionState defaults, TransitionEngine lifecycle
(update, get_blend_factor, blend_outputs), and module-level helpers
(_lerp_channel, _pad_commands).
"""

from __future__ import annotations

import math

import pytest

from lumina.control.protocol import FixtureCommand
from lumina.lighting.transitions import (
    TransitionEngine,
    TransitionState,
    _crossfade,
    _ease_in_out,
    _lerp_channel,
    _linear,
    _pad_commands,
)


# ── Helpers ────────────────────────────────────────────────────────────


def _cmd(
    fixture_id: int = 1,
    red: int = 0,
    green: int = 0,
    blue: int = 0,
    white: int = 0,
    strobe_rate: int = 0,
    strobe_intensity: int = 0,
    special: int = 0,
) -> FixtureCommand:
    """Shorthand for creating a FixtureCommand."""
    return FixtureCommand(
        fixture_id=fixture_id,
        red=red,
        green=green,
        blue=blue,
        white=white,
        strobe_rate=strobe_rate,
        strobe_intensity=strobe_intensity,
        special=special,
    )


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture()
def engine() -> TransitionEngine:
    """Default TransitionEngine with ease_in_out curve."""
    return TransitionEngine(default_duration=2.0, curve="ease_in_out")


# ── Easing functions ──────────────────────────────────────────────────


class TestLinearEasing:
    """Tests for _linear easing function."""

    def test_at_zero(self) -> None:
        assert _linear(0.0) == 0.0

    def test_at_one(self) -> None:
        assert _linear(1.0) == 1.0

    def test_at_midpoint(self) -> None:
        assert _linear(0.5) == 0.5

    def test_at_quarter(self) -> None:
        assert _linear(0.25) == 0.25

    def test_clamps_below_zero(self) -> None:
        assert _linear(-0.5) == 0.0

    def test_clamps_above_one(self) -> None:
        assert _linear(1.5) == 1.0


class TestEaseInOutEasing:
    """Tests for _ease_in_out cubic easing function."""

    def test_at_zero(self) -> None:
        assert _ease_in_out(0.0) == 0.0

    def test_at_one(self) -> None:
        assert _ease_in_out(1.0) == pytest.approx(1.0, abs=1e-9)

    def test_at_midpoint(self) -> None:
        # Cubic ease-in-out at 0.5: 4 * 0.125 = 0.5
        assert _ease_in_out(0.5) == pytest.approx(0.5, abs=1e-9)

    def test_starts_slow(self) -> None:
        # At t=0.1, ease-in-out should be less than linear
        assert _ease_in_out(0.1) < 0.1

    def test_ends_slow(self) -> None:
        # At t=0.9, ease-in-out should be greater than linear
        assert _ease_in_out(0.9) > 0.9

    def test_symmetric(self) -> None:
        # f(t) + f(1-t) should equal 1.0 for cubic ease-in-out
        for t in (0.1, 0.2, 0.3, 0.4):
            assert _ease_in_out(t) + _ease_in_out(1.0 - t) == pytest.approx(
                1.0, abs=1e-9
            )

    def test_clamps_below_zero(self) -> None:
        assert _ease_in_out(-1.0) == 0.0

    def test_clamps_above_one(self) -> None:
        assert _ease_in_out(2.0) == pytest.approx(1.0, abs=1e-9)


class TestCrossfadeEasing:
    """Tests for _crossfade (sqrt) easing function."""

    def test_at_zero(self) -> None:
        assert _crossfade(0.0) == 0.0

    def test_at_one(self) -> None:
        assert _crossfade(1.0) == 1.0

    def test_at_midpoint(self) -> None:
        assert _crossfade(0.5) == pytest.approx(math.sqrt(0.5), abs=1e-9)

    def test_at_quarter(self) -> None:
        assert _crossfade(0.25) == pytest.approx(0.5, abs=1e-9)

    def test_above_linear_in_first_half(self) -> None:
        # sqrt curve rises faster than linear for t in (0, 1)
        for t in (0.1, 0.2, 0.3, 0.4, 0.5):
            assert _crossfade(t) > t

    def test_clamps_below_zero(self) -> None:
        assert _crossfade(-0.5) == 0.0

    def test_clamps_above_one(self) -> None:
        assert _crossfade(1.5) == 1.0


# ── TransitionState ──────────────────────────────────────────────────


class TestTransitionState:
    """Tests for the TransitionState dataclass."""

    def test_creation_with_defaults(self) -> None:
        ts = TransitionState(from_profile="rage_trap", to_profile="psych_rnb")
        assert ts.from_profile == "rage_trap"
        assert ts.to_profile == "psych_rnb"
        assert ts.progress == 0.0
        assert ts.duration == 2.0
        assert ts.start_time == 0.0
        assert ts.curve == "ease_in_out"

    def test_creation_with_custom_values(self) -> None:
        ts = TransitionState(
            from_profile="festival_edm",
            to_profile="uk_bass",
            progress=0.5,
            duration=4.0,
            start_time=10.0,
            curve="crossfade",
        )
        assert ts.from_profile == "festival_edm"
        assert ts.to_profile == "uk_bass"
        assert ts.progress == 0.5
        assert ts.duration == 4.0
        assert ts.start_time == 10.0
        assert ts.curve == "crossfade"

    def test_progress_is_mutable(self) -> None:
        ts = TransitionState(from_profile="a", to_profile="b")
        ts.progress = 0.75
        assert ts.progress == 0.75


# ── TransitionEngine.update ──────────────────────────────────────────


class TestTransitionEngineUpdate:
    """Tests for TransitionEngine.update() state machine."""

    def test_first_call_sets_profile_no_transition(
        self, engine: TransitionEngine
    ) -> None:
        """First call bootstraps _last_profile and returns None."""
        result = engine.update("rage_trap", "verse", timestamp=0.0)
        assert result is None
        assert engine.last_profile == "rage_trap"
        assert engine.active_transition is None

    def test_same_profile_no_transition(self, engine: TransitionEngine) -> None:
        """Repeated calls with the same profile produce no transition."""
        engine.update("rage_trap", "verse", timestamp=0.0)
        result = engine.update("rage_trap", "verse", timestamp=0.5)
        assert result is None
        assert engine.active_transition is None

    def test_profile_change_starts_transition(
        self, engine: TransitionEngine
    ) -> None:
        """Changing profile creates a new TransitionState."""
        engine.update("rage_trap", "verse", timestamp=0.0)
        result = engine.update("psych_rnb", "verse", timestamp=1.0)

        assert result is not None
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0
        assert engine.active_transition is not None
        assert engine.active_transition.from_profile == "rage_trap"
        assert engine.active_transition.to_profile == "psych_rnb"

    def test_transition_progresses_over_time(
        self, engine: TransitionEngine
    ) -> None:
        """Blend factor increases as time passes during a transition."""
        engine.update("rage_trap", "verse", timestamp=0.0)
        f1 = engine.update("psych_rnb", "verse", timestamp=1.0)
        f2 = engine.update("psych_rnb", "verse", timestamp=2.0)

        assert f1 is not None
        assert f2 is not None
        assert f2 > f1

    def test_transition_completes_returns_none(
        self, engine: TransitionEngine
    ) -> None:
        """After transition duration elapses, update returns None
        and commits the new profile."""
        engine.update("rage_trap", "verse", timestamp=0.0)
        engine.update("psych_rnb", "verse", timestamp=1.0)

        # default_duration=2.0, started at t=1.0, so complete at t=3.0
        result = engine.update("psych_rnb", "verse", timestamp=3.5)
        assert result is None
        assert engine.last_profile == "psych_rnb"
        assert engine.active_transition is None

    def test_transition_last_profile_updated_after_completion(
        self, engine: TransitionEngine
    ) -> None:
        """After transition completes, last_profile reflects the new profile."""
        engine.update("rage_trap", "verse", timestamp=0.0)
        engine.update("psych_rnb", "verse", timestamp=1.0)
        engine.update("psych_rnb", "verse", timestamp=10.0)  # well past duration

        assert engine.last_profile == "psych_rnb"

    def test_new_transition_interrupts_existing(
        self, engine: TransitionEngine
    ) -> None:
        """A third profile change mid-transition starts a new transition."""
        engine.update("rage_trap", "verse", timestamp=0.0)
        engine.update("psych_rnb", "verse", timestamp=1.0)

        # Mid-transition, switch to a different profile
        result = engine.update("festival_edm", "verse", timestamp=1.5)
        assert result is not None
        assert engine.active_transition is not None
        assert engine.active_transition.to_profile == "festival_edm"


# ── Segment-aware durations ──────────────────────────────────────────


class TestSegmentAwareDurations:
    """Transition durations should vary by segment type."""

    def test_drop_segment_fast_transition(self) -> None:
        engine = TransitionEngine(default_duration=2.0)
        engine.update("rage_trap", "drop", timestamp=0.0)
        engine.update("psych_rnb", "drop", timestamp=1.0)

        tr = engine.active_transition
        assert tr is not None
        assert tr.duration == pytest.approx(0.1)

    def test_breakdown_segment_slow_transition(self) -> None:
        engine = TransitionEngine(default_duration=2.0)
        engine.update("rage_trap", "breakdown", timestamp=0.0)
        engine.update("psych_rnb", "breakdown", timestamp=1.0)

        tr = engine.active_transition
        assert tr is not None
        assert tr.duration == pytest.approx(3.0)

    def test_chorus_segment_moderate_transition(self) -> None:
        engine = TransitionEngine(default_duration=2.0)
        engine.update("rage_trap", "chorus", timestamp=0.0)
        engine.update("psych_rnb", "chorus", timestamp=1.0)

        tr = engine.active_transition
        assert tr is not None
        assert tr.duration == pytest.approx(1.5)

    def test_verse_segment_uses_default_duration(self) -> None:
        engine = TransitionEngine(default_duration=2.0)
        engine.update("rage_trap", "verse", timestamp=0.0)
        engine.update("psych_rnb", "verse", timestamp=1.0)

        tr = engine.active_transition
        assert tr is not None
        assert tr.duration == pytest.approx(2.0)

    def test_unknown_segment_uses_default_duration(self) -> None:
        engine = TransitionEngine(default_duration=2.0)
        engine.update("rage_trap", "unknown_segment", timestamp=0.0)
        engine.update("psych_rnb", "unknown_segment", timestamp=1.0)

        tr = engine.active_transition
        assert tr is not None
        assert tr.duration == pytest.approx(2.0)


# ── blend_outputs ────────────────────────────────────────────────────


class TestBlendOutputs:
    """Tests for TransitionEngine.blend_outputs()."""

    def test_factor_zero_returns_from_commands(
        self, engine: TransitionEngine
    ) -> None:
        """At factor=0.0, output should be fully from_cmds."""
        from_cmds = [_cmd(fixture_id=1, red=255, green=0, blue=0)]
        to_cmds = [_cmd(fixture_id=1, red=0, green=0, blue=255)]

        result = engine.blend_outputs(from_cmds, to_cmds, factor=0.0)
        assert len(result) == 1
        assert result[0].red == 255
        assert result[0].blue == 0

    def test_factor_one_returns_to_commands(
        self, engine: TransitionEngine
    ) -> None:
        """At factor=1.0, output should be fully to_cmds."""
        from_cmds = [_cmd(fixture_id=1, red=255, green=0, blue=0)]
        to_cmds = [_cmd(fixture_id=1, red=0, green=0, blue=255)]

        result = engine.blend_outputs(from_cmds, to_cmds, factor=1.0)
        assert len(result) == 1
        assert result[0].red == 0
        assert result[0].blue == 255

    def test_factor_half_interpolates_rgbw(
        self, engine: TransitionEngine
    ) -> None:
        """At factor=0.5, RGBW channels should be midpoint values."""
        from_cmds = [_cmd(fixture_id=1, red=200, green=0, blue=0, white=100)]
        to_cmds = [_cmd(fixture_id=1, red=0, green=200, blue=0, white=0)]

        result = engine.blend_outputs(from_cmds, to_cmds, factor=0.5)
        assert result[0].red == 100
        assert result[0].green == 100
        assert result[0].blue == 0
        assert result[0].white == 50

    def test_rgbw_interpolation_at_quarter(
        self, engine: TransitionEngine
    ) -> None:
        """At factor=0.25, values lean toward from_cmds."""
        from_cmds = [_cmd(fixture_id=1, red=200, green=0)]
        to_cmds = [_cmd(fixture_id=1, red=0, green=200)]

        result = engine.blend_outputs(from_cmds, to_cmds, factor=0.25)
        # red: 200 * 0.75 + 0 * 0.25 = 150
        assert result[0].red == 150
        # green: 0 * 0.75 + 200 * 0.25 = 50
        assert result[0].green == 50

    def test_special_byte_interpolated(
        self, engine: TransitionEngine
    ) -> None:
        """Special byte should be linearly interpolated like RGBW."""
        from_cmds = [_cmd(fixture_id=1, special=200)]
        to_cmds = [_cmd(fixture_id=1, special=100)]

        result = engine.blend_outputs(from_cmds, to_cmds, factor=0.5)
        assert result[0].special == 150

    def test_strobe_hard_switch_below_midpoint(
        self, engine: TransitionEngine
    ) -> None:
        """Below factor=0.5, strobe should come from from_cmds."""
        from_cmds = [
            _cmd(fixture_id=1, strobe_rate=200, strobe_intensity=180),
        ]
        to_cmds = [
            _cmd(fixture_id=1, strobe_rate=100, strobe_intensity=90),
        ]

        result = engine.blend_outputs(from_cmds, to_cmds, factor=0.49)
        assert result[0].strobe_rate == 200
        assert result[0].strobe_intensity == 180

    def test_strobe_hard_switch_at_midpoint(
        self, engine: TransitionEngine
    ) -> None:
        """At factor=0.5, strobe switches to to_cmds."""
        from_cmds = [
            _cmd(fixture_id=1, strobe_rate=200, strobe_intensity=180),
        ]
        to_cmds = [
            _cmd(fixture_id=1, strobe_rate=100, strobe_intensity=90),
        ]

        result = engine.blend_outputs(from_cmds, to_cmds, factor=0.5)
        assert result[0].strobe_rate == 100
        assert result[0].strobe_intensity == 90

    def test_strobe_hard_switch_above_midpoint(
        self, engine: TransitionEngine
    ) -> None:
        """Above factor=0.5, strobe should come from to_cmds."""
        from_cmds = [
            _cmd(fixture_id=1, strobe_rate=200, strobe_intensity=180),
        ]
        to_cmds = [
            _cmd(fixture_id=1, strobe_rate=100, strobe_intensity=90),
        ]

        result = engine.blend_outputs(from_cmds, to_cmds, factor=0.8)
        assert result[0].strobe_rate == 100
        assert result[0].strobe_intensity == 90

    def test_unequal_length_pads_shorter_list(
        self, engine: TransitionEngine
    ) -> None:
        """When lists differ in length, shorter is padded with blackout."""
        from_cmds = [
            _cmd(fixture_id=1, red=200),
            _cmd(fixture_id=2, green=200),
        ]
        to_cmds = [
            _cmd(fixture_id=1, red=100),
        ]

        result = engine.blend_outputs(from_cmds, to_cmds, factor=0.5)
        assert len(result) == 2
        # Second fixture: from_cmd has green=200, to_cmd padded to 0
        assert result[1].green == 100  # 200 * 0.5 + 0 * 0.5

    def test_multiple_fixtures_blended_independently(
        self, engine: TransitionEngine
    ) -> None:
        """Each fixture is blended independently."""
        from_cmds = [
            _cmd(fixture_id=1, red=255),
            _cmd(fixture_id=2, blue=255),
        ]
        to_cmds = [
            _cmd(fixture_id=1, red=0),
            _cmd(fixture_id=2, blue=0),
        ]

        result = engine.blend_outputs(from_cmds, to_cmds, factor=0.5)
        assert result[0].red == 128
        assert result[1].blue == 128

    def test_factor_clamped_below_zero(
        self, engine: TransitionEngine
    ) -> None:
        """Negative factor is clamped to 0.0."""
        from_cmds = [_cmd(fixture_id=1, red=200)]
        to_cmds = [_cmd(fixture_id=1, red=50)]

        result = engine.blend_outputs(from_cmds, to_cmds, factor=-0.5)
        assert result[0].red == 200

    def test_factor_clamped_above_one(
        self, engine: TransitionEngine
    ) -> None:
        """Factor > 1.0 is clamped to 1.0."""
        from_cmds = [_cmd(fixture_id=1, red=200)]
        to_cmds = [_cmd(fixture_id=1, red=50)]

        result = engine.blend_outputs(from_cmds, to_cmds, factor=1.5)
        assert result[0].red == 50

    def test_all_channels_in_valid_range(
        self, engine: TransitionEngine
    ) -> None:
        """All output channels stay within 0-255."""
        from_cmds = [_cmd(fixture_id=1, red=255, green=255, blue=255, white=255, special=255)]
        to_cmds = [_cmd(fixture_id=1, red=0, green=0, blue=0, white=0, special=0)]

        for factor in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0):
            result = engine.blend_outputs(from_cmds, to_cmds, factor=factor)
            for cmd in result:
                for attr in ("red", "green", "blue", "white", "special",
                             "strobe_rate", "strobe_intensity"):
                    val = getattr(cmd, attr)
                    assert 0 <= val <= 255, f"{attr}={val} at factor={factor}"


# ── get_blend_factor ─────────────────────────────────────────────────


class TestGetBlendFactor:
    """Tests for TransitionEngine.get_blend_factor() read-only access."""

    def test_no_transition_returns_none(self, engine: TransitionEngine) -> None:
        assert engine.get_blend_factor(0.0) is None

    def test_returns_factor_during_transition(
        self, engine: TransitionEngine
    ) -> None:
        engine.update("rage_trap", "verse", timestamp=0.0)
        engine.update("psych_rnb", "verse", timestamp=1.0)

        factor = engine.get_blend_factor(2.0)
        assert factor is not None
        assert 0.0 <= factor <= 1.0

    def test_does_not_advance_state(self, engine: TransitionEngine) -> None:
        """get_blend_factor should not complete or clear the transition."""
        engine.update("rage_trap", "verse", timestamp=0.0)
        engine.update("psych_rnb", "verse", timestamp=1.0)

        # Read factor well past completion time
        engine.get_blend_factor(100.0)

        # Transition should still be stored (get_blend_factor is read-only)
        assert engine.active_transition is not None

    def test_returns_consistent_value_with_update(
        self, engine: TransitionEngine
    ) -> None:
        """get_blend_factor at the same timestamp should return same value
        as the most recent update."""
        engine.update("rage_trap", "verse", timestamp=0.0)
        factor_from_update = engine.update("psych_rnb", "verse", timestamp=1.0)
        factor_from_getter = engine.get_blend_factor(1.0)

        assert factor_from_update is not None
        assert factor_from_getter is not None
        assert factor_from_update == pytest.approx(factor_from_getter, abs=1e-9)


# ── Engine configuration ─────────────────────────────────────────────


class TestTransitionEngineConfig:
    """Tests for TransitionEngine initialization and configuration."""

    def test_unknown_curve_falls_back(self) -> None:
        """An unknown easing curve name should fall back to ease_in_out."""
        engine = TransitionEngine(curve="nonexistent_curve")
        # Should not raise; falls back internally
        engine.update("a", "verse", timestamp=0.0)
        engine.update("b", "verse", timestamp=1.0)
        assert engine.active_transition is not None
        assert engine.active_transition.curve == "ease_in_out"

    def test_linear_curve_engine(self) -> None:
        """Engine with linear curve should use linear easing."""
        engine = TransitionEngine(default_duration=1.0, curve="linear")
        engine.update("a", "verse", timestamp=0.0)
        engine.update("b", "verse", timestamp=1.0)

        # At t=1.5, raw = 0.5/1.0 = 0.5; linear(0.5) = 0.5
        factor = engine.get_blend_factor(1.5)
        assert factor == pytest.approx(0.5, abs=1e-9)

    def test_crossfade_curve_engine(self) -> None:
        """Engine with crossfade curve should use sqrt easing."""
        engine = TransitionEngine(default_duration=1.0, curve="crossfade")
        engine.update("a", "verse", timestamp=0.0)
        engine.update("b", "verse", timestamp=1.0)

        # At t=1.5, raw = 0.5; crossfade(0.5) = sqrt(0.5)
        factor = engine.get_blend_factor(1.5)
        assert factor == pytest.approx(math.sqrt(0.5), abs=1e-9)


# ── _lerp_channel ───────────────────────────────────────────────────


class TestLerpChannel:
    """Tests for the _lerp_channel module-level helper."""

    def test_factor_zero_returns_a(self) -> None:
        assert _lerp_channel(100, 200, 0.0, 1.0) == 100

    def test_factor_one_returns_b(self) -> None:
        assert _lerp_channel(100, 200, 1.0, 0.0) == 200

    def test_midpoint_interpolation(self) -> None:
        assert _lerp_channel(0, 100, 0.5, 0.5) == 50

    def test_quarter_interpolation(self) -> None:
        # 10 * 0.75 + 250 * 0.25 = 7.5 + 62.5 = 70
        assert _lerp_channel(10, 250, 0.25, 0.75) == 70

    def test_rounding(self) -> None:
        # 100 * 0.7 + 200 * 0.3 = 70 + 60 = 130
        assert _lerp_channel(100, 200, 0.3, 0.7) == 130

    def test_clamp_upper_bound(self) -> None:
        """Result should never exceed 255."""
        assert _lerp_channel(255, 255, 0.5, 0.5) == 255

    def test_clamp_lower_bound(self) -> None:
        """Result should never go below 0."""
        assert _lerp_channel(0, 0, 0.5, 0.5) == 0

    def test_same_values(self) -> None:
        """When a == b, result equals a regardless of factor."""
        assert _lerp_channel(128, 128, 0.3, 0.7) == 128

    def test_full_range(self) -> None:
        """Interpolating between 0 and 255 at 0.5 should give 128."""
        assert _lerp_channel(0, 255, 0.5, 0.5) == 128


# ── _pad_commands ───────────────────────────────────────────────────


class TestPadCommands:
    """Tests for the _pad_commands module-level helper."""

    def test_no_padding_needed(self) -> None:
        """If list already meets target length, return it unchanged."""
        cmds = [FixtureCommand(fixture_id=1), FixtureCommand(fixture_id=2)]
        result = _pad_commands(cmds, 2)
        assert len(result) == 2
        assert result[0].fixture_id == 1
        assert result[1].fixture_id == 2

    def test_longer_than_target_returns_original(self) -> None:
        """If list is already longer than target, return as-is."""
        cmds = [FixtureCommand(fixture_id=i) for i in range(5)]
        result = _pad_commands(cmds, 3)
        assert len(result) == 5

    def test_padding_adds_blackout_commands(self) -> None:
        """Padded entries should be all-zero FixtureCommands."""
        cmds = [FixtureCommand(fixture_id=1, red=255)]
        result = _pad_commands(cmds, 4)
        assert len(result) == 4
        # Original preserved.
        assert result[0].fixture_id == 1
        assert result[0].red == 255
        # Padded entries are blackout (all zeros).
        for padded in result[1:]:
            assert padded.fixture_id == 0
            assert padded.red == 0
            assert padded.green == 0
            assert padded.blue == 0
            assert padded.white == 0
            assert padded.strobe_rate == 0
            assert padded.strobe_intensity == 0
            assert padded.special == 0

    def test_empty_list_padded(self) -> None:
        """An empty list padded to target_len should be all blackout."""
        result = _pad_commands([], 3)
        assert len(result) == 3
        assert all(cmd.fixture_id == 0 for cmd in result)

    def test_target_zero_with_empty_list(self) -> None:
        result = _pad_commands([], 0)
        assert len(result) == 0

    def test_does_not_mutate_original(self) -> None:
        """Padding should not modify the original list."""
        cmds = [FixtureCommand(fixture_id=1)]
        original_len = len(cmds)
        _pad_commands(cmds, 5)
        assert len(cmds) == original_len


# ── blend_outputs edge cases ────────────────────────────────────────


class TestBlendOutputsEdgeCases:
    """Additional edge case tests for blend_outputs."""

    def test_empty_inputs(self, engine: TransitionEngine) -> None:
        """Both lists empty should produce an empty result."""
        result = engine.blend_outputs([], [], 0.5)
        assert result == []

    def test_uses_incoming_fixture_id(self, engine: TransitionEngine) -> None:
        """Blended commands should use the incoming (to) fixture_id."""
        from_cmds = [_cmd(fixture_id=1, red=100)]
        to_cmds = [_cmd(fixture_id=5, red=100)]
        result = engine.blend_outputs(from_cmds, to_cmds, 0.3)
        assert result[0].fixture_id == 5

    def test_from_shorter_pads_correctly(self, engine: TransitionEngine) -> None:
        """When from_cmds is shorter, extra to_cmds blend against blackout."""
        from_cmds = [_cmd(fixture_id=1, red=100)]
        to_cmds = [_cmd(fixture_id=1, red=200), _cmd(fixture_id=2, green=80)]
        result = engine.blend_outputs(from_cmds, to_cmds, 1.0)
        assert len(result) == 2
        assert result[0].red == 200
        assert result[1].green == 80

    def test_to_shorter_pads_correctly(self, engine: TransitionEngine) -> None:
        """When to_cmds is shorter, extra from_cmds blend against blackout."""
        from_cmds = [_cmd(fixture_id=1, red=100), _cmd(fixture_id=2, blue=200)]
        to_cmds = [_cmd(fixture_id=1, red=0)]
        result = engine.blend_outputs(from_cmds, to_cmds, 0.0)
        assert len(result) == 2
        assert result[0].red == 100
        assert result[1].blue == 200
