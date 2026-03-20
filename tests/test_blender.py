"""Tests for lumina.lighting.blender — ProfileBlender and blend_commands."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from lumina.audio.models import MusicState
from lumina.control.protocol import FixtureCommand
from lumina.lighting.blender import ProfileBlender, _weighted_avg_channel, blend_commands
from lumina.lighting.fixture_map import FixtureMap
from lumina.lighting.profiles.base import BaseProfile
from lumina.lighting.profiles.psych_rnb import PsychRnbProfile
from lumina.lighting.profiles.rage_trap import RageTrapProfile


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def fixture_map() -> FixtureMap:
    """Default 15-fixture map."""
    return FixtureMap()


@pytest.fixture
def base_state() -> MusicState:
    """A basic MusicState with sensible defaults for testing."""
    return MusicState(
        timestamp=1.0,
        bpm=120.0,
        beat_phase=0.0,
        bar_phase=0.0,
        is_beat=True,
        is_downbeat=True,
        energy=0.5,
        energy_derivative=0.0,
        segment="verse",
        genre_weights={"rage_trap": 0.6, "psych_rnb": 0.4},
        vocal_energy=0.3,
        spectral_centroid=3000.0,
        sub_bass_energy=0.4,
    )


@pytest.fixture
def rage_profile(fixture_map: FixtureMap) -> RageTrapProfile:
    return RageTrapProfile(fixture_map)


@pytest.fixture
def psych_profile(fixture_map: FixtureMap) -> PsychRnbProfile:
    return PsychRnbProfile(fixture_map)


def _make_cmd(
    fixture_id: int = 1,
    red: int = 0,
    green: int = 0,
    blue: int = 0,
    white: int = 0,
    strobe_rate: int = 0,
    strobe_intensity: int = 0,
    special: int = 0,
) -> FixtureCommand:
    """Shorthand for creating a FixtureCommand with keyword defaults."""
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


# ── _weighted_avg_channel ───────────────────────────────────────────


class TestWeightedAvgChannel:
    """Tests for the _weighted_avg_channel helper."""

    def test_empty_returns_zero(self) -> None:
        assert _weighted_avg_channel([]) == 0

    def test_single_value_full_weight(self) -> None:
        assert _weighted_avg_channel([(200, 1.0)]) == 200

    def test_two_equal_weights(self) -> None:
        # (100 * 0.5) + (200 * 0.5) = 150
        result = _weighted_avg_channel([(100, 0.5), (200, 0.5)])
        assert result == 150

    def test_unequal_weights(self) -> None:
        # (255 * 0.75) + (0 * 0.25) = 191.25 -> 191
        result = _weighted_avg_channel([(255, 0.75), (0, 0.25)])
        assert result == 191

    def test_clamps_above_255(self) -> None:
        # Weights that don't sum to 1.0 can produce >255
        result = _weighted_avg_channel([(255, 1.0), (255, 1.0)])
        assert result == 255

    def test_clamps_to_zero(self) -> None:
        result = _weighted_avg_channel([(0, 1.0)])
        assert result == 0

    def test_rounds_correctly(self) -> None:
        # (100 * 0.33) + (200 * 0.67) = 33 + 134 = 167
        result = _weighted_avg_channel([(100, 0.33), (200, 0.67)])
        assert result == 167

    def test_three_sources(self) -> None:
        # (255 * 0.5) + (0 * 0.25) + (128 * 0.25) = 127.5 + 0 + 32 = 159.5 -> 160
        result = _weighted_avg_channel([(255, 0.5), (0, 0.25), (128, 0.25)])
        assert result == 160


# ── blend_commands ──────────────────────────────────────────────────


class TestBlendCommands:
    """Tests for the blend_commands function."""

    def test_empty_sources_returns_blackout(self) -> None:
        result = blend_commands([], fixture_count=3)
        assert len(result) == 3
        for i, cmd in enumerate(result, 1):
            assert cmd.fixture_id == i
            assert cmd.red == 0
            assert cmd.green == 0
            assert cmd.blue == 0
            assert cmd.white == 0

    def test_single_source_full_weight(self) -> None:
        cmds = [_make_cmd(fixture_id=1, red=200, green=100, blue=50, white=30)]
        result = blend_commands([(cmds, 1.0)], fixture_count=1)
        assert len(result) == 1
        assert result[0].red == 200
        assert result[0].green == 100
        assert result[0].blue == 50
        assert result[0].white == 30

    def test_two_sources_weighted_average_rgbw(self) -> None:
        src_a = [_make_cmd(fixture_id=1, red=255, green=0, blue=0)]
        src_b = [_make_cmd(fixture_id=1, red=0, green=0, blue=255)]
        result = blend_commands([(src_a, 0.5), (src_b, 0.5)], fixture_count=1)
        assert len(result) == 1
        # Equal weights: (255*0.5 + 0*0.5) = 128 for red, same for blue
        assert result[0].red == 128
        assert result[0].blue == 128

    def test_strobe_from_highest_weighted_source(self) -> None:
        src_a = [_make_cmd(fixture_id=1, strobe_rate=200, strobe_intensity=180)]
        src_b = [_make_cmd(fixture_id=1, strobe_rate=100, strobe_intensity=90)]
        result = blend_commands([(src_a, 0.7), (src_b, 0.3)], fixture_count=1)
        # Source A has higher weight and has strobe active
        assert result[0].strobe_rate == 200
        assert result[0].strobe_intensity == 180

    def test_strobe_skips_inactive_sources(self) -> None:
        # Source A has no strobe (rate=0), source B does
        src_a = [_make_cmd(fixture_id=1, red=255, strobe_rate=0)]
        src_b = [_make_cmd(fixture_id=1, red=0, strobe_rate=150, strobe_intensity=200)]
        result = blend_commands([(src_a, 0.8), (src_b, 0.2)], fixture_count=1)
        # Even though source B has lower weight, it's the only one with strobe
        assert result[0].strobe_rate == 150
        assert result[0].strobe_intensity == 200

    def test_no_strobe_when_all_inactive(self) -> None:
        src_a = [_make_cmd(fixture_id=1, red=255, strobe_rate=0)]
        src_b = [_make_cmd(fixture_id=1, red=128, strobe_rate=0)]
        result = blend_commands([(src_a, 0.5), (src_b, 0.5)], fixture_count=1)
        assert result[0].strobe_rate == 0
        assert result[0].strobe_intensity == 0

    def test_missing_fixture_gets_blackout(self) -> None:
        # Source only provides fixture 1 but fixture_count=2
        src = [_make_cmd(fixture_id=1, red=200)]
        result = blend_commands([(src, 1.0)], fixture_count=2)
        assert len(result) == 2
        assert result[0].red == 200  # fixture 1
        assert result[1].red == 0  # fixture 2 defaults to black

    def test_zero_weight_source_ignored(self) -> None:
        src_a = [_make_cmd(fixture_id=1, red=255)]
        src_b = [_make_cmd(fixture_id=1, red=0)]
        result = blend_commands([(src_a, 1.0), (src_b, 0.0)], fixture_count=1)
        assert result[0].red == 255

    def test_special_channel_is_weighted_average(self) -> None:
        src_a = [_make_cmd(fixture_id=1, special=200)]
        src_b = [_make_cmd(fixture_id=1, special=100)]
        result = blend_commands([(src_a, 0.5), (src_b, 0.5)], fixture_count=1)
        assert result[0].special == 150

    def test_result_sorted_by_fixture_id(self) -> None:
        src = [
            _make_cmd(fixture_id=3, red=30),
            _make_cmd(fixture_id=1, red=10),
            _make_cmd(fixture_id=2, red=20),
        ]
        result = blend_commands([(src, 1.0)], fixture_count=3)
        assert [cmd.fixture_id for cmd in result] == [1, 2, 3]

    def test_multiple_fixtures_blended_independently(self) -> None:
        src_a = [
            _make_cmd(fixture_id=1, red=255, green=0),
            _make_cmd(fixture_id=2, red=0, green=255),
        ]
        src_b = [
            _make_cmd(fixture_id=1, red=0, green=255),
            _make_cmd(fixture_id=2, red=255, green=0),
        ]
        result = blend_commands([(src_a, 0.5), (src_b, 0.5)], fixture_count=2)
        # Both fixtures should blend to 128/128
        assert result[0].red == 128
        assert result[0].green == 128
        assert result[1].red == 128
        assert result[1].green == 128


# ── ProfileBlender ──────────────────────────────────────────────────


class TestProfileBlender:
    """Tests for the ProfileBlender class."""

    def test_profile_names_sorted(
        self, rage_profile: RageTrapProfile, psych_profile: PsychRnbProfile
    ) -> None:
        blender = ProfileBlender(
            {"rage_trap": rage_profile, "psych_rnb": psych_profile}
        )
        assert blender.profile_names == ["psych_rnb", "rage_trap"]

    def test_single_active_profile_delegates(
        self,
        fixture_map: FixtureMap,
        base_state: MusicState,
    ) -> None:
        """When only one profile meets min_weight, it delegates directly."""
        mock_profile = MagicMock(spec=BaseProfile)
        mock_profile.generate.return_value = [_make_cmd(fixture_id=1, red=100)]

        blender = ProfileBlender({"test_profile": mock_profile}, min_weight=0.1)
        state = MusicState(
            genre_weights={"test_profile": 0.9},
            energy=0.5,
            segment="verse",
        )
        result = blender.generate(state)

        mock_profile.generate.assert_called_once_with(state)
        assert len(result) == 1
        assert result[0].red == 100

    def test_multiple_active_profiles_blend(
        self,
        fixture_map: FixtureMap,
    ) -> None:
        """When multiple profiles meet min_weight, results are blended."""
        mock_a = MagicMock(spec=BaseProfile)
        mock_a.generate.return_value = [_make_cmd(fixture_id=1, red=255, green=0)]

        mock_b = MagicMock(spec=BaseProfile)
        mock_b.generate.return_value = [_make_cmd(fixture_id=1, red=0, green=255)]

        blender = ProfileBlender(
            {"profile_a": mock_a, "profile_b": mock_b}, min_weight=0.1
        )
        state = MusicState(
            genre_weights={"profile_a": 0.5, "profile_b": 0.5},
            energy=0.5,
            segment="verse",
        )
        result = blender.generate(state)

        mock_a.generate.assert_called_once()
        mock_b.generate.assert_called_once()
        assert len(result) == 1
        # Equal blend of red=255 and red=0 -> 128
        assert result[0].red == 128
        assert result[0].green == 128

    def test_fallback_when_no_profile_meets_min_weight(
        self, fixture_map: FixtureMap
    ) -> None:
        """Falls back to highest-weighted profile even below threshold."""
        mock_profile = MagicMock(spec=BaseProfile)
        mock_profile.generate.return_value = [_make_cmd(fixture_id=1, red=42)]

        blender = ProfileBlender({"test": mock_profile}, min_weight=0.5)
        state = MusicState(
            genre_weights={"test": 0.2},  # below min_weight of 0.5
            energy=0.5,
            segment="verse",
        )
        result = blender.generate(state)

        mock_profile.generate.assert_called_once_with(state)
        assert result[0].red == 42

    def test_empty_genre_weights_uses_first_profile(
        self, fixture_map: FixtureMap
    ) -> None:
        """Empty genre_weights falls back to first registered profile."""
        mock_profile = MagicMock(spec=BaseProfile)
        mock_profile.name = "fallback"
        mock_profile.generate.return_value = [_make_cmd(fixture_id=1, blue=99)]

        blender = ProfileBlender({"fallback": mock_profile}, min_weight=0.1)
        state = MusicState(genre_weights={}, energy=0.5, segment="verse")
        result = blender.generate(state)

        mock_profile.generate.assert_called_once_with(state)
        assert result[0].blue == 99

    def test_no_profiles_registered_returns_empty(self) -> None:
        """With no profiles at all, returns empty list."""
        blender = ProfileBlender({}, min_weight=0.1)
        state = MusicState(genre_weights={"anything": 1.0}, energy=0.5, segment="verse")
        result = blender.generate(state)
        assert result == []

    def test_profiles_below_min_weight_not_called(
        self, fixture_map: FixtureMap
    ) -> None:
        """Profiles with weight below min_weight are not called."""
        mock_main = MagicMock(spec=BaseProfile)
        mock_main.generate.return_value = [_make_cmd(fixture_id=1, red=255)]

        mock_minor = MagicMock(spec=BaseProfile)
        mock_minor.generate.return_value = [_make_cmd(fixture_id=1, red=0)]

        blender = ProfileBlender(
            {"main": mock_main, "minor": mock_minor}, min_weight=0.2
        )
        state = MusicState(
            genre_weights={"main": 0.9, "minor": 0.05},
            energy=0.5,
            segment="verse",
        )
        result = blender.generate(state)

        # Only main should be called (single profile -> delegate)
        mock_main.generate.assert_called_once()
        mock_minor.generate.assert_not_called()

    def test_generate_with_real_profiles(
        self,
        rage_profile: RageTrapProfile,
        psych_profile: PsychRnbProfile,
        base_state: MusicState,
    ) -> None:
        """Integration test: blending real profiles produces valid commands."""
        blender = ProfileBlender(
            {"rage_trap": rage_profile, "psych_rnb": psych_profile},
            min_weight=0.1,
        )
        result = blender.generate(base_state)

        # Should produce commands (one per fixture in the default map)
        assert len(result) > 0
        for cmd in result:
            assert 0 <= cmd.red <= 255
            assert 0 <= cmd.green <= 255
            assert 0 <= cmd.blue <= 255
            assert 0 <= cmd.white <= 255
            assert 0 <= cmd.strobe_rate <= 255
            assert 0 <= cmd.strobe_intensity <= 255

    def test_fallback_picks_highest_weight_among_registered(
        self, fixture_map: FixtureMap
    ) -> None:
        """Fallback selects the highest-weight profile that is registered."""
        mock_a = MagicMock(spec=BaseProfile)
        mock_a.generate.return_value = [_make_cmd(fixture_id=1, red=10)]
        mock_b = MagicMock(spec=BaseProfile)
        mock_b.generate.return_value = [_make_cmd(fixture_id=1, red=20)]

        blender = ProfileBlender(
            {"prof_a": mock_a, "prof_b": mock_b}, min_weight=0.9
        )
        state = MusicState(
            genre_weights={"prof_a": 0.3, "prof_b": 0.5, "unknown": 0.2},
            energy=0.5,
            segment="verse",
        )
        result = blender.generate(state)

        # prof_b has higher weight among registered profiles
        mock_b.generate.assert_called_once_with(state)
        mock_a.generate.assert_not_called()
        assert result[0].red == 20

    def test_weight_normalization_in_blend(
        self, fixture_map: FixtureMap
    ) -> None:
        """Weights are normalized before blending so ratios are preserved."""
        mock_a = MagicMock(spec=BaseProfile)
        mock_a.generate.return_value = [_make_cmd(fixture_id=1, red=255)]
        mock_b = MagicMock(spec=BaseProfile)
        mock_b.generate.return_value = [_make_cmd(fixture_id=1, red=0)]

        blender = ProfileBlender(
            {"a": mock_a, "b": mock_b}, min_weight=0.1
        )
        # Weights 0.6 and 0.2 -> normalized to 0.75 and 0.25
        state = MusicState(
            genre_weights={"a": 0.6, "b": 0.2},
            energy=0.5,
            segment="verse",
        )
        result = blender.generate(state)

        # 255 * 0.75 + 0 * 0.25 = 191.25 -> 191
        assert result[0].red == 191

    def test_unknown_genre_weight_key_ignored(
        self, fixture_map: FixtureMap
    ) -> None:
        """Genre weight keys with no matching profile are gracefully ignored."""
        mock_profile = MagicMock(spec=BaseProfile)
        mock_profile.generate.return_value = [_make_cmd(fixture_id=1, red=77)]

        blender = ProfileBlender({"known": mock_profile}, min_weight=0.1)
        state = MusicState(
            genre_weights={"nonexistent": 0.8, "known": 0.2},
            energy=0.5,
            segment="verse",
        )
        result = blender.generate(state)

        # "known" is the only qualifying profile, so it delegates directly
        mock_profile.generate.assert_called_once_with(state)
        assert result[0].red == 77

    def test_all_output_channels_in_valid_range(
        self,
        rage_profile: RageTrapProfile,
        psych_profile: PsychRnbProfile,
    ) -> None:
        """All blended channels stay within 0-255 across varied inputs."""
        blender = ProfileBlender(
            {"rage_trap": rage_profile, "psych_rnb": psych_profile},
            min_weight=0.1,
        )
        for energy in (0.0, 0.3, 0.7, 1.0):
            for segment in ("verse", "chorus", "drop", "breakdown"):
                state = MusicState(
                    timestamp=2.0,
                    bpm=140.0,
                    energy=energy,
                    segment=segment,
                    genre_weights={"rage_trap": 0.5, "psych_rnb": 0.5},
                    is_beat=True,
                    is_downbeat=(segment == "drop"),
                )
                result = blender.generate(state)
                for cmd in result:
                    for attr in ("red", "green", "blue", "white",
                                 "strobe_rate", "strobe_intensity", "special"):
                        val = getattr(cmd, attr)
                        assert 0 <= val <= 255, (
                            f"{attr}={val} out of range for energy={energy}, "
                            f"segment={segment}"
                        )
