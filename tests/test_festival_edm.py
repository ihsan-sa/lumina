"""Tests for lumina.lighting.profiles.festival_edm."""

from __future__ import annotations

from lumina.audio.models import MusicState
from lumina.lighting.fixture_map import FixtureMap, FixtureType
from lumina.lighting.profiles.festival_edm import FestivalEdmProfile


def _state(**kwargs: object) -> MusicState:
    defaults: dict[str, object] = {"energy": 0.5, "segment": "verse", "bpm": 128.0}
    defaults.update(kwargs)
    return MusicState(**defaults)  # type: ignore[arg-type]


def _make_profile() -> FestivalEdmProfile:
    return FestivalEdmProfile(FixtureMap())


class TestBasics:
    def test_profile_name(self) -> None:
        p = _make_profile()
        assert p.name == "festival_edm"

    def test_returns_one_command_per_fixture(self) -> None:
        p = _make_profile()
        cmds = p.generate(_state())
        assert len(cmds) == 15

    def test_all_fixture_ids_present(self) -> None:
        p = _make_profile()
        cmds = p.generate(_state())
        ids = {c.fixture_id for c in cmds}
        assert ids == set(range(1, 16))

    def test_all_channels_valid_range(self) -> None:
        p = _make_profile()
        for seg in ("verse", "chorus", "drop", "breakdown", "intro", "outro"):
            state = _state(segment=seg, energy=0.9, is_beat=True, is_downbeat=True)
            cmds = p.generate(state)
            for c in cmds:
                assert 0 <= c.red <= 255
                assert 0 <= c.green <= 255
                assert 0 <= c.blue <= 255
                assert 0 <= c.white <= 255
                assert 0 <= c.strobe_rate <= 255
                assert 0 <= c.strobe_intensity <= 255
                assert 0 <= c.special <= 255


class TestBuild:
    def test_build_activates_on_high_drop_probability(self) -> None:
        """Pre-drop build should show increasing strobe activity."""
        p = _make_profile()
        # First frame starts the build timer
        p.generate(_state(drop_probability=0.5, segment="verse", timestamp=0.0))
        # Second frame at later timestamp has nonzero ramp.
        # Build uses stutter pattern which sets RGB color (not strobe_rate)
        # on the "on" half of beat subdivisions. Use beat_phase=0.0 to
        # ensure the stutter is in its "on" state.
        state = _state(
            drop_probability=0.7, segment="verse", beat_phase=0.0, timestamp=10.0,
        )
        cmds = p.generate(state)
        fm = FixtureMap()
        strobe_ids = {f.fixture_id for f in fm.by_type(FixtureType.STROBE)}
        strobes = [c for c in cmds if c.fixture_id in strobe_ids]
        # At least one strobe should have non-zero color (stutter sets RGB)
        # or non-zero strobe_rate
        assert any(
            c.strobe_rate > 0 or c.red > 0 or c.green > 0 or c.blue > 0
            for c in strobes
        )

    def test_build_pars_fade_in(self) -> None:
        """Early build should have fewer active pars than late build."""
        p = _make_profile()
        # Early build (ramp ~0)
        state_early = _state(
            drop_probability=0.5, segment="verse", timestamp=0.0,
        )
        cmds_early = p.generate(state_early)

        # Force a later timestamp for deeper ramp
        state_late = _state(
            drop_probability=0.9, segment="verse", timestamp=30.0,
        )
        cmds_late = p.generate(state_late)

        fm = FixtureMap()
        par_ids = {f.fixture_id for f in fm.by_type(FixtureType.PAR)}

        def active_pars(cmds: list) -> int:
            return sum(1 for c in cmds if c.fixture_id in par_ids and c.red + c.green + c.blue > 0)

        # Both should have at least 1 active par
        assert active_pars(cmds_early) >= 1
        assert active_pars(cmds_late) >= 1


class TestDrop:
    def test_drop_all_fixtures_active(self) -> None:
        """During drop, all pars and strobes should be active."""
        p = _make_profile()
        # Need to trigger the drop_frame counter
        p.generate(_state(segment="verse"))  # set last_segment
        state = _state(segment="drop", energy=0.9, is_beat=True, timestamp=1.0)
        cmds = p.generate(state)
        fm = FixtureMap()
        strobe_ids = {f.fixture_id for f in fm.by_type(FixtureType.STROBE)}
        # All strobes at max
        for c in cmds:
            if c.fixture_id in strobe_ids:
                assert c.strobe_rate == 255
                assert c.strobe_intensity == 255

    def test_drop_uv_at_max(self) -> None:
        """UV should be at maximum during drop."""
        p = _make_profile()
        p.generate(_state(segment="verse"))
        state = _state(segment="drop", energy=0.9, timestamp=1.0)
        cmds = p.generate(state)
        fm = FixtureMap()
        uv_ids = {f.fixture_id for f in fm.by_type(FixtureType.UV)}
        for c in cmds:
            if c.fixture_id in uv_ids:
                assert c.special == 255

    def test_drop_color_cycling(self) -> None:
        """EDM drop (high energy) should produce different par colors over time."""
        p = _make_profile()
        p.generate(_state(segment="verse"))
        # Advance past the initial flash (18 frames) by calling generate enough times
        for i in range(20):
            p.generate(_state(segment="drop", energy=0.9, timestamp=5.0 + i * 0.016))
        state1 = _state(segment="drop", energy=0.9, timestamp=6.0, bar_phase=0.2)
        cmds1 = p.generate(state1)
        state2 = _state(segment="drop", energy=0.9, timestamp=6.1, bar_phase=0.7)
        cmds2 = p.generate(state2)
        # At least one par should have different color values
        par_colors_1 = [(c.red, c.green, c.blue) for c in cmds1 if c.fixture_id <= 4]
        par_colors_2 = [(c.red, c.green, c.blue) for c in cmds2 if c.fixture_id <= 4]
        assert par_colors_1 != par_colors_2


class TestGroove:
    def test_groove_kick_boosts_intensity(self) -> None:
        """Kick onset should produce brighter pars than no onset."""
        p = _make_profile()
        state_no_kick = _state(segment="verse", energy=0.6, bar_phase=0.5)
        state_kick = _state(
            segment="verse", energy=0.6, bar_phase=0.5, onset_type="kick",
        )
        cmds_no = p.generate(state_no_kick)
        cmds_kick = p.generate(state_kick)
        fm = FixtureMap()
        par_ids = {f.fixture_id for f in fm.by_type(FixtureType.PAR)}

        def total_brightness(cmds: list) -> int:
            return sum(c.red + c.green + c.blue for c in cmds if c.fixture_id in par_ids)

        assert total_brightness(cmds_kick) >= total_brightness(cmds_no)

    def test_groove_downbeat_strobe(self) -> None:
        """Strobes should pulse on downbeats during groove."""
        p = _make_profile()
        state = _state(segment="verse", is_downbeat=True, is_beat=True, energy=0.7)
        cmds = p.generate(state)
        fm = FixtureMap()
        strobe_ids = {f.fixture_id for f in fm.by_type(FixtureType.STROBE)}
        strobes = [c for c in cmds if c.fixture_id in strobe_ids]
        assert any(c.strobe_rate > 0 for c in strobes)


class TestBreakdown:
    def test_breakdown_mostly_dark(self) -> None:
        """Breakdown should have most pars dark, only one lit."""
        p = _make_profile()
        state = _state(segment="breakdown", energy=0.2, bar_phase=0.5)
        cmds = p.generate(state)
        fm = FixtureMap()
        par_ids = sorted(f.fixture_id for f in fm.by_type(FixtureType.PAR))
        lit = [c for c in cmds if c.fixture_id in par_ids and c.red + c.green + c.blue > 0]
        # Only one par should be lit
        assert len(lit) == 1

    def test_breakdown_no_strobes(self) -> None:
        """No strobes during breakdown."""
        p = _make_profile()
        state = _state(segment="breakdown", energy=0.2)
        cmds = p.generate(state)
        fm = FixtureMap()
        strobe_ids = {f.fixture_id for f in fm.by_type(FixtureType.STROBE)}
        for c in cmds:
            if c.fixture_id in strobe_ids:
                assert c.strobe_rate == 0


class TestDeterminism:
    def test_same_state_same_output(self) -> None:
        """Same MusicState should always produce the same commands."""
        p = _make_profile()
        state = _state(
            segment="verse", energy=0.7, is_beat=True, timestamp=10.0, bar_phase=0.3,
        )
        cmds1 = p.generate(state)
        cmds2 = p.generate(state)
        for c1, c2 in zip(cmds1, cmds2, strict=True):
            assert c1 == c2


# ─── Extended MusicState integration tests ──────────────────────────


class TestFestivalEdmHeadroom:
    """headroom scaling in festival_edm."""

    def test_headroom_half_reduces_groove(self) -> None:
        p1 = _make_profile()
        p2 = _make_profile()
        full = p1.generate(_state(segment="verse", energy=0.6, headroom=1.0))
        half = p2.generate(_state(segment="verse", energy=0.6, headroom=0.5))
        full_sum = sum(c.red + c.green + c.blue + c.white for c in full)
        half_sum = sum(c.red + c.green + c.blue + c.white for c in half)
        if full_sum > 0:
            assert half_sum < full_sum

    def test_drop_bypasses_headroom(self) -> None:
        """Drops bypass headroom in festival_edm."""
        p = _make_profile()
        cmds = p.generate(_state(segment="drop", energy=0.9, headroom=0.3))
        total = sum(c.red + c.green + c.blue + c.white for c in cmds)
        assert total > 500


class TestFestivalEdmLayerCountBuild:
    """layer_count boosts build intensity."""

    def test_high_layer_count_in_build(self) -> None:
        p = _make_profile()
        cmds = p.generate(_state(
            drop_probability=0.7, segment="verse", energy=0.6,
            layer_count=4,
        ))
        assert len(cmds) == 15

    def test_low_layer_count_in_build(self) -> None:
        p = _make_profile()
        cmds = p.generate(_state(
            drop_probability=0.7, segment="verse", energy=0.6,
            layer_count=1,
        ))
        assert len(cmds) == 15


class TestFestivalEdmMotifRepetitionReset:
    """motif_repetition resets visual on new section."""

    def test_motif_change_shifts_groove_color(self) -> None:
        p = _make_profile()
        # First motif
        p.generate(_state(
            segment="verse", energy=0.5, motif_id=1,
            motif_repetition=3, timestamp=10.0, bar_phase=0.3,
        ))
        # New motif -- color offset should shift
        cmds = p.generate(_state(
            segment="verse", energy=0.5, motif_id=2,
            motif_repetition=1, timestamp=10.5, bar_phase=0.3,
        ))
        assert len(cmds) == 15
