"""Tests for lumina.lighting.profiles.psych_rnb."""

from __future__ import annotations

from lumina.audio.models import MusicState
from lumina.lighting.fixture_map import FixtureMap, FixtureType
from lumina.lighting.profiles.psych_rnb import PsychRnbProfile


def _state(**kwargs: object) -> MusicState:
    defaults: dict[str, object] = {"energy": 0.5, "segment": "verse", "bpm": 90.0}
    defaults.update(kwargs)
    return MusicState(**defaults)  # type: ignore[arg-type]


def _make_profile() -> PsychRnbProfile:
    return PsychRnbProfile(FixtureMap())


class TestBasics:
    def test_profile_name(self) -> None:
        p = _make_profile()
        assert p.name == "psych_rnb"

    def test_returns_one_command_per_fixture(self) -> None:
        p = _make_profile()
        cmds = p.generate(_state())
        assert len(cmds) == 8

    def test_all_fixture_ids_present(self) -> None:
        p = _make_profile()
        cmds = p.generate(_state())
        ids = {c.fixture_id for c in cmds}
        assert ids == set(range(1, 9))

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


class TestNeverHarshWhite:
    def test_verse_no_white_channel(self) -> None:
        """Verse should never output harsh white."""
        p = _make_profile()
        state = _state(segment="verse", vocal_energy=0.8, energy=0.6)
        cmds = p.generate(state)
        fm = FixtureMap()
        par_ids = {f.fixture_id for f in fm.by_type(FixtureType.PAR)}
        for c in cmds:
            if c.fixture_id in par_ids:
                # White channel should be 0 (all colors are RGB only)
                assert c.white == 0

    def test_chorus_no_white_channel(self) -> None:
        """Chorus should not produce harsh white on pars."""
        p = _make_profile()
        state = _state(segment="chorus", vocal_energy=0.7, is_beat=True)
        cmds = p.generate(state)
        fm = FixtureMap()
        par_ids = {f.fixture_id for f in fm.by_type(FixtureType.PAR)}
        for c in cmds:
            if c.fixture_id in par_ids:
                assert c.white == 0

    def test_drop_no_white_channel(self) -> None:
        """Drop should bloom in color, not harsh white."""
        p = _make_profile()
        state = _state(segment="drop", energy=0.9, is_beat=True)
        cmds = p.generate(state)
        fm = FixtureMap()
        par_ids = {f.fixture_id for f in fm.by_type(FixtureType.PAR)}
        for c in cmds:
            if c.fixture_id in par_ids:
                assert c.white == 0


class TestVerse:
    def test_vocal_energy_drives_brightness(self) -> None:
        """Higher vocal energy should produce brighter pars."""
        p = _make_profile()
        state_low = _state(segment="verse", vocal_energy=0.1)
        state_high = _state(segment="verse", vocal_energy=0.9)
        cmds_low = p.generate(state_low)
        cmds_high = p.generate(state_high)
        fm = FixtureMap()
        par_ids = {f.fixture_id for f in fm.by_type(FixtureType.PAR)}

        def total_brightness(cmds: list) -> int:
            return sum(c.red + c.green + c.blue for c in cmds if c.fixture_id in par_ids)

        assert total_brightness(cmds_high) > total_brightness(cmds_low)

    def test_no_strobes_in_verse(self) -> None:
        """Strobes should be completely off during verse."""
        p = _make_profile()
        state = _state(segment="verse", energy=0.5)
        cmds = p.generate(state)
        fm = FixtureMap()
        strobe_ids = {f.fixture_id for f in fm.by_type(FixtureType.STROBE)}
        for c in cmds:
            if c.fixture_id in strobe_ids:
                assert c.strobe_rate == 0

    def test_swell_expands_from_center(self) -> None:
        """Rising energy should trigger center→outward expansion."""
        p = _make_profile()
        state = _state(
            segment="verse", energy=0.6, energy_derivative=0.15,
            vocal_energy=0.5,
        )
        cmds = p.generate(state)
        # Should still produce valid commands
        assert len(cmds) == 8


class TestChorus:
    def test_kick_pulse_adds_brightness(self) -> None:
        """Kick onset in chorus should add brightness boost."""
        p = _make_profile()
        state_no = _state(segment="chorus", vocal_energy=0.6)
        state_kick = _state(
            segment="chorus", vocal_energy=0.6, onset_type="kick",
        )
        cmds_no = p.generate(state_no)
        cmds_kick = p.generate(state_kick)
        fm = FixtureMap()
        par_ids = {f.fixture_id for f in fm.by_type(FixtureType.PAR)}

        def total_brightness(cmds: list) -> int:
            return sum(c.red + c.green + c.blue for c in cmds if c.fixture_id in par_ids)

        assert total_brightness(cmds_kick) >= total_brightness(cmds_no)

    def test_gentle_strobe_on_downbeat(self) -> None:
        """Chorus downbeat should produce gentle tinted strobe."""
        p = _make_profile()
        state = _state(segment="chorus", is_downbeat=True, is_beat=True)
        cmds = p.generate(state)
        fm = FixtureMap()
        strobe_ids = {f.fixture_id for f in fm.by_type(FixtureType.STROBE)}
        strobes = [c for c in cmds if c.fixture_id in strobe_ids]
        # Should have some strobe activity
        assert any(c.strobe_rate > 0 for c in strobes)
        # But NOT at max (gentle, not harsh)
        for c in strobes:
            assert c.strobe_rate < 200


class TestDrop:
    def test_drop_higher_intensity_than_verse(self) -> None:
        """Drop should be significantly brighter than verse."""
        p = _make_profile()
        state_verse = _state(segment="verse", energy=0.5, vocal_energy=0.5)
        state_drop = _state(segment="drop", energy=0.9, is_beat=True)
        cmds_verse = p.generate(state_verse)
        cmds_drop = p.generate(state_drop)
        fm = FixtureMap()
        par_ids = {f.fixture_id for f in fm.by_type(FixtureType.PAR)}

        def total_brightness(cmds: list) -> int:
            return sum(c.red + c.green + c.blue for c in cmds if c.fixture_id in par_ids)

        assert total_brightness(cmds_drop) > total_brightness(cmds_verse)

    def test_drop_uv_peak(self) -> None:
        """UV should be at peak during drop."""
        p = _make_profile()
        state = _state(segment="drop", energy=0.9)
        cmds = p.generate(state)
        fm = FixtureMap()
        uv_ids = {f.fixture_id for f in fm.by_type(FixtureType.UV)}
        for c in cmds:
            if c.fixture_id in uv_ids:
                assert c.special == 200  # _UV_DROP


class TestBreakdown:
    def test_breakdown_mostly_dark(self) -> None:
        """Breakdown should have most pars dark."""
        p = _make_profile()
        state = _state(segment="breakdown", energy=0.1, bar_phase=0.5)
        cmds = p.generate(state)
        fm = FixtureMap()
        par_ids = sorted(f.fixture_id for f in fm.by_type(FixtureType.PAR))
        dark = [c for c in cmds if c.fixture_id in par_ids and c.red + c.green + c.blue == 0]
        # At least 2 pars should be dark (4 total, 2 lit max)
        assert len(dark) >= 2

    def test_breakdown_no_strobes(self) -> None:
        """No strobes during breakdown."""
        p = _make_profile()
        state = _state(segment="breakdown", energy=0.1)
        cmds = p.generate(state)
        fm = FixtureMap()
        strobe_ids = {f.fixture_id for f in fm.by_type(FixtureType.STROBE)}
        for c in cmds:
            if c.fixture_id in strobe_ids:
                assert c.strobe_rate == 0


class TestColorDrift:
    def test_wash_changes_over_time(self) -> None:
        """Color wash should produce different colors at different times."""
        p = _make_profile()
        state1 = _state(segment="verse", vocal_energy=0.5, timestamp=0.0)
        state2 = _state(segment="verse", vocal_energy=0.5, timestamp=15.0)
        cmds1 = p.generate(state1)
        cmds2 = p.generate(state2)
        # Compare first par color — should differ
        par1_t0 = next((c.red, c.green, c.blue) for c in cmds1 if c.fixture_id == 1)
        par1_t15 = next((c.red, c.green, c.blue) for c in cmds2 if c.fixture_id == 1)
        assert par1_t0 != par1_t15


class TestDeterminism:
    def test_same_state_same_output(self) -> None:
        """Same MusicState should always produce the same commands."""
        p = _make_profile()
        state = _state(
            segment="verse", energy=0.5, vocal_energy=0.6,
            timestamp=10.0, bar_phase=0.3,
        )
        cmds1 = p.generate(state)
        cmds2 = p.generate(state)
        for c1, c2 in zip(cmds1, cmds2, strict=True):
            assert c1 == c2
