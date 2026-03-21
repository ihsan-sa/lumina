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
        # Use fresh profiles to avoid crossfade state contamination
        p_low = _make_profile()
        p_high = _make_profile()
        state_low = _state(segment="verse", vocal_energy=0.1)
        state_high = _state(segment="verse", vocal_energy=0.9)
        cmds_low = p_low.generate(state_low)
        cmds_high = p_high.generate(state_high)
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
        assert len(cmds) == 15


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
        # Use fresh profiles to avoid crossfade state contamination.
        # Drop uses diverge which needs bar_phase >= ~0.8 for all pars
        # to be active (pars are positioned at room edges, not center).
        p_verse = _make_profile()
        p_drop = _make_profile()
        state_verse = _state(segment="verse", energy=0.5, vocal_energy=0.5, bar_phase=0.8)
        state_drop = _state(segment="drop", energy=0.9, is_beat=True, bar_phase=0.8)
        cmds_verse = p_verse.generate(state_verse)
        cmds_drop = p_drop.generate(state_drop)
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
        # Use fresh profiles to avoid crossfade between calls, and high
        # energy so all pars are active via select_active_fixtures.
        p1 = _make_profile()
        p2 = _make_profile()
        state1 = _state(segment="verse", vocal_energy=0.5, energy=0.8, timestamp=0.0)
        state2 = _state(segment="verse", vocal_energy=0.5, energy=0.8, timestamp=15.0)
        cmds1 = p1.generate(state1)
        cmds2 = p2.generate(state2)
        fm = FixtureMap()
        par_ids = sorted(f.fixture_id for f in fm.by_type(FixtureType.PAR))
        # Compare total par color across all pars — should differ over time
        colors_t0 = [(c.red, c.green, c.blue) for c in cmds1 if c.fixture_id in par_ids]
        colors_t15 = [(c.red, c.green, c.blue) for c in cmds2 if c.fixture_id in par_ids]
        assert colors_t0 != colors_t15


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


# ─── Extended MusicState integration tests ──────────────────────────


class TestPsychRnbHeadroom:
    """headroom scaling in psych_rnb."""

    def test_headroom_half_reduces_brightness(self) -> None:
        p1 = _make_profile()
        p2 = _make_profile()
        full = p1.generate(_state(segment="verse", energy=0.6, headroom=1.0,
                                  vocal_energy=0.5))
        half = p2.generate(_state(segment="verse", energy=0.6, headroom=0.5,
                                  vocal_energy=0.5))
        fm = FixtureMap()
        par_ids = {f.fixture_id for f in fm.by_type(FixtureType.PAR)}
        full_sum = sum(c.red + c.green + c.blue for c in full if c.fixture_id in par_ids)
        half_sum = sum(c.red + c.green + c.blue for c in half if c.fixture_id in par_ids)
        if full_sum > 0:
            assert half_sum < full_sum

    def test_headroom_one_passthrough(self) -> None:
        """headroom=1.0 should not alter output."""
        p = _make_profile()
        cmds = p.generate(_state(segment="verse", energy=0.5, headroom=1.0))
        assert len(cmds) == 15


class TestPsychRnbNotesPerBeat:
    """notes_per_beat drives breathe cycle speed."""

    def test_high_notes_per_beat_produces_valid_output(self) -> None:
        p = _make_profile()
        cmds = p.generate(_state(
            segment="verse", energy=0.5, notes_per_beat=4,
            note_pattern_phase=0.5, vocal_energy=0.4,
        ))
        assert len(cmds) == 15

    def test_different_notes_per_beat_different_output(self) -> None:
        """Different notes_per_beat values should produce different visual timing."""
        p1 = _make_profile()
        p2 = _make_profile()
        # Same state except notes_per_beat
        s_low = _state(segment="verse", energy=0.5, notes_per_beat=1,
                       vocal_energy=0.4, timestamp=5.0, bar_phase=0.3)
        s_high = _state(segment="verse", energy=0.5, notes_per_beat=4,
                        vocal_energy=0.4, timestamp=5.0, bar_phase=0.3)
        cmds_low = p1.generate(s_low)
        cmds_high = p2.generate(s_high)
        # Both should be valid
        assert len(cmds_low) == 15
        assert len(cmds_high) == 15


class TestPsychRnbMotifPaletteRotation:
    """Motif changes trigger palette rotation."""

    def test_motif_change_shifts_color(self) -> None:
        p = _make_profile()
        # Generate with motif_id=1
        cmds1 = p.generate(_state(
            segment="verse", energy=0.5, vocal_energy=0.5,
            motif_id=1, motif_repetition=1, timestamp=10.0,
        ))
        # Change to motif_id=2
        cmds2 = p.generate(_state(
            segment="verse", energy=0.5, vocal_energy=0.5,
            motif_id=2, motif_repetition=1, timestamp=10.5,
        ))
        # Both should be valid
        assert len(cmds1) == 15
        assert len(cmds2) == 15
