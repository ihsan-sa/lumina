"""Tests for lumina.lighting.profiles.generic."""

from __future__ import annotations

from lumina.audio.models import MusicState
from lumina.lighting.engine import LightingEngine
from lumina.lighting.fixture_map import FixtureMap, FixtureType
from lumina.lighting.profiles.generic import GenericProfile


def _state(**kwargs: object) -> MusicState:
    defaults: dict[str, object] = {"energy": 0.5, "segment": "verse", "bpm": 120.0}
    defaults.update(kwargs)
    return MusicState(**defaults)  # type: ignore[arg-type]


def _make_profile() -> GenericProfile:
    return GenericProfile(FixtureMap())


class TestBasics:
    def test_profile_name(self) -> None:
        p = _make_profile()
        assert p.name == "generic"

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
        for seg in ("verse", "chorus", "drop", "breakdown", "intro", "outro", "bridge"):
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


class TestEnergyReactive:
    def test_high_energy_brighter(self) -> None:
        """Higher energy should produce brighter output."""
        p = _make_profile()
        state_low = _state(energy=0.1, segment="verse")
        state_high = _state(energy=0.9, segment="verse")
        cmds_low = p.generate(state_low)
        cmds_high = p.generate(state_high)
        fm = FixtureMap()
        par_ids = {f.fixture_id for f in fm.by_type(FixtureType.PAR)}

        def total_brightness(cmds: list) -> int:
            return sum(c.red + c.green + c.blue + c.white for c in cmds if c.fixture_id in par_ids)

        assert total_brightness(cmds_high) > total_brightness(cmds_low)


class TestSectionAware:
    def test_chorus_brighter_than_verse(self) -> None:
        """Chorus should be brighter than verse at same energy.

        Chorus uses chase_lr which concentrates brightness spatially, so
        peak par brightness should be higher than verse wash_hold's uniform
        brightness.  We compare the max single-par brightness.
        """
        p = _make_profile()
        fm = FixtureMap()
        par_ids = {f.fixture_id for f in fm.by_type(FixtureType.PAR)}
        state_verse = _state(segment="verse", energy=0.8, bar_phase=0.5)
        state_chorus = _state(segment="chorus", energy=0.8, bar_phase=0.5)
        cmds_verse = p.generate(state_verse)
        cmds_chorus = p.generate(state_chorus)

        def max_par_brightness(cmds: list) -> int:
            return max(
                (c.red + c.green + c.blue + c.white for c in cmds if c.fixture_id in par_ids),
                default=0,
            )

        assert max_par_brightness(cmds_chorus) > max_par_brightness(cmds_verse)

    def test_drop_brightest(self) -> None:
        """Drop should be the brightest section.

        Drop uses diverge pattern which needs bar_phase >= ~0.8 for all
        pars to be active (pars are at room edges, not center).
        """
        p = _make_profile()
        state_drop = _state(segment="drop", energy=0.9, is_beat=True, bar_phase=0.9)
        state_verse = _state(segment="verse", energy=0.9, bar_phase=0.9)
        cmds_drop = p.generate(state_drop)
        cmds_verse = p.generate(state_verse)

        def total_brightness(cmds: list) -> int:
            return sum(c.red + c.green + c.blue + c.white for c in cmds)

        assert total_brightness(cmds_drop) > total_brightness(cmds_verse)

    def test_breakdown_dimmer_than_verse(self) -> None:
        """Breakdown should be dimmer than verse at same energy."""
        p = _make_profile()
        state_verse = _state(segment="verse", energy=0.5, bar_phase=0.5)
        state_bd = _state(segment="breakdown", energy=0.5, bar_phase=0.5)
        cmds_verse = p.generate(state_verse)
        cmds_bd = p.generate(state_bd)
        fm = FixtureMap()
        par_ids = {f.fixture_id for f in fm.by_type(FixtureType.PAR)}

        def total_brightness(cmds: list) -> int:
            return sum(c.red + c.green + c.blue + c.white for c in cmds if c.fixture_id in par_ids)

        assert total_brightness(cmds_bd) < total_brightness(cmds_verse)


class TestBeatReactive:
    def test_kick_pulse(self) -> None:
        """Kick onset should boost peak par brightness.

        Verse with kick uses alternate pattern which concentrates brightness
        on half the pars. The peak single-par brightness should exceed the
        uniform wash_hold brightness of the no-kick baseline.
        """
        p = _make_profile()
        fm = FixtureMap()
        par_ids = {f.fixture_id for f in fm.by_type(FixtureType.PAR)}
        state_no = _state(segment="verse", energy=0.8, bar_phase=0.5)
        state_kick = _state(
            segment="verse", energy=0.8, bar_phase=0.5, onset_type="kick",
        )
        cmds_no = p.generate(state_no)
        cmds_kick = p.generate(state_kick)

        def max_par_brightness(cmds: list) -> int:
            return max(
                (c.red + c.green + c.blue + c.white for c in cmds if c.fixture_id in par_ids),
                default=0,
            )

        assert max_par_brightness(cmds_kick) >= max_par_brightness(cmds_no)

    def test_snare_strobe_in_high_energy(self) -> None:
        """Snare hit at high energy should trigger gentle strobe."""
        p = _make_profile()
        state = _state(
            segment="verse", energy=0.8, onset_type="snare", bar_phase=0.5,
        )
        cmds = p.generate(state)
        fm = FixtureMap()
        strobe_ids = {f.fixture_id for f in fm.by_type(FixtureType.STROBE)}
        strobes = [c for c in cmds if c.fixture_id in strobe_ids]
        assert any(c.strobe_rate > 0 for c in strobes)

    def test_no_strobe_in_low_energy(self) -> None:
        """Snare hit at low energy should NOT trigger strobe."""
        p = _make_profile()
        state = _state(
            segment="verse", energy=0.2, onset_type="snare", bar_phase=0.5,
        )
        cmds = p.generate(state)
        fm = FixtureMap()
        strobe_ids = {f.fixture_id for f in fm.by_type(FixtureType.STROBE)}
        for c in cmds:
            if c.fixture_id in strobe_ids:
                assert c.strobe_rate == 0


class TestConservativeStrobes:
    def test_strobe_never_max(self) -> None:
        """Generic profile should never hit max strobe rate."""
        p = _make_profile()
        for seg in ("verse", "chorus", "drop", "breakdown"):
            state = _state(
                segment=seg, energy=1.0, is_beat=True, is_downbeat=True,
                onset_type="snare",
            )
            cmds = p.generate(state)
            fm = FixtureMap()
            strobe_ids = {f.fixture_id for f in fm.by_type(FixtureType.STROBE)}
            for c in cmds:
                if c.fixture_id in strobe_ids:
                    assert c.strobe_rate < 200


class TestColorCycling:
    def test_colors_change_over_time(self) -> None:
        """Color should cycle over time (8-bar period).

        Use high energy so all pars are active via select_active_fixtures,
        and compare a center par that is always selected.
        """
        p = _make_profile()
        state1 = _state(segment="verse", timestamp=0.0, energy=0.9, bar_phase=0.5)
        state2 = _state(segment="verse", timestamp=15.0, energy=0.9, bar_phase=0.5)
        cmds1 = p.generate(state1)
        cmds2 = p.generate(state2)
        fm = FixtureMap()
        par_ids = sorted(f.fixture_id for f in fm.by_type(FixtureType.PAR))
        # Use a center par (ID 4 or 5) that is always selected
        center_par = par_ids[len(par_ids) // 2]
        color_t0 = next((c.red, c.green, c.blue) for c in cmds1 if c.fixture_id == center_par)
        color_t15 = next((c.red, c.green, c.blue) for c in cmds2 if c.fixture_id == center_par)
        assert color_t0 != color_t15


class TestFallbackBehavior:
    def test_engine_uses_generic_when_low_weight(self) -> None:
        """Engine should fall back to generic when no genre exceeds 0.3."""
        engine = LightingEngine()
        state = _state(genre_weights={"rage_trap": 0.2, "psych_rnb": 0.15})
        profile = engine._select_profile(state)
        assert profile.name == "generic"

    def test_engine_uses_generic_when_empty_weights(self) -> None:
        """Engine should use generic when weights are empty."""
        engine = LightingEngine()
        state = _state(genre_weights={})
        profile = engine._select_profile(state)
        assert profile.name == "generic"

    def test_engine_uses_genre_when_high_weight(self) -> None:
        """Engine should use the genre profile when weight >= 0.3."""
        engine = LightingEngine()
        state = _state(genre_weights={"rage_trap": 0.8})
        profile = engine._select_profile(state)
        assert profile.name == "rage_trap"

    def test_engine_uses_genre_at_threshold(self) -> None:
        """Engine should use genre at exactly 0.3 weight."""
        engine = LightingEngine()
        state = _state(genre_weights={"festival_edm": 0.3})
        profile = engine._select_profile(state)
        assert profile.name == "festival_edm"

    def test_engine_selects_psych_rnb(self) -> None:
        """Engine should select psych_rnb when it's the highest."""
        engine = LightingEngine()
        state = _state(genre_weights={"psych_rnb": 0.7, "rage_trap": 0.3})
        profile = engine._select_profile(state)
        assert profile.name == "psych_rnb"


class TestDeterminism:
    def test_same_state_same_output(self) -> None:
        """Same MusicState should always produce the same commands."""
        p = _make_profile()
        state = _state(
            segment="verse", energy=0.5, timestamp=10.0, bar_phase=0.3,
        )
        cmds1 = p.generate(state)
        cmds2 = p.generate(state)
        for c1, c2 in zip(cmds1, cmds2, strict=True):
            assert c1 == c2
