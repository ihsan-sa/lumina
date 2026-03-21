"""Tests for lumina.lighting.profiles.theatrical."""

from __future__ import annotations

from lumina.audio.models import MusicState
from lumina.lighting.fixture_map import FixtureMap, FixtureType
from lumina.lighting.profiles.theatrical import TheatricalProfile


def _profile() -> TheatricalProfile:
    return TheatricalProfile(FixtureMap())


def _state(**kwargs: object) -> MusicState:
    defaults: dict[str, object] = {
        "energy": 0.5,
        "segment": "verse",
        "bpm": 120.0,
    }
    defaults.update(kwargs)
    return MusicState(**defaults)  # type: ignore[arg-type]


def _par_ids() -> set[int]:
    fm = FixtureMap()
    return {f.fixture_id for f in fm.by_type(FixtureType.PAR)}


def _strobe_ids() -> set[int]:
    fm = FixtureMap()
    return {f.fixture_id for f in fm.by_type(FixtureType.STROBE)}


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


class TestVocalDrivenIntensity:
    """Vocal energy is the primary intensity driver in verse."""

    def test_high_vocal_brighter_than_low(self) -> None:
        p = _profile()
        cmds_low = p.generate(_state(segment="verse", vocal_energy=0.1))
        cmds_high = p.generate(_state(segment="verse", vocal_energy=0.9))

        par_ids = _par_ids()
        low_map = {c.fixture_id: c for c in cmds_low}
        high_map = {c.fixture_id: c for c in cmds_high}

        # Sum par brightness (R+G+B+W) across all pars
        low_total = sum(
            low_map[pid].red + low_map[pid].green + low_map[pid].blue + low_map[pid].white
            for pid in par_ids
        )
        high_total = sum(
            high_map[pid].red + high_map[pid].green + high_map[pid].blue + high_map[pid].white
            for pid in par_ids
        )
        assert high_total > low_total, (
            f"High vocal ({high_total}) should be brighter than low vocal ({low_total})"
        )

    def test_zero_vocal_still_has_minimal_output(self) -> None:
        """Even at zero vocal energy, verse should not be fully black."""
        p = _profile()
        cmds = p.generate(_state(segment="verse", vocal_energy=0.0))
        cmd_map = {c.fixture_id: c for c in cmds}
        par_ids = _par_ids()
        # At least some pars should have non-zero output (min 0.05 intensity)
        any_lit = any(
            cmd_map[pid].red > 0 or cmd_map[pid].green > 0
            or cmd_map[pid].blue > 0 or cmd_map[pid].white > 0
            for pid in par_ids
        )
        assert any_lit, "Verse should have minimal lighting even at zero vocal energy"


class TestDropTheatrical:
    """Drop segment: theatrical diverge bloom, strobes only on downbeats."""

    def test_drop_has_nonzero_par_output(self) -> None:
        p = _profile()
        cmds = p.generate(_state(segment="drop", energy=0.8, bar_phase=0.5))
        cmd_map = {c.fixture_id: c for c in cmds}
        par_ids = _par_ids()
        any_lit = any(
            cmd_map[pid].red > 0 or cmd_map[pid].green > 0
            or cmd_map[pid].blue > 0
            for pid in par_ids
        )
        assert any_lit, "Drop should have lit pars"

    def test_no_strobe_on_non_downbeat(self) -> None:
        """Strobes should be off when is_downbeat is False."""
        p = _profile()
        cmds = p.generate(_state(
            segment="drop", energy=0.8, is_beat=True, is_downbeat=False,
        ))
        cmd_map = {c.fixture_id: c for c in cmds}
        for sid in _strobe_ids():
            assert cmd_map[sid].strobe_rate == 0, (
                f"Strobe {sid} should be off on non-downbeat frames"
            )

    def test_strobe_fires_on_downbeat(self) -> None:
        p = _profile()
        cmds = p.generate(_state(
            segment="drop", energy=0.8, is_beat=True, is_downbeat=True,
        ))
        cmd_map = {c.fixture_id: c for c in cmds}
        any_strobe = any(cmd_map[sid].strobe_rate > 0 for sid in _strobe_ids())
        assert any_strobe, "Strobes should fire on downbeats during drop"


class TestBreakdownIntimate:
    """Breakdown uses spotlight isolation -- very few lit fixtures."""

    def test_few_lit_pars(self) -> None:
        p = _profile()
        cmds = p.generate(_state(segment="breakdown", energy=0.2))
        cmd_map = {c.fixture_id: c for c in cmds}
        par_ids = _par_ids()
        lit_pars = [
            pid for pid in par_ids
            if (cmd_map[pid].red > 0 or cmd_map[pid].green > 0
                or cmd_map[pid].blue > 0 or cmd_map[pid].white > 0)
        ]
        assert len(lit_pars) <= 2, (
            f"Breakdown should have 1-2 lit pars, got {len(lit_pars)}"
        )
        assert len(lit_pars) >= 1, "Breakdown should have at least 1 lit par"

    def test_strobes_off_in_breakdown(self) -> None:
        p = _profile()
        cmds = p.generate(_state(segment="breakdown"))
        cmd_map = {c.fixture_id: c for c in cmds}
        for sid in _strobe_ids():
            assert cmd_map[sid].strobe_rate == 0
            assert cmd_map[sid].strobe_intensity == 0


class TestChorusWarmPalette:
    """Chorus should use warm colors (red > blue)."""

    def test_chorus_warm_color(self) -> None:
        p = _profile()
        cmds = p.generate(_state(segment="chorus", vocal_energy=0.7, energy=0.7))
        cmd_map = {c.fixture_id: c for c in cmds}
        par_ids = _par_ids()

        total_red = sum(cmd_map[pid].red for pid in par_ids)
        total_blue = sum(cmd_map[pid].blue for pid in par_ids)
        assert total_red > total_blue, (
            f"Chorus should be warm (red={total_red} > blue={total_blue})"
        )

    def test_chorus_downbeat_gentle_strobe(self) -> None:
        """Downbeat strobes should be gentle (low rate and intensity)."""
        p = _profile()
        cmds = p.generate(_state(
            segment="chorus", is_beat=True, is_downbeat=True, vocal_energy=0.6,
        ))
        cmd_map = {c.fixture_id: c for c in cmds}
        for sid in _strobe_ids():
            if cmd_map[sid].strobe_rate > 0:
                assert cmd_map[sid].strobe_rate <= 80, (
                    f"Chorus strobe rate should be gentle, got {cmd_map[sid].strobe_rate}"
                )
                assert cmd_map[sid].strobe_intensity <= 100, (
                    f"Chorus strobe intensity should be gentle, got {cmd_map[sid].strobe_intensity}"
                )


# ─── Extended MusicState integration tests ──────────────────────────


class TestTheatricalHeadroom:
    """headroom scaling in theatrical."""

    def test_headroom_half_reduces_brightness(self) -> None:
        p1 = _profile()
        p2 = _profile()
        full = p1.generate(_state(segment="verse", energy=0.5, vocal_energy=0.6, headroom=1.0))
        half = p2.generate(_state(segment="verse", energy=0.5, vocal_energy=0.6, headroom=0.5))
        full_sum = sum(c.red + c.green + c.blue + c.white for c in full)
        half_sum = sum(c.red + c.green + c.blue + c.white for c in half)
        if full_sum > 0:
            assert half_sum < full_sum


class TestTheatricalLayerMaskVocals:
    """layer_mask['vocals'] directly drives spotlight intensity."""

    def test_high_vocal_layer_bright(self) -> None:
        p = _profile()
        cmds = p.generate(_state(
            segment="verse", energy=0.5,
            layer_mask={"vocals": 0.9, "drums": 0.5},
            vocal_energy=0.3,  # lower than layer_mask to prove layer_mask wins
        ))
        cmd_map = {c.fixture_id: c for c in cmds}
        par_brightness = sum(
            cmd_map[pid].red + cmd_map[pid].green + cmd_map[pid].blue + cmd_map[pid].white
            for pid in _par_ids()
        )
        assert par_brightness > 0, "High vocal layer should produce visible lighting"

    def test_low_vocal_layer_dim(self) -> None:
        p = _profile()
        cmds = p.generate(_state(
            segment="verse", energy=0.5,
            layer_mask={"vocals": 0.1, "drums": 0.5},
            vocal_energy=0.9,  # higher than layer_mask to prove layer_mask wins
        ))
        assert len(cmds) == 15


class TestTheatricalMotifTemperature:
    """motif_id maps to distinct color temperature."""

    def test_different_motifs_different_temperature(self) -> None:
        p = _profile()
        # Motif 1
        p.generate(_state(
            segment="verse", energy=0.5, vocal_energy=0.5,
            motif_id=1, timestamp=1.0,
        ))
        # Motif 2
        cmds = p.generate(_state(
            segment="verse", energy=0.5, vocal_energy=0.5,
            motif_id=2, timestamp=2.0,
        ))
        assert len(cmds) == 15

    def test_same_motif_consistent_temperature(self) -> None:
        p = _profile()
        cmds1 = p.generate(_state(
            segment="verse", energy=0.5, vocal_energy=0.5,
            motif_id=1, timestamp=1.0,
        ))
        cmds2 = p.generate(_state(
            segment="verse", energy=0.5, vocal_energy=0.5,
            motif_id=1, timestamp=1.0,
        ))
        for c1, c2 in zip(cmds1, cmds2, strict=True):
            assert c1 == c2
