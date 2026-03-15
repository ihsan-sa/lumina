"""Profile 4: French Hard Rap -- Kaaris.

Core philosophy: REGIMENTED SYMMETRY, COLD PALETTE, DELIBERATE HITS.

Lighting language:
- Colors: ICE WHITE, STEEL BLUE, COLD WHITE only. No warm tones.
  Every color choice reinforces the cold, military, industrial feel.
- Contrast: Binary on/off feel. Fixtures snap between states --
  no smooth fades. Every light hit is deliberate like a punch.
- Verse: chase_mirror at strict 1 sweep/bar, steel blue.
  Binary brightness. Kick onsets trigger bump + par flash.
  LED bars at 10% steel blue.
- Chorus: alternate on pars (ice white vs black) locked to beat.
  Full strobe_burst on every kick. LED bars at 30% ice white.
  All pars active -- wall of light on hooks.
- Drop: First 6 frames blinder entry. Then chase_mirror fast
  (speed=3.0) + strobe_chase on corners. LED bars full. Laser active.
- Breakdown: Single spotlight, steel blue, no movement.
  spotlight_isolate on 1 par at 25%. Everything else off.
- Intro/outro: breathe on 2 pars, steel blue, very slow
  (period_bars=4.0), low intensity.
- Sub-bass: increases white channel (color.w += sub_bass * 0.3).
- Spectral centroid: steel blue (low) -> ice white (high) via
  _color_temperature().
"""

from __future__ import annotations

from lumina.audio.models import MusicState
from lumina.control.protocol import FixtureCommand
from lumina.lighting.fixture_map import FixtureMap, FixtureType
from lumina.lighting.patterns import (
    alternate,
    breathe,
    chase_mirror,
    make_command,
    select_active_fixtures,
    spotlight_isolate,
    strobe_burst,
    strobe_chase,
    wash_hold,
)
from lumina.lighting.profiles.base import (
    BLACK,
    BaseProfile,
    BumpTracker,
    Color,
    FixtureInfo,
)

# --- Cold palette -----------------------------------------------------------

ICE_WHITE = Color(0.85, 0.9, 1.0, 1.0)
STEEL_BLUE = Color(0.3, 0.4, 0.7, 0.2)
COLD_WHITE = Color(0.7, 0.75, 0.85, 0.8)

# Laser patterns
_LASER_OFF = 0
_LASER_DROP = 6

# Drop timing (frames at 60fps)
_DROP_BLINDER_FRAMES = 6  # 100ms at 60fps


class FrenchHardProfile(BaseProfile):
    """French Hard Rap lighting profile (Kaaris).

    Regimented symmetry: every hit is deliberate like a punch.
    Cold palette (ice white, steel blue) with strict left-right
    mirror patterns. Binary brightness -- on or off, no fades.

    Args:
        fixture_map: Venue fixture layout.
    """

    name = "french_hard"

    def __init__(self, fixture_map: FixtureMap) -> None:
        super().__init__(fixture_map)
        self._pars = self._map.by_type(FixtureType.PAR)
        self._strobes = self._map.by_type(FixtureType.STROBE)
        self._led_bars = self._map.by_type(FixtureType.LED_BAR)
        self._lasers = self._map.by_type(FixtureType.LASER)
        self._pars_lr = self._map.sorted_by_x(self._pars)

        # 40ms deliberate snap -- tighter than default, slightly softer than rage
        self._bump = BumpTracker(decay_rate=25.0)

        # State tracking
        self._segment_start_time: float = 0.0
        self._last_segment: str = ""
        self._segment_changed: bool = False
        self._drop_frame_count: int = 0

    @property
    def motif_pattern_preferences(self) -> list[str]:
        """French hard prefers symmetric, regimented patterns for motifs."""
        return ["chase_mirror", "alternate", "strobe_burst", "converge"]

    @property
    def motif_color_palette(self) -> list[Color]:
        """French hard color cycle: cold tones only."""
        return [ICE_WHITE, STEEL_BLUE, COLD_WHITE, Color(0.5, 0.55, 0.7, 0.5)]

    def _headroom_scale(self, state: MusicState, intensity: float) -> float:
        """Apply headroom to intensity, french-hard style.

        During drops, headroom is bypassed. Strobes always full.

        Args:
            state: Current music state.
            intensity: Raw intensity value.

        Returns:
            Headroom-adjusted intensity.
        """
        if state.segment == "drop":
            return intensity
        return intensity * max(0.15, state.headroom)

    def _get_active_pars(self, state: MusicState) -> list[FixtureInfo]:
        """Get active pars based on layer count or energy.

        Args:
            state: Current music state.

        Returns:
            List of active par fixtures.
        """
        if state.layer_count > 0:
            n = self._layer_fixture_count(
                state.layer_count, state.energy, len(self._pars)
            )
            return self._pars_lr[:n]
        return select_active_fixtures(self._pars, state.energy)

    def _apply_sub_bass_white(self, color: Color, sub_bass: float) -> Color:
        """Boost white channel based on sub-bass energy.

        Sub-bass presence pushes the cold palette even colder by
        increasing the white LED channel.

        Args:
            color: Base color.
            sub_bass: Sub-bass energy 0.0-1.0.

        Returns:
            Color with boosted white channel.
        """
        if sub_bass < 0.1:
            return color
        w = min(1.0, color.w + sub_bass * 0.3)
        return Color(color.r, color.g, color.b, w)

    def generate(self, state: MusicState) -> list[FixtureCommand]:
        """Generate french-hard fixture commands for the current frame.

        Decision hierarchy:
        1. Arc layer: headroom caps intensity (except drops)
        2. Note pattern: if regular notes detected, cycle fixtures
        3. Segment layer: verse/chorus/drop/breakdown routing
        4. Reactive layer: onset reactions capped by headroom

        Args:
            state: Current music analysis frame.

        Returns:
            One FixtureCommand per fixture (15 total).
        """
        self._begin_debug_frame()
        segment = state.segment

        # Track segment transitions
        if segment != self._last_segment:
            self._segment_start_time = state.timestamp
            self._segment_changed = True
            if segment == "drop":
                self._drop_frame_count = 0
            self._last_segment = segment
        else:
            self._segment_changed = False

        if segment == "drop":
            self._drop_frame_count += 1

        # --- Note-level pattern (each note = different light) ---
        if state.notes_per_beat > 0 and segment not in (
            "breakdown", "bridge", "intro", "outro"
        ):
            active_pars = self._get_active_pars(state)
            note_cmds = self._apply_note_pattern(state, active_pars, STEEL_BLUE)
            if note_cmds is not None:
                self._note_patterns("note_pattern")
                for f in self._strobes + self._led_bars:
                    note_cmds[f.fixture_id] = make_command(f, BLACK, 0.0)
                for f in self._lasers:
                    note_cmds[f.fixture_id] = make_command(f, BLACK, special=_LASER_OFF)
                return self._merge_commands(note_cmds)

        # --- Segment-based routing ---

        if segment == "drop":
            self._note_patterns("drop")
            return self._drop(state)

        if segment == "chorus":
            self._note_patterns("chorus")
            return self._chorus(state)

        if segment in ("breakdown", "bridge"):
            self._note_patterns("breakdown")
            return self._breakdown(state)

        if segment in ("intro", "outro"):
            self._note_patterns("intro_outro")
            return self._intro_outro(state)

        # Default: verse
        self._note_patterns("verse")
        return self._verse(state)

    # --- Segment handlers ---------------------------------------------------

    def _verse(self, state: MusicState) -> list[FixtureCommand]:
        """Verse: chase_mirror at 1 sweep/bar, steel blue. Binary feel.

        Kick onsets trigger bump + par flash. LED bars at 10% steel blue.
        Sub-bass boosts white channel. Spectral centroid shifts temperature.
        """
        commands: dict[int, FixtureCommand] = {}
        energy = state.energy

        # Active pars: layer-driven or energy-based
        active_pars = self._get_active_pars(state)

        # Color temperature: steel blue (low centroid) -> ice white (high)
        verse_color = self._color_temperature(
            state.spectral_centroid, STEEL_BLUE, ICE_WHITE
        )

        # Sub-bass: boost white channel
        verse_color = self._apply_sub_bass_white(verse_color, state.sub_bass_energy)

        # Onset reactions
        if state.onset_type == "kick":
            self._bump.trigger("pars", state.timestamp)
            kick_int = self._headroom_scale(state, 1.0)
            par_cmds = wash_hold(
                active_pars, state, state.timestamp, ICE_WHITE, intensity=kick_int,
            )
            commands.update(par_cmds)
            # Strobes: single deliberate flash on kick
            burst = strobe_burst(self._strobes, state, state.timestamp, COLD_WHITE)
            commands.update(burst)

        elif state.is_beat:
            # Chase mirror at strict 1 sweep/bar
            beat_int = self._headroom_scale(state, 0.80)
            par_cmds = chase_mirror(
                active_pars, state, state.timestamp, verse_color,
                speed=1.0, width=0.35, intensity=beat_int,
            )
            commands.update(par_cmds)
            for f in self._strobes:
                commands[f.fixture_id] = make_command(f, BLACK, 0.0)

        else:
            # Between beats: bump decay for professional tail
            floor = 0.0 if energy < 0.4 else 0.05
            floor = self._headroom_scale(state, floor)
            between = self._bump.get_intensity(
                "pars", state.timestamp, peak=0.80, floor=floor,
            )
            par_cmds = chase_mirror(
                active_pars, state, state.timestamp, verse_color,
                speed=1.0, width=0.35, intensity=between,
            )
            commands.update(par_cmds)
            for f in self._strobes:
                commands[f.fixture_id] = make_command(f, BLACK, 0.0)

        # Blackout inactive pars
        for f in self._pars:
            if f.fixture_id not in commands:
                commands[f.fixture_id] = make_command(f, BLACK, 0.0)

        # LED bars: 10% steel blue
        bar_int = self._headroom_scale(state, 0.10)
        bar_cmds = wash_hold(
            self._led_bars, state, state.timestamp, STEEL_BLUE, intensity=bar_int,
        )
        commands.update(bar_cmds)

        # Laser off during verse
        for f in self._lasers:
            commands[f.fixture_id] = make_command(f, BLACK, special=_LASER_OFF)

        return self._merge_commands(commands)

    def _chorus(self, state: MusicState) -> list[FixtureCommand]:
        """Chorus: alternate on pars (ice white vs black) locked to beat.

        Full strobe_burst on every kick. LED bars at 30% ice white.
        All pars active -- wall of light on hooks.
        """
        commands: dict[int, FixtureCommand] = {}

        # Chorus always uses all pars -- wall of light
        chorus_pars = self._pars

        # Color: ice white with sub-bass white boost
        chorus_color = self._apply_sub_bass_white(ICE_WHITE, state.sub_bass_energy)

        # Alternating pattern: ice white vs black, locked to beat phase
        par_cmds = alternate(
            chorus_pars, state, state.timestamp, chorus_color,
            color_b=BLACK, intensity=self._headroom_scale(state, 0.90),
        )
        commands.update(par_cmds)

        # Kick: full strobe burst on every kick
        if state.onset_type == "kick":
            self._bump.trigger("pars", state.timestamp)
            burst = strobe_burst(
                self._strobes, state, state.timestamp, COLD_WHITE,
            )
            commands.update(burst)
        else:
            for f in self._strobes:
                commands[f.fixture_id] = make_command(f, BLACK, 0.0)

        # LED bars: 30% ice white
        bar_int = self._headroom_scale(state, 0.30)
        bar_cmds = wash_hold(
            self._led_bars, state, state.timestamp, ICE_WHITE, intensity=bar_int,
        )
        commands.update(bar_cmds)

        # Laser off during chorus
        for f in self._lasers:
            commands[f.fixture_id] = make_command(f, BLACK, special=_LASER_OFF)

        return self._merge_commands(commands)

    def _drop(self, state: MusicState) -> list[FixtureCommand]:
        """Drop: blinder entry then fast chase_mirror + strobe_chase.

        First 6 frames: blinder() on all fixtures.
        After: chase_mirror fast (speed=3.0) + strobe_chase on corners.
        LED bars full. Laser active.
        """
        commands: dict[int, FixtureCommand] = {}

        # First 6 frames: BLINDER
        if self._drop_frame_count <= _DROP_BLINDER_FRAMES:
            commands.update({c.fixture_id: c for c in [
                make_command(f, ICE_WHITE, intensity=1.0)
                for f in self._map.all
                if f.fixture_type not in (FixtureType.STROBE, FixtureType.LASER)
            ]})
            # Strobes at max
            for f in self._strobes:
                commands[f.fixture_id] = make_command(
                    f, COLD_WHITE, intensity=1.0,
                    strobe_rate=255, strobe_intensity=255,
                )
            # Laser active
            for f in self._lasers:
                commands[f.fixture_id] = make_command(f, BLACK, special=_LASER_DROP)
            return self._merge_commands(commands)

        # Post-blinder: fast chase_mirror + strobe_chase
        drop_color = self._apply_sub_bass_white(ICE_WHITE, state.sub_bass_energy)

        # Pars: fast symmetric chase
        par_cmds = chase_mirror(
            self._pars, state, state.timestamp, drop_color,
            speed=3.0, width=0.25, intensity=1.0,
        )
        commands.update(par_cmds)

        # Strobes: rotating strobe chase on corners
        strobe_cmds = strobe_chase(
            self._strobes, state, state.timestamp, COLD_WHITE,
            speed=2.0, intensity=1.0,
        )
        commands.update(strobe_cmds)

        # LED bars: full intensity
        bar_cmds = wash_hold(
            self._led_bars, state, state.timestamp, ICE_WHITE, intensity=1.0,
        )
        commands.update(bar_cmds)

        # Laser active during drop
        for f in self._lasers:
            commands[f.fixture_id] = make_command(f, BLACK, special=_LASER_DROP)

        return self._merge_commands(commands)

    def _breakdown(self, state: MusicState) -> list[FixtureCommand]:
        """Breakdown: single spotlight, steel blue, no movement.

        spotlight_isolate on 1 par at 25%. Everything else off.
        Maximum restraint -- the silence before the next hit.
        """
        commands: dict[int, FixtureCommand] = {}

        # Single par spotlight at 25%
        par_cmds = spotlight_isolate(
            self._pars, state, state.timestamp, STEEL_BLUE,
            target_index=0, intensity=0.25, dim_others=0.0,
        )
        commands.update(par_cmds)

        # Everything else off
        for f in self._strobes + self._led_bars:
            commands[f.fixture_id] = make_command(f, BLACK, 0.0)
        for f in self._lasers:
            commands[f.fixture_id] = make_command(f, BLACK, special=_LASER_OFF)

        return self._merge_commands(commands)

    def _intro_outro(self, state: MusicState) -> list[FixtureCommand]:
        """Intro/outro: slow breathing on 2 pars, steel blue.

        breathe on first 2 pars, period_bars=4.0, low intensity.
        Everything else off.
        """
        commands: dict[int, FixtureCommand] = {}

        # Breathe on first 2 pars
        breathing_pars = self._pars[:2] if len(self._pars) >= 2 else self._pars
        seg_time = state.timestamp - self._segment_start_time
        par_cmds = breathe(
            breathing_pars, state, seg_time, STEEL_BLUE,
            min_intensity=0.05, max_intensity=0.20, period_bars=4.0,
        )
        commands.update(par_cmds)

        # Blackout remaining pars
        for f in self._pars:
            if f.fixture_id not in commands:
                commands[f.fixture_id] = make_command(f, BLACK, 0.0)

        # Everything else off
        for f in self._strobes + self._led_bars:
            commands[f.fixture_id] = make_command(f, BLACK, 0.0)
        for f in self._lasers:
            commands[f.fixture_id] = make_command(f, BLACK, special=_LASER_OFF)

        return self._merge_commands(commands)
