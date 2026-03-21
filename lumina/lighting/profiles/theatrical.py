"""Profile 6: Theatrical Electronic -- Stromae.

Core philosophy: STORYTELLING -- lights follow the emotional arc, not
just beats.  Vocal energy is the PRIMARY intensity driver.  The voice IS
the dimmer.

Lighting language:
- Palette changes per segment: cool neutrals (verse), warm gold (chorus),
  deep blue (bridge/breakdown), red/amber (drop).
- Verse: wash_hold at vocal-energy-driven intensity.  Spectral centroid
  shifts color temperature.  Fixture count expands with vocal energy.
- Chorus: color_split (warm gold left, gold right) + vocal energy pulses.
  Breathe overlay on LED bars at 8-bar period.  Gentle downbeat strobes.
- Drop: diverge bloom (slow, theatrical, NOT explosive).  Color drifts
  red->amber via bar_phase.  Strobes on downbeats only.
- Breakdown/bridge: spotlight_isolate, single par, slow color drift.
  BRIDGE_BLUE.  Intimate.
- Intro/outro: breathe on 2 pars, VERSE_COOL, low intensity.
- Sub-bass deepens warm colors.  Spectral centroid is the PRIMARY color
  mechanism (very pronounced warm<->cool).
"""

from __future__ import annotations

from lumina.audio.models import MusicState
from lumina.control.protocol import FixtureCommand
from lumina.lighting.fixture_map import FixtureMap, FixtureType
from lumina.lighting.patterns import (
    breathe,
    color_split,
    diverge,
    make_command,
    spotlight_isolate,
    wash_hold,
)
from lumina.lighting.profiles.base import (
    BLACK,
    BaseProfile,
    BumpTracker,
    Color,
    energy_brightness,
    lerp_color,
)

# --- Theatrical palette ---------------------------------------------------

# Verse: cool neutrals
VERSE_COOL = Color(0.3, 0.35, 0.5, 0.3)
VERSE_NEUTRAL = Color(0.4, 0.4, 0.45, 0.4)

# Chorus: warm gold
CHORUS_WARM = Color(1.0, 0.75, 0.2, 0.5)
CHORUS_GOLD = Color(1.0, 0.6, 0.1, 0.4)

# Bridge / breakdown: deep blue
BRIDGE_BLUE = Color(0.1, 0.15, 0.7, 0.0)

# Drop: red / amber
DROP_RED = Color(1.0, 0.2, 0.0, 0.2)
DROP_AMBER = Color(1.0, 0.5, 0.1, 0.3)

# Laser patterns
_LASER_OFF = 0
_LASER_DROP = 4


class TheatricalProfile(BaseProfile):
    """Theatrical Electronic lighting profile (Stromae).

    Story-driven: lights follow the emotional arc, not just beats.
    Vocal energy is the primary intensity driver -- the voice IS the
    dimmer.  Spectral centroid shifts color temperature across warm/cool.

    Args:
        fixture_map: Venue fixture layout.
    """

    name = "theatrical"

    def __init__(self, fixture_map: FixtureMap) -> None:
        super().__init__(fixture_map)
        self._pars = self._map.by_type(FixtureType.PAR)
        self._strobes = self._map.by_type(FixtureType.STROBE)
        self._led_bars = self._map.by_type(FixtureType.LED_BAR)
        self._lasers = self._map.by_type(FixtureType.LASER)
        self._pars_lr = self._map.sorted_by_x(self._pars)

        # 330ms theatrical fades (slower than rage trap, faster than psych)
        self._bump = BumpTracker(decay_rate=3.0)

        # Segment tracking
        self._last_segment: str = ""
        self._segment_start_time: float = 0.0

        # Motif-to-color-temperature mapping
        self._motif_temp_map: dict[int, float] = {}
        self._next_temp_value: float = 0.0

    # --- Motif overrides ---------------------------------------------------

    @property
    def motif_pattern_preferences(self) -> list[str]:
        """Theatrical prefers sweeping, emotional patterns for motifs."""
        return ["diverge", "color_split", "breathe", "wash_hold"]

    @property
    def motif_color_palette(self) -> list[Color]:
        """Theatrical color cycle follows the segment palette."""
        return [CHORUS_WARM, BRIDGE_BLUE, DROP_RED, VERSE_COOL]

    # --- Main generate -----------------------------------------------------

    def _get_motif_temperature(self, motif_id: int | None) -> float:
        """Get or assign a color temperature value for a motif.

        Each distinct motif gets its own warm/cool bias (0.0=warm, 1.0=cool).
        The mapping is built up as new motifs appear.

        Args:
            motif_id: Current motif identifier (None = no motif).

        Returns:
            Temperature value 0.0-1.0.
        """
        if motif_id is None:
            return 0.5  # neutral
        if motif_id not in self._motif_temp_map:
            self._motif_temp_map[motif_id] = self._next_temp_value
            # Cycle through distinct temperature values
            self._next_temp_value = (self._next_temp_value + 0.3) % 1.0
        return self._motif_temp_map[motif_id]

    def _vocal_layer_intensity(self, state: MusicState) -> float:
        """Compute spotlight intensity from vocal layer in layer_mask.

        When layer_mask contains a 'vocals' key, use it directly to drive
        spotlight intensity (the voice IS the dimmer, amplified by layer
        awareness).  Falls back to vocal_energy.

        Args:
            state: Current music state.

        Returns:
            Vocal-driven intensity 0.0-1.0.
        """
        layer_mask = getattr(state, "layer_mask", {})
        if isinstance(layer_mask, dict) and "vocals" in layer_mask:
            return max(0.0, min(1.0, layer_mask["vocals"]))
        return max(0.0, min(1.0, state.vocal_energy))

    def generate(self, state: MusicState) -> list[FixtureCommand]:
        """Generate theatrical fixture commands for the current frame.

        Decision hierarchy: vocal energy drives intensity, spectral
        centroid drives color temperature, segment drives palette and
        spatial pattern.

        Extended MusicState integration:
        - layer_mask['vocals'] directly drives spotlight intensity
        - motif_id maps to distinct color temperature per musical theme
        - headroom scales final output

        Args:
            state: Current audio analysis frame.

        Returns:
            One FixtureCommand per fixture (15 total).
        """
        self._begin_debug_frame()
        self._store_headroom(state)
        segment = state.segment

        # Track segment transitions
        if segment != self._last_segment:
            self._segment_start_time = state.timestamp
            self._last_segment = segment

        # Segment routing
        if segment == "drop":
            self._note_patterns("drop")
            return self._apply_headroom(self._drop(state))

        if segment == "chorus":
            self._note_patterns("chorus")
            return self._apply_headroom(self._chorus(state))

        if segment in ("breakdown", "bridge"):
            self._note_patterns("breakdown")
            return self._apply_headroom(self._breakdown(state))

        if segment in ("intro", "outro"):
            self._note_patterns("intro_outro")
            return self._apply_headroom(self._intro_outro(state))

        # Default: verse
        self._note_patterns("verse")
        return self._apply_headroom(self._verse(state))

    # --- Segment handlers --------------------------------------------------

    def _verse(self, state: MusicState) -> list[FixtureCommand]:
        """Verse: vocal-energy-driven wash with spectral color temperature.

        2-4 pars expanding with vocal energy.  LED bars dim.
        Strobes off.  Laser off.
        layer_mask['vocals'] drives intensity directly.
        motif_id shifts color temperature per musical theme.
        """
        commands: dict[int, FixtureCommand] = {}
        vocal = self._vocal_layer_intensity(state)

        # Color temperature driven by spectral centroid
        verse_color = self._color_temperature(
            state.spectral_centroid, VERSE_COOL, VERSE_NEUTRAL,
        )

        # Motif-driven temperature: each musical theme gets its own warmth
        motif_id = getattr(state, "motif_id", None)
        motif_temp = self._get_motif_temperature(motif_id)
        motif_color = lerp_color(VERSE_COOL, VERSE_NEUTRAL, motif_temp)
        verse_color = lerp_color(verse_color, motif_color, 0.3)

        # Sub-bass deepens warm tones
        if state.sub_bass_energy > 0.5:
            verse_color = self._bass_saturate(state.sub_bass_energy, verse_color)

        # Fixture count expands with vocal energy: 2 at silence, all at full voice
        count = max(2, int(vocal * len(self._pars)))
        active_pars = self._pars_lr[:count]

        # Vocal energy (from layer_mask or vocal_energy) IS the intensity
        vocal_intensity = energy_brightness(vocal, gamma=0.5) * 0.6 + 0.05

        par_cmds = wash_hold(
            active_pars, state, state.timestamp, verse_color, intensity=vocal_intensity,
        )
        commands.update(par_cmds)

        # Blackout inactive pars
        for f in self._pars:
            if f.fixture_id not in commands:
                commands[f.fixture_id] = make_command(f, BLACK, 0.0)

        # LED bars: dim atmospheric
        bar_intensity = vocal_intensity * 0.3
        bar_cmds = wash_hold(
            self._led_bars, state, state.timestamp, verse_color, intensity=bar_intensity,
        )
        commands.update(bar_cmds)

        # Strobes off
        for f in self._strobes:
            commands[f.fixture_id] = make_command(f, BLACK, 0.0)

        # Laser off
        for f in self._lasers:
            commands[f.fixture_id] = make_command(f, BLACK, special=_LASER_OFF)

        return self._merge_commands(commands)

    def _chorus(self, state: MusicState) -> list[FixtureCommand]:
        """Chorus: warm gold color_split + vocal energy pulses.

        All pars active.  Breathe overlay on LED bars at 8-bar period.
        Gentle downbeat strobes (tinted warm, rate=60, intensity=80).
        layer_mask['vocals'] drives intensity directly.
        """
        commands: dict[int, FixtureCommand] = {}
        vocal = self._vocal_layer_intensity(state)

        # Vocal-driven intensity
        vocal_intensity = energy_brightness(vocal, gamma=0.5) * 0.7 + 0.20

        # Color split: warm gold left, gold right
        par_cmds = color_split(
            self._pars, state, state.timestamp, CHORUS_WARM,
            color_right=CHORUS_GOLD, intensity=vocal_intensity,
        )
        commands.update(par_cmds)

        # LED bars: breathe at 8-bar period
        bar_cmds = breathe(
            self._led_bars, state, state.timestamp, CHORUS_WARM,
            min_intensity=0.15, max_intensity=0.45, period_bars=8.0,
        )
        commands.update(bar_cmds)

        # Strobes: gentle warm tint on downbeats only
        if state.is_downbeat:
            for f in self._strobes:
                commands[f.fixture_id] = make_command(
                    f, CHORUS_WARM, strobe_rate=60, strobe_intensity=80,
                )
        else:
            for f in self._strobes:
                commands[f.fixture_id] = make_command(f, BLACK, 0.0)

        # Laser off in chorus
        for f in self._lasers:
            commands[f.fixture_id] = make_command(f, BLACK, special=_LASER_OFF)

        return self._merge_commands(commands)

    def _drop(self, state: MusicState) -> list[FixtureCommand]:
        """Drop: theatrical diverge bloom (slow, NOT explosive).

        Color drifts from DROP_RED to DROP_AMBER based on bar_phase.
        Strobes only on downbeats.  LED bars full.  Laser active.
        """
        commands: dict[int, FixtureCommand] = {}

        # Color drifts through bar phase: red -> amber
        drop_color = lerp_color(DROP_RED, DROP_AMBER, state.bar_phase)

        # Diverge bloom: center-out, theatrical pace
        par_cmds = diverge(
            self._pars, state, state.timestamp, drop_color, intensity=0.85,
        )
        commands.update(par_cmds)

        # LED bars: full intensity in drop color
        bar_cmds = wash_hold(
            self._led_bars, state, state.timestamp, drop_color, intensity=0.80,
        )
        commands.update(bar_cmds)

        # Strobes on downbeats only (theatrical, not every beat)
        if state.is_downbeat:
            for f in self._strobes:
                commands[f.fixture_id] = make_command(
                    f, drop_color, strobe_rate=120, strobe_intensity=160,
                )
        else:
            for f in self._strobes:
                commands[f.fixture_id] = make_command(f, BLACK, 0.0)

        # Laser active during drop
        for f in self._lasers:
            commands[f.fixture_id] = make_command(f, BLACK, special=_LASER_DROP)

        return self._merge_commands(commands)

    def _breakdown(self, state: MusicState) -> list[FixtureCommand]:
        """Breakdown/bridge: single spotlight, slow color drift.

        Spotlight_isolate with rotating target based on timestamp.
        BRIDGE_BLUE color.  Intimate -- very few fixtures lit.
        """
        commands: dict[int, FixtureCommand] = {}

        # Slow-rotating spotlight target (one par at a time)
        n_pars = len(self._pars)
        target = int(state.timestamp * 0.2) % max(n_pars, 1)

        par_cmds = spotlight_isolate(
            self._pars, state, state.timestamp, BRIDGE_BLUE,
            target_index=target, intensity=0.40, dim_others=0.0,
        )
        commands.update(par_cmds)

        # LED bars: very faint blue wash
        bar_cmds = wash_hold(
            self._led_bars, state, state.timestamp, BRIDGE_BLUE, intensity=0.08,
        )
        commands.update(bar_cmds)

        # Strobes off
        for f in self._strobes:
            commands[f.fixture_id] = make_command(f, BLACK, 0.0)

        # Laser off
        for f in self._lasers:
            commands[f.fixture_id] = make_command(f, BLACK, special=_LASER_OFF)

        return self._merge_commands(commands)

    def _intro_outro(self, state: MusicState) -> list[FixtureCommand]:
        """Intro/outro: slow breathe on 2 pars, VERSE_COOL, low intensity."""
        commands: dict[int, FixtureCommand] = {}

        # Only first 2 pars breathe
        intro_pars = self._pars_lr[:2]

        seg_time = state.timestamp - self._segment_start_time
        par_cmds = breathe(
            intro_pars, state, seg_time, VERSE_COOL,
            min_intensity=0.15, max_intensity=0.25, period_bars=4.0,
        )
        commands.update(par_cmds)

        # Blackout remaining pars
        for f in self._pars:
            if f.fixture_id not in commands:
                commands[f.fixture_id] = make_command(f, BLACK, 0.0)

        # LED bars off
        for f in self._led_bars:
            commands[f.fixture_id] = make_command(f, BLACK, 0.0)

        # Strobes off
        for f in self._strobes:
            commands[f.fixture_id] = make_command(f, BLACK, 0.0)

        # Laser off
        for f in self._lasers:
            commands[f.fixture_id] = make_command(f, BLACK, special=_LASER_OFF)

        return self._merge_commands(commands)
