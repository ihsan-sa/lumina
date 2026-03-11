"""Profile 3: French Melodic Rap — Ninho, Jul.

Core philosophy: WARM PALETTE, HI-HAT-DRIVEN BOUNCE, FIRE FLICKER.

Lighting language:
- Colors: Warm gold, amber, sunset orange, soft coral. No cold blues.
  White channel adds warmth, not clinical harshness.
- Rhythm: Hi-hat onsets drive LED bar brightness ripple and accelerate
  chase speed. Kicks trigger par bumps. Bouncy, celebratory energy.
- Verse: chase_bounce on active pars (L-R-L), flicker layered on LED
  bars at 15% for fire effect. Fixture count builds with energy.
- Chorus: chase_mirror on all pars + color_pop on kicks. LED bars full
  warm wash. Gentle amber-tinted strobes on downbeats only.
- Drop: rainbow_roll with warm hue range (orange->red only). Strobes
  on kicks with bump decay. Initial 12-frame strobe burst.
- Breakdown: flicker on 2-3 pars (candle effect) at low intensity.
  Everything else off.
- Intro/outro: Single par spotlight, warm gold, low intensity.
- Sub-bass: deepens warm saturation. Spectral centroid: gold (low) ->
  coral (high).
- Laser: always off (Phase 1 constraint).
"""

from __future__ import annotations

import math

from lumina.audio.models import MusicState
from lumina.control.protocol import FixtureCommand
from lumina.lighting.fixture_map import FixtureMap, FixtureType
from lumina.lighting.patterns import (
    chase_bounce,
    chase_mirror,
    color_pop,
    flicker,
    make_command,
    rainbow_roll,
    select_active_fixtures,
    spotlight_isolate,
    strobe_burst,
    wash_hold,
)
from lumina.lighting.profiles.base import (
    BLACK,
    BaseProfile,
    BumpTracker,
    Color,
    FixtureInfo,
    energy_brightness,
    lerp_color,
)

# ─── French melodic palette ──────────────────────────────────────────

WARM_GOLD = Color(1.0, 0.75, 0.15, 0.35)
AMBER = Color(1.0, 0.5, 0.0, 0.25)
SUNSET_ORANGE = Color(1.0, 0.35, 0.05, 0.15)
SOFT_CORAL = Color(1.0, 0.4, 0.3, 0.2)
FIRE_RED = Color(1.0, 0.15, 0.0, 0.0)

# Laser patterns (stub — always off in Phase 1)
_LASER_OFF = 0

# Drop timing (frames at 60fps)
_DROP_BURST_FRAMES = 12  # 200ms initial strobe burst


class FrenchMelodicProfile(BaseProfile):
    """French Melodic Rap lighting profile (Ninho, Jul).

    Warm, bouncy, celebratory lighting driven by hi-hat rhythms.
    Gold/amber palette with fire-flicker atmospheric effects on LED
    bars. Chase patterns bounce L-R-L in sync with the groove.

    Args:
        fixture_map: Venue fixture layout.
    """

    name = "french_melodic"

    def __init__(self, fixture_map: FixtureMap) -> None:
        super().__init__(fixture_map)
        self._pars = self._map.by_type(FixtureType.PAR)
        self._strobes = self._map.by_type(FixtureType.STROBE)
        self._led_bars = self._map.by_type(FixtureType.LED_BAR)
        self._lasers = self._map.by_type(FixtureType.LASER)
        self._pars_lr = self._map.sorted_by_x(self._pars)

        # Override bump tracker: 80ms fast bounce — snappy but not harsh
        self._bump = BumpTracker(decay_rate=12.0)

        # State tracking
        self._segment_start_time: float = 0.0
        self._last_segment: str = ""
        self._segment_changed: bool = False
        self._drop_frame_count: int = 0

    @property
    def motif_pattern_preferences(self) -> list[str]:
        """French melodic prefers bouncy, warm patterns for motifs."""
        return ["chase_bounce", "chase_mirror", "color_pop", "alternate"]

    @property
    def motif_color_palette(self) -> list[Color]:
        """Warm color cycle: golds, ambers, and corals."""
        return [WARM_GOLD, AMBER, SUNSET_ORANGE, SOFT_CORAL]

    def _headroom_scale(self, state: MusicState, intensity: float) -> float:
        """Apply headroom to intensity, french-melodic style.

        During drops, headroom is relaxed (minimum 0.5). Otherwise
        headroom scales normally with a floor of 0.2.

        Args:
            state: Current music state.
            intensity: Raw intensity value.

        Returns:
            Headroom-adjusted intensity.
        """
        if state.segment == "drop":
            return intensity * max(0.5, state.headroom)
        return intensity * max(0.2, state.headroom)

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

    def _warm_color(self, state: MusicState) -> Color:
        """Pick a warm color based on spectral centroid and sub-bass.

        Low centroid (bassy) -> gold, high centroid (bright) -> coral.
        Sub-bass deepens the saturation of the dominant warm channel.

        Args:
            state: Current music state.

        Returns:
            Warm Color adapted to the current spectral character.
        """
        base = self._color_temperature(state.spectral_centroid, WARM_GOLD, SOFT_CORAL)
        if state.sub_bass_energy > 0.3:
            base = self._bass_saturate(state.sub_bass_energy, base, boost=0.25)
        return base

    def generate(self, state: MusicState) -> list[FixtureCommand]:
        """Generate french-melodic fixture commands for the current frame.

        Decision hierarchy:
        1. Arc layer: headroom caps intensity
        2. Note pattern: if regular notes detected, cycle fixtures
        3. Segment layer: verse/chorus/drop/breakdown routing
        4. Reactive layer: onset reactions (kick bump, hi-hat ripple)

        Args:
            state: Current audio analysis frame.

        Returns:
            One FixtureCommand per fixture (15 total).
        """
        self._begin_debug_frame()
        segment = state.segment
        energy = state.energy

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

        # ─── Hi-hat: always trigger bar bump regardless of segment ──
        if state.onset_type == "hihat":
            self._bump.trigger("bars", state.timestamp)
            self._bump.trigger("chase_speed", state.timestamp)

        # ─── Note-level pattern (each note = different fixture) ─────
        if (
            state.notes_per_beat > 0
            and segment not in ("breakdown", "bridge", "intro", "outro")
        ):
            active_pars = self._get_active_pars(state)
            note_cmds = self._apply_note_pattern(state, active_pars, WARM_GOLD)
            if note_cmds is not None:
                self._note_patterns("note_pattern")
                for f in self._strobes + self._led_bars:
                    note_cmds[f.fixture_id] = make_command(f, BLACK, 0.0)
                for f in self._lasers:
                    note_cmds[f.fixture_id] = make_command(f, BLACK, special=_LASER_OFF)
                return self._merge_commands(note_cmds)

        # ─── Segment-based routing ──────────────────────────────────

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

        self._note_patterns("verse")
        return self._verse(state)

    # ─── Segment handlers ────────────────────────────────────────────

    def _verse(self, state: MusicState) -> list[FixtureCommand]:
        """Verse: bouncy chase on pars, fire flicker on LED bars.

        Active par count scales with energy. Kicks trigger bump decay.
        LED bars hold a low-intensity fire flicker for atmosphere.
        """
        commands: dict[int, FixtureCommand] = {}
        energy = state.energy

        # Active par selection
        active_pars = self._get_active_pars(state)
        if not active_pars:
            active_pars = select_active_fixtures(
                self._pars, energy, low_count=3, mid_count=6,
            )

        # Color adapts to spectral character
        verse_color = self._warm_color(state)

        # Chase speed boosted by hi-hat bumps
        base_speed = 0.8 + energy * 0.4
        hihat_boost = self._bump.get_intensity(
            "chase_speed", state.timestamp, peak=0.6, floor=0.0,
        )
        chase_speed = base_speed + hihat_boost

        # Onset reactions
        if state.onset_type == "kick":
            self._bump.trigger("pars", state.timestamp)
            kick_int = self._headroom_scale(state, 1.0)
            par_cmds = wash_hold(
                active_pars, state, state.timestamp, AMBER, intensity=kick_int,
            )
            commands.update(par_cmds)
        else:
            # Bouncy chase with bump decay
            bump_int = self._bump.get_intensity(
                "pars", state.timestamp, peak=0.8, floor=0.0,
            )
            chase_int = self._headroom_scale(
                state, energy_brightness(energy, gamma=0.5) * 0.6 + bump_int * 0.3,
            )
            par_cmds = chase_bounce(
                active_pars, state, state.timestamp, verse_color,
                speed=chase_speed, width=0.35, intensity=chase_int,
            )
            commands.update(par_cmds)

        # Blackout inactive pars
        for f in self._pars:
            if f.fixture_id not in commands:
                commands[f.fixture_id] = make_command(f, BLACK, 0.0)

        # Strobes off during verse
        for f in self._strobes:
            commands[f.fixture_id] = make_command(f, BLACK, 0.0)

        # LED bars: fire flicker at 15% + hi-hat bump
        bar_base = 0.15
        bar_bump = self._bump.get_intensity(
            "bars", state.timestamp, peak=0.35, floor=0.0,
        )
        bar_int = self._headroom_scale(state, bar_base + bar_bump)
        bar_cmds = flicker(
            self._led_bars, state, state.timestamp, WARM_GOLD,
            intensity=bar_int, jitter=0.3,
        )
        commands.update(bar_cmds)

        # Laser off
        for f in self._lasers:
            commands[f.fixture_id] = make_command(f, BLACK, special=_LASER_OFF)

        return self._merge_commands(commands)

    def _chorus(self, state: MusicState) -> list[FixtureCommand]:
        """Chorus: mirror chase on all pars, color pop on kicks.

        Full fixture count. LED bars full warm wash. Gentle amber
        strobes on downbeats only.
        """
        commands: dict[int, FixtureCommand] = {}
        energy = state.energy

        # Chorus uses all pars for wide coverage
        chorus_pars = self._pars
        chorus_color = self._warm_color(state)

        # Chase speed boosted by hi-hat bumps
        base_speed = 1.0 + energy * 0.3
        hihat_boost = self._bump.get_intensity(
            "chase_speed", state.timestamp, peak=0.5, floor=0.0,
        )
        chase_speed = base_speed + hihat_boost

        # Onset reactions
        if state.onset_type == "kick":
            self._bump.trigger("pars", state.timestamp)
            # Color pop: complementary flash on kicks
            par_cmds = color_pop(
                chorus_pars, state, state.timestamp, chorus_color,
                intensity=self._headroom_scale(state, 1.0),
            )
            commands.update(par_cmds)
        else:
            # Mirror chase for symmetric bounce
            bump_int = self._bump.get_intensity(
                "pars", state.timestamp, peak=0.7, floor=0.0,
            )
            chase_int = self._headroom_scale(
                state, energy_brightness(energy, gamma=0.5) * 0.7 + bump_int * 0.2,
            )
            par_cmds = chase_mirror(
                chorus_pars, state, state.timestamp, chorus_color,
                speed=chase_speed, width=0.4, intensity=chase_int,
            )
            commands.update(par_cmds)

        # Strobes: gentle amber tint on downbeats only
        if state.is_downbeat:
            for f in self._strobes:
                commands[f.fixture_id] = make_command(
                    f, AMBER, strobe_rate=100, strobe_intensity=120,
                )
        else:
            for f in self._strobes:
                commands[f.fixture_id] = make_command(f, BLACK, 0.0)

        # LED bars: full warm wash
        bar_int = self._headroom_scale(state, 0.6 + energy * 0.3)
        bar_bump = self._bump.get_intensity(
            "bars", state.timestamp, peak=0.15, floor=0.0,
        )
        bar_cmds = wash_hold(
            self._led_bars, state, state.timestamp, WARM_GOLD,
            intensity=bar_int + bar_bump,
        )
        commands.update(bar_cmds)

        # Laser off
        for f in self._lasers:
            commands[f.fixture_id] = make_command(f, BLACK, special=_LASER_OFF)

        return self._merge_commands(commands)

    def _drop(self, state: MusicState) -> list[FixtureCommand]:
        """Drop: warm rainbow roll with strobe accents on kicks.

        Initial 12-frame strobe burst on entry. After: rainbow_roll
        with hue range restricted to orange->red. Strobes fire on
        kicks with bump decay. LED bars full.
        """
        commands: dict[int, FixtureCommand] = {}
        energy = state.energy

        # Initial burst on drop entry (200ms)
        if self._segment_changed or self._drop_frame_count <= _DROP_BURST_FRAMES:
            burst = strobe_burst(
                self._strobes, state, state.timestamp, AMBER,
            )
            commands.update(burst)
            par_cmds = wash_hold(
                self._pars, state, state.timestamp, WARM_GOLD, intensity=1.0,
            )
            commands.update(par_cmds)
            bar_cmds = wash_hold(
                self._led_bars, state, state.timestamp, WARM_GOLD, intensity=1.0,
            )
            commands.update(bar_cmds)
            for f in self._lasers:
                commands[f.fixture_id] = make_command(f, BLACK, special=_LASER_OFF)
            return self._merge_commands(commands)

        # Rainbow roll: warm hue range only (0.0=red -> 0.15=orange)
        roll_int = self._headroom_scale(state, 0.7 + energy * 0.3)
        par_cmds = rainbow_roll(
            self._pars, state, state.timestamp, WARM_GOLD,
            speed=1.0, hue_min=0.0, hue_max=0.15, intensity=roll_int,
        )
        commands.update(par_cmds)

        # Strobes on kicks with bump decay
        if state.onset_type == "kick":
            self._bump.trigger("strobes", state.timestamp)

        strobe_int = self._bump.get_intensity(
            "strobes", state.timestamp, peak=1.0, floor=0.0,
        )
        if strobe_int > 0.1:
            for f in self._strobes:
                commands[f.fixture_id] = make_command(
                    f, AMBER,
                    strobe_rate=int(180 * strobe_int),
                    strobe_intensity=int(200 * strobe_int),
                )
        else:
            for f in self._strobes:
                commands[f.fixture_id] = make_command(f, BLACK, 0.0)

        # LED bars full warm wash
        bar_cmds = wash_hold(
            self._led_bars, state, state.timestamp, SUNSET_ORANGE,
            intensity=self._headroom_scale(state, 0.8),
        )
        commands.update(bar_cmds)

        # Laser off
        for f in self._lasers:
            commands[f.fixture_id] = make_command(f, BLACK, special=_LASER_OFF)

        return self._merge_commands(commands)

    def _breakdown(self, state: MusicState) -> list[FixtureCommand]:
        """Breakdown: candle-like flicker on 2-3 pars. Everything else off.

        Minimal, atmospheric. The fire flicker creates an intimate
        candlelit mood during stripped-back sections.
        """
        commands: dict[int, FixtureCommand] = {}

        # Select limited pars for candle effect
        active_pars = select_active_fixtures(
            self._pars, state.energy, low_count=2, mid_count=3, mid_threshold=0.5,
        )

        # Candle flicker at very low intensity
        flicker_int = 0.08 + state.energy * 0.07  # 0.08-0.15 range
        par_cmds = flicker(
            active_pars, state, state.timestamp, WARM_GOLD,
            intensity=flicker_int, jitter=0.5,
        )
        commands.update(par_cmds)

        # Blackout inactive pars
        for f in self._pars:
            if f.fixture_id not in commands:
                commands[f.fixture_id] = make_command(f, BLACK, 0.0)

        # Everything else off
        for f in self._strobes + self._led_bars:
            commands[f.fixture_id] = make_command(f, BLACK, 0.0)
        for f in self._lasers:
            commands[f.fixture_id] = make_command(f, BLACK, special=_LASER_OFF)

        return self._merge_commands(commands)

    def _intro_outro(self, state: MusicState) -> list[FixtureCommand]:
        """Intro/outro: single par spotlight, warm gold, low intensity."""
        commands: dict[int, FixtureCommand] = {}

        par_cmds = spotlight_isolate(
            self._pars, state, state.timestamp, WARM_GOLD,
            target_index=0, intensity=0.25, dim_others=0.0,
        )
        commands.update(par_cmds)

        for f in self._strobes + self._led_bars:
            commands[f.fixture_id] = make_command(f, BLACK, 0.0)
        for f in self._lasers:
            commands[f.fixture_id] = make_command(f, BLACK, special=_LASER_OFF)

        return self._merge_commands(commands)
