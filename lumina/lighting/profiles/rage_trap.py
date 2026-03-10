"""Profile 1: Rage / Experimental Trap — Playboi Carti, Travis Scott.

Core philosophy: EXTREME CONTRAST — BLINDING or DARK, nothing in between.

Lighting language:
- Colors: RED and WHITE only. Deep blood red (hsv 0/355, full sat).
  White = raw strobe white. No pastels, no blending, no gradients.
- Contrast: Binary. Fixtures are either at 100% or at 0%.
  No smooth fades. Transitions are instant (1-frame cuts).
- Verse: wash_hold on pars at dark red 15% + breathe on 1 par.
  Fixture count builds over bars via select_active_fixtures.
- Build: stutter on 1 strobe (accelerating) + converge on pars.
- Drop/808 hit: strobe_burst on all strobes then instant blackout.
  Between hits: everything OFF (binary: blinding or dark).
- Ad-libs: random_scatter on pars with red/orange palette.
- Beat switch: 500ms full blackout then new color palette.
- Breakdown: breathe on pars, deep red, 10-20% range.
- LED bars: follow pars in verse, full during drops, off in breakdown.
- Laser: off in verse/breakdown, pattern active during drops.
"""

from __future__ import annotations

import hashlib
import math

from lumina.audio.models import MusicState
from lumina.control.protocol import FixtureCommand
from lumina.lighting.fixture_map import FixtureMap, FixtureType
from lumina.lighting.patterns import (
    breathe,
    chase_lr,
    converge,
    make_command,
    random_scatter,
    select_active_fixtures,
    spotlight_isolate,
    strobe_burst,
    stutter,
    wash_hold,
)
from lumina.lighting.profiles.base import (
    BLACK,
    BaseProfile,
    Color,
    lerp_color,
)

# ─── Rage palette ───────────────────────────────────────────────────

BLOOD_RED = Color(1.0, 0.0, 0.0, 0.0)
DEEP_RED = Color(0.7, 0.0, 0.0, 0.0)
COOL_RED = Color(0.6, 0.0, 0.1, 0.0)
WARM_RED = Color(1.0, 0.05, 0.0, 0.0)
DARK_ORANGE = Color(1.0, 0.2, 0.0, 0.0)
STROBE_WHITE = Color(1.0, 1.0, 1.0, 1.0)

# Drop timing (frames at 60fps)
_DROP_HIT_WHITE_FRAMES = 12  # 200ms at 60fps
_BLACKOUT_GAP_BARS = 12
_BLACKOUT_GAP_DURATION_S = 0.5
_BARS_PER_FIXTURE_ADD = 4

# Laser patterns
_LASER_OFF = 0
_LASER_DROP_PATTERN = 5


class RageTrapProfile(BaseProfile):
    """Rage / Experimental Trap lighting profile.

    Binary contrast philosophy: every fixture is either fully ON or
    fully OFF.  No smooth fades, no gradients, no in-between states.
    RED and WHITE only.

    Args:
        fixture_map: Venue fixture layout.
    """

    name = "rage_trap"

    def __init__(self, fixture_map: FixtureMap) -> None:
        super().__init__(fixture_map)
        self._pars = self._map.by_type(FixtureType.PAR)
        self._strobes = self._map.by_type(FixtureType.STROBE)
        self._led_bars = self._map.by_type(FixtureType.LED_BAR)
        self._lasers = self._map.by_type(FixtureType.LASER)
        self._pars_lr = self._map.sorted_by_x(self._pars)

        # State tracking
        self._segment_start_time: float = 0.0
        self._last_segment: str = ""
        self._segment_changed: bool = False
        self._drop_frame_count: int = 0
        self._was_pre_drop: bool = False

    def generate(self, state: MusicState) -> list[FixtureCommand]:
        """Generate rage-trap fixture commands for the current frame.

        Args:
            state: Current music analysis frame.

        Returns:
            One FixtureCommand per fixture (15 total).
        """
        self._begin_debug_frame()
        segment = state.segment
        energy = state.energy

        # Track segment transitions — instant reset on change
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

        # Pre-drop: accelerating strobe
        if state.drop_probability > 0.6 and segment != "drop":
            self._was_pre_drop = True
            self._note_patterns("pre_drop_build")
            return self._pre_drop_build(state)

        # Drop hit
        if segment == "drop" and energy > 0.5:
            self._note_patterns("drop_explosion")
            result = self._drop_explosion(state)
            self._was_pre_drop = False
            return result

        self._was_pre_drop = False

        # Breakdown / bridge
        if segment in ("breakdown", "bridge"):
            self._note_patterns("breakdown_breathe")
            return self._breakdown_breathe(state)

        # Intro / outro
        if segment in ("intro", "outro"):
            self._note_patterns("intro_outro")
            return self._intro_outro(state)

        # Chorus / hook — brighter and wider than verse
        if segment == "chorus":
            self._note_patterns("chorus_reactive")
            return self._chorus_reactive(state)

        # Vocal calm
        if state.vocal_energy > 0.6:
            self._note_patterns("vocal_calm")
            return self._vocal_calm(state)

        self._note_patterns("verse_reactive")
        return self._verse_reactive(state)

    # ─── Segment handlers ────────────────────────────────────────

    def _pre_drop_build(self, state: MusicState) -> list[FixtureCommand]:
        """Pre-drop: stutter on strobe (accelerating) + converge on pars."""
        commands: dict[int, FixtureCommand] = {}

        # Map drop_probability 0.6→1.0 to ramp 0→1
        ramp = max(0.0, min(1.0, (state.drop_probability - 0.6) / 0.4))

        # Pars: converge to center with increasing red intensity
        par_intensity = 0.30 + ramp * 0.50
        par_cmds = converge(
            self._pars, state, state.timestamp, BLOOD_RED, intensity=par_intensity,
        )
        commands.update(par_cmds)

        # Strobes: stutter with accelerating rate (2→8 subdivisions)
        stutter_rate = 2.0 + ramp * 6.0
        strobe_cmds = stutter(
            self._strobes, state, state.timestamp, STROBE_WHITE,
            rate=stutter_rate, intensity=1.0,
        )
        commands.update(strobe_cmds)

        # LED bars: follow par intensity
        bar_cmds = wash_hold(
            self._led_bars, state, state.timestamp, BLOOD_RED, intensity=par_intensity * 0.7,
        )
        commands.update(bar_cmds)

        # Laser: activate in final 25% of ramp
        for f in self._lasers:
            sp = _LASER_DROP_PATTERN if ramp > 0.75 else _LASER_OFF
            commands[f.fixture_id] = make_command(f, BLACK, special=sp)

        return self._merge_commands(commands)

    def _drop_explosion(self, state: MusicState) -> list[FixtureCommand]:
        """Drop: initial white flash then binary red/black alternating.

        First 200ms: ALL fixtures max white (strobe_burst on all).
        After: full red on beat, energy-aware floor on offbeat.
        LED bars full. Laser active.
        """
        commands: dict[int, FixtureCommand] = {}
        energy = state.energy

        # Initial white flash (200ms = ~12 frames at 60fps)
        if self._was_pre_drop and self._drop_frame_count <= _DROP_HIT_WHITE_FRAMES:
            # Strobe burst on strobes
            burst = strobe_burst(
                self._strobes, state, state.timestamp, STROBE_WHITE,
            )
            commands.update(burst)
            # All pars + bars full white
            par_cmds = wash_hold(
                self._pars, state, state.timestamp, STROBE_WHITE, intensity=1.0,
            )
            commands.update(par_cmds)
            bar_cmds = wash_hold(
                self._led_bars, state, state.timestamp, STROBE_WHITE, intensity=1.0,
            )
            commands.update(bar_cmds)
            for f in self._lasers:
                commands[f.fixture_id] = make_command(f, BLACK, special=_LASER_DROP_PATTERN)
            return self._merge_commands(commands)

        # Binary alternating: on-beat = full red, off-beat = blackout
        on_beat = state.beat_phase < 0.5

        if on_beat:
            # All pars blood red
            par_cmds = wash_hold(
                self._pars, state, state.timestamp, BLOOD_RED, intensity=1.0,
            )
            commands.update(par_cmds)
            # Strobes fire
            burst = strobe_burst(
                self._strobes, state, state.timestamp, STROBE_WHITE,
            )
            commands.update(burst)
            # LED bars full red
            bar_cmds = wash_hold(
                self._led_bars, state, state.timestamp, BLOOD_RED, intensity=1.0,
            )
            commands.update(bar_cmds)
        else:
            # Off-beat: energy-aware floor (not full blackout when energy is high)
            if energy > 0.6:
                floor = 0.20 + (energy - 0.6) / 0.4 * 0.10  # 20-30% deep red
                par_cmds = wash_hold(
                    self._pars, state, state.timestamp, DEEP_RED, intensity=floor,
                )
                commands.update(par_cmds)
            else:
                for f in self._pars:
                    commands[f.fixture_id] = make_command(f, BLACK, 0.0)
            for f in self._strobes + self._led_bars:
                commands[f.fixture_id] = make_command(f, BLACK, 0.0)

        # Laser active during drop
        for f in self._lasers:
            sp = _LASER_DROP_PATTERN if on_beat else _LASER_OFF
            commands[f.fixture_id] = make_command(f, BLACK, special=sp)

        return self._merge_commands(commands)

    def _breakdown_breathe(self, state: MusicState) -> list[FixtureCommand]:
        """Slow deep-red breathing pulse. No strobes, LED bars off, laser off.

        Only 2-4 pars breathing at 10-20% intensity.
        """
        commands: dict[int, FixtureCommand] = {}

        # Select limited pars based on low energy
        active_pars = select_active_fixtures(
            self._pars, state.energy, low_count=2, mid_count=4, mid_threshold=0.5,
        )

        # Breathing pattern on active pars — use segment-relative time for instant reset
        seg_time = state.timestamp - self._segment_start_time
        par_cmds = breathe(
            active_pars, state, seg_time, DEEP_RED,
            min_intensity=0.10, max_intensity=0.20, period_bars=1.0,
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
        """Minimal lighting: spotlight on single par, low red."""
        commands: dict[int, FixtureCommand] = {}

        par_cmds = spotlight_isolate(
            self._pars, state, state.timestamp, DEEP_RED,
            target_index=0, intensity=0.30, dim_others=0.0,
        )
        commands.update(par_cmds)

        for f in self._strobes + self._led_bars:
            commands[f.fixture_id] = make_command(f, BLACK, 0.0)
        for f in self._lasers:
            commands[f.fixture_id] = make_command(f, BLACK, special=_LASER_OFF)

        return self._merge_commands(commands)

    def _vocal_calm(self, state: MusicState) -> list[FixtureCommand]:
        """Choir / vocal calm — intimate, haunted. 2 pars at low red."""
        commands: dict[int, FixtureCommand] = {}

        # Rotating spotlight
        n_pars = len(self._pars)
        target = int(state.timestamp * 0.5) % max(n_pars, 1)
        par_cmds = spotlight_isolate(
            self._pars, state, state.timestamp, DEEP_RED,
            target_index=target, intensity=0.30, dim_others=0.05,
        )
        commands.update(par_cmds)

        for f in self._strobes + self._led_bars:
            commands[f.fixture_id] = make_command(f, BLACK, 0.0)
        for f in self._lasers:
            commands[f.fixture_id] = make_command(f, BLACK, special=_LASER_OFF)

        return self._merge_commands(commands)

    def _verse_reactive(self, state: MusicState) -> list[FixtureCommand]:
        """Evolving verse with fixture build-up, chases, and blackout gaps.

        Uses select_active_fixtures for escalation. Onset reactions via
        patterns: kicks use wash_hold flash, snares use random_scatter,
        claps use random_scatter (ad-lib scatter).
        """
        commands: dict[int, FixtureCommand] = {}
        energy = state.energy

        # Time since segment started (reset on each segment transition)
        dt = max(0.0, state.timestamp - self._segment_start_time)
        bpm = max(60.0, state.bpm)
        bar_duration = 60.0 / bpm * 4.0
        bars_elapsed = dt / bar_duration

        # Blackout gap: 500ms of darkness every N bars
        bar_cycle = bars_elapsed % _BLACKOUT_GAP_BARS
        gap_start = _BLACKOUT_GAP_BARS - (_BLACKOUT_GAP_DURATION_S / bar_duration)
        if bar_cycle >= gap_start:
            return self._blackout()

        # Fixture build-up via energy escalation
        active_pars = select_active_fixtures(
            self._pars, state.energy,
            low_count=max(1, 1 + int(bars_elapsed / _BARS_PER_FIXTURE_ADD)),
            mid_count=6, high_threshold=0.8,
        )

        # Color temperature: cooler → warmer as energy rises
        verse_color = lerp_color(COOL_RED, WARM_RED, state.energy)

        # Onset reactions
        if state.onset_type == "kick":
            # Full red flash on active pars
            par_cmds = wash_hold(
                active_pars, state, state.timestamp, BLOOD_RED, intensity=1.0,
            )
            commands.update(par_cmds)
            # Strobes alternate left/right on kicks
            beat_idx = int(state.timestamp * bpm / 60.0)
            left_strobes = self._map.get_by_group("strobe_left")
            right_strobes = self._map.get_by_group("strobe_right")
            active_strobes = left_strobes if beat_idx % 2 == 0 else right_strobes
            inactive_strobes = right_strobes if beat_idx % 2 == 0 else left_strobes
            for f in active_strobes:
                commands[f.fixture_id] = make_command(
                    f, STROBE_WHITE, strobe_rate=200, strobe_intensity=220,
                )
            for f in inactive_strobes:
                commands[f.fixture_id] = make_command(f, BLACK, 0.0)

        elif state.onset_type == "snare":
            # Single random fixture white flash
            par_cmds = random_scatter(
                active_pars, state, state.timestamp, STROBE_WHITE,
                density=1.0 / max(len(active_pars), 1), intensity=1.0,
            )
            commands.update(par_cmds)
            for f in self._strobes:
                commands[f.fixture_id] = make_command(f, BLACK, 0.0)

        elif state.onset_type == "clap":
            # Ad-lib scatter: random_scatter with red/orange
            par_cmds = random_scatter(
                self._pars, state, state.timestamp, DARK_ORANGE,
                density=0.4, intensity=1.0,
            )
            commands.update(par_cmds)
            for f in self._strobes:
                commands[f.fixture_id] = make_command(f, BLACK, 0.0)

        elif state.is_beat:
            # Beat pulse: chase sweep on active pars
            par_cmds = chase_lr(
                active_pars, state, state.timestamp, verse_color,
                speed=1.0, width=0.4, intensity=0.85,
            )
            commands.update(par_cmds)
            for f in self._strobes:
                commands[f.fixture_id] = make_command(f, BLACK, 0.0)
        else:
            # Between beats: energy-aware floor (binary contrast, but lifted at high energy)
            if energy > 0.6:
                between_intensity = 0.20 + (energy - 0.6) / 0.4 * 0.10  # 20-30%
            elif energy < 0.4:
                between_intensity = 0.0  # full blackout at low energy
            else:
                between_intensity = 0.05  # near-dark at mid energy
            par_cmds = wash_hold(
                active_pars, state, state.timestamp, verse_color, intensity=between_intensity,
            )
            commands.update(par_cmds)
            for f in self._strobes:
                commands[f.fixture_id] = make_command(f, BLACK, 0.0)

        # Blackout inactive pars
        for f in self._pars:
            if f.fixture_id not in commands:
                commands[f.fixture_id] = make_command(f, BLACK, 0.0)

        # LED bars: follow pars at reduced intensity during verse
        bar_cmds = wash_hold(
            self._led_bars, state, state.timestamp, verse_color, intensity=0.10,
        )
        commands.update(bar_cmds)

        # Laser off during verse
        for f in self._lasers:
            commands[f.fixture_id] = make_command(f, BLACK, special=_LASER_OFF)

        return self._merge_commands(commands)

    def _chorus_reactive(self, state: MusicState) -> list[FixtureCommand]:
        """Chorus / hook — brighter and wider than verse.

        All pars active at 35-50% base intensity. Beat-driven strobes fire
        on every beat (not just alternating). LED bars at 40%. Laser off.
        Energy-aware: high energy pushes toward full brightness.
        """
        commands: dict[int, FixtureCommand] = {}
        energy = state.energy
        bpm = max(60.0, state.bpm)

        # All pars active — chorus is wider than verse
        chorus_color = lerp_color(WARM_RED, BLOOD_RED, energy)
        base_intensity = 0.35 + energy * 0.15  # 35-50%

        # Onset reactions
        if state.onset_type == "kick":
            # Full-intensity flash on all pars
            par_cmds = wash_hold(
                self._pars, state, state.timestamp, BLOOD_RED, intensity=1.0,
            )
            commands.update(par_cmds)
            # Both strobes fire on kick
            burst = strobe_burst(self._strobes, state, state.timestamp, STROBE_WHITE)
            commands.update(burst)

        elif state.onset_type == "snare":
            # Wide scatter across all pars
            par_cmds = random_scatter(
                self._pars, state, state.timestamp, STROBE_WHITE,
                density=0.5, intensity=1.0,
            )
            commands.update(par_cmds)
            for f in self._strobes:
                commands[f.fixture_id] = make_command(f, BLACK, 0.0)

        elif state.onset_type == "clap":
            # Orange scatter across all pars
            par_cmds = random_scatter(
                self._pars, state, state.timestamp, DARK_ORANGE,
                density=0.6, intensity=1.0,
            )
            commands.update(par_cmds)
            for f in self._strobes:
                commands[f.fixture_id] = make_command(f, BLACK, 0.0)

        elif state.is_beat:
            # Beat: chase at higher intensity across all pars
            par_cmds = chase_lr(
                self._pars, state, state.timestamp, chorus_color,
                speed=1.0, width=0.5, intensity=0.90,
            )
            commands.update(par_cmds)
            # Single strobe flicker on beat
            beat_idx = int(state.timestamp * bpm / 60.0)
            left_strobes = self._map.get_by_group("strobe_left")
            right_strobes = self._map.get_by_group("strobe_right")
            active_strobes = left_strobes if beat_idx % 2 == 0 else right_strobes
            inactive_strobes = right_strobes if beat_idx % 2 == 0 else left_strobes
            for f in active_strobes:
                commands[f.fixture_id] = make_command(
                    f, STROBE_WHITE, strobe_rate=180, strobe_intensity=200,
                )
            for f in inactive_strobes:
                commands[f.fixture_id] = make_command(f, BLACK, 0.0)

        else:
            # Between beats: raised floor (chorus is hotter than verse)
            par_cmds = wash_hold(
                self._pars, state, state.timestamp, chorus_color, intensity=base_intensity,
            )
            commands.update(par_cmds)
            for f in self._strobes:
                commands[f.fixture_id] = make_command(f, BLACK, 0.0)

        # LED bars at 40% during chorus (higher than verse's 10%)
        bar_cmds = wash_hold(
            self._led_bars, state, state.timestamp, chorus_color, intensity=0.40,
        )
        commands.update(bar_cmds)

        # Laser off during chorus
        for f in self._lasers:
            commands[f.fixture_id] = make_command(f, BLACK, special=_LASER_OFF)

        return self._merge_commands(commands)
