"""Profile 2: Psychedelic Trap / Dark R&B — Don Toliver, The Weeknd.

Core philosophy: SMOOTH AND FLOWING — transitions over bars, not beats.

Lighting language:
- Palette: Purple, cyan, magenta, hot pink, neon blue. NEVER harsh white.
- Verse: breathe on pars with per-fixture phase offset (out-of-phase
  drift) + intensity tied to vocal_energy.
- Chorus: alternate on pars (warm palette, 30% depth on kick) +
  color_split (left=magenta, right=cyan).
- Drop: diverge (smooth over 2 bars, not instant). Bloom, not explosion.
- Synth swell (rising energy_derivative): diverge timing.
- Breakdown: spotlight_isolate on 2 pars, low purple + breathe.
- No strobe_burst ever. Transitions use 2-4 bar crossfades.
- LED bars: follow par wash at 50% intensity. Full during drops.
- Laser: always off (doesn't fit smooth aesthetic).
"""

from __future__ import annotations

import math

from lumina.audio.models import MusicState
from lumina.control.protocol import FixtureCommand
from lumina.lighting.fixture_map import FixtureMap, FixtureType
from lumina.lighting.patterns import (
    alternate,
    breathe,
    color_split,
    diverge,
    make_command,
    select_active_fixtures,
    spotlight_isolate,
    wash_hold,
)
from lumina.lighting.profiles.base import (
    BLACK,
    BaseProfile,
    Color,
    energy_brightness,
    lerp_color,
    sine_pulse,
)

# ─── Psychedelic R&B palette ──────────────────────────────────────

DEEP_PURPLE = Color(0.5, 0.0, 1.0, 0.0)
NEON_CYAN = Color(0.0, 0.89, 1.0, 0.0)
HOT_MAGENTA = Color(1.0, 0.0, 0.67, 0.0)
HOT_PINK = Color(1.0, 0.1, 0.4, 0.0)
NEON_BLUE = Color(0.1, 0.2, 1.0, 0.0)
DARK_VIOLET = Color(0.5, 0.0, 1.0, 0.0)

# Wash color sets
_VERSE_COLORS = [DEEP_PURPLE, NEON_CYAN, NEON_BLUE]
_CHORUS_COLORS = [HOT_MAGENTA, HOT_PINK, DEEP_PURPLE]

# Intensity ranges
_VERSE_MIN = 0.30
_VERSE_MAX = 0.60
_CHORUS_KICK_PULSE = 0.25
_BREAKDOWN_INTENSITY = 0.12
_DROP_PEAK = 0.95

# Crossfade duration in seconds (~2-4 bars at 128bpm)
_CROSSFADE_DURATION_S = 4.0

# Laser always off
_LASER_OFF = 0


class PsychRnbProfile(BaseProfile):
    """Psychedelic Trap / Dark R&B lighting profile.

    Smooth, flowing, atmospheric. The room breathes with the music.
    No harsh whites, no instant cuts. Purple/cyan/magenta palette
    with vocal-energy-driven intensity.

    Args:
        fixture_map: Venue fixture layout.
    """

    name = "psych_rnb"

    def __init__(self, fixture_map: FixtureMap) -> None:
        super().__init__(fixture_map)
        self._pars = self._map.by_type(FixtureType.PAR)
        self._strobes = self._map.by_type(FixtureType.STROBE)
        self._led_bars = self._map.by_type(FixtureType.LED_BAR)
        self._lasers = self._map.by_type(FixtureType.LASER)
        self._pars_lr = self._map.sorted_by_x(self._pars)

        # Crossfade state
        self._last_segment: str = ""
        self._segment_transition_time: float = -1.0
        self._prev_commands: dict[int, FixtureCommand] = {}

    def generate(self, state: MusicState) -> list[FixtureCommand]:
        """Generate psychedelic R&B fixture commands.

        Args:
            state: Current audio analysis frame.

        Returns:
            One FixtureCommand per fixture (15 total).
        """
        self._begin_debug_frame()
        segment = state.segment

        # Detect segment transitions for crossfade
        if segment != self._last_segment:
            self._segment_transition_time = state.timestamp
            self._last_segment = segment

        # Generate commands for current segment
        if segment == "drop":
            self._note_patterns("drop")
            current = self._drop(state)
        elif segment in ("breakdown", "bridge"):
            self._note_patterns("breakdown")
            current = self._breakdown(state)
        elif segment in ("intro", "outro"):
            self._note_patterns("intro_outro")
            current = self._intro_outro(state)
        elif segment == "chorus":
            self._note_patterns("chorus")
            current = self._chorus(state)
        else:
            self._note_patterns("verse")
            current = self._verse(state)

        # Apply crossfade if within transition window
        if self._prev_commands and self._segment_transition_time >= 0:
            dt = state.timestamp - self._segment_transition_time
            if dt < _CROSSFADE_DURATION_S:
                t = dt / _CROSSFADE_DURATION_S
                current = self._crossfade(self._prev_commands, current, t)

        # Store for next frame's crossfade
        self._prev_commands = {cmd.fixture_id: cmd for cmd in current}

        return current

    # ─── Segment handlers ──────────────────────────────────────────

    def _verse(self, state: MusicState) -> list[FixtureCommand]:
        """Verse: breathe with per-fixture phase offset for out-of-phase drift.

        Intensity tied to vocal energy. Color drifts through purple/cyan palette.
        """
        commands: dict[int, FixtureCommand] = {}

        # Base intensity from vocal energy
        vocal = max(0.0, min(1.0, state.vocal_energy))
        eb = energy_brightness(state.energy)
        base_intensity = _VERSE_MIN + max(vocal, eb) * (_VERSE_MAX - _VERSE_MIN)

        # Fixture escalation
        active_pars = select_active_fixtures(
            self._pars, state.energy,
            low_count=3, mid_count=6, mid_threshold=0.4,
        )

        # Swell: if energy is rising, use diverge
        if state.energy_derivative > 0.05:
            swell_boost = min(0.15, state.energy_derivative * 0.5)
            swell_intensity = min(0.75, base_intensity + swell_boost)
            color = self._wash_color(state.timestamp, _VERSE_COLORS, 0.0)
            par_cmds = diverge(
                active_pars, state, state.timestamp, color, intensity=swell_intensity,
            )
            commands.update(par_cmds)
        else:
            # Per-fixture breathing with phase offset for drift
            color = self._wash_color(state.timestamp, _VERSE_COLORS, 0.0)
            par_cmds = breathe(
                active_pars, state, state.timestamp, color,
                min_intensity=base_intensity * 0.6,
                max_intensity=base_intensity,
                period_bars=8.0,
                phase_offset_per_fixture=0.15,
            )
            commands.update(par_cmds)

        # Blackout inactive pars
        for f in self._pars:
            if f.fixture_id not in commands:
                commands[f.fixture_id] = make_command(f, BLACK, 0.0)

        # Strobes: OFF (never harsh in psych_rnb)
        for f in self._strobes:
            commands[f.fixture_id] = make_command(f, BLACK, 0.0)

        # LED bars: follow par wash at 50%
        color = self._wash_color(state.timestamp, _VERSE_COLORS, 0.0)
        bar_cmds = wash_hold(
            self._led_bars, state, state.timestamp, color,
            intensity=base_intensity * 0.5,
        )
        commands.update(bar_cmds)

        # Laser always off
        for f in self._lasers:
            commands[f.fixture_id] = make_command(f, BLACK, special=_LASER_OFF)

        return self._merge_commands(commands)

    def _chorus(self, state: MusicState) -> list[FixtureCommand]:
        """Chorus: color_split (magenta/cyan) + alternate on kicks.

        Warmer palette with gentle kick pulse layered on top.
        """
        commands: dict[int, FixtureCommand] = {}

        vocal = max(0.0, min(1.0, state.vocal_energy))
        eb = energy_brightness(state.energy)
        base_intensity = max(0.50, _VERSE_MIN + max(vocal, eb) * (_VERSE_MAX - _VERSE_MIN) + 0.15)

        # Kick pulse
        kick_boost = 0.0
        if state.onset_type == "kick":
            kick_boost = _CHORUS_KICK_PULSE
        elif state.is_beat:
            kick_boost = _CHORUS_KICK_PULSE * 0.5

        # Fixture escalation
        active_pars = select_active_fixtures(
            self._pars, state.energy,
            low_count=4, mid_count=6, high_threshold=0.7,
        )

        # Color split: magenta left, cyan right
        split_cmds = color_split(
            active_pars, state, state.timestamp, HOT_MAGENTA,
            color_right=NEON_CYAN, intensity=min(1.0, base_intensity + kick_boost),
        )
        commands.update(split_cmds)

        # Layer alternate at 30% depth on kicks
        if state.is_beat:
            alt_color = self._wash_color(state.timestamp, _CHORUS_COLORS, 0.0)
            alt_cmds = alternate(
                active_pars, state, state.timestamp, alt_color,
                color_b=DEEP_PURPLE, intensity=base_intensity * 0.3,
            )
            # Blend: take max brightness to layer alternate on top
            for fid, cmd in alt_cmds.items():
                if fid in commands:
                    existing = commands[fid]
                    # Add alternate contribution on top
                    commands[fid] = FixtureCommand(
                        fixture_id=fid,
                        red=min(255, existing.red + cmd.red // 3),
                        green=min(255, existing.green + cmd.green // 3),
                        blue=min(255, existing.blue + cmd.blue // 3),
                        white=min(255, existing.white + cmd.white // 3),
                        strobe_rate=0, strobe_intensity=0,
                        special=existing.special,
                    )

        # Blackout inactive pars
        for f in self._pars:
            if f.fixture_id not in commands:
                commands[f.fixture_id] = make_command(f, BLACK, 0.0)

        # Strobes: very gentle on downbeats only, tinted (never harsh)
        if state.is_downbeat:
            for f in self._strobes:
                commands[f.fixture_id] = make_command(
                    f, HOT_MAGENTA, strobe_rate=60, strobe_intensity=80,
                )
        else:
            for f in self._strobes:
                commands[f.fixture_id] = make_command(f, BLACK, 0.0)

        # LED bars: follow color split at 60%
        bar_cmds = wash_hold(
            self._led_bars, state, state.timestamp, DEEP_PURPLE,
            intensity=base_intensity * 0.6,
        )
        commands.update(bar_cmds)

        # Laser off
        for f in self._lasers:
            commands[f.fixture_id] = make_command(f, BLACK, special=_LASER_OFF)

        return self._merge_commands(commands)

    def _drop(self, state: MusicState) -> list[FixtureCommand]:
        """Drop: slow bloom via diverge over 2 bars.

        Colors reach peak saturation. No explosion — smooth bloom.
        All pars active. LED bars full. Still no harsh white.
        """
        commands: dict[int, FixtureCommand] = {}

        # Bloom intensity
        bloom = min(1.0, state.energy * 1.2)
        intensity = _VERSE_MAX + bloom * (_DROP_PEAK - _VERSE_MAX)

        # Diverge: center out, smooth bloom
        color = self._wash_color(state.timestamp * 0.5, _CHORUS_COLORS, 0.0)
        par_cmds = diverge(
            self._pars, state, state.timestamp, color, intensity=intensity,
        )
        commands.update(par_cmds)

        # Strobes: gentle pulsing, tinted (never harsh white)
        if state.is_beat:
            for f in self._strobes:
                commands[f.fixture_id] = make_command(
                    f, HOT_PINK, strobe_rate=100, strobe_intensity=120,
                )
        else:
            for f in self._strobes:
                commands[f.fixture_id] = make_command(f, BLACK, 0.0)

        # LED bars full intensity
        bar_cmds = wash_hold(
            self._led_bars, state, state.timestamp, color, intensity=intensity,
        )
        commands.update(bar_cmds)

        # Laser off
        for f in self._lasers:
            commands[f.fixture_id] = make_command(f, BLACK, special=_LASER_OFF)

        return self._merge_commands(commands)

    def _breakdown(self, state: MusicState) -> list[FixtureCommand]:
        """Breakdown: spotlight_isolate on 2 pars + breathe. Near-darkness."""
        commands: dict[int, FixtureCommand] = {}

        # Select 2 pars for minimal atmosphere
        active_pars = self._pars[:2] if len(self._pars) >= 2 else self._pars

        # Breathing on active pars
        par_cmds = breathe(
            active_pars, state, state.timestamp, DARK_VIOLET,
            min_intensity=_BREAKDOWN_INTENSITY,
            max_intensity=_BREAKDOWN_INTENSITY + 0.08,
            period_bars=2.0,
        )
        commands.update(par_cmds)

        # All other pars off
        for f in self._pars:
            if f.fixture_id not in commands:
                commands[f.fixture_id] = make_command(f, BLACK, 0.0)

        # Strobes off
        for f in self._strobes:
            commands[f.fixture_id] = make_command(f, BLACK, 0.0)

        # LED bars very dim
        bar_cmds = wash_hold(
            self._led_bars, state, state.timestamp, DARK_VIOLET, intensity=0.05,
        )
        commands.update(bar_cmds)

        # Laser off
        for f in self._lasers:
            commands[f.fixture_id] = make_command(f, BLACK, special=_LASER_OFF)

        return self._merge_commands(commands)

    def _intro_outro(self, state: MusicState) -> list[FixtureCommand]:
        """Intro/outro: minimal purple glow, slow breathing."""
        commands: dict[int, FixtureCommand] = {}

        par_cmds = breathe(
            self._pars, state, state.timestamp, DEEP_PURPLE,
            min_intensity=0.20, max_intensity=0.30, period_bars=2.0,
            phase_offset_per_fixture=0.1,
        )
        commands.update(par_cmds)

        for f in self._strobes:
            commands[f.fixture_id] = make_command(f, BLACK, 0.0)

        bar_cmds = wash_hold(
            self._led_bars, state, state.timestamp, DEEP_PURPLE, intensity=0.15,
        )
        commands.update(bar_cmds)

        for f in self._lasers:
            commands[f.fixture_id] = make_command(f, BLACK, special=_LASER_OFF)

        return self._merge_commands(commands)

    # ─── Helpers ───────────────────────────────────────────────────

    @staticmethod
    def _wash_color(
        timestamp: float,
        palette: list[Color],
        spatial_offset: float,
    ) -> Color:
        """Compute a slowly drifting wash color from a palette.

        Cycles through the palette over ~30s. Each fixture can have a
        spatial_offset for drift across the room.

        Args:
            timestamp: Current time in seconds.
            palette: List of colors to cycle through.
            spatial_offset: Per-fixture phase offset (0.0-1.0).

        Returns:
            Blended Color at current position.
        """
        n = len(palette)
        if n == 0:
            return BLACK
        cycle_pos = ((timestamp / 30.0) + spatial_offset) % 1.0
        idx = int(cycle_pos * n) % n
        next_idx = (idx + 1) % n
        blend_t = (cycle_pos * n) % 1.0
        smooth_t = (math.sin((blend_t - 0.5) * math.pi) + 1.0) / 2.0
        return lerp_color(palette[idx], palette[next_idx], smooth_t)

    def _crossfade(
        self,
        prev: dict[int, FixtureCommand],
        current: list[FixtureCommand],
        t: float,
    ) -> list[FixtureCommand]:
        """Crossfade between previous and current commands over time t.

        Args:
            prev: Previous frame's commands by fixture ID.
            current: Current segment's commands.
            t: Crossfade progress 0.0 (all prev) to 1.0 (all current).

        Returns:
            Blended command list.
        """
        t = max(0.0, min(1.0, t))
        # Smooth the transition
        smooth_t = (math.sin((t - 0.5) * math.pi) + 1.0) / 2.0

        result: list[FixtureCommand] = []
        for cmd in current:
            if cmd.fixture_id in prev:
                p = prev[cmd.fixture_id]
                blended = FixtureCommand(
                    fixture_id=cmd.fixture_id,
                    red=int(p.red + (cmd.red - p.red) * smooth_t),
                    green=int(p.green + (cmd.green - p.green) * smooth_t),
                    blue=int(p.blue + (cmd.blue - p.blue) * smooth_t),
                    white=int(p.white + (cmd.white - p.white) * smooth_t),
                    strobe_rate=int(p.strobe_rate + (cmd.strobe_rate - p.strobe_rate) * smooth_t),
                    strobe_intensity=int(
                        p.strobe_intensity + (cmd.strobe_intensity - p.strobe_intensity) * smooth_t
                    ),
                    special=cmd.special,
                )
                result.append(blended)
            else:
                result.append(cmd)
        return result
