"""Profile 7: Festival EDM / Electro House / Trance — Guetta, Armin, Edward Maya.

Core philosophy: THE BUILD-DROP CYCLE — everything serves tension or release.

Lighting language:
- Build-up: Monochromatic start, intensity rises linearly over 16-32 bars.
  Colors shift cool (blue/cyan) to warm (gold/white). Strobe frequency
  increases from off to rapid. Pars fade in one by one.
- Drop: FULL EXPLOSION. Every fixture at maximum. Rapid color cycling
  across pars, all strobes firing at max. UV at full. The room turns into
  a wall of light for 4-8 bars.
- Groove: Rhythmic patterns locked to kick. 4-bar color cycles through a
  warm palette. Pars sweep left-to-right on beat phase. Strobes pulse on
  downbeats. Chase patterns during sustained grooves.
- Breakdown: Near-blackout. Single color wash (blue or cyan) on one par,
  slow breathing synced to bar. Everything else dark.
- Trance variant: Longer builds (32-64 bars), emotional release instead
  of violent explosion. Warm white/gold on drop instead of rapid cycling.
  Smooth expansion center-to-outward.
- Spatial: Pars sweep spatially on beat, strobes alternate L/R during
  builds, chase patterns during grooves.
"""

from __future__ import annotations

import math

from lumina.audio.models import MusicState
from lumina.control.protocol import FixtureCommand
from lumina.lighting.fixture_map import FixtureMap, FixtureType
from lumina.lighting.profiles.base import (
    BLACK,
    BaseProfile,
    Color,
    color_from_hsv,
    energy_brightness,
    lerp_color,
    sine_pulse,
)

# ─── Festival palette ────────────────────────────────────────────────

# Cool end (build starts here)
ICE_BLUE = Color(0.0, 0.3, 1.0, 0.1)
CYAN = Color(0.0, 0.8, 1.0, 0.1)
DEEP_BLUE = Color(0.0, 0.0, 1.0, 0.0)

# Warm end (build peaks here)
GOLD = Color(1.0, 0.75, 0.0, 0.4)
HOT_WHITE = Color(1.0, 0.9, 0.7, 1.0)
AMBER = Color(1.0, 0.5, 0.0, 0.2)

# Drop cycling palette (high saturation, full value)
_DROP_HUES = [0.0, 0.08, 0.15, 0.55, 0.65, 0.75, 0.85]

# Groove palette (warm club colors, 4-bar cycle — white at 30-40%)
_GROOVE_COLORS = [
    Color(1.0, 0.2, 0.0, 0.35),   # warm red-orange
    Color(1.0, 0.67, 0.0, 0.40),  # amber-gold
    Color(0.0, 0.5, 1.0, 0.30),   # electric blue
    Color(0.6, 0.0, 1.0, 0.30),   # purple
]

# Breakdown (max channel at 1.0 for blue)
BREAKDOWN_BLUE = Color(0.0, 0.25, 1.0, 0.0)

# UV levels
_UV_BUILD_BASE = 40
_UV_DROP = 255
_UV_GROOVE = 120
_UV_BREAKDOWN = 30

# Timing
_BUILD_BARS_SHORT = 16  # normal EDM build
_BUILD_BARS_LONG = 32   # trance build


class FestivalEdmProfile(BaseProfile):
    """Festival EDM / Electro House / Trance lighting profile.

    Everything revolves around the build-drop cycle. Builds are
    slow-burning tension ramps. Drops are maximum sensory overload.
    Grooves are locked rhythmic patterns. Breakdowns are silence.

    Args:
        fixture_map: Venue fixture layout.
    """

    name = "festival_edm"

    def __init__(self, fixture_map: FixtureMap) -> None:
        super().__init__(fixture_map)
        self._pars = self._map.by_type(FixtureType.PAR)
        self._strobes = self._map.by_type(FixtureType.STROBE)
        self._uvs = self._map.by_type(FixtureType.UV)
        self._pars_lr = self._map.sorted_by_x(self._pars)

        # State
        self._build_start_time: float = -1.0
        self._last_segment: str = ""
        self._drop_frame: int = 0

    def generate(self, state: MusicState) -> list[FixtureCommand]:
        """Generate festival EDM fixture commands.

        Args:
            state: Current audio analysis frame.

        Returns:
            One FixtureCommand per fixture.
        """
        segment = state.segment

        # Track segment transitions
        if segment != self._last_segment:
            if segment == "drop":
                self._drop_frame = 0
            self._last_segment = segment

        if segment == "drop":
            self._drop_frame += 1

        # ── Pre-drop build: rising tension ──────────────────────
        if state.drop_probability > 0.4 and segment != "drop":
            self._build_start_time = (
                state.timestamp
                if self._build_start_time < 0
                else self._build_start_time
            )
            return self._build(state)

        # Reset build tracking when not building
        if state.drop_probability <= 0.4 and segment != "drop":
            self._build_start_time = -1.0

        # ── Drop: full explosion ────────────────────────────────
        if segment == "drop":
            return self._drop(state)

        # ── Breakdown / bridge: near-blackout breathing ─────────
        if segment in ("breakdown", "bridge"):
            return self._breakdown(state)

        # ── Intro / outro: minimal ──────────────────────────────
        if segment in ("intro", "outro"):
            return self._intro_outro(state)

        # ── Chorus / verse: groove patterns ─────────────────────
        return self._groove(state)

    # ─── Segment handlers ──────────────────────────────────────────

    def _build(self, state: MusicState) -> list[FixtureCommand]:
        """Build-up: monochromatic start → intensity and strobe ramp.

        Colors shift cool→warm. Pars fade in sequentially. Strobe
        frequency rises from 0 to rapid. Build duration adapts:
        16 bars for EDM, up to 32 for trance-leaning tracks.
        """
        commands: dict[int, FixtureCommand] = {}

        # Compute build progress 0→1
        bpm = max(60.0, state.bpm)
        bar_duration = 60.0 / bpm * 4.0
        dt = max(0.0, state.timestamp - self._build_start_time)
        build_bars = _BUILD_BARS_SHORT
        ramp = min(1.0, dt / (build_bars * bar_duration))

        # Color: cool blue → warm gold
        color = lerp_color(ICE_BLUE, GOLD, ramp)

        # Intensity: 0.30 → 1.0
        intensity = 0.30 + ramp * 0.70

        # Pars: fade in one by one over the build
        n_pars = len(self._pars_lr)
        active_count = max(1, math.ceil(ramp * n_pars))
        for i, f in enumerate(self._pars_lr):
            if i < active_count:
                # Earlier pars slightly brighter
                par_intensity = intensity * (1.0 - 0.1 * (i / max(n_pars - 1, 1)))
                commands[f.fixture_id] = self._cmd(f, color, par_intensity)
            else:
                commands[f.fixture_id] = self._cmd(f, BLACK, intensity=0.0)

        # Strobes: rate ramps from off → rapid, alternating L/R below 70%
        strobe_rate = int(ramp * 220)
        strobe_int = int(ramp * 200)
        if ramp < 0.7:
            left_on = state.beat_phase < 0.5
            for i, f in enumerate(self._strobes):
                if (i % 2 == 0) == left_on:
                    commands[f.fixture_id] = self._cmd(
                        f, color, strobe_rate=strobe_rate, strobe_intensity=strobe_int,
                    )
                else:
                    commands[f.fixture_id] = self._cmd(f, BLACK, intensity=0.0)
        else:
            # All synced at high tension
            for f in self._strobes:
                commands[f.fixture_id] = self._cmd(
                    f, color, strobe_rate=strobe_rate, strobe_intensity=strobe_int,
                )

        # UV: ramp from base to medium
        uv_level = int(_UV_BUILD_BASE + ramp * (_UV_GROOVE - _UV_BUILD_BASE))
        for f in self._uvs:
            commands[f.fixture_id] = self._cmd(f, special=uv_level)

        return self._merge_commands(commands)

    def _drop(self, state: MusicState) -> list[FixtureCommand]:
        """Drop: FULL EXPLOSION — rapid color cycling, all strobes max.

        First 300ms: warm white flash on everything. Then rapid hue
        cycling across pars with all strobes firing continuously.
        Trance variant uses sustained warm white/gold instead of cycling.
        """
        commands: dict[int, FixtureCommand] = {}

        # Detect trance-leaning: lower energy drops are more trance
        is_trance = state.energy < 0.75

        # Initial flash (300ms ≈ 18 frames at 60fps)
        if self._drop_frame <= 18:
            flash_color = GOLD if is_trance else HOT_WHITE
            for f in self._pars:
                commands[f.fixture_id] = self._cmd(f, flash_color, intensity=1.0)
            for f in self._strobes:
                commands[f.fixture_id] = self._cmd(
                    f, HOT_WHITE, strobe_rate=255, strobe_intensity=255,
                )
            for f in self._uvs:
                commands[f.fixture_id] = self._cmd(f, special=_UV_DROP)
            return self._merge_commands(commands)

        if is_trance:
            # Trance: sustained warm glow expanding center→outward
            expand_phase = min(1.0, (self._drop_frame - 18) / 60.0)
            expand = self._focus_expand(expand_phase, GOLD, intensity=0.9)
            commands.update(expand)
        else:
            # EDM: rapid color cycling — each par gets a different hue
            # White at 40% adds perceived brightness during drops
            cycle_speed = state.timestamp * 4.0  # 4 full cycles per second
            n_pars = len(self._pars_lr)
            for i, f in enumerate(self._pars_lr):
                hue_offset = i / max(n_pars, 1)
                hue = (cycle_speed + hue_offset) % 1.0
                color = color_from_hsv(hue, 1.0, 1.0)
                color = Color(color.r, color.g, color.b, w=0.40)
                commands[f.fixture_id] = self._cmd(f, color, intensity=1.0)

        # Strobes: continuous max
        for f in self._strobes:
            commands[f.fixture_id] = self._cmd(
                f, HOT_WHITE, strobe_rate=255, strobe_intensity=255,
            )

        # UV max
        for f in self._uvs:
            commands[f.fixture_id] = self._cmd(f, special=_UV_DROP)

        return self._merge_commands(commands)

    def _groove(self, state: MusicState) -> list[FixtureCommand]:
        """Groove: kick-locked rhythm, 4-bar color cycle, L→R par sweep.

        Pars sweep left-to-right on beat phase. Colors cycle through
        warm palette every 4 bars. Strobes pulse on downbeats.
        """
        commands: dict[int, FixtureCommand] = {}

        bpm = max(60.0, state.bpm)
        bar_duration = 60.0 / bpm * 4.0

        # 4-bar color cycle
        bar_index = state.timestamp / bar_duration
        cycle_pos = (bar_index / 4.0) % 1.0
        n_colors = len(_GROOVE_COLORS)
        color_idx = int(cycle_pos * n_colors) % n_colors
        next_idx = (color_idx + 1) % n_colors
        blend_t = (cycle_pos * n_colors) % 1.0
        groove_color = lerp_color(_GROOVE_COLORS[color_idx], _GROOVE_COLORS[next_idx], blend_t)

        # Pars: sweep L→R on beat phase (min 50% in groove)
        base_intensity = 0.50 + energy_brightness(state.energy) * 0.35
        # Kick pulse: brief brightness boost
        if state.onset_type == "kick":
            base_intensity = min(1.0, base_intensity + 0.15)

        sweep = self._sweep_x(
            state.beat_phase, groove_color, width=0.4, intensity=base_intensity,
        )
        commands.update(sweep)

        # Strobes: pulse on downbeats only
        if state.is_downbeat:
            for f in self._strobes:
                commands[f.fixture_id] = self._cmd(
                    f, groove_color, strobe_rate=180, strobe_intensity=200,
                )
        elif state.is_beat:
            for f in self._strobes:
                commands[f.fixture_id] = self._cmd(
                    f, groove_color, strobe_rate=80, strobe_intensity=100,
                )
        else:
            for f in self._strobes:
                commands[f.fixture_id] = self._cmd(f, BLACK, intensity=0.0)

        # UV steady
        for f in self._uvs:
            commands[f.fixture_id] = self._cmd(f, special=_UV_GROOVE)

        return self._merge_commands(commands)

    def _breakdown(self, state: MusicState) -> list[FixtureCommand]:
        """Breakdown: near-blackout, single par breathing a cool wash.

        One par (front-left) breathes a blue wash synced to bar phase.
        Everything else is dark. Creates anticipation for the next build.
        """
        commands: dict[int, FixtureCommand] = {}

        # Breathing intensity on bar phase (10-20% range)
        breath = sine_pulse(state.bar_phase)
        intensity = 0.10 + breath * 0.12

        # Only the first par
        for i, f in enumerate(self._pars):
            if i == 0:
                commands[f.fixture_id] = self._cmd(f, BREAKDOWN_BLUE, intensity=intensity)
            else:
                commands[f.fixture_id] = self._cmd(f, BLACK, intensity=0.0)

        # Strobes off
        for f in self._strobes:
            commands[f.fixture_id] = self._cmd(f, BLACK, intensity=0.0)

        # UV very low
        for f in self._uvs:
            commands[f.fixture_id] = self._cmd(f, special=_UV_BREAKDOWN)

        return self._merge_commands(commands)

    def _intro_outro(self, state: MusicState) -> list[FixtureCommand]:
        """Intro/outro: slow blue fade on pars, no strobes."""
        commands: dict[int, FixtureCommand] = {}

        breath = sine_pulse(state.bar_phase, power=0.5)
        intensity = 0.25 + breath * 0.15

        for f in self._pars:
            commands[f.fixture_id] = self._cmd(f, DEEP_BLUE, intensity=intensity)

        for f in self._strobes:
            commands[f.fixture_id] = self._cmd(f, BLACK, intensity=0.0)

        for f in self._uvs:
            commands[f.fixture_id] = self._cmd(f, special=_UV_BREAKDOWN)

        return self._merge_commands(commands)
