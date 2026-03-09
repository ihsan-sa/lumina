"""Profile 7: Festival EDM / Electro House / Trance — Guetta, Armin, Edward Maya.

Core philosophy: THE BUILD-DROP CYCLE — everything serves tension or release.

Lighting language:
- Build-up: chase_lr on pars (slow→fast, blue→cyan→white) + stutter on
  1 strobe (8th→16th→32nd over 16 bars) + converge as build peaks.
- Drop: diverge(pars, full white, 300ms) → chase_lr(pars, fast, rapid
  color cycling) + strobe_burst on every kick + overhead bars full white.
- Groove: alternate(pars, 2 colors on beat) + chase_lr(overhead, slow
  4-bar cycle).
- Breakdown: spotlight_isolate(1 par, blue) + breathe(4-bar period).
  Everything else off.
- LED bars: full white during drops, warm wash during groove.
- Laser: slow pattern in build, fast during drop, off in breakdown.
"""

from __future__ import annotations

from lumina.audio.models import MusicState
from lumina.control.protocol import FixtureCommand
from lumina.lighting.fixture_map import FixtureMap, FixtureType
from lumina.lighting.patterns import (
    alternate,
    breathe,
    chase_lr,
    converge,
    diverge,
    make_command,
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
    color_from_hsv,
    energy_brightness,
    lerp_color,
    sine_pulse,
)

# ─── Festival palette ────────────────────────────────────────────────

ICE_BLUE = Color(0.0, 0.3, 1.0, 0.1)
CYAN = Color(0.0, 0.8, 1.0, 0.1)
DEEP_BLUE = Color(0.0, 0.0, 1.0, 0.0)
GOLD = Color(1.0, 0.75, 0.0, 0.4)
HOT_WHITE = Color(1.0, 0.9, 0.7, 1.0)
AMBER = Color(1.0, 0.5, 0.0, 0.2)
BREAKDOWN_BLUE = Color(0.0, 0.25, 1.0, 0.0)

# Groove palette (warm club colors, 4-bar cycle)
_GROOVE_COLORS = [
    Color(1.0, 0.2, 0.0, 0.35),   # warm red-orange
    Color(1.0, 0.67, 0.0, 0.40),  # amber-gold
    Color(0.0, 0.5, 1.0, 0.30),   # electric blue
    Color(0.6, 0.0, 1.0, 0.30),   # purple
]

# Timing
_BUILD_BARS_SHORT = 16

# Laser patterns
_LASER_OFF = 0
_LASER_SLOW = 2
_LASER_FAST = 8


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
        self._led_bars = self._map.by_type(FixtureType.LED_BAR)
        self._lasers = self._map.by_type(FixtureType.LASER)
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
            One FixtureCommand per fixture (15 total).
        """
        segment = state.segment

        if segment != self._last_segment:
            if segment == "drop":
                self._drop_frame = 0
            self._last_segment = segment

        if segment == "drop":
            self._drop_frame += 1

        # Pre-drop build
        if state.drop_probability > 0.4 and segment != "drop":
            self._build_start_time = (
                state.timestamp
                if self._build_start_time < 0
                else self._build_start_time
            )
            return self._build(state)

        if state.drop_probability <= 0.4 and segment != "drop":
            self._build_start_time = -1.0

        if segment == "drop":
            return self._drop(state)

        if segment in ("breakdown", "bridge"):
            return self._breakdown(state)

        if segment in ("intro", "outro"):
            return self._intro_outro(state)

        return self._groove(state)

    # ─── Segment handlers ──────────────────────────────────────────

    def _build(self, state: MusicState) -> list[FixtureCommand]:
        """Build-up: chase_lr (slow→fast) + stutter on strobe + converge.

        Colors shift blue→cyan→white. Fixture count escalates. Strobe
        stutter accelerates from 8th notes to 32nd notes over build.
        """
        commands: dict[int, FixtureCommand] = {}

        bpm = max(60.0, state.bpm)
        bar_duration = 60.0 / bpm * 4.0
        dt = max(0.0, state.timestamp - self._build_start_time)
        ramp = min(1.0, dt / (_BUILD_BARS_SHORT * bar_duration))

        # Color: blue → cyan → white
        color = lerp_color(ICE_BLUE, HOT_WHITE, ramp)

        # Fixture escalation: start with few pars, expand to all
        active_pars = select_active_fixtures(
            self._pars, ramp,
            low_count=2, mid_count=5, mid_threshold=0.3, high_threshold=0.7,
        )

        # Chase with increasing speed
        chase_speed = 0.5 + ramp * 2.0
        par_cmds = chase_lr(
            active_pars, state, state.timestamp, color,
            speed=chase_speed, width=0.35, intensity=0.30 + ramp * 0.70,
        )
        commands.update(par_cmds)

        # At high ramp, add converge overlay
        if ramp > 0.6:
            converge_cmds = converge(
                active_pars, state, state.timestamp, color,
                intensity=ramp * 0.5,
            )
            for fid, cmd in converge_cmds.items():
                if fid in commands:
                    existing = commands[fid]
                    if cmd.red + cmd.green + cmd.blue > existing.red + existing.green + existing.blue:
                        commands[fid] = cmd
                else:
                    commands[fid] = cmd

        # Blackout inactive pars
        for f in self._pars:
            if f.fixture_id not in commands:
                commands[f.fixture_id] = make_command(f, BLACK, 0.0)

        # Strobes: stutter with accelerating rate
        stutter_rate = 2.0 + ramp * 6.0
        if self._strobes:
            strobe_cmds = stutter(
                self._strobes[:1], state, state.timestamp, color,
                rate=stutter_rate, intensity=ramp,
            )
            commands.update(strobe_cmds)
            for f in self._strobes[1:]:
                if ramp < 0.7:
                    commands[f.fixture_id] = make_command(f, BLACK, 0.0)
                else:
                    commands.update(
                        stutter([f], state, state.timestamp, color,
                                rate=stutter_rate, intensity=ramp)
                    )

        # LED bars: activate at ~50% ramp
        if ramp > 0.5:
            bar_intensity = (ramp - 0.5) * 2.0
            bar_cmds = wash_hold(
                self._led_bars, state, state.timestamp, color,
                intensity=bar_intensity * 0.6,
            )
            commands.update(bar_cmds)
        else:
            for f in self._led_bars:
                commands[f.fixture_id] = make_command(f, BLACK, 0.0)

        # Laser: activate in final 25%
        for f in self._lasers:
            sp = _LASER_SLOW if ramp > 0.75 else _LASER_OFF
            commands[f.fixture_id] = make_command(f, BLACK, special=sp)

        return self._merge_commands(commands)

    def _drop(self, state: MusicState) -> list[FixtureCommand]:
        """Drop: diverge flash → chase_lr fast + strobe_burst on kicks.

        First 300ms: diverge with full white. Then rapid color cycling
        chase on pars + strobe_burst on every kick + overhead bars full.
        """
        commands: dict[int, FixtureCommand] = {}

        is_trance = state.energy < 0.75

        # Initial flash (300ms ≈ 18 frames)
        if self._drop_frame <= 18:
            flash_color = GOLD if is_trance else HOT_WHITE
            par_cmds = diverge(
                self._pars, state, state.timestamp, flash_color, intensity=1.0,
            )
            commands.update(par_cmds)
            burst = strobe_burst(
                self._strobes, state, state.timestamp, HOT_WHITE,
            )
            commands.update(burst)
            bar_cmds = wash_hold(
                self._led_bars, state, state.timestamp, flash_color, intensity=1.0,
            )
            commands.update(bar_cmds)
            for f in self._lasers:
                commands[f.fixture_id] = make_command(f, BLACK, special=_LASER_FAST)
            return self._merge_commands(commands)

        if is_trance:
            par_cmds = diverge(
                self._pars, state, state.timestamp, GOLD, intensity=0.9,
            )
            commands.update(par_cmds)
        else:
            # Rapid color cycling per par
            cycle_speed = state.timestamp * 4.0
            n_pars = len(self._pars_lr)
            for i, f in enumerate(self._pars_lr):
                hue_offset = i / max(n_pars, 1)
                hue = (cycle_speed + hue_offset) % 1.0
                color = color_from_hsv(hue, 1.0, 1.0)
                color = Color(color.r, color.g, color.b, w=0.40)
                commands[f.fixture_id] = make_command(f, color, intensity=1.0)

        # Strobes: burst on kicks/beats
        if state.onset_type == "kick" or state.is_beat:
            burst = strobe_burst(
                self._strobes, state, state.timestamp, HOT_WHITE,
            )
            commands.update(burst)
        else:
            for f in self._strobes:
                commands[f.fixture_id] = make_command(f, BLACK, 0.0)

        # LED bars full white
        bar_cmds = wash_hold(
            self._led_bars, state, state.timestamp, HOT_WHITE, intensity=1.0,
        )
        commands.update(bar_cmds)

        # Laser fast
        for f in self._lasers:
            commands[f.fixture_id] = make_command(f, BLACK, special=_LASER_FAST)

        return self._merge_commands(commands)

    def _groove(self, state: MusicState) -> list[FixtureCommand]:
        """Groove: alternate on pars + chase on overhead bars."""
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
        groove_color_b = lerp_color(
            _GROOVE_COLORS[(color_idx + 2) % n_colors],
            _GROOVE_COLORS[(next_idx + 2) % n_colors], blend_t,
        )

        base_intensity = 0.50 + energy_brightness(state.energy) * 0.35
        if state.onset_type == "kick":
            base_intensity = min(1.0, base_intensity + 0.15)

        # Fixture escalation
        active_pars = select_active_fixtures(
            self._pars, state.energy,
            low_count=4, mid_count=6, high_threshold=0.8,
        )

        par_cmds = alternate(
            active_pars, state, state.timestamp, groove_color,
            color_b=groove_color_b, intensity=base_intensity,
        )
        commands.update(par_cmds)

        for f in self._pars:
            if f.fixture_id not in commands:
                commands[f.fixture_id] = make_command(f, BLACK, 0.0)

        # LED bars: slow chase
        bar_cmds = chase_lr(
            self._led_bars, state, state.timestamp, groove_color,
            speed=0.25, width=0.6, intensity=0.6,
        )
        commands.update(bar_cmds)

        # Strobes: pulse on downbeats
        if state.is_downbeat:
            burst = strobe_burst(
                self._strobes, state, state.timestamp, groove_color,
                rate=180, burst_intensity=200,
            )
            commands.update(burst)
        elif state.is_beat:
            for f in self._strobes:
                commands[f.fixture_id] = make_command(
                    f, groove_color, strobe_rate=80, strobe_intensity=100,
                )
        else:
            for f in self._strobes:
                commands[f.fixture_id] = make_command(f, BLACK, 0.0)

        # Laser off
        for f in self._lasers:
            commands[f.fixture_id] = make_command(f, BLACK, special=_LASER_OFF)

        return self._merge_commands(commands)

    def _breakdown(self, state: MusicState) -> list[FixtureCommand]:
        """Breakdown: spotlight on 1 par with 4-bar breathing."""
        commands: dict[int, FixtureCommand] = {}

        # All pars off
        par_cmds = spotlight_isolate(
            self._pars, state, state.timestamp, BREAKDOWN_BLUE,
            target_index=0, intensity=0.0, dim_others=0.0,
        )
        commands.update(par_cmds)

        # Override first par with breathing
        if self._pars:
            breath_cmds = breathe(
                self._pars[:1], state, state.timestamp, BREAKDOWN_BLUE,
                min_intensity=0.10, max_intensity=0.22, period_bars=4.0,
            )
            commands.update(breath_cmds)

        for f in self._strobes + self._led_bars:
            commands[f.fixture_id] = make_command(f, BLACK, 0.0)
        for f in self._lasers:
            commands[f.fixture_id] = make_command(f, BLACK, special=_LASER_OFF)

        return self._merge_commands(commands)

    def _intro_outro(self, state: MusicState) -> list[FixtureCommand]:
        """Intro/outro: slow blue breathe on pars."""
        commands: dict[int, FixtureCommand] = {}

        par_cmds = breathe(
            self._pars, state, state.timestamp, DEEP_BLUE,
            min_intensity=0.20, max_intensity=0.35, period_bars=2.0,
        )
        commands.update(par_cmds)

        for f in self._strobes + self._led_bars:
            commands[f.fixture_id] = make_command(f, BLACK, 0.0)
        for f in self._lasers:
            commands[f.fixture_id] = make_command(f, BLACK, special=_LASER_OFF)

        return self._merge_commands(commands)
