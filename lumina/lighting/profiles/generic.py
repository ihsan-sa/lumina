"""Generic fallback profile — looks good on ANY music.

Core philosophy: NEVER UGLY, NEVER BORING, NEVER EXTREME.

This is the safety net. When genre classification is uncertain (no genre
has >0.3 weight), this profile takes over. It must produce acceptable
lighting for any genre, tempo, or mood.

Lighting language:
- Verse: wash_hold on pars (blue/purple) + gentle alternate on kicks.
- Chorus: chase_lr on pars (medium speed) + strobe on snares only.
- Drop: diverge on pars + wash_hold on overhead (full) + strobes on kicks.
- Breakdown: breathe on 2 pars (slow) + everything else dim.
- LED bars: follow pars at reduced intensity across all segments.
- Laser: off except during drops (low-speed pattern).
- Fixture count escalation: 2-4 at low energy, 6-10 medium, all 15 high.
"""

from __future__ import annotations

from lumina.audio.models import MusicState
from lumina.control.protocol import FixtureCommand
from lumina.lighting.fixture_map import FixtureMap, FixtureType
from lumina.lighting.patterns import (
    alternate,
    breathe,
    chase_lr,
    diverge,
    make_command,
    select_active_fixtures,
    strobe_burst,
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

# ─── Generic palette ──────────────────────────────────────────────

SOFT_BLUE = Color(0.12, 0.25, 1.0, 0.15)
WARM_PURPLE = Color(0.57, 0.14, 1.0, 0.15)
WARM_WHITE = Color(0.8, 0.6, 0.4, 0.50)
GENTLE_CYAN = Color(0.14, 0.71, 1.0, 0.15)

# 8-bar color cycle
_CYCLE_COLORS = [SOFT_BLUE, WARM_PURPLE, WARM_WHITE, GENTLE_CYAN]

# Section intensity multipliers
_SECTION_INTENSITY: dict[str, float] = {
    "intro": 0.4,
    "verse": 0.6,
    "chorus": 0.9,
    "drop": 1.0,
    "breakdown": 0.3,
    "bridge": 0.5,
    "outro": 0.4,
}

# Strobe limits (conservative)
_SNARE_STROBE_RATE = 100
_SNARE_STROBE_INTENSITY = 120

# Laser
_LASER_OFF = 0
_LASER_DROP = 3


class GenericProfile(BaseProfile):
    """Generic fallback lighting profile.

    Energy-reactive, beat-reactive, section-aware. Smooth color cycling
    on 8-bar loops. Never ugly, never boring, never extreme. Always
    looks acceptable regardless of genre.

    Args:
        fixture_map: Venue fixture layout.
    """

    name = "generic"

    def __init__(self, fixture_map: FixtureMap) -> None:
        super().__init__(fixture_map)
        self._bump = BumpTracker(decay_rate=8.0)
        self._pars = self._map.by_type(FixtureType.PAR)
        self._strobes = self._map.by_type(FixtureType.STROBE)
        self._led_bars = self._map.by_type(FixtureType.LED_BAR)
        self._lasers = self._map.by_type(FixtureType.LASER)
        self._pars_lr = self._map.sorted_by_x(self._pars)

    def generate(self, state: MusicState) -> list[FixtureCommand]:
        """Generate generic fixture commands for any music.

        Args:
            state: Current audio analysis frame.

        Returns:
            One FixtureCommand per fixture (15 total).
        """
        self._begin_debug_frame()
        segment = state.segment

        if segment in ("breakdown", "bridge"):
            self._note_patterns("breathing")
            return self._breathing(state)

        if segment == "drop":
            self._note_patterns("drop")
            return self._drop(state)

        if segment in ("intro", "outro"):
            self._note_patterns("gentle")
            return self._gentle(state)

        self._note_patterns("reactive")
        return self._reactive(state)

    # ─── Segment handlers ──────────────────────────────────────────

    def _reactive(self, state: MusicState) -> list[FixtureCommand]:
        """Standard beat-reactive mode for verse/chorus.

        Verse: wash_hold + gentle alternate on kicks.
        Chorus: chase_lr at medium speed + strobe on snares.
        """
        commands: dict[int, FixtureCommand] = {}

        section_mult = _SECTION_INTENSITY.get(state.segment, 0.6)
        eb = energy_brightness(state.energy)
        base_intensity = (0.25 + eb * 0.55) * section_mult

        # Sub-bass intensity boost on PARs
        if state.sub_bass_energy > 0.4:
            base_intensity = min(1.0, base_intensity + state.sub_bass_energy * 0.1)

        # Kick bump: trigger on kick, read decay every frame
        if state.onset_type == "kick":
            self._bump.trigger("pars", state.timestamp)
        kick_boost = self._bump.get_intensity(
            "pars", state.timestamp, peak=0.2, floor=0.0,
        )

        color = self._cycle_color(state)
        color = self._color_temperature(state.spectral_centroid, color, GENTLE_CYAN)

        # Fixture escalation
        active_pars = select_active_fixtures(
            self._pars, state.energy,
            low_count=3, mid_count=6, high_threshold=0.7,
        )

        if state.segment == "chorus":
            # Chase L→R at medium speed
            chorus_color = Color(color.r, color.g, color.b, w=max(color.w, 0.30))
            par_cmds = chase_lr(
                active_pars, state, state.timestamp, chorus_color,
                speed=1.0, width=0.4, intensity=min(1.0, base_intensity + kick_boost),
            )
            commands.update(par_cmds)
        else:
            # Verse: wash_hold + gentle alternate on kicks
            if state.onset_type == "kick":
                par_cmds = alternate(
                    active_pars, state, state.timestamp, color,
                    color_b=color.scaled(0.3),
                    intensity=min(1.0, base_intensity + kick_boost),
                )
            else:
                par_cmds = wash_hold(
                    active_pars, state, state.timestamp, color,
                    intensity=base_intensity,
                )
            commands.update(par_cmds)

        # Blackout inactive pars
        for f in self._pars:
            if f.fixture_id not in commands:
                commands[f.fixture_id] = make_command(f, BLACK, 0.0)

        # Strobes: gentle on snare hits in high-energy sections
        if state.onset_type == "snare" and state.energy > 0.5:
            for f in self._strobes:
                commands[f.fixture_id] = make_command(
                    f, color, strobe_rate=_SNARE_STROBE_RATE,
                    strobe_intensity=_SNARE_STROBE_INTENSITY,
                )
        elif state.is_downbeat and state.energy > 0.7:
            for f in self._strobes:
                commands[f.fixture_id] = make_command(
                    f, color, strobe_rate=60, strobe_intensity=80,
                )
        else:
            for f in self._strobes:
                commands[f.fixture_id] = make_command(f, BLACK, 0.0)

        # LED bars: follow pars at reduced intensity
        bar_cmds = wash_hold(
            self._led_bars, state, state.timestamp, color,
            intensity=base_intensity * 0.5,
        )
        commands.update(bar_cmds)

        # Laser off
        for f in self._lasers:
            commands[f.fixture_id] = make_command(f, BLACK, special=_LASER_OFF)

        return self._merge_commands(commands)

    def _drop(self, state: MusicState) -> list[FixtureCommand]:
        """Drop: diverge on pars + overhead full + strobes on kicks."""
        commands: dict[int, FixtureCommand] = {}

        eb = energy_brightness(state.energy)
        base_intensity = max(0.80, 0.70 + eb * 0.30)
        color = self._cycle_color(state)
        color = Color(color.r, color.g, color.b, w=max(color.w, 0.40))

        if state.onset_type == "kick" or state.is_beat:
            base_intensity = min(1.0, base_intensity + 0.10)

        # Diverge on all pars (center-out bloom)
        par_cmds = diverge(
            self._pars, state, state.timestamp, color, intensity=base_intensity,
        )
        commands.update(par_cmds)

        # Strobes on beats
        if state.is_beat:
            burst = strobe_burst(
                self._strobes, state, state.timestamp, WARM_WHITE,
                rate=140, burst_intensity=160,
            )
            commands.update(burst)
        else:
            for f in self._strobes:
                commands[f.fixture_id] = make_command(f, BLACK, 0.0)

        # LED bars: full intensity during drops
        bar_cmds = wash_hold(
            self._led_bars, state, state.timestamp, color, intensity=base_intensity,
        )
        commands.update(bar_cmds)

        # Laser: low-speed pattern during drops
        for f in self._lasers:
            commands[f.fixture_id] = make_command(f, BLACK, special=_LASER_DROP)

        return self._merge_commands(commands)

    def _breathing(self, state: MusicState) -> list[FixtureCommand]:
        """Breakdown/bridge: slow breathing on 2 pars, everything else dim."""
        commands: dict[int, FixtureCommand] = {}

        color = self._cycle_color(state)

        # Only 2 pars breathing
        active = self._pars[:2] if len(self._pars) >= 2 else self._pars
        par_cmds = breathe(
            active, state, state.timestamp, color,
            min_intensity=0.10, max_intensity=0.25, period_bars=2.0,
        )
        commands.update(par_cmds)

        # Other pars off
        for f in self._pars:
            if f.fixture_id not in commands:
                commands[f.fixture_id] = make_command(f, BLACK, 0.0)

        # Everything else off
        for f in self._strobes:
            commands[f.fixture_id] = make_command(f, BLACK, 0.0)

        # LED bars very dim
        bar_cmds = wash_hold(
            self._led_bars, state, state.timestamp, color, intensity=0.05,
        )
        commands.update(bar_cmds)

        # Laser off
        for f in self._lasers:
            commands[f.fixture_id] = make_command(f, BLACK, special=_LASER_OFF)

        return self._merge_commands(commands)

    def _gentle(self, state: MusicState) -> list[FixtureCommand]:
        """Intro/outro: gentle low breathe on all pars."""
        commands: dict[int, FixtureCommand] = {}

        par_cmds = breathe(
            self._pars, state, state.timestamp, SOFT_BLUE,
            min_intensity=0.20, max_intensity=0.35, period_bars=2.0,
        )
        commands.update(par_cmds)

        for f in self._strobes:
            commands[f.fixture_id] = make_command(f, BLACK, 0.0)

        bar_cmds = wash_hold(
            self._led_bars, state, state.timestamp, SOFT_BLUE, intensity=0.15,
        )
        commands.update(bar_cmds)

        for f in self._lasers:
            commands[f.fixture_id] = make_command(f, BLACK, special=_LASER_OFF)

        return self._merge_commands(commands)

    # ─── Helpers ───────────────────────────────────────────────────

    @staticmethod
    def _cycle_color(state: MusicState) -> Color:
        """Smooth 8-bar color cycle through the palette.

        Args:
            state: Current music state (uses timestamp and bpm).

        Returns:
            Blended color at current position in the cycle.
        """
        bpm = max(60.0, state.bpm)
        bar_duration = 60.0 / bpm * 4.0
        cycle_pos = (state.timestamp / (bar_duration * 8.0)) % 1.0

        n = len(_CYCLE_COLORS)
        idx = int(cycle_pos * n) % n
        next_idx = (idx + 1) % n
        blend_t = (cycle_pos * n) % 1.0
        return lerp_color(_CYCLE_COLORS[idx], _CYCLE_COLORS[next_idx], blend_t)
