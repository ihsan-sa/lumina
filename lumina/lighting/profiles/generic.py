"""Generic fallback profile — looks good on ANY music.

Core philosophy: NEVER UGLY, NEVER BORING, NEVER EXTREME.

This is the safety net. When genre classification is uncertain (no genre
has >0.3 weight), this profile takes over. It must produce acceptable
lighting for any genre, tempo, or mood.

Lighting language:
- Palette: Moderate blues, purples, warm whites. Safe, universally
  pleasing colors that work with any music.
- Energy-reactive: Global brightness follows the energy envelope.
  High energy = bright room. Low energy = dim room. Simple and correct.
- Beat-reactive: Subtle par pulse on kicks (brief brightness bump, not
  a flash). Gentle strobe on snares (low rate, low intensity).
- Color cycling: Smooth 8-bar color loop through blues → purple → warm
  white → back. Gradual enough that you barely notice it changing.
- Section-aware: Brighter on chorus/drop, dimmer on verse/breakdown.
  Multiplicative intensity modifier by section.
- Spatial: Slow L→R sweep on pars during verses, all-fill during
  choruses, breathing on breakdowns.
- UV: Low constant glow. Higher during drops.
- Strobes: Conservative. Only on snare hits during high-energy sections.
  Never aggressive.
"""

from __future__ import annotations

from lumina.audio.models import MusicState
from lumina.control.protocol import FixtureCommand
from lumina.lighting.fixture_map import FixtureMap, FixtureType
from lumina.lighting.profiles.base import (
    BLACK,
    BaseProfile,
    Color,
    energy_brightness,
    lerp_color,
    sine_pulse,
)

# ─── Generic palette (max RGB channel = 1.0, white at 15-40%) ───────

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

# UV levels
_UV_BASE = 60
_UV_DROP = 150

# Strobe limits (conservative)
_SNARE_STROBE_RATE = 100
_SNARE_STROBE_INTENSITY = 120


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
        self._pars = self._map.by_type(FixtureType.PAR)
        self._strobes = self._map.by_type(FixtureType.STROBE)
        self._uvs = self._map.by_type(FixtureType.UV)
        self._pars_lr = self._map.sorted_by_x(self._pars)

    def generate(self, state: MusicState) -> list[FixtureCommand]:
        """Generate generic fixture commands for any music.

        Args:
            state: Current audio analysis frame.

        Returns:
            One FixtureCommand per fixture.
        """
        segment = state.segment

        if segment in ("breakdown", "bridge"):
            return self._breathing(state)

        if segment == "drop":
            return self._drop(state)

        if segment in ("intro", "outro"):
            return self._gentle(state)

        # Verse / chorus: standard reactive
        return self._reactive(state)

    # ─── Segment handlers ──────────────────────────────────────────

    def _reactive(self, state: MusicState) -> list[FixtureCommand]:
        """Standard beat-reactive mode for verse/chorus.

        Color cycling with energy-driven brightness. Kick pulses on
        pars. Gentle strobe on snares during high energy. Sweep during
        verse, full fill during chorus.
        """
        commands: dict[int, FixtureCommand] = {}

        # Section modifier
        section_mult = _SECTION_INTENSITY.get(state.segment, 0.6)

        # Base intensity from energy (boosted curve, floor 25% verse / 50% chorus)
        eb = energy_brightness(state.energy)
        base_intensity = (0.25 + eb * 0.55) * section_mult

        # Kick pulse
        kick_boost = 0.0
        if state.onset_type == "kick":
            kick_boost = 0.2
        elif state.is_beat:
            kick_boost = 0.1

        # Color from 8-bar cycle
        color = self._cycle_color(state)

        # Pars: sweep during verse, full fill during chorus
        if state.segment == "chorus":
            # White at 30% during chorus for brightness
            chorus_color = Color(color.r, color.g, color.b, w=max(color.w, 0.30))
            for f in self._pars:
                intensity = min(1.0, base_intensity + kick_boost)
                commands[f.fixture_id] = self._cmd(f, chorus_color, intensity)
        else:
            # Slow L→R sweep
            sweep = self._sweep_x(
                state.bar_phase, color, width=0.5,
                intensity=min(1.0, base_intensity + kick_boost),
            )
            commands.update(sweep)

        # Strobes: gentle on snare hits in high-energy sections
        if state.onset_type == "snare" and state.energy > 0.5:
            for f in self._strobes:
                commands[f.fixture_id] = self._cmd(
                    f, color, strobe_rate=_SNARE_STROBE_RATE,
                    strobe_intensity=_SNARE_STROBE_INTENSITY,
                )
        elif state.is_downbeat and state.energy > 0.7:
            for f in self._strobes:
                commands[f.fixture_id] = self._cmd(
                    f, color, strobe_rate=60, strobe_intensity=80,
                )
        else:
            for f in self._strobes:
                commands[f.fixture_id] = self._cmd(f, BLACK, intensity=0.0)

        # UV
        for f in self._uvs:
            commands[f.fixture_id] = self._cmd(f, special=_UV_BASE)

        return self._merge_commands(commands)

    def _drop(self, state: MusicState) -> list[FixtureCommand]:
        """Drop: brighter version of reactive with UV boost (min 80%)."""
        commands: dict[int, FixtureCommand] = {}

        eb = energy_brightness(state.energy)
        base_intensity = max(0.80, 0.70 + eb * 0.30)
        color = self._cycle_color(state)
        # Add white at 40% during drops for extra brightness
        color = Color(color.r, color.g, color.b, w=max(color.w, 0.40))

        # Kick boost
        if state.onset_type == "kick" or state.is_beat:
            base_intensity = min(1.0, base_intensity + 0.10)

        # All pars full fill
        for f in self._pars:
            commands[f.fixture_id] = self._cmd(f, color, base_intensity)

        # Strobes on beats
        if state.is_beat:
            for f in self._strobes:
                commands[f.fixture_id] = self._cmd(
                    f, WARM_WHITE, strobe_rate=140, strobe_intensity=160,
                )
        else:
            for f in self._strobes:
                commands[f.fixture_id] = self._cmd(f, BLACK, intensity=0.0)

        # UV higher
        for f in self._uvs:
            commands[f.fixture_id] = self._cmd(f, special=_UV_DROP)

        return self._merge_commands(commands)

    def _breathing(self, state: MusicState) -> list[FixtureCommand]:
        """Breakdown/bridge: slow breathing, dim wash on two pars only."""
        commands: dict[int, FixtureCommand] = {}

        section_mult = _SECTION_INTENSITY.get(state.segment, 0.3)
        breath = sine_pulse(state.bar_phase, power=0.5)
        intensity = section_mult * (0.2 + breath * 0.3)

        color = self._cycle_color(state)

        # Only first two pars — keep the room dim
        for i, f in enumerate(self._pars):
            if i < 2:
                commands[f.fixture_id] = self._cmd(f, color, intensity)
            else:
                commands[f.fixture_id] = self._cmd(f, BLACK, intensity=0.0)

        for f in self._strobes:
            commands[f.fixture_id] = self._cmd(f, BLACK, intensity=0.0)

        for f in self._uvs:
            commands[f.fixture_id] = self._cmd(f, special=_UV_BASE)

        return self._merge_commands(commands)

    def _gentle(self, state: MusicState) -> list[FixtureCommand]:
        """Intro/outro: gentle low wash."""
        commands: dict[int, FixtureCommand] = {}

        section_mult = _SECTION_INTENSITY.get(state.segment, 0.4)
        breath = sine_pulse(state.bar_phase, power=0.5)
        intensity = section_mult * (0.5 + breath * 0.3)

        for f in self._pars:
            commands[f.fixture_id] = self._cmd(f, SOFT_BLUE, intensity)

        for f in self._strobes:
            commands[f.fixture_id] = self._cmd(f, BLACK, intensity=0.0)

        for f in self._uvs:
            commands[f.fixture_id] = self._cmd(f, special=_UV_BASE)

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
        # Full cycle every 8 bars
        cycle_pos = (state.timestamp / (bar_duration * 8.0)) % 1.0

        n = len(_CYCLE_COLORS)
        idx = int(cycle_pos * n) % n
        next_idx = (idx + 1) % n
        blend_t = (cycle_pos * n) % 1.0
        return lerp_color(_CYCLE_COLORS[idx], _CYCLE_COLORS[next_idx], blend_t)
