"""Profile 2: Psychedelic Trap / Dark R&B — Don Toliver, The Weeknd.

Core philosophy: SMOOTH AND FLOWING — transitions over bars, not beats.

Lighting language:
- Palette: Purple, cyan, magenta, hot pink, neon blue. NEVER harsh white.
  All colors are saturated but deep — neon vibes, not pastel.
- Verse: Slow color wash drifting between 2-3 colors over 8-16 bars.
  Intensity is tied to vocal energy — the room breathes with the singer.
  Low ambient glow, not darkness.
- Chorus: Add a rhythmic kick pulse at 30-40% brightness on top of the
  existing wash. Wash colors shift slightly warmer (more magenta/pink).
  Still smooth — no harsh jumps.
- Synth swells: When energy_derivative is positive and energy is rising,
  expand intensity from center outward. The room inflates.
- Drop: Slow bloom to full intensity, not an explosion. Colors saturate
  fully. UV peaks. Still smooth — even drops feel like exhales, not punches.
- Breakdown: Minimal. One or two pars at very low purple. Near-darkness
  but not total blackout — maintain the atmosphere.
- Everything flows: No instant cuts. Transitions happen over bars.
  The room should feel like it's breathing with the music.
- Spatial: Color washes drift across fixtures. Center→outward expansion
  on swells. Corner isolation on quiet vocal moments.
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
    energy_brightness,
    lerp_color,
    sine_pulse,
)

# ─── Psychedelic R&B palette (max RGB channel = 1.0 for brightness) ──

DEEP_PURPLE = Color(0.5, 0.0, 1.0, 0.0)
NEON_CYAN = Color(0.0, 0.89, 1.0, 0.0)
HOT_MAGENTA = Color(1.0, 0.0, 0.67, 0.0)
HOT_PINK = Color(1.0, 0.1, 0.4, 0.0)
NEON_BLUE = Color(0.1, 0.2, 1.0, 0.0)
DARK_VIOLET = Color(0.5, 0.0, 1.0, 0.0)

# Wash color sets (drifts between these)
_VERSE_COLORS = [DEEP_PURPLE, NEON_CYAN, NEON_BLUE]
_CHORUS_COLORS = [HOT_MAGENTA, HOT_PINK, DEEP_PURPLE]

# UV levels (UV enhances the neon feel)
_UV_VERSE = 100
_UV_CHORUS = 150
_UV_DROP = 200
_UV_BREAKDOWN = 50

# Intensity ranges (raised floors: verse 25%+, chorus 50%+, drop 80%+)
_VERSE_MIN = 0.30
_VERSE_MAX = 0.60
_CHORUS_KICK_PULSE = 0.25  # pulse on top of already-bright wash
_BREAKDOWN_INTENSITY = 0.12
_DROP_PEAK = 0.95


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
        self._uvs = self._map.by_type(FixtureType.UV)
        self._pars_lr = self._map.sorted_by_x(self._pars)

    def generate(self, state: MusicState) -> list[FixtureCommand]:
        """Generate psychedelic R&B fixture commands.

        Args:
            state: Current audio analysis frame.

        Returns:
            One FixtureCommand per fixture.
        """
        segment = state.segment

        # ── Drop: slow bloom, full saturation ────────────────────
        if segment == "drop":
            return self._drop(state)

        # ── Breakdown / bridge: near-darkness, atmosphere ────────
        if segment in ("breakdown", "bridge"):
            return self._breakdown(state)

        # ── Intro / outro: minimal purple glow ───────────────────
        if segment in ("intro", "outro"):
            return self._intro_outro(state)

        # ── Chorus: wash + kick pulse ────────────────────────────
        if segment == "chorus":
            return self._chorus(state)

        # ── Verse: slow drifting wash ────────────────────────────
        return self._verse(state)

    # ─── Segment handlers ──────────────────────────────────────────

    def _verse(self, state: MusicState) -> list[FixtureCommand]:
        """Verse: slow color wash drifting between 2-3 colors.

        Intensity tied to vocal energy. Each par gets a slightly
        different phase offset for spatial drift across the room.
        """
        commands: dict[int, FixtureCommand] = {}

        # Base intensity from vocal energy, boosted by energy curve
        vocal = max(0.0, min(1.0, state.vocal_energy))
        eb = energy_brightness(state.energy)
        base_intensity = _VERSE_MIN + max(vocal, eb) * (_VERSE_MAX - _VERSE_MIN)

        # Swell: if energy is rising, expand from center
        if state.energy_derivative > 0.05:
            swell_boost = min(0.15, state.energy_derivative * 0.5)
            base_intensity = min(0.75, base_intensity + swell_boost)
            swell_cmds = self._focus_expand(
                min(1.0, state.energy * 1.2),
                self._wash_color(state.timestamp, _VERSE_COLORS, 0.0),
                intensity=base_intensity,
            )
            commands.update(swell_cmds)
        else:
            # Slow drift: each par at slightly different hue phase
            n_pars = len(self._pars_lr)
            for i, f in enumerate(self._pars_lr):
                offset = i / max(n_pars, 1) * 0.3  # spatial offset
                color = self._wash_color(state.timestamp, _VERSE_COLORS, offset)
                commands[f.fixture_id] = self._cmd(f, color, base_intensity)

        # Strobes: OFF in verse (never harsh)
        for f in self._strobes:
            commands[f.fixture_id] = self._cmd(f, BLACK, intensity=0.0)

        # UV ambient
        for f in self._uvs:
            commands[f.fixture_id] = self._cmd(f, special=_UV_VERSE)

        return self._merge_commands(commands)

    def _chorus(self, state: MusicState) -> list[FixtureCommand]:
        """Chorus: drifting wash with kick pulse layered on top.

        Colors shift warmer (more magenta/pink). Kick adds a brief
        30-40% brightness pulse on top of the base wash.
        """
        commands: dict[int, FixtureCommand] = {}

        vocal = max(0.0, min(1.0, state.vocal_energy))
        eb = energy_brightness(state.energy)
        base_intensity = max(0.50, _VERSE_MIN + max(vocal, eb) * (_VERSE_MAX - _VERSE_MIN) + 0.15)

        # Kick pulse: smooth bump
        kick_boost = 0.0
        if state.onset_type == "kick":
            kick_boost = _CHORUS_KICK_PULSE
        elif state.is_beat:
            kick_boost = _CHORUS_KICK_PULSE * 0.5

        # Pars with spatial drift (warmer palette)
        n_pars = len(self._pars_lr)
        for i, f in enumerate(self._pars_lr):
            offset = i / max(n_pars, 1) * 0.3
            color = self._wash_color(state.timestamp, _CHORUS_COLORS, offset)
            par_intensity = min(1.0, base_intensity + kick_boost)
            commands[f.fixture_id] = self._cmd(f, color, par_intensity)

        # Strobes: very gentle on downbeats only, tinted
        if state.is_downbeat:
            for f in self._strobes:
                commands[f.fixture_id] = self._cmd(
                    f, HOT_MAGENTA, strobe_rate=60, strobe_intensity=80,
                )
        else:
            for f in self._strobes:
                commands[f.fixture_id] = self._cmd(f, BLACK, intensity=0.0)

        # UV higher in chorus
        for f in self._uvs:
            commands[f.fixture_id] = self._cmd(f, special=_UV_CHORUS)

        return self._merge_commands(commands)

    def _drop(self, state: MusicState) -> list[FixtureCommand]:
        """Drop: slow bloom to full saturation, not an explosion.

        Colors reach peak saturation. Intensity blooms smoothly.
        All pars at high intensity. UV peaks. Still no harsh white.
        """
        commands: dict[int, FixtureCommand] = {}

        # Bloom intensity: ramps up with energy
        bloom = min(1.0, state.energy * 1.2)
        intensity = _VERSE_MAX + bloom * (_DROP_PEAK - _VERSE_MAX)

        # Color: saturated magenta/purple cycle, slower than EDM
        # Each par gets a slightly different phase for spatial richness
        n_pars = len(self._pars_lr)
        for i, f in enumerate(self._pars_lr):
            offset = i / max(n_pars, 1) * 0.2
            color = self._wash_color(state.timestamp * 0.5, _CHORUS_COLORS, offset)
            commands[f.fixture_id] = self._cmd(f, color, intensity)

        # Strobes: gentle pulsing, tinted (never harsh white)
        if state.is_beat:
            for f in self._strobes:
                commands[f.fixture_id] = self._cmd(
                    f, HOT_PINK, strobe_rate=100, strobe_intensity=120,
                )
        else:
            for f in self._strobes:
                commands[f.fixture_id] = self._cmd(f, BLACK, intensity=0.0)

        # UV peak
        for f in self._uvs:
            commands[f.fixture_id] = self._cmd(f, special=_UV_DROP)

        return self._merge_commands(commands)

    def _breakdown(self, state: MusicState) -> list[FixtureCommand]:
        """Breakdown: near-darkness, one or two pars at low purple.

        Maintains atmosphere without going to total black. Slow
        breathing on bar phase.
        """
        commands: dict[int, FixtureCommand] = {}

        breath = sine_pulse(state.bar_phase, power=0.5)
        intensity = _BREAKDOWN_INTENSITY + breath * 0.08

        # Only first two pars, low purple
        for i, f in enumerate(self._pars):
            if i < 2:
                commands[f.fixture_id] = self._cmd(f, DARK_VIOLET, intensity=intensity)
            else:
                commands[f.fixture_id] = self._cmd(f, BLACK, intensity=0.0)

        # Strobes off
        for f in self._strobes:
            commands[f.fixture_id] = self._cmd(f, BLACK, intensity=0.0)

        # UV low but present
        for f in self._uvs:
            commands[f.fixture_id] = self._cmd(f, special=_UV_BREAKDOWN)

        return self._merge_commands(commands)

    def _intro_outro(self, state: MusicState) -> list[FixtureCommand]:
        """Intro/outro: minimal purple glow, slow breathing."""
        commands: dict[int, FixtureCommand] = {}

        breath = sine_pulse(state.bar_phase, power=0.5)
        intensity = 0.25 + breath * 0.10

        for f in self._pars:
            commands[f.fixture_id] = self._cmd(f, DEEP_PURPLE, intensity=intensity)

        for f in self._strobes:
            commands[f.fixture_id] = self._cmd(f, BLACK, intensity=0.0)

        for f in self._uvs:
            commands[f.fixture_id] = self._cmd(f, special=_UV_BREAKDOWN)

        return self._merge_commands(commands)

    # ─── Helpers ───────────────────────────────────────────────────

    @staticmethod
    def _wash_color(
        timestamp: float,
        palette: list[Color],
        spatial_offset: float,
    ) -> Color:
        """Compute a slowly drifting wash color from a palette.

        Cycles through the palette over ~16 bars (~30s at 128bpm).
        Each fixture can have a spatial_offset for drift across the room.

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
        # Full cycle every ~30 seconds
        cycle_pos = ((timestamp / 30.0) + spatial_offset) % 1.0
        idx = int(cycle_pos * n) % n
        next_idx = (idx + 1) % n
        blend_t = (cycle_pos * n) % 1.0
        # Smooth the blend with sine for organic feel
        smooth_t = (math.sin((blend_t - 0.5) * math.pi) + 1.0) / 2.0
        return lerp_color(palette[idx], palette[next_idx], smooth_t)
