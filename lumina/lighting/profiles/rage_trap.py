"""Profile 1: Rage / Experimental Trap — Playboi Carti, Travis Scott.

Core philosophy: EXTREME CONTRAST — BLINDING or DARK, nothing in between.

Lighting language:
- Colors: RED and WHITE only. Deep blood red (hsv 0°/355°, full sat).
  White = raw strobe white. No pastels, no blending, no gradients.
- Contrast: Binary. Fixtures are either at 100% or at 0%.
  No smooth fades. Transitions are instant (1-frame cuts).
- 808 hits: Maximum strobe violence. All strobes fire simultaneously
  at max rate for the duration of the 808 sustain.
- Kick/snare: Full red flash on all pars, instant cut to black.
- Hi-hat: Ignored (too fast to be meaningful at this energy level).
- Drops: Total blackout for 2-4 beats before the drop, then
  instant full-white strobe explosion on the downbeat.
- Breakdowns/bridges: Slow deep-red pulse on pars only (no strobes).
  Breathing pattern synced to bar phase.
- Choir/vocal calm: Only 1-2 pars at low red. Everything else dark.
  Creates intimate, haunted atmosphere.
- Ad-libs ("what", "slatt"): Random single-fixture scatter.
  One random par flashes white for one beat, others stay dark.
- UV: Constant low-level glow during verses, off during drops/breakdowns.
- Spatial: Corner isolation during calm sections. All-fixture blast
  during high energy. Front-to-back sweep on kick patterns.
"""

from __future__ import annotations

import hashlib

from lumina.audio.models import MusicState
from lumina.control.protocol import FixtureCommand
from lumina.lighting.fixture_map import FixtureMap, FixtureType
from lumina.lighting.profiles.base import (
    BLACK,
    BaseProfile,
    Color,
)

# ─── Rage palette ───────────────────────────────────────────────────

BLOOD_RED = Color(1.0, 0.0, 0.0, 0.0)
DEEP_RED = Color(0.7, 0.0, 0.0, 0.0)
STROBE_WHITE = Color(1.0, 1.0, 1.0, 1.0)

# Strobe parameters
_808_STROBE_RATE = 255  # Maximum rate (~25Hz)
_808_STROBE_INTENSITY = 255
_KICK_STROBE_RATE = 200
_KICK_STROBE_INTENSITY = 220

# UV levels
_UV_VERSE = 80  # Low ambient glow
_UV_OFF = 0

# Breakdown breathing
_BREAKDOWN_MIN_INTENSITY = 0.05
_BREAKDOWN_MAX_INTENSITY = 0.4

# Choir / calm section
_CHOIR_INTENSITY = 0.15
_CHOIR_NUM_PARS = 2


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
        self._uvs = self._map.by_type(FixtureType.UV)

    def generate(self, state: MusicState) -> list[FixtureCommand]:
        """Generate rage-trap fixture commands for the current frame.

        Decision tree:
        1. Drop incoming (drop_probability > 0.8) → pre-drop blackout
        2. Drop/chorus at high energy → strobe explosion
        3. 808 onset → strobe violence
        4. Kick/snare onset → red flash
        5. Breakdown/bridge → slow red breathing
        6. Verse with vocals (choir) → corner isolation
        7. Verse normal → beat-reactive red

        Args:
            state: Current music analysis frame.

        Returns:
            One FixtureCommand per fixture.
        """
        segment = state.segment
        energy = state.energy

        # ── Pre-drop blackout ────────────────────────────────────
        if state.drop_probability > 0.8 and segment != "drop":
            return self._pre_drop_blackout(state)

        # ── Drop / chorus at high energy: strobe explosion ───────
        if segment in ("drop", "chorus") and energy > 0.7:
            return self._drop_explosion(state)

        # ── Breakdown / bridge: slow breathing ───────────────────
        if segment in ("breakdown", "bridge"):
            return self._breakdown_breathe(state)

        # ── Intro / outro: minimal ───────────────────────────────
        if segment in ("intro", "outro"):
            return self._intro_outro(state)

        # ── Verse: beat-reactive or vocal-calm ───────────────────
        if state.vocal_energy > 0.6:
            return self._vocal_calm(state)

        return self._verse_reactive(state)

    # ─── Segment handlers ────────────────────────────────────────

    def _pre_drop_blackout(self, state: MusicState) -> list[FixtureCommand]:
        """Total blackout before a drop — builds tension through darkness."""
        return self._blackout()

    def _drop_explosion(self, state: MusicState) -> list[FixtureCommand]:
        """Maximum strobe + red blast on drop/chorus.

        808 onsets get strobe violence; kicks get red flash;
        downbeats get full white strobe; otherwise sustain red.
        """
        commands: dict[int, FixtureCommand] = {}

        # Strobe behavior
        if state.onset_type == "kick":
            # 808 / kick: strobe violence
            for f in self._strobes:
                commands[f.fixture_id] = self._cmd(
                    f,
                    STROBE_WHITE,
                    strobe_rate=_808_STROBE_RATE,
                    strobe_intensity=_808_STROBE_INTENSITY,
                )
            # All pars full red
            for f in self._pars:
                commands[f.fixture_id] = self._cmd(f, BLOOD_RED, intensity=1.0)
        elif state.is_downbeat:
            # Downbeat: white strobe flash
            for f in self._strobes:
                commands[f.fixture_id] = self._cmd(
                    f,
                    STROBE_WHITE,
                    strobe_rate=_808_STROBE_RATE,
                    strobe_intensity=_808_STROBE_INTENSITY,
                )
            for f in self._pars:
                commands[f.fixture_id] = self._cmd(f, BLOOD_RED, intensity=1.0)
        elif state.is_beat:
            # Beat: red flash, strobes pulse
            for f in self._strobes:
                commands[f.fixture_id] = self._cmd(
                    f,
                    STROBE_WHITE,
                    strobe_rate=_KICK_STROBE_RATE,
                    strobe_intensity=_KICK_STROBE_INTENSITY,
                )
            for f in self._pars:
                commands[f.fixture_id] = self._cmd(f, BLOOD_RED, intensity=1.0)
        else:
            # Between beats: instant black on pars, strobes off
            for f in self._strobes:
                commands[f.fixture_id] = self._cmd(f, BLACK, intensity=0.0)
            for f in self._pars:
                commands[f.fixture_id] = self._cmd(f, BLACK, intensity=0.0)

        # UV off during drops
        for f in self._uvs:
            commands[f.fixture_id] = self._cmd(f, special=_UV_OFF)

        return self._merge_commands(commands)

    def _breakdown_breathe(self, state: MusicState) -> list[FixtureCommand]:
        """Slow deep-red breathing pulse synced to bar phase.

        No strobes. Just pars pulsing like a heartbeat.
        """
        commands: dict[int, FixtureCommand] = {}

        # Breathing intensity from bar phase (0→peak→0 per bar)
        import math

        breath = (math.sin(state.bar_phase * math.pi * 2.0 - math.pi / 2.0) + 1.0) / 2.0
        intensity = _BREAKDOWN_MIN_INTENSITY + breath * (
            _BREAKDOWN_MAX_INTENSITY - _BREAKDOWN_MIN_INTENSITY
        )

        for f in self._pars:
            commands[f.fixture_id] = self._cmd(f, DEEP_RED, intensity=intensity)

        # Strobes off
        for f in self._strobes:
            commands[f.fixture_id] = self._cmd(f)

        # UV off
        for f in self._uvs:
            commands[f.fixture_id] = self._cmd(f, special=_UV_OFF)

        return self._merge_commands(commands)

    def _intro_outro(self, state: MusicState) -> list[FixtureCommand]:
        """Minimal lighting for intro/outro — single par, low red."""
        commands: dict[int, FixtureCommand] = {}

        # Only the first par at low intensity
        for i, f in enumerate(self._pars):
            if i == 0:
                commands[f.fixture_id] = self._cmd(f, DEEP_RED, intensity=0.2)
            else:
                commands[f.fixture_id] = self._cmd(f, BLACK, intensity=0.0)

        for f in self._strobes:
            commands[f.fixture_id] = self._cmd(f)

        for f in self._uvs:
            commands[f.fixture_id] = self._cmd(f, special=_UV_OFF)

        return self._merge_commands(commands)

    def _vocal_calm(self, state: MusicState) -> list[FixtureCommand]:
        """Choir / vocal calm section — intimate, haunted.

        Only 1-2 pars at low red. Everything else dark.
        Which pars are lit rotates based on bar phase.
        """
        commands: dict[int, FixtureCommand] = {}

        # Pick which pars to light based on bar count
        # Use timestamp to rotate which pair is lit
        n_pars = len(self._pars)
        if n_pars > 0:
            offset = int(state.timestamp * 0.5) % n_pars
            for i, f in enumerate(self._pars):
                if i >= offset and i < offset + _CHOIR_NUM_PARS:
                    commands[f.fixture_id] = self._cmd(
                        f, DEEP_RED, intensity=_CHOIR_INTENSITY
                    )
                else:
                    commands[f.fixture_id] = self._cmd(f, BLACK, intensity=0.0)

        for f in self._strobes:
            commands[f.fixture_id] = self._cmd(f)

        # Low UV for atmosphere
        for f in self._uvs:
            commands[f.fixture_id] = self._cmd(f, special=_UV_VERSE // 2)

        return self._merge_commands(commands)

    def _verse_reactive(self, state: MusicState) -> list[FixtureCommand]:
        """Standard verse: beat-reactive red with ad-lib scatter.

        Kicks: all pars flash red, instant cut.
        Snares: front-to-back sweep.
        Ad-libs (clap onset): random single fixture scatter.
        Between beats: black.
        """
        commands: dict[int, FixtureCommand] = {}

        if state.onset_type == "clap":
            # Ad-lib scatter: pseudo-random single par flash
            return self._adlib_scatter(state)

        if state.onset_type == "kick":
            # Full red flash
            for f in self._pars:
                commands[f.fixture_id] = self._cmd(f, BLOOD_RED, intensity=1.0)
            # Strobes pulse
            for f in self._strobes:
                commands[f.fixture_id] = self._cmd(
                    f,
                    STROBE_WHITE,
                    strobe_rate=_KICK_STROBE_RATE,
                    strobe_intensity=_KICK_STROBE_INTENSITY,
                )
        elif state.onset_type == "snare":
            # Front-to-back sweep
            sweep = self._sweep_y(state.beat_phase, BLOOD_RED, width=0.4, intensity=1.0)
            commands.update(sweep)
        elif state.is_beat:
            # Beat pulse: red flash
            for f in self._pars:
                commands[f.fixture_id] = self._cmd(f, BLOOD_RED, intensity=0.8)
        else:
            # Between beats: black
            for f in self._pars:
                commands[f.fixture_id] = self._cmd(f, BLACK, intensity=0.0)

        # Strobes off unless kick
        if state.onset_type != "kick":
            for f in self._strobes:
                commands[f.fixture_id] = self._cmd(f, BLACK, intensity=0.0)

        # UV low glow during verses
        for f in self._uvs:
            commands[f.fixture_id] = self._cmd(f, special=_UV_VERSE)

        return self._merge_commands(commands)

    def _adlib_scatter(self, state: MusicState) -> list[FixtureCommand]:
        """Random single-fixture scatter for ad-libs.

        Deterministic pseudo-random selection based on timestamp
        so the same frame always produces the same output.
        """
        commands: dict[int, FixtureCommand] = {}

        # Deterministic "random" fixture selection
        n_pars = len(self._pars)
        if n_pars > 0:
            # Use timestamp hash for deterministic scatter
            ts_hash = int(
                hashlib.md5(  # noqa: S324
                    str(state.timestamp).encode()
                ).hexdigest()[:8],
                16,
            )
            chosen_idx = ts_hash % n_pars
            for i, f in enumerate(self._pars):
                if i == chosen_idx:
                    commands[f.fixture_id] = self._cmd(
                        f, STROBE_WHITE, intensity=1.0
                    )
                else:
                    commands[f.fixture_id] = self._cmd(f, BLACK, intensity=0.0)

        for f in self._strobes:
            commands[f.fixture_id] = self._cmd(f)

        for f in self._uvs:
            commands[f.fixture_id] = self._cmd(f, special=_UV_OFF)

        return self._merge_commands(commands)
