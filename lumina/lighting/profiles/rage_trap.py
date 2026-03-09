"""Profile 1: Rage / Experimental Trap — Playboi Carti, Travis Scott.

Core philosophy: EXTREME CONTRAST — BLINDING or DARK, nothing in between.

Lighting language:
- Colors: RED and WHITE only. Deep blood red (hsv 0/355, full sat).
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

Verse evolution:
- Fixtures build over bars: start with 1 par, add every 4 bars.
- Strobes alternate left/right (never fire together in verses).
- Chase patterns during grooves sweep around the room.
- Intensity breathing even during steady sections.
- Blackout gaps (500ms darkness) every 8-16 bars for contrast.
- Color temperature shifts cooler→warmer as energy rises.
- Kick = quick full-room red flash then instant decay.
- Snare = single random fixture white flash.
- Between beats = near-darkness, not dim lighting.

Drop behavior:
- Pre-drop: rapid strobe acceleration over 4 bars.
- Drop hit: ALL fixtures max white for 200ms then blackout.
- During drop: binary alternating — full red on beat, total
  blackout on offbeat. Nothing in between.
"""

from __future__ import annotations

import hashlib
import math

from lumina.audio.models import MusicState
from lumina.control.protocol import FixtureCommand
from lumina.lighting.fixture_map import FixtureMap, FixtureType
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
STROBE_WHITE = Color(1.0, 1.0, 1.0, 1.0)

# Strobe parameters
_808_STROBE_RATE = 255  # Maximum rate (~25Hz)
_808_STROBE_INTENSITY = 255
_KICK_STROBE_RATE = 200
_KICK_STROBE_INTENSITY = 220

# UV levels
_UV_VERSE = 80  # Low ambient glow
_UV_OFF = 0

# Breakdown breathing (10-20% range per spec)
_BREAKDOWN_MIN_INTENSITY = 0.10
_BREAKDOWN_MAX_INTENSITY = 0.20

# Choir / calm section
_CHOIR_INTENSITY = 0.30
_CHOIR_NUM_PARS = 2

# Verse evolution
_BARS_PER_FIXTURE_ADD = 4  # Add a new fixture every N bars
_BLACKOUT_GAP_BARS = 12  # Blackout gap every N bars
_BLACKOUT_GAP_DURATION_S = 0.5  # 500ms of total darkness
_VERSE_BREATHE_DEPTH = 0.15  # Intensity wobble depth

# Drop timing (frames at 60fps)
_DROP_HIT_WHITE_FRAMES = 12  # 200ms at 60fps


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
        # Ordered pars for chase: FL, FR, BR, BL (clockwise)
        self._chase_order = self._map.sorted_by_x(self._pars)

        # State tracking for verse evolution and drop
        self._verse_start_time: float = -1.0
        self._last_segment: str = ""
        self._drop_frame_count: int = 0
        self._was_pre_drop: bool = False

    def generate(self, state: MusicState) -> list[FixtureCommand]:
        """Generate rage-trap fixture commands for the current frame.

        Args:
            state: Current music analysis frame.

        Returns:
            One FixtureCommand per fixture.
        """
        segment = state.segment
        energy = state.energy

        # Track segment transitions
        if segment != self._last_segment:
            if segment in ("verse", "chorus"):
                self._verse_start_time = state.timestamp
            if segment == "drop":
                self._drop_frame_count = 0
            self._last_segment = segment

        # Count frames in drop for the initial white flash
        if segment == "drop":
            self._drop_frame_count += 1

        # ── Pre-drop: accelerating strobe over 4 bars ──────────
        if state.drop_probability > 0.6 and segment != "drop":
            self._was_pre_drop = True
            return self._pre_drop_build(state)

        # ── Drop hit: max white flash then binary alternating ──
        if segment == "drop" and energy > 0.5:
            result = self._drop_explosion(state)
            self._was_pre_drop = False
            return result

        self._was_pre_drop = False

        # ── Breakdown / bridge: slow breathing ─────────────────
        if segment in ("breakdown", "bridge"):
            return self._breakdown_breathe(state)

        # ── Intro / outro: minimal ─────────────────────────────
        if segment in ("intro", "outro"):
            return self._intro_outro(state)

        # ── Verse: evolving beat-reactive or vocal-calm ────────
        if state.vocal_energy > 0.6:
            return self._vocal_calm(state)

        return self._verse_reactive(state)

    # ─── Segment handlers ────────────────────────────────────────

    def _pre_drop_build(self, state: MusicState) -> list[FixtureCommand]:
        """Pre-drop: accelerating strobe build-up over bars.

        Strobe rate ramps from slow to max based on drop_probability.
        Pars pulse red with increasing intensity.
        """
        commands: dict[int, FixtureCommand] = {}

        # Map drop_probability 0.6→1.0 to ramp 0→1
        ramp = max(0.0, min(1.0, (state.drop_probability - 0.6) / 0.4))

        # Strobe rate accelerates: 40→255
        rate = int(40 + ramp * 215)
        intensity = int(80 + ramp * 175)

        # Alternate strobes at lower ramp, sync at high ramp
        if ramp < 0.7:
            left_on = state.beat_phase < 0.5
            for i, f in enumerate(self._strobes):
                is_left = i % 2 == 0
                if (is_left and left_on) or (not is_left and not left_on):
                    commands[f.fixture_id] = self._cmd(
                        f,
                        STROBE_WHITE,
                        strobe_rate=rate,
                        strobe_intensity=intensity,
                    )
                else:
                    commands[f.fixture_id] = self._cmd(f, BLACK, intensity=0.0)
        else:
            # All strobes synced at high tension
            for f in self._strobes:
                commands[f.fixture_id] = self._cmd(
                    f,
                    STROBE_WHITE,
                    strobe_rate=rate,
                    strobe_intensity=intensity,
                )

        # Pars: pulsing red at increasing intensity
        par_intensity = 0.30 + ramp * 0.50
        if state.is_beat:
            par_intensity = min(1.0, par_intensity + 0.2)
        for f in self._pars:
            commands[f.fixture_id] = self._cmd(f, BLOOD_RED, intensity=par_intensity)

        # UV off
        for f in self._uvs:
            commands[f.fixture_id] = self._cmd(f, special=_UV_OFF)

        return self._merge_commands(commands)

    def _drop_explosion(self, state: MusicState) -> list[FixtureCommand]:
        """Drop: initial white flash then binary red/black alternating.

        First 200ms: ALL fixtures max white.
        After: full red on beat, total blackout on offbeat.
        """
        commands: dict[int, FixtureCommand] = {}

        # Initial white flash (200ms = ~12 frames at 60fps)
        if self._was_pre_drop and self._drop_frame_count <= _DROP_HIT_WHITE_FRAMES:
            for f in self._pars:
                commands[f.fixture_id] = self._cmd(f, STROBE_WHITE, intensity=1.0)
            for f in self._strobes:
                commands[f.fixture_id] = self._cmd(
                    f,
                    STROBE_WHITE,
                    strobe_rate=255,
                    strobe_intensity=255,
                )
            for f in self._uvs:
                commands[f.fixture_id] = self._cmd(f, special=255)
            return self._merge_commands(commands)

        # Binary alternating: on-beat = full red, off-beat = blackout
        on_beat = state.beat_phase < 0.5

        if on_beat:
            for f in self._pars:
                commands[f.fixture_id] = self._cmd(f, BLOOD_RED, intensity=1.0)
            for f in self._strobes:
                commands[f.fixture_id] = self._cmd(
                    f,
                    STROBE_WHITE,
                    strobe_rate=_808_STROBE_RATE,
                    strobe_intensity=_808_STROBE_INTENSITY,
                )
        else:
            for f in self._pars:
                commands[f.fixture_id] = self._cmd(f, BLACK, intensity=0.0)
            for f in self._strobes:
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

        breath = (math.sin(state.bar_phase * math.pi * 2.0 - math.pi / 2.0) + 1.0) / 2.0
        intensity = _BREAKDOWN_MIN_INTENSITY + breath * (
            _BREAKDOWN_MAX_INTENSITY - _BREAKDOWN_MIN_INTENSITY
        )

        for f in self._pars:
            commands[f.fixture_id] = self._cmd(f, DEEP_RED, intensity=intensity)

        for f in self._strobes:
            commands[f.fixture_id] = self._cmd(f)

        for f in self._uvs:
            commands[f.fixture_id] = self._cmd(f, special=_UV_OFF)

        return self._merge_commands(commands)

    def _intro_outro(self, state: MusicState) -> list[FixtureCommand]:
        """Minimal lighting for intro/outro — single par, low red."""
        commands: dict[int, FixtureCommand] = {}

        for i, f in enumerate(self._pars):
            if i == 0:
                commands[f.fixture_id] = self._cmd(f, DEEP_RED, intensity=0.30)
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

        n_pars = len(self._pars)
        if n_pars > 0:
            offset = int(state.timestamp * 0.5) % n_pars
            for i, f in enumerate(self._pars):
                if i >= offset and i < offset + _CHOIR_NUM_PARS:
                    commands[f.fixture_id] = self._cmd(f, DEEP_RED, intensity=_CHOIR_INTENSITY)
                else:
                    commands[f.fixture_id] = self._cmd(f, BLACK, intensity=0.0)

        for f in self._strobes:
            commands[f.fixture_id] = self._cmd(f)

        for f in self._uvs:
            commands[f.fixture_id] = self._cmd(f, special=_UV_VERSE // 2)

        return self._merge_commands(commands)

    def _verse_reactive(self, state: MusicState) -> list[FixtureCommand]:
        """Evolving verse with fixture build-up, chases, and blackout gaps.

        Fixtures build over bars: start with 1 par, add every 4 bars.
        Strobes alternate left/right. Chase patterns during grooves.
        Blackout gaps (500ms) every 8-16 bars for dramatic contrast.
        """
        commands: dict[int, FixtureCommand] = {}

        # Time since verse started
        dt = max(0.0, state.timestamp - self._verse_start_time)
        bpm = max(60.0, state.bpm)
        bar_duration = 60.0 / bpm * 4.0
        bars_elapsed = dt / bar_duration

        # ── Blackout gap: 500ms of darkness every N bars ──────
        bar_cycle = bars_elapsed % _BLACKOUT_GAP_BARS
        gap_start = _BLACKOUT_GAP_BARS - (_BLACKOUT_GAP_DURATION_S / bar_duration)
        if bar_cycle >= gap_start:
            return self._blackout()

        # ── Fixture build-up: add a par every 4 bars ─────────
        n_pars = len(self._pars)
        active_pars = min(n_pars, 1 + int(bars_elapsed / _BARS_PER_FIXTURE_ADD))

        # ── Color temperature: cooler → warmer as energy rises ─
        verse_color = lerp_color(COOL_RED, WARM_RED, state.energy)

        # ── Intensity breathing: subtle wobble ────────────────
        breathe = 1.0 - _VERSE_BREATHE_DEPTH * (
            math.sin(state.timestamp * math.pi * 0.5) * 0.5 + 0.5
        )

        # ── Onset reactions ───────────────────────────────────
        if state.onset_type == "kick":
            # Quick full-room red flash — all active pars at max
            for i, f in enumerate(self._pars):
                if i < active_pars:
                    commands[f.fixture_id] = self._cmd(f, BLOOD_RED, intensity=1.0)
                else:
                    commands[f.fixture_id] = self._cmd(f, BLACK, intensity=0.0)
            # Alternate strobes on kicks (left on even beats, right on odd)
            beat_idx = int(state.timestamp * bpm / 60.0)
            for i, f in enumerate(self._strobes):
                if i % 2 == beat_idx % 2:
                    commands[f.fixture_id] = self._cmd(
                        f,
                        STROBE_WHITE,
                        strobe_rate=_KICK_STROBE_RATE,
                        strobe_intensity=_KICK_STROBE_INTENSITY,
                    )
                else:
                    commands[f.fixture_id] = self._cmd(f, BLACK, intensity=0.0)
        elif state.onset_type == "snare":
            # Single random fixture white flash
            idx = _deterministic_index(state.timestamp, n_pars)
            for i, f in enumerate(self._pars):
                if i == idx:
                    commands[f.fixture_id] = self._cmd(f, STROBE_WHITE, intensity=1.0)
                else:
                    commands[f.fixture_id] = self._cmd(f, BLACK, intensity=0.0)
            for f in self._strobes:
                commands[f.fixture_id] = self._cmd(f, BLACK, intensity=0.0)
        elif state.onset_type == "clap":
            # Ad-lib scatter
            return self._adlib_scatter(state)
        elif state.is_beat:
            # Beat pulse: chase sweep on active pars
            chase = self._chase(
                self._chase_order[:active_pars],
                state.bar_phase,
                verse_color,
                width=0.4,
                intensity=0.85 * breathe,
            )
            commands.update(chase)
            # Black out inactive pars
            for i, f in enumerate(self._pars):
                if i >= active_pars and f.fixture_id not in commands:
                    commands[f.fixture_id] = self._cmd(f, BLACK, intensity=0.0)
            for f in self._strobes:
                commands[f.fixture_id] = self._cmd(f, BLACK, intensity=0.0)
        else:
            # Between beats: near-dark glow (rage_trap = binary contrast)
            base_glow = 0.05 * breathe
            for i, f in enumerate(self._pars):
                if i < active_pars:
                    commands[f.fixture_id] = self._cmd(f, verse_color, intensity=base_glow)
                else:
                    commands[f.fixture_id] = self._cmd(f, BLACK, intensity=0.0)
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

        n_pars = len(self._pars)
        if n_pars > 0:
            chosen_idx = _deterministic_index(state.timestamp, n_pars)
            for i, f in enumerate(self._pars):
                if i == chosen_idx:
                    commands[f.fixture_id] = self._cmd(f, STROBE_WHITE, intensity=1.0)
                else:
                    commands[f.fixture_id] = self._cmd(f, BLACK, intensity=0.0)

        for f in self._strobes:
            commands[f.fixture_id] = self._cmd(f)

        for f in self._uvs:
            commands[f.fixture_id] = self._cmd(f, special=_UV_OFF)

        return self._merge_commands(commands)


def _deterministic_index(timestamp: float, n: int) -> int:
    """Deterministic pseudo-random index from timestamp.

    Args:
        timestamp: Time value for seed.
        n: Range upper bound (exclusive).

    Returns:
        Index in [0, n).
    """
    if n <= 0:
        return 0
    ts_hash = int(
        hashlib.md5(str(timestamp).encode()).hexdigest()[:8],
        16,
    )
    return ts_hash % n
