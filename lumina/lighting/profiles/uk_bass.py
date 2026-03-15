"""Profile 8: UK Bass / Dubstep / Grime — Fred again.., Boiler Room sets.

Core philosophy: UNDERGROUND RAVE — raw, DIY, imperfect by design.

Lighting language:
- Colors: Industrial palette — sodium amber, deep green, cold white,
  dirty pink.  Never clean primary colors.
- Strobes: Imprecise, organic.  Not perfectly on-beat — add jitter.
  Fire slightly early or late (hash-based timing offset).
- Verse/groove: flicker on pars (organic jitter like fire), dirty
  amber/green wash.  Low intensity.
- Build: stutter on strobes (slower than EDM builds).  Converge on
  pars with cold white.
- Drop: strobe_burst + blinder, then alternating left/right
  random_scatter with green/amber.
- Breakdown: Single par breathe at very low intensity.  Everything
  else off.  Intimate.
- LED bars: gradient_y (front-to-back) during drops, low flicker
  during verse, off in breakdown.
- Laser: off in most sections, low-speed pattern during drops only.
"""

from __future__ import annotations

import hashlib

from lumina.audio.models import MusicState
from lumina.control.protocol import FixtureCommand
from lumina.lighting.fixture_map import FixtureMap, FixtureType
from lumina.lighting.patterns import (
    blinder,
    breathe,
    converge,
    flicker,
    gradient_y,
    make_command,
    random_scatter,
    select_active_fixtures,
    strobe_burst,
    stutter,
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

# ─── UK Bass palette ─────────────────────────────────────────────────
# Industrial / underground: never clean primaries.  Dirty, desaturated,
# sodium-lit warehouse aesthetic.

SODIUM_AMBER = Color(1.0, 0.55, 0.05, 0.3)
DEEP_GREEN = Color(0.05, 0.65, 0.15, 0.0)
COLD_WHITE = Color(0.7, 0.75, 0.9, 0.8)
DIRTY_PINK = Color(0.85, 0.25, 0.35, 0.1)
GRIME_GREEN = Color(0.2, 1.0, 0.1, 0.0)
NEON_GREEN = Color(0.0, 1.0, 0.2, 0.0)
HARSH_WHITE = Color(1.0, 1.0, 1.0, 1.0)
UV_PURPLE = Color(0.3, 0.0, 0.6, 0.0)

# Verse flicker palette (cycled per bar)
_VERSE_COLORS = [SODIUM_AMBER, DEEP_GREEN, DIRTY_PINK]

# Drop alternating palette
_DROP_LEFT = GRIME_GREEN
_DROP_RIGHT = SODIUM_AMBER

# Timing
_DROP_BLINDER_FRAMES = 10  # ~167ms at 60fps
_JITTER_MAX_MS = 25  # max strobe jitter in ms (±)

# Laser patterns
_LASER_OFF = 0
_LASER_DROP_SLOW = 3

# Build timing
_BUILD_STUTTER_MIN = 1.5  # slower stutter start than EDM
_BUILD_STUTTER_MAX = 6.0


def _jitter_offset(fixture_id: int, timestamp: float) -> float:
    """Compute a deterministic timing jitter for a fixture.

    UK bass strobes are intentionally imprecise — they fire slightly
    early or late relative to the grid.  This uses a hash of fixture_id
    and quantized timestamp to produce a stable offset per fixture per
    beat that shifts the perceived timing.

    Args:
        fixture_id: Fixture identifier for seed.
        timestamp: Current timestamp in seconds.

    Returns:
        Offset in seconds, range [-JITTER_MAX_MS, +JITTER_MAX_MS] ms.
    """
    t_quant = int(timestamp * 8)  # ~125ms quantization (per beat feel)
    seed = f"jitter_{t_quant}_{fixture_id}".encode()
    h = int(hashlib.md5(seed).hexdigest()[:8], 16)
    normalized = (h & 0xFFFF) / 0xFFFF  # 0.0-1.0
    return (normalized - 0.5) * 2.0 * (_JITTER_MAX_MS / 1000.0)


class UkBassProfile(BaseProfile):
    """UK Bass / Dubstep / Grime lighting profile.

    Underground rave philosophy: raw, DIY, imperfect by design.
    Organic flicker replaces clean chases.  Strobe timing has
    intentional jitter.  Breakdowns are intimate single-fixture
    moments.  Drops alternate scatter patterns with green/amber.

    Args:
        fixture_map: Venue fixture layout.
    """

    name = "uk_bass"

    def __init__(self, fixture_map: FixtureMap) -> None:
        super().__init__(fixture_map)
        self._pars = self._map.by_type(FixtureType.PAR)
        self._strobes = self._map.by_type(FixtureType.STROBE)
        self._led_bars = self._map.by_type(FixtureType.LED_BAR)
        self._lasers = self._map.by_type(FixtureType.LASER)
        self._pars_lr = self._map.sorted_by_x(self._pars)

        # Slower decay than rage (200ms half-life) — more organic
        self._bump = BumpTracker(decay_rate=5.0)

        # State tracking
        self._segment_start_time: float = 0.0
        self._last_segment: str = ""
        self._segment_changed: bool = False
        self._drop_frame_count: int = 0
        self._was_pre_drop: bool = False

    @property
    def motif_pattern_preferences(self) -> list[str]:
        """UK bass prefers raw, organic patterns for motifs."""
        return ["flicker", "random_scatter", "chase_lr", "stutter"]

    @property
    def motif_color_palette(self) -> list[Color]:
        """UK bass color cycle: industrial warehouse tones."""
        return [SODIUM_AMBER, DEEP_GREEN, DIRTY_PINK, GRIME_GREEN]

    def _headroom_scale(self, state: MusicState, intensity: float) -> float:
        """Apply headroom to intensity, UK bass style.

        Drops ignore headroom.  Other segments scale by headroom with
        a higher floor than rage (underground always has some light).

        Args:
            state: Current music state.
            intensity: Raw intensity value.

        Returns:
            Headroom-adjusted intensity.
        """
        if state.segment == "drop":
            return intensity
        return intensity * max(0.20, state.headroom)

    def _get_verse_color(self, state: MusicState) -> Color:
        """Select verse color from palette based on bar position.

        Cycles through sodium amber, deep green, and dirty pink on
        a 4-bar rotation with smooth interpolation between them.

        Args:
            state: Current music state.

        Returns:
            Interpolated verse color.
        """
        bpm = max(60.0, state.bpm)
        bar_duration = 60.0 / bpm * 4.0
        dt = max(0.0, state.timestamp - self._segment_start_time)
        bar_index = dt / bar_duration
        cycle_pos = (bar_index / 4.0) % 1.0

        n = len(_VERSE_COLORS)
        idx = int(cycle_pos * n) % n
        next_idx = (idx + 1) % n
        blend = (cycle_pos * n) % 1.0
        return lerp_color(_VERSE_COLORS[idx], _VERSE_COLORS[next_idx], blend)

    def generate(self, state: MusicState) -> list[FixtureCommand]:
        """Generate UK bass fixture commands for the current frame.

        Decision hierarchy:
        1. Track segment transitions
        2. Pre-drop build detection (stutter + converge)
        3. Drop (blinder flash then scatter)
        4. Breakdown (single par breathe)
        5. Intro/outro (minimal)
        6. Default verse/groove (flicker)

        Args:
            state: Current audio analysis frame.

        Returns:
            One FixtureCommand per fixture.
        """
        self._begin_debug_frame()
        segment = state.segment

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

        # Pre-drop build
        if state.drop_probability > 0.5 and segment != "drop":
            self._was_pre_drop = True
            self._note_patterns("build")
            return self._build(state)

        # Drop
        if segment == "drop" and state.energy > 0.4:
            self._note_patterns("drop")
            result = self._drop(state)
            self._was_pre_drop = False
            return result

        self._was_pre_drop = False

        # Breakdown / bridge
        if segment in ("breakdown", "bridge"):
            self._note_patterns("breakdown")
            return self._breakdown(state)

        # Intro / outro
        if segment in ("intro", "outro"):
            self._note_patterns("intro_outro")
            return self._intro_outro(state)

        # Chorus — wider flicker, more energy
        if segment == "chorus":
            self._note_patterns("verse_groove")
            return self._verse_groove(state, chorus_boost=True)

        # Default: verse/groove
        self._note_patterns("verse_groove")
        return self._verse_groove(state)

    # ─── Segment handlers ────────────────────────────────────────

    def _verse_groove(
        self,
        state: MusicState,
        *,
        chorus_boost: bool = False,
    ) -> list[FixtureCommand]:
        """Verse/groove: organic flicker on pars, dirty amber/green wash.

        Low intensity flickering like a warehouse with bad wiring.
        Hi-hat onsets bump LED bars.  MC delivery (vocal energy) can
        push intensity up slightly.  Strobes fire with jitter on
        kicks — imprecise, organic.

        Args:
            state: Current music state.
            chorus_boost: If True, use wider fixture count and higher
                base intensity (for chorus sections).

        Returns:
            Fixture command list.
        """
        commands: dict[int, FixtureCommand] = {}
        energy = state.energy

        verse_color = self._get_verse_color(state)

        # Sub-bass: wobble the color toward green when sub-bass is heavy
        if state.sub_bass_energy > 0.4:
            verse_color = lerp_color(verse_color, DEEP_GREEN, state.sub_bass_energy * 0.4)
            verse_color = self._bass_saturate(state.sub_bass_energy, verse_color)

        # Fixture count: chorus uses more pars
        if chorus_boost:
            active_pars = select_active_fixtures(
                self._pars, energy,
                low_count=4, mid_count=6, high_threshold=0.7,
            )
            base_intensity = 0.30 + energy_brightness(energy) * 0.25
        else:
            active_pars = select_active_fixtures(
                self._pars, energy,
                low_count=2, mid_count=4, mid_threshold=0.4, high_threshold=0.75,
            )
            base_intensity = 0.15 + energy_brightness(energy) * 0.20

        base_intensity = self._headroom_scale(state, base_intensity)

        # Vocal energy (MC delivery) pushes intensity up
        if state.vocal_energy > 0.5:
            vocal_boost = (state.vocal_energy - 0.5) * 0.15
            base_intensity = min(1.0, base_intensity + vocal_boost)

        # Pars: organic flicker (the signature UK bass look)
        par_cmds = flicker(
            active_pars, state, state.timestamp, verse_color,
            intensity=base_intensity, jitter=0.45,
        )
        commands.update(par_cmds)

        # Kick: trigger bump for punchy decay on pars
        if state.onset_type == "kick":
            self._bump.trigger("pars", state.timestamp)

        # Bump decay blends into flicker base
        bump_level = self._bump.get_intensity(
            "pars", state.timestamp, peak=base_intensity * 1.5, floor=0.0,
        )
        if bump_level > base_intensity:
            # Override active pars with bump-boosted wash during decay
            bump_cmds = wash_hold(
                active_pars, state, state.timestamp, verse_color,
                intensity=min(1.0, bump_level),
            )
            commands.update(bump_cmds)

        # Blackout inactive pars
        for f in self._pars:
            if f.fixture_id not in commands:
                commands[f.fixture_id] = make_command(f, BLACK, 0.0)

        # Strobes: fire with jitter on kicks (imprecise, organic)
        if state.onset_type == "kick":
            for f in self._strobes:
                jitter = _jitter_offset(f.fixture_id, state.timestamp)
                # Jitter affects perceived rate — slightly off-grid
                rate = 180 + int(jitter * 1000)  # ±25 variation
                rate = max(140, min(220, rate))
                commands[f.fixture_id] = make_command(
                    f, COLD_WHITE, strobe_rate=rate, strobe_intensity=200,
                )
        elif state.onset_type == "snare":
            # Snare: quick scatter flash on strobes
            scatter = random_scatter(
                self._strobes, state, state.timestamp, HARSH_WHITE,
                density=0.6, intensity=0.8,
            )
            commands.update(scatter)
        else:
            for f in self._strobes:
                commands[f.fixture_id] = make_command(f, BLACK, 0.0)

        # Hi-hat: bump LED bars
        if state.onset_type == "hihat":
            self._bump.trigger("bars", state.timestamp)

        # LED bars: low flicker during verse
        bar_bump = self._bump.get_intensity(
            "bars", state.timestamp, peak=0.30, floor=0.0,
        )
        bar_intensity = self._headroom_scale(state, 0.10 + bar_bump)
        bar_cmds = flicker(
            self._led_bars, state, state.timestamp, verse_color,
            intensity=bar_intensity, jitter=0.35,
        )
        commands.update(bar_cmds)

        # Laser: off during verse
        for f in self._lasers:
            commands[f.fixture_id] = make_command(f, BLACK, special=_LASER_OFF)

        return self._merge_commands(commands)

    def _build(self, state: MusicState) -> list[FixtureCommand]:
        """Build: stutter on strobes (slower than EDM) + converge on pars.

        UK bass builds are less polished than EDM builds.  The stutter
        is slower, the convergence is grittier.  Cold white replaces
        the warm amber as tension mounts.

        Args:
            state: Current music state.

        Returns:
            Fixture command list.
        """
        commands: dict[int, FixtureCommand] = {}

        # Map drop_probability 0.5→1.0 to ramp 0→1
        ramp = max(0.0, min(1.0, (state.drop_probability - 0.5) / 0.5))

        # Color: dirty amber → cold white as build progresses
        build_color = lerp_color(SODIUM_AMBER, COLD_WHITE, ramp)

        # Pars: converge with increasing intensity
        par_intensity = 0.20 + ramp * 0.60
        par_cmds = converge(
            self._pars, state, state.timestamp, build_color,
            intensity=par_intensity,
        )
        commands.update(par_cmds)

        # Strobes: stutter, slower acceleration than festival EDM
        stutter_rate = _BUILD_STUTTER_MIN + ramp * (
            _BUILD_STUTTER_MAX - _BUILD_STUTTER_MIN
        )
        strobe_cmds = stutter(
            self._strobes, state, state.timestamp, COLD_WHITE,
            rate=stutter_rate, intensity=0.6 + ramp * 0.4,
        )
        commands.update(strobe_cmds)

        # LED bars: activate at ~40% ramp, converging cold white
        if ramp > 0.4:
            bar_intensity = (ramp - 0.4) / 0.6 * 0.5
            bar_cmds = wash_hold(
                self._led_bars, state, state.timestamp, build_color,
                intensity=bar_intensity,
            )
            commands.update(bar_cmds)
        else:
            for f in self._led_bars:
                commands[f.fixture_id] = make_command(f, BLACK, 0.0)

        # Laser: off during build (saves it for the drop)
        for f in self._lasers:
            commands[f.fixture_id] = make_command(f, BLACK, special=_LASER_OFF)

        return self._merge_commands(commands)

    def _drop(self, state: MusicState) -> list[FixtureCommand]:
        """Drop: blinder flash then alternating L/R scatter with green/amber.

        First ~167ms: blinder on all fixtures (raw white explosion).
        After: random_scatter alternating between left and right pars
        with grime green and sodium amber.  LED bars get gradient_y.
        Wobble bass = wobble lights (sub-bass modulates intensity).

        Args:
            state: Current music state.

        Returns:
            Fixture command list.
        """
        commands: dict[int, FixtureCommand] = {}

        # Initial blinder flash
        if self._was_pre_drop and self._drop_frame_count <= _DROP_BLINDER_FRAMES:
            all_fixtures = self._pars + self._strobes + self._led_bars
            blind_cmds = blinder(
                all_fixtures, state, state.timestamp, HARSH_WHITE,
            )
            commands.update(blind_cmds)
            for f in self._lasers:
                commands[f.fixture_id] = make_command(
                    f, BLACK, special=_LASER_DROP_SLOW,
                )
            return self._merge_commands(commands)

        # Wobble bass modulation: sub_bass_energy modulates scatter intensity
        wobble = 0.7 + state.sub_bass_energy * 0.3

        # Alternating L/R scatter: swap sides every half-bar
        left_pars = [f for f in self._pars_lr if f.position[0] < 2.5]
        right_pars = [f for f in self._pars_lr if f.position[0] >= 2.5]

        # Alternate on half-bar boundary
        first_half = state.bar_phase < 0.5

        if first_half:
            active_pars, active_color = left_pars, _DROP_LEFT
            inactive_pars, inactive_color = right_pars, _DROP_RIGHT
        else:
            active_pars, active_color = right_pars, _DROP_LEFT
            inactive_pars, inactive_color = left_pars, _DROP_RIGHT

        # Active side: random scatter at high intensity
        if active_pars:
            scatter_cmds = random_scatter(
                active_pars, state, state.timestamp, active_color,
                density=0.7, intensity=wobble,
            )
            commands.update(scatter_cmds)

        # Inactive side: dim scatter of the other color
        if inactive_pars:
            dim_cmds = random_scatter(
                inactive_pars, state, state.timestamp, inactive_color,
                density=0.3, intensity=wobble * 0.3,
            )
            commands.update(dim_cmds)

        # Strobes: burst on kicks with jitter, scatter on beats
        if state.onset_type == "kick" or state.is_downbeat:
            burst = strobe_burst(
                self._strobes, state, state.timestamp, HARSH_WHITE,
            )
            commands.update(burst)
        elif state.is_beat:
            for f in self._strobes:
                jitter = _jitter_offset(f.fixture_id, state.timestamp)
                rate = 200 + int(jitter * 800)
                rate = max(160, min(240, rate))
                commands[f.fixture_id] = make_command(
                    f, GRIME_GREEN, strobe_rate=rate, strobe_intensity=180,
                )
        else:
            for f in self._strobes:
                commands[f.fixture_id] = make_command(f, BLACK, 0.0)

        # LED bars: gradient front-to-back (neon green → amber)
        bar_cmds = gradient_y(
            self._led_bars, state, state.timestamp, GRIME_GREEN,
            color_back=SODIUM_AMBER, intensity=wobble * 0.8,
        )
        commands.update(bar_cmds)

        # Laser: low-speed pattern during drops only
        for f in self._lasers:
            commands[f.fixture_id] = make_command(
                f, BLACK, special=_LASER_DROP_SLOW,
            )

        return self._merge_commands(commands)

    def _breakdown(self, state: MusicState) -> list[FixtureCommand]:
        """Breakdown: single par breathe at very low intensity.

        Everything else off.  Intimate, quiet, personal.  The room
        almost disappears.  This is the UK bass equivalent of the
        crowd singing along in the dark.

        Args:
            state: Current music state.

        Returns:
            Fixture command list.
        """
        commands: dict[int, FixtureCommand] = {}

        # Single par breathing — pick the first one
        if self._pars:
            seg_time = state.timestamp - self._segment_start_time
            breath_cmds = breathe(
                self._pars[:1], state, seg_time, SODIUM_AMBER,
                min_intensity=0.05, max_intensity=0.15, period_bars=2.0,
            )
            commands.update(breath_cmds)

        # All other pars off
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
        """Intro/outro: minimal dirty amber wash, slow breathe.

        Low and atmospheric.  The room feels like it is just waking
        up (intro) or winding down (outro).

        Args:
            state: Current music state.

        Returns:
            Fixture command list.
        """
        commands: dict[int, FixtureCommand] = {}

        # Slow breathe on 2 pars — sodium amber at very low intensity
        active_pars = self._pars[:2] if len(self._pars) >= 2 else self._pars
        par_cmds = breathe(
            active_pars, state, state.timestamp, SODIUM_AMBER,
            min_intensity=0.08, max_intensity=0.20, period_bars=2.0,
            phase_offset_per_fixture=0.5,
        )
        commands.update(par_cmds)

        # Blackout remaining pars
        for f in self._pars:
            if f.fixture_id not in commands:
                commands[f.fixture_id] = make_command(f, BLACK, 0.0)

        # Everything else off
        for f in self._strobes + self._led_bars:
            commands[f.fixture_id] = make_command(f, BLACK, 0.0)
        for f in self._lasers:
            commands[f.fixture_id] = make_command(f, BLACK, special=_LASER_OFF)

        return self._merge_commands(commands)
