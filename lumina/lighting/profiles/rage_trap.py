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

from lumina.audio.models import MusicState
from lumina.control.protocol import FixtureCommand
from lumina.lighting.fixture_map import FixtureMap, FixtureType
from lumina.lighting.patterns import (
    blinder,
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
    BumpTracker,
    Color,
    FixtureInfo,
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

        # Override bump tracker: 50ms snap — rage is sharp
        self._bump = BumpTracker(decay_rate=20.0)

        # State tracking
        self._segment_start_time: float = 0.0
        self._last_segment: str = ""
        self._segment_changed: bool = False
        self._drop_frame_count: int = 0
        self._was_pre_drop: bool = False

    @property
    def motif_pattern_preferences(self) -> list[str]:
        """Rage trap prefers aggressive, binary patterns for motifs."""
        return ["chase_lr", "alternate", "strobe_burst", "random_scatter"]

    @property
    def motif_color_palette(self) -> list[Color]:
        """Rage trap color cycle: reds and whites only."""
        return [BLOOD_RED, DEEP_RED, WARM_RED, DARK_ORANGE]

    def _headroom_scale(self, state: MusicState, intensity: float) -> float:
        """Apply headroom to intensity, rage-trap style.

        Rage trap keeps strobes at full but scales par/bar intensity.
        During drops, headroom is ignored (always full).

        Args:
            state: Current music state.
            intensity: Raw intensity value.

        Returns:
            Headroom-adjusted intensity.
        """
        if state.segment == "drop":
            return intensity  # drops are always full
        return intensity * max(0.15, state.headroom)

    def _get_active_pars(self, state: MusicState) -> list[FixtureInfo]:
        """Get active pars based on layer count (if available) or energy.

        Uses layer_count for fixture selection when available, falling
        back to energy-based selection.

        Args:
            state: Current music state.

        Returns:
            List of active par fixtures.
        """
        if state.layer_count > 0:
            n = self._layer_fixture_count(
                state.layer_count, state.energy, len(self._pars)
            )
            return self._pars_lr[:n]
        # Fallback to energy-based selection
        return select_active_fixtures(self._pars, state.energy)

    def generate(self, state: MusicState) -> list[FixtureCommand]:
        """Generate rage-trap fixture commands for the current frame.

        Decision hierarchy:
        1. Arc layer: headroom caps intensity (except drops)
        2. Motif repetition: escalate to strobe burst when crowd knows the part
        3. Note pattern: if regular notes detected, cycle fixtures
        4. Segment layer: existing verse/chorus/drop routing
        5. Reactive layer: onset reactions capped by headroom

        Args:
            state: Current music analysis frame.

        Returns:
            One FixtureCommand per fixture (15 total).
        """
        self._begin_debug_frame()
        self._store_headroom(state)
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

        # Pre-drop: accelerating strobe (bypass motif/note layers)
        if state.drop_probability > 0.6 and segment != "drop":
            self._was_pre_drop = True
            self._note_patterns("pre_drop_build")
            return self._apply_headroom(self._pre_drop_build(state))

        # Drop hit (bypass motif/note layers — drops are always full)
        if segment == "drop" and energy > 0.5:
            self._note_patterns("drop_explosion")
            result = self._drop_explosion(state)
            self._was_pre_drop = False
            return result  # drops bypass headroom

        self._was_pre_drop = False

        # ─── Motif repetition escalation ──────────────────────────
        # When a motif has repeated 4+ times, the crowd knows this part.
        # Escalate to strobe burst on beats for visceral recognition.
        motif_rep = getattr(state, "motif_repetition", 0)
        if (
            motif_rep > 3
            and state.is_beat
            and segment not in ("breakdown", "bridge", "intro", "outro")
        ):
            self._note_patterns("motif_escalation")
            commands: dict[int, FixtureCommand] = {}
            burst = strobe_burst(
                self._strobes, state, state.timestamp, STROBE_WHITE,
            )
            commands.update(burst)
            active_pars = self._get_active_pars(state)
            par_cmds = wash_hold(
                active_pars, state, state.timestamp, BLOOD_RED,
                intensity=self._headroom_scale(state, 1.0),
            )
            commands.update(par_cmds)
            for f in self._pars:
                if f.fixture_id not in commands:
                    commands[f.fixture_id] = make_command(f, BLACK, 0.0)
            for f in self._led_bars:
                commands[f.fixture_id] = make_command(f, BLACK, 0.0)
            for f in self._lasers:
                commands[f.fixture_id] = make_command(f, BLACK, special=_LASER_OFF)
            return self._apply_headroom(self._merge_commands(commands))

        # ─── Layer count sparse mode ──────────────────────────────
        # layer_count < 2: single red spotlight only (minimal)
        layer_count = getattr(state, "layer_count", 0)
        if (
            layer_count > 0
            and layer_count < 2
            and segment not in ("drop", "breakdown", "bridge", "intro", "outro")
        ):
            self._note_patterns("sparse_spotlight")
            return self._apply_headroom(self._sparse_spotlight(state))

        # ─── Note-level pattern (each note = different light) ─────
        if state.notes_per_beat > 0 and segment not in ("breakdown", "bridge", "intro", "outro"):
            active_pars = self._get_active_pars(state)
            note_cmds = self._apply_note_pattern(state, active_pars, BLOOD_RED)
            if note_cmds is not None:
                self._note_patterns("note_pattern")
                # Add strobes/bars/lasers as quiet background
                for f in self._strobes + self._led_bars:
                    note_cmds[f.fixture_id] = make_command(f, BLACK, 0.0)
                for f in self._lasers:
                    note_cmds[f.fixture_id] = make_command(f, BLACK, special=_LASER_OFF)
                return self._apply_headroom(self._merge_commands(note_cmds))

        # ─── Segment-based routing (with headroom awareness) ──────

        # Breakdown / bridge
        if segment in ("breakdown", "bridge"):
            self._note_patterns("breakdown_breathe")
            return self._apply_headroom(self._breakdown_breathe(state))

        # Intro / outro
        if segment in ("intro", "outro"):
            self._note_patterns("intro_outro")
            return self._apply_headroom(self._intro_outro(state))

        # Chorus / hook — brighter and wider than verse
        if segment == "chorus":
            self._note_patterns("chorus_reactive")
            return self._apply_headroom(self._chorus_reactive(state))

        # Vocal calm
        if state.vocal_energy > 0.6:
            self._note_patterns("vocal_calm")
            return self._apply_headroom(self._vocal_calm(state))

        self._note_patterns("verse_reactive")
        return self._apply_headroom(self._verse_reactive(state))

    # ─── Extended MusicState handlers ─────────────────────────────

    def _sparse_spotlight(self, state: MusicState) -> list[FixtureCommand]:
        """Single red spotlight when layer_count < 2 (sparse instrumentation).

        Only one par is lit at very low intensity. Everything else dark.
        Sparse music deserves sparse lighting.

        Args:
            state: Current music state.

        Returns:
            Fixture command list.
        """
        commands: dict[int, FixtureCommand] = {}

        par_cmds = spotlight_isolate(
            self._pars, state, state.timestamp, DEEP_RED,
            target_index=0, intensity=self._headroom_scale(state, 0.25),
            dim_others=0.0,
        )
        commands.update(par_cmds)

        for f in self._strobes + self._led_bars:
            commands[f.fixture_id] = make_command(f, BLACK, 0.0)
        for f in self._lasers:
            commands[f.fixture_id] = make_command(f, BLACK, special=_LASER_OFF)

        return self._merge_commands(commands)

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
            if self._drop_frame_count <= 6:
                # Frames 1-6: blinder — every fixture at absolute max
                all_fixtures = self._pars + self._strobes + self._led_bars
                blind_cmds = blinder(
                    all_fixtures, state, state.timestamp, STROBE_WHITE,
                )
                commands.update(blind_cmds)
            else:
                # Frames 7-12: strobe burst + white wash (existing behavior)
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

        Uses layer_count for fixture escalation (with energy fallback).
        Onset reactions capped by headroom. Headroom scales par intensity.
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

        # Fixture count: layer-driven (with energy/time fallback)
        active_pars = self._get_active_pars(state)
        if not active_pars:
            # Fallback: energy-based escalation
            active_pars = select_active_fixtures(
                self._pars, energy,
                low_count=max(1, 1 + int(bars_elapsed / _BARS_PER_FIXTURE_ADD)),
                mid_count=6, high_threshold=0.8,
            )

        # Color temperature: cooler → warmer as energy rises
        verse_color = lerp_color(COOL_RED, WARM_RED, energy)

        # Sub-bass saturation: deepen color when sub-bass is heavy
        if state.sub_bass_energy > 0.5:
            verse_color = self._bass_saturate(state.sub_bass_energy, verse_color)

        # Onset reactions (headroom caps intensity)
        if state.onset_type == "kick":
            self._bump.trigger("pars", state.timestamp)
            kick_int = self._headroom_scale(state, 1.0)
            par_cmds = wash_hold(
                active_pars, state, state.timestamp, BLOOD_RED, intensity=kick_int,
            )
            commands.update(par_cmds)
            # Strobes alternate left/right on kicks (strobes bypass headroom)
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
            snare_int = self._headroom_scale(state, 1.0)
            par_cmds = random_scatter(
                active_pars, state, state.timestamp, STROBE_WHITE,
                density=1.0 / max(len(active_pars), 1), intensity=snare_int,
            )
            commands.update(par_cmds)
            for f in self._strobes:
                commands[f.fixture_id] = make_command(f, BLACK, 0.0)

        elif state.onset_type == "clap":
            clap_int = self._headroom_scale(state, 1.0)
            par_cmds = random_scatter(
                self._pars, state, state.timestamp, DARK_ORANGE,
                density=0.4, intensity=clap_int,
            )
            commands.update(par_cmds)
            for f in self._strobes:
                commands[f.fixture_id] = make_command(f, BLACK, 0.0)

        elif state.is_beat:
            beat_int = self._headroom_scale(state, 0.85)
            par_cmds = chase_lr(
                active_pars, state, state.timestamp, verse_color,
                speed=1.0, width=0.4, intensity=beat_int,
            )
            commands.update(par_cmds)
            for f in self._strobes:
                commands[f.fixture_id] = make_command(f, BLACK, 0.0)
        else:
            # Between beats: energy-aware floor scaled by headroom
            if energy > 0.6:
                between_intensity = 0.20 + (energy - 0.6) / 0.4 * 0.10
            elif energy < 0.4:
                between_intensity = 0.0
            else:
                between_intensity = 0.05
            between_intensity = self._headroom_scale(state, between_intensity)
            # Bump decay: smooth tail after kick instead of instant drop
            between_intensity = self._bump.get_intensity(
                "pars", state.timestamp, peak=1.0, floor=between_intensity,
            )
            par_cmds = wash_hold(
                active_pars, state, state.timestamp, verse_color, intensity=between_intensity,
            )
            commands.update(par_cmds)
            for f in self._strobes:
                commands[f.fixture_id] = make_command(f, BLACK, 0.0)

        # Hi-hat: bump LED bars on hi-hat onsets
        if state.onset_type == "hihat":
            self._bump.trigger("bars", state.timestamp)

        # Blackout inactive pars
        for f in self._pars:
            if f.fixture_id not in commands:
                commands[f.fixture_id] = make_command(f, BLACK, 0.0)

        # LED bars: follow pars at reduced intensity during verse + hi-hat bump
        bar_int = self._headroom_scale(state, 0.10)
        bar_int = self._bump.get_intensity(
            "bars", state.timestamp, peak=0.40, floor=bar_int,
        )
        bar_cmds = wash_hold(
            self._led_bars, state, state.timestamp, verse_color, intensity=bar_int,
        )
        commands.update(bar_cmds)

        # Laser off during verse
        for f in self._lasers:
            commands[f.fixture_id] = make_command(f, BLACK, special=_LASER_OFF)

        return self._merge_commands(commands)

    def _chorus_reactive(self, state: MusicState) -> list[FixtureCommand]:
        """Chorus / hook — brighter and wider than verse.

        Layer-aware fixture count. Headroom scales par intensity.
        Beat-driven strobes. LED bars scaled by headroom. Laser off.
        """
        commands: dict[int, FixtureCommand] = {}
        energy = state.energy
        bpm = max(60.0, state.bpm)

        # Chorus uses more pars than verse — layer-driven or all
        if state.layer_count >= 3:
            chorus_pars = self._pars  # dense = all pars
        else:
            chorus_pars = self._get_active_pars(state)
            if len(chorus_pars) < len(self._pars) // 2:
                chorus_pars = self._pars  # chorus should be wide

        chorus_color = lerp_color(WARM_RED, BLOOD_RED, energy)

        # Sub-bass saturation: deepen color when sub-bass is heavy
        if state.sub_bass_energy > 0.5:
            chorus_color = self._bass_saturate(state.sub_bass_energy, chorus_color)

        base_intensity = self._headroom_scale(state, 0.35 + energy * 0.15)

        # Onset reactions (headroom scales par intensity)
        if state.onset_type == "kick":
            self._bump.trigger("pars", state.timestamp)
            kick_int = self._headroom_scale(state, 1.0)
            par_cmds = wash_hold(
                chorus_pars, state, state.timestamp, BLOOD_RED, intensity=kick_int,
            )
            commands.update(par_cmds)
            burst = strobe_burst(self._strobes, state, state.timestamp, STROBE_WHITE)
            commands.update(burst)

        elif state.onset_type == "snare":
            snare_int = self._headroom_scale(state, 1.0)
            par_cmds = random_scatter(
                chorus_pars, state, state.timestamp, STROBE_WHITE,
                density=0.5, intensity=snare_int,
            )
            commands.update(par_cmds)
            for f in self._strobes:
                commands[f.fixture_id] = make_command(f, BLACK, 0.0)

        elif state.onset_type == "clap":
            clap_int = self._headroom_scale(state, 1.0)
            par_cmds = random_scatter(
                chorus_pars, state, state.timestamp, DARK_ORANGE,
                density=0.6, intensity=clap_int,
            )
            commands.update(par_cmds)
            for f in self._strobes:
                commands[f.fixture_id] = make_command(f, BLACK, 0.0)

        elif state.is_beat:
            beat_int = self._headroom_scale(state, 0.90)
            par_cmds = chase_lr(
                chorus_pars, state, state.timestamp, chorus_color,
                speed=1.0, width=0.5, intensity=beat_int,
            )
            commands.update(par_cmds)
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
            # Bump decay: smooth tail after kick instead of instant drop
            chorus_between = self._bump.get_intensity(
                "pars", state.timestamp, peak=1.0, floor=base_intensity,
            )
            par_cmds = wash_hold(
                chorus_pars, state, state.timestamp, chorus_color, intensity=chorus_between,
            )
            commands.update(par_cmds)
            for f in self._strobes:
                commands[f.fixture_id] = make_command(f, BLACK, 0.0)

        # Hi-hat: bump LED bars on hi-hat onsets
        if state.onset_type == "hihat":
            self._bump.trigger("bars", state.timestamp)

        # Blackout inactive pars in chorus
        for f in self._pars:
            if f.fixture_id not in commands:
                commands[f.fixture_id] = make_command(f, BLACK, 0.0)

        # LED bars scaled by headroom (40% base) + hi-hat bump
        bar_int = self._headroom_scale(state, 0.40)
        bar_int = self._bump.get_intensity(
            "bars", state.timestamp, peak=0.60, floor=bar_int,
        )
        bar_cmds = wash_hold(
            self._led_bars, state, state.timestamp, chorus_color, intensity=bar_int,
        )
        commands.update(bar_cmds)

        # Laser off during chorus
        for f in self._lasers:
            commands[f.fixture_id] = make_command(f, BLACK, special=_LASER_OFF)

        return self._merge_commands(commands)
