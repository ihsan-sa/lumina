"""Profile 5: European Alt Hip-Hop -- AyVe, Exetra Archive.

Core philosophy: VISUAL SILENCE, GALLERY RESTRAINT, MONOCHROME REVEAL.

Lighting language:
- Colors: WHITE only (warm/cool/dim via white channel). No color except
  rare ACCENT_REVEAL (muted orange-red) every 8 bars in chorus.
- Restraint: Most fixtures OFF most of the time. Darkness is the
  dominant visual. One or two lit fixtures carry the entire verse.
- Verse: spotlight_isolate rotating through pars (one at a time, 20-30%).
  No strobes ever. LED bars at 5%.
- Chorus: 2-3 pars active via gradient_y. Rare accent color reveal
  every 8 bars -- one par flashes ACCENT_REVEAL for 1 frame.
- Drop: converge on all pars (first time all pars are lit). Moderate
  intensity (70% max). LED bars at 60%. No strobes.
- Breakdown: complete blackout except 1 LED bar at 10%.
- Intro/outro: breathe on 1 par, DIM_WHITE, period_bars=4.0.
- Spectral centroid: warm white -> cool white via _color_temperature().
- No strobes in any segment. Laser always off.
"""

from __future__ import annotations

from lumina.audio.models import MusicState
from lumina.control.protocol import FixtureCommand
from lumina.lighting.fixture_map import FixtureMap, FixtureType
from lumina.lighting.patterns import (
    breathe,
    converge,
    gradient_y,
    make_command,
    select_active_fixtures,
    spotlight_isolate,
    wash_hold,
)
from lumina.lighting.profiles.base import (
    BLACK,
    BaseProfile,
    BumpTracker,
    Color,
    lerp_color,
)

# --- Euro Alt palette -------------------------------------------------------

WARM_WHITE = Color(0.0, 0.0, 0.0, 0.6)
COOL_WHITE = Color(0.0, 0.0, 0.0, 0.8)
ACCENT_REVEAL = Color(0.8, 0.2, 0.1, 0.0)
DIM_WHITE = Color(0.0, 0.0, 0.0, 0.3)

_LASER_OFF = 0

# Accent reveal fires every N bars on a downbeat
_ACCENT_INTERVAL_BARS = 8


class EuroAltProfile(BaseProfile):
    """European Alt Hip-Hop lighting profile.

    Gallery restraint philosophy: visual silence is the primary tool.
    Most fixtures are dark most of the time. One or two white fixtures
    carry the verse. Color appears only as a rare punctuation mark.
    No strobes in any segment.

    Args:
        fixture_map: Venue fixture layout.
    """

    name = "euro_alt"

    def __init__(self, fixture_map: FixtureMap) -> None:
        super().__init__(fixture_map)
        self._pars = self._map.by_type(FixtureType.PAR)
        self._strobes = self._map.by_type(FixtureType.STROBE)
        self._led_bars = self._map.by_type(FixtureType.LED_BAR)
        self._lasers = self._map.by_type(FixtureType.LASER)

        # Gentle bump decay (200ms half-life)
        self._bump = BumpTracker(decay_rate=5.0)

        # Track accent reveals: fires every 8 bars on downbeat
        self._accent_bar_counter: int = 0

        # Motif-driven temperature shift state
        self._last_motif_id: int | None = None
        self._white_temp_offset: float = 0.0

    @property
    def motif_pattern_preferences(self) -> list[str]:
        """Euro alt prefers restrained, spatial patterns for motifs."""
        return ["spotlight_isolate", "gradient_y", "breathe", "converge"]

    @property
    def motif_color_palette(self) -> list[Color]:
        """Euro alt color cycle: whites with rare accent."""
        return [WARM_WHITE, COOL_WHITE, DIM_WHITE, ACCENT_REVEAL]

    def generate(self, state: MusicState) -> list[FixtureCommand]:
        """Generate euro-alt fixture commands for the current frame.

        Decision hierarchy:
        1. Segment routing (verse/chorus/drop/breakdown/intro/outro)
        2. Color temperature from spectral centroid
        3. Accent reveal counter on downbeats

        No strobes ever. Laser always off.

        Args:
            state: Current audio analysis frame.

        Returns:
            One FixtureCommand per fixture (15 total).
        """
        self._begin_debug_frame()
        self._store_headroom(state)
        segment = state.segment

        # Count bars for accent reveal timing
        if state.is_downbeat:
            self._accent_bar_counter += 1

        # Track motif changes for slow white temperature shift
        motif_id = getattr(state, "motif_id", None)
        if motif_id is not None and motif_id != self._last_motif_id:
            # Each motif change shifts the warm/cool balance by 0.2
            self._white_temp_offset = (self._white_temp_offset + 0.2) % 1.0
            self._last_motif_id = motif_id

        # Layer count < 2: extreme single-spotlight restraint (override segment)
        layer_count = getattr(state, "layer_count", 0)
        if (
            layer_count > 0
            and layer_count < 2
            and segment not in ("drop",)
        ):
            self._note_patterns("sparse_single_spotlight")
            return self._apply_headroom(self._sparse_single_spotlight(state))

        if segment == "drop":
            self._note_patterns("drop_converge")
            return self._apply_headroom(self._drop(state))

        if segment == "chorus":
            self._note_patterns("chorus_gradient")
            return self._apply_headroom(self._chorus(state))

        if segment in ("breakdown", "bridge"):
            self._note_patterns("breakdown_blackout")
            return self._apply_headroom(self._breakdown(state))

        if segment in ("intro", "outro"):
            self._note_patterns("intro_outro_breathe")
            return self._apply_headroom(self._intro_outro(state))

        # Default: verse
        self._note_patterns("verse_spotlight")
        return self._apply_headroom(self._verse(state))

    # --- Extended MusicState handlers ----------------------------------------

    def _motif_white_color(self, state: MusicState) -> Color:
        """Compute white color with motif-driven temperature shift.

        Each new motif slowly shifts the warm/cool balance, giving each
        musical theme a subtly different visual warmth.

        Args:
            state: Current music state.

        Returns:
            Temperature-shifted white Color.
        """
        base = self._color_temperature(
            state.spectral_centroid, WARM_WHITE, COOL_WHITE,
        )
        # Apply motif-driven temperature offset
        if self._white_temp_offset > 0.01:
            shifted = lerp_color(WARM_WHITE, COOL_WHITE, self._white_temp_offset)
            base = lerp_color(base, shifted, 0.3)
        return base

    def _sparse_single_spotlight(self, state: MusicState) -> list[FixtureCommand]:
        """Single spotlight when layer_count < 2 (extreme restraint).

        One par at very low intensity. Everything else completely dark.
        This is the most minimal state for euro_alt -- nearly invisible.

        Args:
            state: Current music state.

        Returns:
            Fixture command list.
        """
        commands: dict[int, FixtureCommand] = {}

        color = self._motif_white_color(state)

        par_cmds = spotlight_isolate(
            self._pars, state, state.timestamp, color,
            target_index=0, intensity=0.15, dim_others=0.0,
        )
        commands.update(par_cmds)

        for f in self._led_bars:
            commands[f.fixture_id] = make_command(f, BLACK, 0.0)
        for f in self._strobes:
            commands[f.fixture_id] = make_command(f, BLACK, 0.0)
        for f in self._lasers:
            commands[f.fixture_id] = make_command(f, BLACK, special=_LASER_OFF)

        return self._merge_commands(commands)

    # --- Segment handlers ---------------------------------------------------

    def _verse(self, state: MusicState) -> list[FixtureCommand]:
        """Verse: single rotating spotlight, extreme restraint.

        One par at a time at 20-30%. No strobes. LED bars at 5%.
        Motif changes shift white temperature.
        """
        commands: dict[int, FixtureCommand] = {}

        # Color temperature from spectral centroid + motif-driven shift
        color = self._motif_white_color(state)

        # Rotating spotlight: one par at a time, slow rotation
        n_pars = len(self._pars)
        target_index = int(state.timestamp * 0.25) % max(n_pars, 1)
        intensity = 0.20 + state.energy * 0.10  # 20-30%

        par_cmds = spotlight_isolate(
            self._pars, state, state.timestamp, color,
            target_index=target_index, intensity=intensity, dim_others=0.0,
        )
        commands.update(par_cmds)

        # LED bars at 5%
        bar_cmds = wash_hold(
            self._led_bars, state, state.timestamp, DIM_WHITE, intensity=0.05,
        )
        commands.update(bar_cmds)

        # Strobes always off
        for f in self._strobes:
            commands[f.fixture_id] = make_command(f, BLACK, 0.0)

        # Laser always off
        for f in self._lasers:
            commands[f.fixture_id] = make_command(f, BLACK, special=_LASER_OFF)

        return self._merge_commands(commands)

    def _chorus(self, state: MusicState) -> list[FixtureCommand]:
        """Chorus: 2-3 pars with gradient, rare accent reveal.

        Gradient_y (front-to-back brightness) using white. Every 8 bars
        on a downbeat, one par flashes ACCENT_REVEAL for 1 frame.
        Motif changes shift white temperature.
        """
        commands: dict[int, FixtureCommand] = {}

        # Color temperature from spectral centroid + motif-driven shift
        color = self._motif_white_color(state)

        # Select 2-3 pars based on energy
        active_pars = select_active_fixtures(
            self._pars, state.energy,
            low_count=2, mid_count=3, high_threshold=0.9,
        )

        # Gradient front-to-back using white
        par_cmds = gradient_y(
            active_pars, state, state.timestamp, color,
            intensity=0.35 + state.energy * 0.15,
        )
        commands.update(par_cmds)

        # Blackout inactive pars
        for f in self._pars:
            if f.fixture_id not in commands:
                commands[f.fixture_id] = make_command(f, BLACK, 0.0)

        # Rare accent reveal: every 8 bars on downbeat, flash one par
        if (state.is_downbeat
                and self._accent_bar_counter > 0
                and self._accent_bar_counter % _ACCENT_INTERVAL_BARS == 0
                and active_pars):
            # Flash the first active par with accent color
            accent_target = active_pars[0]
            commands[accent_target.fixture_id] = make_command(
                accent_target, ACCENT_REVEAL, 0.80,
            )

        # LED bars at 15% during chorus
        bar_cmds = wash_hold(
            self._led_bars, state, state.timestamp, DIM_WHITE, intensity=0.15,
        )
        commands.update(bar_cmds)

        # Strobes always off
        for f in self._strobes:
            commands[f.fixture_id] = make_command(f, BLACK, 0.0)

        # Laser always off
        for f in self._lasers:
            commands[f.fixture_id] = make_command(f, BLACK, special=_LASER_OFF)

        return self._merge_commands(commands)

    def _drop(self, state: MusicState) -> list[FixtureCommand]:
        """Drop: converge on ALL pars -- the only segment with full activation.

        Moderate intensity (70% max). LED bars at 60%. No strobes.
        This is the climax: contrast comes from everything else being dark.
        """
        commands: dict[int, FixtureCommand] = {}

        # Color temperature from spectral centroid
        color = self._color_temperature(state.spectral_centroid, WARM_WHITE, COOL_WHITE)

        # Converge: all pars activate (first time in the song)
        par_cmds = converge(
            self._pars, state, state.timestamp, color,
            intensity=0.70,
        )
        commands.update(par_cmds)

        # LED bars at 60%
        bar_cmds = wash_hold(
            self._led_bars, state, state.timestamp, color, intensity=0.60,
        )
        commands.update(bar_cmds)

        # Strobes always off
        for f in self._strobes:
            commands[f.fixture_id] = make_command(f, BLACK, 0.0)

        # Laser always off
        for f in self._lasers:
            commands[f.fixture_id] = make_command(f, BLACK, special=_LASER_OFF)

        return self._merge_commands(commands)

    def _breakdown(self, state: MusicState) -> list[FixtureCommand]:
        """Breakdown: near-complete blackout. 1 LED bar at 10%.

        Everything else off. Visual silence at its most extreme.
        """
        commands: dict[int, FixtureCommand] = {}

        # All pars off
        for f in self._pars:
            commands[f.fixture_id] = make_command(f, BLACK, 0.0)

        # 1 LED bar at 10% (spotlight_isolate on led_bars)
        if self._led_bars:
            bar_cmds = spotlight_isolate(
                self._led_bars, state, state.timestamp, DIM_WHITE,
                target_index=0, intensity=0.10, dim_others=0.0,
            )
            commands.update(bar_cmds)

        # Strobes always off
        for f in self._strobes:
            commands[f.fixture_id] = make_command(f, BLACK, 0.0)

        # Laser always off
        for f in self._lasers:
            commands[f.fixture_id] = make_command(f, BLACK, special=_LASER_OFF)

        return self._merge_commands(commands)

    def _intro_outro(self, state: MusicState) -> list[FixtureCommand]:
        """Intro/outro: single par breathing slowly, DIM_WHITE.

        Period = 4 bars. Intensity 15-20%. Everything else off.
        """
        commands: dict[int, FixtureCommand] = {}

        # Breathe on 1 par only
        if self._pars:
            par_cmds = breathe(
                self._pars[:1], state, state.timestamp, DIM_WHITE,
                min_intensity=0.15, max_intensity=0.20, period_bars=4.0,
            )
            commands.update(par_cmds)

        # Blackout remaining pars
        for f in self._pars[1:]:
            commands[f.fixture_id] = make_command(f, BLACK, 0.0)

        # LED bars off
        for f in self._led_bars:
            commands[f.fixture_id] = make_command(f, BLACK, 0.0)

        # Strobes always off
        for f in self._strobes:
            commands[f.fixture_id] = make_command(f, BLACK, 0.0)

        # Laser always off
        for f in self._lasers:
            commands[f.fixture_id] = make_command(f, BLACK, special=_LASER_OFF)

        return self._merge_commands(commands)
