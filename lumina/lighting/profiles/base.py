"""Base class for genre-specific lighting profiles.

All genre profiles inherit from BaseProfile and override ``generate()``
to produce per-fixture commands from the current MusicState.  The base
class provides spatial pattern helpers, color math, intensity curves,
strobe timing, and blackout utilities so profiles can focus on artistic
decisions rather than low-level fixture wrangling.
"""

from __future__ import annotations

import colorsys
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

from lumina.audio.models import MusicState
from lumina.control.protocol import FixtureCommand
from lumina.lighting.fixture_map import FixtureInfo, FixtureMap, FixtureType


# ─── Color helpers ──────────────────────────────────────────────────


@dataclass(slots=True)
class Color:
    """RGBW color value (all channels 0.0-1.0 float).

    Args:
        r: Red channel.
        g: Green channel.
        b: Blue channel.
        w: White channel.
    """

    r: float = 0.0
    g: float = 0.0
    b: float = 0.0
    w: float = 0.0

    def scaled(self, intensity: float) -> Color:
        """Return a copy scaled by *intensity* (0.0-1.0).

        Args:
            intensity: Scale factor.

        Returns:
            New Color with all channels multiplied.
        """
        return Color(
            r=self.r * intensity,
            g=self.g * intensity,
            b=self.b * intensity,
            w=self.w * intensity,
        )

    def to_bytes(self) -> tuple[int, int, int, int]:
        """Convert to 0-255 integer tuple (R, G, B, W).

        Returns:
            Clamped 8-bit RGBW tuple.
        """
        return (
            _clamp8(self.r),
            _clamp8(self.g),
            _clamp8(self.b),
            _clamp8(self.w),
        )


def clamp8(v: float) -> int:
    """Clamp float 0.0-1.0 to int 0-255."""
    return max(0, min(255, int(v * 255)))


# Keep the old name as an alias for backward compatibility
_clamp8 = clamp8


BLACK = Color(0.0, 0.0, 0.0, 0.0)
WHITE = Color(1.0, 1.0, 1.0, 1.0)
RED = Color(1.0, 0.0, 0.0, 0.0)


def color_from_hsv(h: float, s: float, v: float) -> Color:
    """Create a Color from HSV values (all 0.0-1.0).

    Args:
        h: Hue (0.0-1.0, wraps).
        s: Saturation (0.0-1.0).
        v: Value / brightness (0.0-1.0).

    Returns:
        RGB Color (white channel is 0).
    """
    r, g, b = colorsys.hsv_to_rgb(h % 1.0, s, v)
    return Color(r=r, g=g, b=b, w=0.0)


def lerp_color(a: Color, b: Color, t: float) -> Color:
    """Linearly interpolate between two colors.

    Args:
        a: Start color.
        b: End color.
        t: Interpolation factor (0.0 = a, 1.0 = b).

    Returns:
        Blended Color.
    """
    t = max(0.0, min(1.0, t))
    return Color(
        r=a.r + (b.r - a.r) * t,
        g=a.g + (b.g - a.g) * t,
        b=a.b + (b.b - a.b) * t,
        w=a.w + (b.w - a.w) * t,
    )


# ─── Intensity curves ──────────────────────────────────────────────


def sine_pulse(phase: float, power: float = 1.0) -> float:
    """Sine-based pulse from 0→1→0 over a full phase cycle.

    Args:
        phase: 0.0-1.0 position in the cycle.
        power: Exponent to sharpen the pulse (>1 = sharper).

    Returns:
        Intensity value 0.0-1.0.
    """
    raw = (math.sin(phase * math.pi * 2.0 - math.pi / 2.0) + 1.0) / 2.0
    return raw**power


def triangle_wave(phase: float) -> float:
    """Triangle wave: 0→1→0 over a full phase cycle.

    Args:
        phase: 0.0-1.0 position.

    Returns:
        Intensity 0.0-1.0.
    """
    p = phase % 1.0
    return 1.0 - abs(2.0 * p - 1.0)


def ease_in_out(t: float) -> float:
    """Smooth ease-in-out (cubic).

    Args:
        t: 0.0-1.0 input.

    Returns:
        Smoothed 0.0-1.0 output.
    """
    t = max(0.0, min(1.0, t))
    if t < 0.5:
        return 4.0 * t * t * t
    return 1.0 - (-2.0 * t + 2.0) ** 3 / 2.0


def energy_brightness(energy: float, gamma: float = 0.5) -> float:
    """Map energy to perceived brightness using a power curve.

    Boosts low and mid energy values so lights feel alive even during
    quieter sections.  Default gamma=0.5 (square root) maps::

        energy 0.2 → 0.45
        energy 0.4 → 0.63
        energy 0.6 → 0.77
        energy 0.8 → 0.89

    Args:
        energy: 0.0-1.0 energy level from audio analysis.
        gamma: Power exponent (<1.0 boosts low values, >1.0 compresses).

    Returns:
        Mapped brightness 0.0-1.0.
    """
    return max(0.0, min(1.0, energy)) ** gamma


# ─── Shared command routing ────────────────────────────────────────


def route_command(
    fixture: FixtureInfo,
    color: Color = BLACK,
    intensity: float = 1.0,
    strobe_rate: int = 0,
    strobe_intensity: int = 0,
    special: int | None = None,
) -> FixtureCommand:
    """Build a FixtureCommand with fixture-type-aware channel routing.

    This is the shared routing logic used by both BaseProfile._cmd() and
    the standalone pattern functions in lumina.lighting.patterns.

    Args:
        fixture: Target fixture.
        color: Desired color (already scaled by intensity if needed).
        intensity: Master intensity 0.0-1.0 (used for auto-deriving special).
        strobe_rate: Strobe rate 0-255.
        strobe_intensity: Strobe brightness 0-255.
        special: Override for the special byte.

    Returns:
        FixtureCommand for this fixture.
    """
    r, g, b, w = color.to_bytes()

    if fixture.fixture_type == FixtureType.PAR:
        sp = special if special is not None else _clamp8(intensity)
        return FixtureCommand(
            fixture_id=fixture.fixture_id,
            red=r, green=g, blue=b, white=w,
            strobe_rate=0, strobe_intensity=0, special=sp,
        )
    elif fixture.fixture_type == FixtureType.STROBE:
        return FixtureCommand(
            fixture_id=fixture.fixture_id,
            red=r, green=g, blue=b, white=w,
            strobe_rate=strobe_rate, strobe_intensity=strobe_intensity,
            special=special if special is not None else 0,
        )
    elif fixture.fixture_type == FixtureType.LED_BAR:
        # LED bars behave like pars: RGBW color wash, special = master dimmer
        sp = special if special is not None else _clamp8(intensity)
        return FixtureCommand(
            fixture_id=fixture.fixture_id,
            red=r, green=g, blue=b, white=w,
            strobe_rate=0, strobe_intensity=0, special=sp,
        )
    elif fixture.fixture_type == FixtureType.LASER:
        # Laser: special = pattern_id (0 = off). RGBW/strobe ignored.
        sp = special if special is not None else 0
        return FixtureCommand(
            fixture_id=fixture.fixture_id,
            red=0, green=0, blue=0, white=0,
            strobe_rate=0, strobe_intensity=0, special=sp,
        )
    else:  # UV
        sp = special if special is not None else _clamp8(intensity)
        return FixtureCommand(
            fixture_id=fixture.fixture_id,
            red=0, green=0, blue=0, white=0,
            strobe_rate=0, strobe_intensity=0, special=sp,
        )


# ─── BaseProfile ────────────────────────────────────────────────────


class BaseProfile(ABC):
    """Abstract base for genre-specific lighting profiles.

    Subclasses must implement ``generate()`` which receives the current
    MusicState and fixture map and returns per-fixture commands.

    The base class provides spatial pattern generators and utility
    methods that profiles combine to create their lighting language.

    Args:
        fixture_map: The venue's fixture layout.
    """

    # Human-readable profile name (override in subclasses)
    name: str = "base"

    def __init__(self, fixture_map: FixtureMap) -> None:
        self._map = fixture_map

    @abstractmethod
    def generate(self, state: MusicState) -> list[FixtureCommand]:
        """Generate fixture commands for the current music state.

        Args:
            state: Current audio analysis frame.

        Returns:
            One FixtureCommand per fixture in the map.
        """
        ...

    # ─── Command builders ────────────────────────────────────────

    def _cmd(
        self,
        fixture: FixtureInfo,
        color: Color = BLACK,
        intensity: float = 1.0,
        strobe_rate: int = 0,
        strobe_intensity: int = 0,
        special: int | None = None,
    ) -> FixtureCommand:
        """Build a FixtureCommand for a fixture.

        Automatically routes channels based on fixture type:
        - PAR: RGBW color, special = master dimmer
        - STROBE: strobe channels, base color for strobe tint
        - UV: special = UV intensity

        Args:
            fixture: Target fixture.
            color: Desired color.
            intensity: Master intensity 0.0-1.0.
            strobe_rate: Strobe rate 0-255.
            strobe_intensity: Strobe brightness 0-255.
            special: Override for the special byte.  If None, pars and UV
                bars auto-derive from intensity.

        Returns:
            FixtureCommand for this fixture.
        """
        scaled = color.scaled(intensity)
        return route_command(
            fixture, color=scaled, intensity=intensity,
            strobe_rate=strobe_rate, strobe_intensity=strobe_intensity,
            special=special,
        )

    def _blackout(self) -> list[FixtureCommand]:
        """All fixtures to black / off.

        Returns:
            List of blackout commands for every fixture.
        """
        return [self._cmd(f, BLACK, intensity=0.0) for f in self._map.all]

    def _all_color(self, color: Color, intensity: float = 1.0) -> list[FixtureCommand]:
        """Set all pars to a single color.

        Args:
            color: Target color.
            intensity: Master intensity.

        Returns:
            Commands for all fixtures (non-pars get blackout).
        """
        commands: list[FixtureCommand] = []
        for f in self._map.all:
            if f.fixture_type in (FixtureType.PAR, FixtureType.LED_BAR):
                commands.append(self._cmd(f, color, intensity))
            else:
                commands.append(self._cmd(f, BLACK, intensity=0.0))
        return commands

    # ─── Spatial patterns ────────────────────────────────────────

    def _chase(
        self,
        fixtures: list[FixtureInfo],
        phase: float,
        color: Color,
        width: float = 0.3,
        intensity: float = 1.0,
    ) -> dict[int, FixtureCommand]:
        """Chase pattern: a bright spot sweeps through fixtures in order.

        Args:
            fixtures: Ordered list of fixtures for the chase.
            phase: 0.0-1.0 position of the bright spot.
            color: Chase color.
            width: Width of the bright spot (0.0-1.0 of the fixture list).
            intensity: Peak intensity.

        Returns:
            Dict of fixture_id → FixtureCommand for fixtures in the chase.
        """
        n = len(fixtures)
        if n == 0:
            return {}

        result: dict[int, FixtureCommand] = {}
        for i, f in enumerate(fixtures):
            pos = i / max(n - 1, 1)
            dist = abs(pos - (phase % 1.0))
            dist = min(dist, 1.0 - dist)  # wrap-around
            brightness = max(0.0, 1.0 - dist / max(width, 0.01)) * intensity
            result[f.fixture_id] = self._cmd(f, color, brightness)
        return result

    def _sweep_x(
        self,
        phase: float,
        color: Color,
        width: float = 0.3,
        intensity: float = 1.0,
        fixtures: list[FixtureInfo] | None = None,
    ) -> dict[int, FixtureCommand]:
        """Left-to-right sweep based on fixture x-position.

        Args:
            phase: 0.0-1.0 sweep position.
            color: Sweep color.
            width: Spread of the sweep beam.
            intensity: Peak brightness.
            fixtures: Subset of fixtures; defaults to all.

        Returns:
            Dict of fixture_id → FixtureCommand.
        """
        return self._chase(
            self._map.sorted_by_x(fixtures), phase, color, width, intensity
        )

    def _sweep_y(
        self,
        phase: float,
        color: Color,
        width: float = 0.3,
        intensity: float = 1.0,
        fixtures: list[FixtureInfo] | None = None,
    ) -> dict[int, FixtureCommand]:
        """Front-to-back sweep based on fixture y-position.

        Args:
            phase: 0.0-1.0 sweep position.
            color: Sweep color.
            width: Spread of the sweep beam.
            intensity: Peak brightness.
            fixtures: Subset of fixtures; defaults to all.

        Returns:
            Dict of fixture_id → FixtureCommand.
        """
        return self._chase(
            self._map.sorted_by_y(fixtures), phase, color, width, intensity
        )

    def _alternating(
        self,
        fixtures: list[FixtureInfo],
        color_a: Color,
        color_b: Color,
        phase: float,
        intensity: float = 1.0,
    ) -> dict[int, FixtureCommand]:
        """Alternating pattern: even/odd fixtures swap colors on phase.

        When phase < 0.5, even-index fixtures get color_a and odd get
        color_b; when phase >= 0.5 they swap.

        Args:
            fixtures: Fixture list.
            color_a: First color.
            color_b: Second color.
            phase: 0.0-1.0 cycle position.
            intensity: Master intensity.

        Returns:
            Dict of fixture_id → FixtureCommand.
        """
        swap = phase >= 0.5
        result: dict[int, FixtureCommand] = {}
        for i, f in enumerate(fixtures):
            is_even = i % 2 == 0
            if (is_even and not swap) or (not is_even and swap):
                result[f.fixture_id] = self._cmd(f, color_a, intensity)
            else:
                result[f.fixture_id] = self._cmd(f, color_b, intensity)
        return result

    def _focus_expand(
        self,
        phase: float,
        color: Color,
        intensity: float = 1.0,
    ) -> dict[int, FixtureCommand]:
        """Focus→expand: center fixtures light first, corners follow.

        At phase=0 only center fixtures are lit; at phase=1 all are lit.

        Args:
            phase: 0.0-1.0 expansion.
            color: Color.
            intensity: Master intensity.

        Returns:
            Dict of fixture_id → FixtureCommand.
        """
        from lumina.lighting.fixture_map import ROOM_DEPTH, ROOM_WIDTH

        cx, cy = ROOM_WIDTH / 2, ROOM_DEPTH / 2
        # Compute max distance for normalization
        max_dist = math.sqrt(cx**2 + cy**2)

        result: dict[int, FixtureCommand] = {}
        for f in self._map.all:
            x, y, _z = f.position
            dist = math.sqrt((x - cx) ** 2 + (y - cy) ** 2) / max_dist
            brightness = max(0.0, 1.0 - dist / max(phase, 0.01)) * intensity
            brightness = min(1.0, brightness)
            result[f.fixture_id] = self._cmd(f, color, brightness)
        return result

    def _corner_isolation(
        self,
        corner_role: str,
        color: Color,
        intensity: float = 1.0,
    ) -> list[FixtureCommand]:
        """Light only one corner, blackout the rest.

        Args:
            corner_role: One of "front_left", "front_right",
                "back_left", "back_right".
            color: Color for the lit corner.
            intensity: Intensity for the lit corner.

        Returns:
            Full fixture command list (one per fixture).
        """
        from lumina.lighting.fixture_map import FixtureRole

        role_map = {
            "front_left": FixtureRole.FRONT_LEFT,
            "front_right": FixtureRole.FRONT_RIGHT,
            "back_left": FixtureRole.BACK_LEFT,
            "back_right": FixtureRole.BACK_RIGHT,
        }
        target_role = role_map.get(corner_role)

        commands: list[FixtureCommand] = []
        for f in self._map.all:
            if f.role == target_role:
                commands.append(self._cmd(f, color, intensity))
            else:
                commands.append(self._cmd(f))
        return commands

    # ─── Utility ─────────────────────────────────────────────────

    def _merge_commands(
        self,
        *sources: dict[int, FixtureCommand],
        base: list[FixtureCommand] | None = None,
    ) -> list[FixtureCommand]:
        """Merge multiple partial command dicts into a full fixture list.

        Later sources override earlier ones for the same fixture ID.
        Any fixture not covered by any source gets a blackout command.

        Args:
            *sources: Dicts of fixture_id → FixtureCommand to merge.
            base: Optional base command list to start from.

        Returns:
            Complete list of commands, one per fixture, sorted by ID.
        """
        merged: dict[int, FixtureCommand] = {}
        if base:
            for cmd in base:
                merged[cmd.fixture_id] = cmd
        else:
            for f in self._map.all:
                merged[f.fixture_id] = self._cmd(f, BLACK, intensity=0.0)

        for source in sources:
            merged.update(source)

        return [merged[fid] for fid in sorted(merged)]

    def _strobe_on_beat(
        self,
        state: MusicState,
        max_rate: int = 255,
        max_intensity: int = 255,
    ) -> tuple[int, int]:
        """Calculate strobe parameters that fire on beats.

        Args:
            state: Current music state.
            max_rate: Maximum strobe rate (on downbeat).
            max_intensity: Maximum strobe brightness.

        Returns:
            Tuple of (strobe_rate, strobe_intensity).
        """
        if state.is_downbeat:
            return max_rate, max_intensity
        if state.is_beat:
            return int(max_rate * 0.7), int(max_intensity * 0.8)
        return 0, 0
