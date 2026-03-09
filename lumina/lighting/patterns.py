"""Spatial pattern library for LUMINA lighting profiles.

Standalone pattern functions that produce partial fixture command dicts.
Each function takes a list of fixtures, MusicState, a Color, and optional
kwargs, and returns ``dict[int, FixtureCommand]``.

Patterns are pure functions with no shared state.  They are composable:
profiles layer multiple patterns by merging their output dicts (later
dicts override earlier ones for the same fixture ID).

All 12 patterns:
    chase_lr, chase_bounce, converge, diverge, alternate,
    random_scatter, breathe, strobe_burst, wash_hold,
    color_split, spotlight_isolate, stutter
"""

from __future__ import annotations

import hashlib
import math

from lumina.audio.models import MusicState
from lumina.control.protocol import FixtureCommand
from lumina.lighting.fixture_map import (
    ROOM_DEPTH,
    ROOM_WIDTH,
    FixtureInfo,
)
from lumina.lighting.profiles.base import (
    BLACK,
    Color,
    route_command,
    triangle_wave,
)


# ─── Shared helpers ────────────────────────────────────────────────


def make_command(
    fixture: FixtureInfo,
    color: Color = BLACK,
    intensity: float = 1.0,
    strobe_rate: int = 0,
    strobe_intensity: int = 0,
    special: int | None = None,
) -> FixtureCommand:
    """Build a FixtureCommand with fixture-type-aware channel routing.

    This is a convenience wrapper around ``route_command`` that scales
    the color by intensity before routing, matching the behavior of
    ``BaseProfile._cmd()``.

    Args:
        fixture: Target fixture.
        color: Desired color (will be scaled by intensity).
        intensity: Master intensity 0.0-1.0.
        strobe_rate: Strobe rate 0-255.
        strobe_intensity: Strobe brightness 0-255.
        special: Override for the special byte.

    Returns:
        FixtureCommand for this fixture.
    """
    scaled = color.scaled(intensity)
    return route_command(
        fixture, color=scaled, intensity=intensity,
        strobe_rate=strobe_rate, strobe_intensity=strobe_intensity,
        special=special,
    )


def select_active_fixtures(
    fixtures: list[FixtureInfo],
    energy: float,
    *,
    low_count: int = 3,
    mid_count: int = 8,
    high_threshold: float = 0.7,
    mid_threshold: float = 0.4,
) -> list[FixtureInfo]:
    """Select a subset of fixtures based on energy level.

    Low energy returns ``low_count`` fixtures (from the center of the
    list), medium energy returns ``mid_count``, and high energy returns
    all fixtures.  This enables the fixture count escalation pattern
    where low energy uses 2-4 fixtures and high energy uses all.

    Args:
        fixtures: Full fixture list to select from.
        energy: Current energy level 0.0-1.0.
        low_count: Number of fixtures for low energy.
        mid_count: Number of fixtures for medium energy.
        high_threshold: Energy above which all fixtures are active.
        mid_threshold: Energy above which mid_count fixtures are active.

    Returns:
        Subset of fixtures appropriate for the energy level.
    """
    n = len(fixtures)
    if n == 0:
        return []

    if energy >= high_threshold:
        return list(fixtures)

    count = low_count if energy < mid_threshold else mid_count
    count = min(count, n)

    # Select from the center outward for balanced spatial distribution
    center = n // 2
    selected: list[FixtureInfo] = []
    for offset in range(n):
        if len(selected) >= count:
            break
        # Alternate left and right of center
        for idx in [center - offset, center + offset]:
            if 0 <= idx < n and fixtures[idx] not in selected:
                selected.append(fixtures[idx])
                if len(selected) >= count:
                    break

    return sorted(selected, key=lambda f: f.fixture_id)


# ─── Pattern functions ──────────────────────────────────────────────


def chase_lr(
    fixtures: list[FixtureInfo],
    state: MusicState,
    timestamp: float,
    color: Color,
    *,
    speed: float = 1.0,
    width: float = 0.3,
    intensity: float = 1.0,
) -> dict[int, FixtureCommand]:
    """Sequential left-to-right activation with configurable overlap.

    Each fixture activates in sequence with a time offset based on
    bar_phase.  The bright spot sweeps at ``speed`` cycles per bar
    (speed=1.0 = one full sweep per bar).

    Args:
        fixtures: Target fixtures (sorted by spatial position).
        state: Current music state.
        timestamp: Current timestamp (unused, phase-driven).
        color: Chase color.
        speed: Cycles per bar (1.0 = one sweep per bar).
        width: Width of the bright spot (0.0-1.0 of fixture span).
        intensity: Peak brightness.

    Returns:
        Dict of fixture_id -> FixtureCommand.
    """
    n = len(fixtures)
    if n == 0:
        return {}

    # Sort by x position for L→R
    ordered = sorted(fixtures, key=lambda f: (f.position[0], f.position[1]))
    phase = (state.bar_phase * speed) % 1.0

    result: dict[int, FixtureCommand] = {}
    for i, f in enumerate(ordered):
        pos = i / max(n - 1, 1)
        dist = abs(pos - phase)
        dist = min(dist, 1.0 - dist)  # wrap-around
        brightness = max(0.0, 1.0 - dist / max(width, 0.01)) * intensity
        result[f.fixture_id] = make_command(f, color, brightness)
    return result


def chase_bounce(
    fixtures: list[FixtureInfo],
    state: MusicState,
    timestamp: float,
    color: Color,
    *,
    speed: float = 1.0,
    width: float = 0.3,
    intensity: float = 1.0,
) -> dict[int, FixtureCommand]:
    """Ping-pong chase: 1->2->3->4->3->2->1.

    Like chase_lr but uses a triangle wave for the phase so the bright
    spot bounces back and forth.  Good for rapid percussive sections.

    Args:
        fixtures: Target fixtures.
        state: Current music state.
        timestamp: Current timestamp.
        color: Chase color.
        speed: Cycles per bar.
        width: Width of the bright spot.
        intensity: Peak brightness.

    Returns:
        Dict of fixture_id -> FixtureCommand.
    """
    n = len(fixtures)
    if n == 0:
        return {}

    ordered = sorted(fixtures, key=lambda f: (f.position[0], f.position[1]))
    phase = triangle_wave((state.bar_phase * speed) % 1.0)

    result: dict[int, FixtureCommand] = {}
    for i, f in enumerate(ordered):
        pos = i / max(n - 1, 1)
        dist = abs(pos - phase)
        brightness = max(0.0, 1.0 - dist / max(width, 0.01)) * intensity
        result[f.fixture_id] = make_command(f, color, brightness)
    return result


def converge(
    fixtures: list[FixtureInfo],
    state: MusicState,
    timestamp: float,
    color: Color,
    *,
    intensity: float = 1.0,
) -> dict[int, FixtureCommand]:
    """Outside-in convergence: edge fixtures fire first, center follows.

    Each fixture's activation is offset by its distance from room center.
    At bar_phase=0, only edge fixtures are lit; at bar_phase=1, all are lit.

    Args:
        fixtures: Target fixtures.
        state: Current music state.
        timestamp: Current timestamp.
        color: Convergence color.
        intensity: Peak brightness.

    Returns:
        Dict of fixture_id -> FixtureCommand.
    """
    if not fixtures:
        return {}

    cx, cy = ROOM_WIDTH / 2, ROOM_DEPTH / 2
    max_dist = math.sqrt(cx**2 + cy**2)

    result: dict[int, FixtureCommand] = {}
    for f in fixtures:
        x, y, _z = f.position
        dist = math.sqrt((x - cx) ** 2 + (y - cy) ** 2) / max_dist
        # Edges (dist~1) activate first (at low bar_phase), center (dist~0) last
        # Invert distance: activation_time = 1.0 - dist
        activation = 1.0 - dist
        # brightness ramps from 0 to intensity as bar_phase reaches activation
        if state.bar_phase >= activation:
            brightness = intensity
        else:
            brightness = max(0.0, 1.0 - (activation - state.bar_phase) / 0.3) * intensity
        result[f.fixture_id] = make_command(f, color, brightness)
    return result


def diverge(
    fixtures: list[FixtureInfo],
    state: MusicState,
    timestamp: float,
    color: Color,
    *,
    intensity: float = 1.0,
) -> dict[int, FixtureCommand]:
    """Center-out divergence: center fixtures fire first, expand outward.

    Used for drops and energy release moments (bloom effect).

    Args:
        fixtures: Target fixtures.
        state: Current music state.
        timestamp: Current timestamp.
        color: Diverge color.
        intensity: Peak brightness.

    Returns:
        Dict of fixture_id -> FixtureCommand.
    """
    if not fixtures:
        return {}

    cx, cy = ROOM_WIDTH / 2, ROOM_DEPTH / 2
    max_dist = math.sqrt(cx**2 + cy**2)

    result: dict[int, FixtureCommand] = {}
    for f in fixtures:
        x, y, _z = f.position
        dist = math.sqrt((x - cx) ** 2 + (y - cy) ** 2) / max_dist
        # Center (dist~0) activates first, edges (dist~1) last
        if state.bar_phase >= dist:
            brightness = intensity
        else:
            brightness = max(0.0, 1.0 - (dist - state.bar_phase) / 0.3) * intensity
        result[f.fixture_id] = make_command(f, color, brightness)
    return result


def alternate(
    fixtures: list[FixtureInfo],
    state: MusicState,
    timestamp: float,
    color: Color,
    *,
    color_b: Color = BLACK,
    intensity: float = 1.0,
) -> dict[int, FixtureCommand]:
    """Odd-indexed vs even-indexed fixtures swap on each beat.

    Even fixtures get ``color`` on first half of beat, ``color_b`` on
    second half.  Odd fixtures get the opposite.

    Args:
        fixtures: Target fixtures.
        state: Current music state.
        timestamp: Current timestamp.
        color: Primary color (even fixtures first half).
        color_b: Secondary color (odd fixtures first half).
        intensity: Master intensity.

    Returns:
        Dict of fixture_id -> FixtureCommand.
    """
    if not fixtures:
        return {}

    swap = state.beat_phase >= 0.5
    result: dict[int, FixtureCommand] = {}
    for i, f in enumerate(fixtures):
        is_even = i % 2 == 0
        if (is_even and not swap) or (not is_even and swap):
            result[f.fixture_id] = make_command(f, color, intensity)
        else:
            result[f.fixture_id] = make_command(f, color_b, intensity)
    return result


def random_scatter(
    fixtures: list[FixtureInfo],
    state: MusicState,
    timestamp: float,
    color: Color,
    *,
    density: float = 0.3,
    intensity: float = 1.0,
) -> dict[int, FixtureCommand]:
    """Deterministic pseudo-random scatter.

    Each fixture independently fires based on a hash of timestamp +
    fixture_id.  ``density`` (0.0-1.0) controls the fraction of fixtures
    that are lit at any given moment.  Intentional chaos for ad-libs,
    glitch sections, and vocal chops.

    Args:
        fixtures: Target fixtures.
        state: Current music state.
        timestamp: Current timestamp (used for deterministic randomness).
        color: Scatter color.
        density: Fraction of fixtures lit (0.0-1.0).
        intensity: Brightness of lit fixtures.

    Returns:
        Dict of fixture_id -> FixtureCommand.
    """
    if not fixtures:
        return {}

    result: dict[int, FixtureCommand] = {}
    # Quantize timestamp to ~50ms intervals for natural scatter
    t_quant = int(timestamp * 20)
    for f in fixtures:
        seed = f"{t_quant}_{f.fixture_id}".encode()
        h = int(hashlib.md5(seed).hexdigest()[:8], 16)  # noqa: S324
        threshold = int(density * 0xFFFFFFFF)
        if h < threshold:
            result[f.fixture_id] = make_command(f, color, intensity)
        else:
            result[f.fixture_id] = make_command(f, BLACK, 0.0)
    return result


def breathe(
    fixtures: list[FixtureInfo],
    state: MusicState,
    timestamp: float,
    color: Color,
    *,
    min_intensity: float = 0.1,
    max_intensity: float = 0.5,
    period_bars: float = 1.0,
    phase_offset_per_fixture: float = 0.0,
) -> dict[int, FixtureCommand]:
    """Sine-wave breathing: all fixtures rise and fall together.

    Intensity oscillates between ``min_intensity`` and ``max_intensity``
    over ``period_bars`` bars using bar_phase.

    Args:
        fixtures: Target fixtures.
        state: Current music state.
        timestamp: Current timestamp.
        color: Breathe color.
        min_intensity: Minimum brightness.
        max_intensity: Maximum brightness.
        period_bars: Period in bars (1.0 = one breathe per bar).
        phase_offset_per_fixture: Phase offset between consecutive
            fixtures (0.0 = all in sync, >0 = wave effect).

    Returns:
        Dict of fixture_id -> FixtureCommand.
    """
    if not fixtures:
        return {}

    result: dict[int, FixtureCommand] = {}
    for i, f in enumerate(fixtures):
        phase = (state.bar_phase / period_bars + i * phase_offset_per_fixture) % 1.0
        sine = (math.sin(phase * math.pi * 2.0 - math.pi / 2.0) + 1.0) / 2.0
        brightness = min_intensity + sine * (max_intensity - min_intensity)
        result[f.fixture_id] = make_command(f, color, brightness)
    return result


def strobe_burst(
    fixtures: list[FixtureInfo],
    state: MusicState,
    timestamp: float,
    color: Color,
    *,
    rate: int = 255,
    burst_intensity: int = 255,
) -> dict[int, FixtureCommand]:
    """All fixtures fire at maximum strobe for a burst effect.

    The single most impactful move — should be used sparingly on major
    hits only.  Only affects STROBE-type fixtures; non-strobes get a
    full-brightness color wash instead.

    Args:
        fixtures: Target fixtures.
        state: Current music state.
        timestamp: Current timestamp.
        color: Burst color (used as tint for strobes, color for others).
        rate: Strobe rate 0-255.
        burst_intensity: Strobe brightness 0-255.

    Returns:
        Dict of fixture_id -> FixtureCommand.
    """
    if not fixtures:
        return {}

    from lumina.lighting.fixture_map import FixtureType

    result: dict[int, FixtureCommand] = {}
    for f in fixtures:
        if f.fixture_type == FixtureType.STROBE:
            result[f.fixture_id] = make_command(
                f, color, intensity=1.0,
                strobe_rate=rate, strobe_intensity=burst_intensity,
            )
        else:
            result[f.fixture_id] = make_command(f, color, intensity=1.0)
    return result


def wash_hold(
    fixtures: list[FixtureInfo],
    state: MusicState,
    timestamp: float,
    color: Color,
    *,
    intensity: float = 1.0,
) -> dict[int, FixtureCommand]:
    """Static color wash: all fixtures set to the same color.

    The background/default state.  Very slow intensity drift via a
    gentle sine wave keeps the wash from feeling completely static.

    Args:
        fixtures: Target fixtures.
        state: Current music state.
        timestamp: Current timestamp.
        color: Wash color.
        intensity: Base intensity (drifts ±5%).

    Returns:
        Dict of fixture_id -> FixtureCommand.
    """
    if not fixtures:
        return {}

    # Gentle intensity drift (±5%) to avoid looking static
    drift = math.sin(timestamp * 0.3) * 0.05
    actual = max(0.0, min(1.0, intensity + drift))

    result: dict[int, FixtureCommand] = {}
    for f in fixtures:
        result[f.fixture_id] = make_command(f, color, actual)
    return result


def color_split(
    fixtures: list[FixtureInfo],
    state: MusicState,
    timestamp: float,
    color: Color,
    *,
    color_right: Color | None = None,
    intensity: float = 1.0,
) -> dict[int, FixtureCommand]:
    """Left/right color split for spatial depth.

    Left-half fixtures get ``color``, right-half get ``color_right``.
    Split is determined by x position relative to room center.

    Args:
        fixtures: Target fixtures.
        state: Current music state.
        timestamp: Current timestamp.
        color: Left-side color.
        color_right: Right-side color.  If None, uses a complementary
            hue shift of the left color.
        intensity: Master intensity.

    Returns:
        Dict of fixture_id -> FixtureCommand.
    """
    if not fixtures:
        return {}

    if color_right is None:
        # Simple complement: swap R and B channels
        color_right = Color(r=color.b, g=color.g, b=color.r, w=color.w)

    mid_x = ROOM_WIDTH / 2
    result: dict[int, FixtureCommand] = {}
    for f in fixtures:
        if f.position[0] < mid_x:
            result[f.fixture_id] = make_command(f, color, intensity)
        else:
            result[f.fixture_id] = make_command(f, color_right, intensity)
    return result


def spotlight_isolate(
    fixtures: list[FixtureInfo],
    state: MusicState,
    timestamp: float,
    color: Color,
    *,
    target_index: int = 0,
    intensity: float = 1.0,
    dim_others: float = 0.0,
) -> dict[int, FixtureCommand]:
    """Spotlight one fixture at high intensity, dim everything else.

    Args:
        fixtures: Target fixtures.
        state: Current music state.
        timestamp: Current timestamp.
        color: Spotlight color.
        target_index: Index into fixtures list for the lit fixture.
        intensity: Spotlight brightness.
        dim_others: Brightness for non-spotlight fixtures (0.0 = off).

    Returns:
        Dict of fixture_id -> FixtureCommand.
    """
    if not fixtures:
        return {}

    target_idx = target_index % len(fixtures)
    result: dict[int, FixtureCommand] = {}
    for i, f in enumerate(fixtures):
        if i == target_idx:
            result[f.fixture_id] = make_command(f, color, intensity)
        else:
            if dim_others > 0.0:
                result[f.fixture_id] = make_command(f, color, dim_others)
            else:
                result[f.fixture_id] = make_command(f, BLACK, 0.0)
    return result


def stutter(
    fixtures: list[FixtureInfo],
    state: MusicState,
    timestamp: float,
    color: Color,
    *,
    rate: float = 4.0,
    intensity: float = 1.0,
) -> dict[int, FixtureCommand]:
    """Rapid on/off stutter at musical subdivisions.

    Binary: fixtures are full-on or full-off based on beat subdivision.
    ``rate`` controls the subdivision: 2.0 = 8th notes, 4.0 = 16th notes,
    8.0 = 32nd notes.  Used in builds where frequency accelerates toward
    the drop.

    Args:
        fixtures: Target fixtures.
        state: Current music state.
        timestamp: Current timestamp.
        color: Stutter color.
        rate: Subdivision rate (flashes per beat).
        intensity: On-state brightness.

    Returns:
        Dict of fixture_id -> FixtureCommand.
    """
    if not fixtures:
        return {}

    # Use beat_phase subdivided by rate
    sub_phase = (state.beat_phase * rate) % 1.0
    is_on = sub_phase < 0.5

    result: dict[int, FixtureCommand] = {}
    for f in fixtures:
        if is_on:
            result[f.fixture_id] = make_command(f, color, intensity)
        else:
            result[f.fixture_id] = make_command(f, BLACK, 0.0)
    return result
