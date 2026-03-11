"""Map LightingIntent to per-fixture FixtureCommands.

Translates the high-level lighting intent predicted by the ML model
into concrete commands for each fixture in the venue layout.  This
bridges the gap between the model's abstract output (dominant color,
spatial distribution, strobe/blackout) and the per-fixture protocol.

Mapping logic:
  - dominant_color -> par RGBW values (HSV -> RGB conversion)
  - spatial_distribution -> per-fixture dimmer levels based on fixture positions
  - strobe_active -> strobe fixture commands (rate + intensity)
  - blackout -> all channels zero
"""

from __future__ import annotations

import colorsys
import logging

from lumina.control.protocol import FixtureCommand
from lumina.lighting.fixture_map import (
    FixtureInfo,
    FixtureMap,
    FixtureType,
    ROOM_WIDTH,
)
from lumina.ml.model.architecture import LightingIntent

logger = logging.getLogger(__name__)

# Strobe rate mapping: intensity 0-1 -> strobe_rate 0-255.
_MAX_STROBE_RATE = 200  # Cap below 255 to avoid dangerously fast strobing.


def _clamp_byte(value: float) -> int:
    """Clamp a float to an integer byte (0-255).

    Args:
        value: Float value to convert.

    Returns:
        Clamped integer in [0, 255].
    """
    return max(0, min(255, round(value)))


def _hsv_to_rgb_bytes(
    hue: float, saturation: float, value: float
) -> tuple[int, int, int]:
    """Convert HSV to RGB byte values.

    Args:
        hue: Hue in degrees (0-360).
        saturation: Saturation (0-1).
        value: Brightness value (0-1).

    Returns:
        Tuple of (R, G, B) each in [0, 255].
    """
    # colorsys expects hue in 0-1 range.
    h_norm = (hue % 360.0) / 360.0
    r, g, b = colorsys.hsv_to_rgb(h_norm, saturation, value)
    return _clamp_byte(r * 255), _clamp_byte(g * 255), _clamp_byte(b * 255)


def _fixture_spatial_weight(
    fixture: FixtureInfo,
    spatial_distribution: tuple[float, float, float],
) -> float:
    """Compute spatial dimming weight for a fixture based on its position.

    Divides the room width into three zones (left, center, right) and
    returns the appropriate brightness multiplier from the spatial
    distribution.

    Args:
        fixture: Fixture metadata with position.
        spatial_distribution: (left, center, right) brightness levels.

    Returns:
        Brightness multiplier in [0, 1].
    """
    x = fixture.position[0]
    left_brightness, center_brightness, right_brightness = spatial_distribution

    # Divide room into three zones by x-position.
    third = ROOM_WIDTH / 3.0

    if x < third:
        # Left zone — blend between left and center.
        t = x / third
        return left_brightness * (1.0 - t) + center_brightness * t
    elif x < 2 * third:
        # Center zone.
        return center_brightness
    else:
        # Right zone — blend between center and right.
        t = (x - 2 * third) / third
        return center_brightness * (1.0 - t) + right_brightness * t


def _generate_par_command(
    fixture: FixtureInfo,
    intent: LightingIntent,
) -> FixtureCommand:
    """Generate a command for an RGBW par fixture.

    Maps dominant_color to RGBW channels with spatial dimming.

    Args:
        fixture: Par fixture metadata.
        intent: High-level lighting intent.

    Returns:
        FixtureCommand for this par fixture.
    """
    if intent.blackout:
        return FixtureCommand(fixture_id=fixture.fixture_id)

    # Convert dominant HSV color to RGB.
    hue, sat, val = intent.dominant_color
    r, g, b = _hsv_to_rgb_bytes(hue, sat, val)

    # Compute white channel: higher when saturation is low (warm wash).
    white = _clamp_byte((1.0 - sat) * val * 255)

    # Apply spatial dimming based on fixture position.
    spatial_weight = _fixture_spatial_weight(fixture, intent.spatial_distribution)
    dimmer = intent.overall_brightness * spatial_weight

    # Apply dimmer to all channels.
    r = _clamp_byte(r * dimmer)
    g = _clamp_byte(g * dimmer)
    b = _clamp_byte(b * dimmer)
    white = _clamp_byte(white * dimmer)

    # Special byte is master dimmer for pars.
    master = _clamp_byte(dimmer * 255)

    return FixtureCommand(
        fixture_id=fixture.fixture_id,
        red=r,
        green=g,
        blue=b,
        white=white,
        strobe_rate=0,
        strobe_intensity=0,
        special=master,
    )


def _generate_strobe_command(
    fixture: FixtureInfo,
    intent: LightingIntent,
) -> FixtureCommand:
    """Generate a command for a strobe fixture.

    Maps strobe_active and strobe_intensity to rate and brightness.
    When strobe is not active, uses DC mode (rate=0) with spatial
    dimming as a simple wash light.

    Args:
        fixture: Strobe fixture metadata.
        intent: High-level lighting intent.

    Returns:
        FixtureCommand for this strobe fixture.
    """
    if intent.blackout:
        return FixtureCommand(fixture_id=fixture.fixture_id)

    if intent.strobe_active:
        # Active strobe: set rate and intensity.
        rate = _clamp_byte(intent.strobe_intensity * _MAX_STROBE_RATE)
        intensity = _clamp_byte(intent.strobe_intensity * 255)

        # Use dominant color for strobe tint.
        hue, sat, val = intent.dominant_color
        r, g, b = _hsv_to_rgb_bytes(hue, sat, val)

        return FixtureCommand(
            fixture_id=fixture.fixture_id,
            red=r,
            green=g,
            blue=b,
            white=0,
            strobe_rate=rate,
            strobe_intensity=intensity,
            special=0,
        )
    else:
        # No strobe — use DC mode with spatial brightness.
        spatial_weight = _fixture_spatial_weight(
            fixture, intent.spatial_distribution
        )
        brightness = intent.overall_brightness * spatial_weight
        intensity = _clamp_byte(brightness * 255)

        hue, sat, val = intent.dominant_color
        r, g, b = _hsv_to_rgb_bytes(hue, sat, val * brightness)

        return FixtureCommand(
            fixture_id=fixture.fixture_id,
            red=r,
            green=g,
            blue=b,
            white=0,
            strobe_rate=0,
            strobe_intensity=intensity,
            special=0,
        )


def _generate_led_bar_command(
    fixture: FixtureInfo,
    intent: LightingIntent,
) -> FixtureCommand:
    """Generate a command for an LED bar fixture.

    LED bars use RGBW for color wash and special byte as master dimmer.

    Args:
        fixture: LED bar fixture metadata.
        intent: High-level lighting intent.

    Returns:
        FixtureCommand for this LED bar fixture.
    """
    if intent.blackout:
        return FixtureCommand(fixture_id=fixture.fixture_id)

    # Use secondary color for LED bars (visual contrast with pars).
    hue, sat, val = intent.secondary_color
    r, g, b = _hsv_to_rgb_bytes(hue, sat, val)

    spatial_weight = _fixture_spatial_weight(fixture, intent.spatial_distribution)
    dimmer = intent.overall_brightness * spatial_weight

    r = _clamp_byte(r * dimmer)
    g = _clamp_byte(g * dimmer)
    b = _clamp_byte(b * dimmer)
    white = _clamp_byte((1.0 - sat) * val * dimmer * 255)
    master = _clamp_byte(dimmer * 255)

    return FixtureCommand(
        fixture_id=fixture.fixture_id,
        red=r,
        green=g,
        blue=b,
        white=white,
        strobe_rate=0,
        strobe_intensity=0,
        special=master,
    )


def _generate_uv_command(
    fixture: FixtureInfo,
    intent: LightingIntent,
) -> FixtureCommand:
    """Generate a command for a UV bar fixture.

    UV bars only use the special byte as UV intensity.  UV is more
    effective at lower overall brightness, so it scales inversely
    with brightness.

    Args:
        fixture: UV fixture metadata.
        intent: High-level lighting intent.

    Returns:
        FixtureCommand for this UV fixture.
    """
    if intent.blackout:
        return FixtureCommand(fixture_id=fixture.fixture_id)

    # UV intensity scales inversely with overall brightness.
    # More UV when the scene is darker for atmospheric effect.
    uv_intensity = intent.overall_brightness * 0.5
    if intent.overall_brightness < 0.3:
        uv_intensity = 0.7  # Boost UV in low-light scenes.

    special = _clamp_byte(uv_intensity * 255)

    return FixtureCommand(
        fixture_id=fixture.fixture_id,
        red=0,
        green=0,
        blue=0,
        white=0,
        strobe_rate=0,
        strobe_intensity=0,
        special=special,
    )


def _generate_laser_command(
    fixture: FixtureInfo,
    intent: LightingIntent,
) -> FixtureCommand:
    """Generate a command for a laser fixture.

    Lasers use the special byte as pattern ID.  Active during high
    energy and strobe moments.

    Args:
        fixture: Laser fixture metadata.
        intent: High-level lighting intent.

    Returns:
        FixtureCommand for this laser fixture.
    """
    if intent.blackout:
        return FixtureCommand(fixture_id=fixture.fixture_id)

    # Lasers activate at high brightness or during strobe.
    if intent.overall_brightness > 0.7 or intent.strobe_active:
        # Use overall brightness to select pattern intensity.
        pattern = _clamp_byte(intent.overall_brightness * 200)
    else:
        pattern = 0

    return FixtureCommand(
        fixture_id=fixture.fixture_id,
        red=0,
        green=0,
        blue=0,
        white=0,
        strobe_rate=0,
        strobe_intensity=0,
        special=pattern,
    )


# Dispatch table: fixture type -> command generator.
_GENERATORS: dict[FixtureType, type] = {}  # Not used; see function below.


def _generate_command(
    fixture: FixtureInfo,
    intent: LightingIntent,
) -> FixtureCommand:
    """Generate a command for any fixture type by dispatching.

    Args:
        fixture: Fixture metadata.
        intent: High-level lighting intent.

    Returns:
        FixtureCommand for this fixture.
    """
    if fixture.fixture_type == FixtureType.PAR:
        return _generate_par_command(fixture, intent)
    elif fixture.fixture_type == FixtureType.STROBE:
        return _generate_strobe_command(fixture, intent)
    elif fixture.fixture_type == FixtureType.LED_BAR:
        return _generate_led_bar_command(fixture, intent)
    elif fixture.fixture_type == FixtureType.UV:
        return _generate_uv_command(fixture, intent)
    elif fixture.fixture_type == FixtureType.LASER:
        return _generate_laser_command(fixture, intent)
    else:
        # Unknown fixture type — return blackout.
        logger.warning("Unknown fixture type: %s", fixture.fixture_type)
        return FixtureCommand(fixture_id=fixture.fixture_id)


def intent_to_commands(
    intent: LightingIntent,
    fixture_map: FixtureMap,
) -> list[FixtureCommand]:
    """Convert an ML model's LightingIntent to per-fixture commands.

    Iterates over all fixtures in the map and generates type-appropriate
    commands based on the lighting intent.

    Args:
        intent: High-level lighting intent from the ML model.
        fixture_map: Venue fixture layout.

    Returns:
        List of FixtureCommand, one per fixture in the map (sorted by ID).
    """
    commands: list[FixtureCommand] = []

    for fixture in fixture_map.all:
        cmd = _generate_command(fixture, intent)
        commands.append(cmd)

    return commands
