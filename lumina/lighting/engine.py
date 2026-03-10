"""Main lighting decision engine.

Receives MusicState at 60fps, selects the active profile(s) based on
genre_weights, runs them through the fixture map, and outputs a final
list of FixtureCommand — one per fixture.

Profile selection: the engine picks the highest-weighted registered
profile from genre_weights.  If no genre exceeds 0.3 weight, the
generic fallback profile is used to guarantee acceptable output.
"""

from __future__ import annotations

import logging

from lumina.audio.models import MusicState
from lumina.control.protocol import FixtureCommand
from lumina.lighting.fixture_map import FixtureMap, FixtureType
from lumina.lighting.profiles.base import BaseProfile
from lumina.lighting.profiles.festival_edm import FestivalEdmProfile
from lumina.lighting.profiles.generic import GenericProfile
from lumina.lighting.profiles.psych_rnb import PsychRnbProfile
from lumina.lighting.profiles.rage_trap import RageTrapProfile

logger = logging.getLogger(__name__)

# Minimum weight for a genre profile to be selected over the fallback.
_MIN_GENRE_WEIGHT = 0.3

# Registry of profile name → class.  New profiles are added here.
_PROFILE_REGISTRY: dict[str, type[BaseProfile]] = {
    "rage_trap": RageTrapProfile,
    "psych_rnb": PsychRnbProfile,
    "festival_edm": FestivalEdmProfile,
    "generic": GenericProfile,
}


def _dominant_colors(
    samples: list[tuple[int, int, int]], max_colors: int = 2
) -> list[str]:
    """Find the 1-2 dominant colors from a list of RGB samples.

    Quantizes each channel to the nearest 32 to group similar colors,
    then returns the most frequent buckets formatted as hex strings.

    Args:
        samples: List of (R, G, B) tuples (0-255).
        max_colors: Maximum number of colors to return.

    Returns:
        List of hex color strings like ["#ff00ff", "#00ffcc"].
    """
    if not samples:
        return []

    def _q(v: int) -> int:
        return min(255, round(v / 32) * 32)

    counts: dict[tuple[int, int, int], int] = {}
    for r, g, b in samples:
        key = (_q(r), _q(g), _q(b))
        counts[key] = counts.get(key, 0) + 1

    sorted_colors = sorted(counts, key=lambda k: -counts[k])
    return [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in sorted_colors[:max_colors]]


class LightingEngine:
    """Central lighting decision engine.

    Takes a FixtureMap and instantiates all registered profiles.
    On each frame, selects the dominant profile (or blends multiple
    profiles) based on MusicState.genre_weights and generates
    per-fixture commands.

    Args:
        fixture_map: Venue fixture layout.  If None, uses the default.
    """

    def __init__(self, fixture_map: FixtureMap | None = None) -> None:
        self._map = fixture_map if fixture_map is not None else FixtureMap()
        self._profiles: dict[str, BaseProfile] = {
            name: cls(self._map) for name, cls in _PROFILE_REGISTRY.items()
        }
        self._fallback_profile = "generic"
        self._last_debug_info: dict[str, object] = {}

    @property
    def fixture_map(self) -> FixtureMap:
        """The venue fixture layout."""
        return self._map

    @property
    def profile_names(self) -> list[str]:
        """Names of all registered profiles."""
        return sorted(self._profiles)

    @property
    def last_debug_info(self) -> dict[str, object]:
        """Debug info dict from the most recent generate() call."""
        return self._last_debug_info

    def generate(self, state: MusicState) -> list[FixtureCommand]:
        """Generate fixture commands for a single frame.

        Selection logic:
        1. Find the registered profile with the highest genre_weight.
        2. If its weight >= 0.3, use it exclusively.
        3. Otherwise, use the generic fallback profile.

        Args:
            state: Current audio analysis frame.

        Returns:
            List of FixtureCommand, one per fixture in the map.
        """
        profile = self._select_profile(state)
        commands = profile.generate(state)
        self._last_debug_info = self._build_debug_info(profile, state, commands)
        return commands

    def _build_debug_info(
        self,
        profile: BaseProfile,
        state: MusicState,
        commands: list[FixtureCommand],
    ) -> dict[str, object]:
        """Compute per-frame debug info from the profile and its output.

        Args:
            profile: The profile that was just run.
            state: The music state frame.
            commands: The commands it generated.

        Returns:
            Dict with keys: profile, segment, patterns, active, total,
            type_counts, colors, strobe_rate_max.
        """
        patterns: list[str] = list(profile._debug_info.get("patterns", []))  # type: ignore[arg-type]

        cmd_map = {c.fixture_id: c for c in commands}
        # type -> [active_count, total_count]
        type_tallies: dict[str, list[int]] = {}
        color_samples: list[tuple[int, int, int]] = []
        strobe_rate_max = 0

        for f in self._map.all:
            ftype = f.fixture_type.value
            tally = type_tallies.setdefault(ftype, [0, 0])
            tally[1] += 1

            c = cmd_map.get(f.fixture_id)
            if c is None:
                continue

            # Determine "active" based on fixture type
            if f.fixture_type == FixtureType.STROBE:
                is_active = c.strobe_intensity > 0
                strobe_rate_max = max(strobe_rate_max, c.strobe_rate)
            else:
                is_active = c.special > 0

            if is_active:
                tally[0] += 1

            # Collect non-black colors from pars and LED bars
            if is_active and f.fixture_type in (FixtureType.PAR, FixtureType.LED_BAR):
                if c.red + c.green + c.blue > 30:
                    color_samples.append((c.red, c.green, c.blue))

        active_total = sum(v[0] for v in type_tallies.values())
        total_total = sum(v[1] for v in type_tallies.values())

        return {
            "profile": profile.name,
            "segment": state.segment,
            "patterns": patterns,
            "active": active_total,
            "total": total_total,
            "type_counts": {k: (v[0], v[1]) for k, v in type_tallies.items()},
            "colors": _dominant_colors(color_samples),
            "strobe_rate_max": strobe_rate_max,
        }

    def _select_profile(self, state: MusicState) -> BaseProfile:
        """Pick the active profile based on genre_weights.

        Falls back to generic when no registered genre exceeds
        ``_MIN_GENRE_WEIGHT`` (0.3).

        Args:
            state: Current music state with genre_weights.

        Returns:
            The selected BaseProfile instance.
        """
        if not state.genre_weights:
            return self._profiles[self._fallback_profile]

        # Find highest-weighted registered profile (excluding generic)
        best_name: str | None = None
        best_weight = -1.0
        for name, weight in state.genre_weights.items():
            if name in self._profiles and name != "generic" and weight > best_weight:
                best_name = name
                best_weight = weight

        # Only use a genre profile if it exceeds the minimum threshold
        if best_name is not None and best_weight >= _MIN_GENRE_WEIGHT:
            return self._profiles[best_name]

        return self._profiles[self._fallback_profile]
