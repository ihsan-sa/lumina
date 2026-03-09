"""Main lighting decision engine.

Receives MusicState at 60fps, selects the active profile(s) based on
genre_weights, runs them through the fixture map, and outputs a final
list of FixtureCommand — one per fixture.

For now only the rage_trap profile is implemented.  As more profiles
are added, the engine will blend their outputs weighted by genre_weights.
"""

from __future__ import annotations

import logging

from lumina.audio.models import MusicState
from lumina.control.protocol import FixtureCommand
from lumina.lighting.fixture_map import FixtureMap
from lumina.lighting.profiles.base import BaseProfile
from lumina.lighting.profiles.rage_trap import RageTrapProfile

logger = logging.getLogger(__name__)

# Registry of profile name → class.  New profiles are added here.
_PROFILE_REGISTRY: dict[str, type[BaseProfile]] = {
    "rage_trap": RageTrapProfile,
}


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
        self._fallback_profile = "rage_trap"

    @property
    def fixture_map(self) -> FixtureMap:
        """The venue fixture layout."""
        return self._map

    @property
    def profile_names(self) -> list[str]:
        """Names of all registered profiles."""
        return sorted(self._profiles)

    def generate(self, state: MusicState) -> list[FixtureCommand]:
        """Generate fixture commands for a single frame.

        Selection logic:
        1. Find the profile with the highest genre_weight.
        2. If it exists in the registry, use it exclusively.
        3. If no genre_weights match any registered profile, use
           the fallback profile.
        4. Future: blend multiple profiles weighted by genre_weights
           when more profiles are available.

        Args:
            state: Current audio analysis frame.

        Returns:
            List of FixtureCommand, one per fixture in the map.
        """
        profile = self._select_profile(state)
        return profile.generate(state)

    def _select_profile(self, state: MusicState) -> BaseProfile:
        """Pick the active profile based on genre_weights.

        Args:
            state: Current music state with genre_weights.

        Returns:
            The selected BaseProfile instance.
        """
        if not state.genre_weights:
            return self._profiles[self._fallback_profile]

        # Find highest-weighted profile that we have registered
        best_name: str | None = None
        best_weight = -1.0
        for name, weight in state.genre_weights.items():
            if name in self._profiles and weight > best_weight:
                best_name = name
                best_weight = weight

        if best_name is not None:
            return self._profiles[best_name]

        return self._profiles[self._fallback_profile]
