"""Profile blending engine for combining multiple genre lighting profiles.

When a track spans multiple genres (e.g. psych_rnb: 0.6, festival_edm: 0.2),
the ProfileBlender runs each active profile and combines their outputs
proportionally.  Color channels use weighted averages, strobe parameters
come from the dominant active source, and special bytes are averaged.
"""

from __future__ import annotations

import logging
from collections import defaultdict

from lumina.audio.models import MusicState
from lumina.control.protocol import FixtureCommand
from lumina.lighting.profiles.base import BaseProfile

logger = logging.getLogger(__name__)


def _clamp(value: int) -> int:
    """Clamp an integer to the 0-255 range.

    Args:
        value: Raw integer value.

    Returns:
        Clamped value between 0 and 255.
    """
    return max(0, min(255, value))


def _weighted_avg_channel(values: list[tuple[int, float]]) -> int:
    """Compute a weighted average of (value, weight) pairs, clamped to 0-255.

    Args:
        values: List of (channel_value, weight) tuples.  Channel values
            are 0-255 integers; weights are positive floats that should
            sum to 1.0.

    Returns:
        Weighted average clamped to 0-255.
    """
    if not values:
        return 0
    total = sum(v * w for v, w in values)
    return _clamp(round(total))


def blend_commands(
    sources: list[tuple[list[FixtureCommand], float]],
    fixture_count: int,
) -> list[FixtureCommand]:
    """Blend fixture commands from multiple profile outputs.

    For each fixture_id present in any source:
    - RGBW channels are weighted averages across all sources.
    - Strobe rate and intensity come from the highest-weighted source
      that has strobe_rate > 0.  This avoids diluting strobe effects
      through averaging (a half-speed strobe looks wrong, not subtle).
    - The special byte is a weighted average (clamped 0-255).

    Args:
        sources: List of (commands, weight) tuples.  Each ``commands``
            list contains FixtureCommands from a single profile run.
            Weights should be positive and ideally sum to 1.0.
        fixture_count: Expected number of fixtures.  Fixture IDs that
            appear in any source are included; missing fixtures get a
            blackout command.

    Returns:
        One FixtureCommand per fixture, sorted by fixture_id.
    """
    if not sources:
        return [
            FixtureCommand(fixture_id=fid)
            for fid in range(1, fixture_count + 1)
        ]

    # Index commands by fixture_id for each source
    # fixture_id -> list of (FixtureCommand, weight)
    by_fixture: dict[int, list[tuple[FixtureCommand, float]]] = defaultdict(list)
    seen_ids: set[int] = set()

    for commands, weight in sources:
        if weight <= 0.0:
            continue
        for cmd in commands:
            by_fixture[cmd.fixture_id].append((cmd, weight))
            seen_ids.add(cmd.fixture_id)

    # Ensure we cover all expected fixture IDs (1-based)
    for fid in range(1, fixture_count + 1):
        seen_ids.add(fid)

    result: list[FixtureCommand] = []
    for fid in sorted(seen_ids):
        entries = by_fixture.get(fid)
        if not entries:
            result.append(FixtureCommand(fixture_id=fid))
            continue

        # Normalize weights for this fixture's entries
        total_weight = sum(w for _, w in entries)
        if total_weight <= 0.0:
            result.append(FixtureCommand(fixture_id=fid))
            continue

        norm_entries = [(cmd, w / total_weight) for cmd, w in entries]

        # Weighted average for RGBW and special
        red = _weighted_avg_channel([(cmd.red, w) for cmd, w in norm_entries])
        green = _weighted_avg_channel([(cmd.green, w) for cmd, w in norm_entries])
        blue = _weighted_avg_channel([(cmd.blue, w) for cmd, w in norm_entries])
        white = _weighted_avg_channel([(cmd.white, w) for cmd, w in norm_entries])
        special = _weighted_avg_channel(
            [(cmd.special, w) for cmd, w in norm_entries]
        )

        # Strobe: use the highest-weighted source that has strobe active
        strobe_rate = 0
        strobe_intensity = 0
        best_strobe_weight = -1.0
        for cmd, w in norm_entries:
            if cmd.strobe_rate > 0 and w > best_strobe_weight:
                strobe_rate = cmd.strobe_rate
                strobe_intensity = cmd.strobe_intensity
                best_strobe_weight = w

        result.append(
            FixtureCommand(
                fixture_id=fid,
                red=red,
                green=green,
                blue=blue,
                white=white,
                strobe_rate=strobe_rate,
                strobe_intensity=strobe_intensity,
                special=special,
            )
        )

    return result


class ProfileBlender:
    """Blends output from multiple genre profiles based on genre weights.

    Instead of picking a single dominant profile, the blender runs all
    profiles whose weight meets a minimum threshold and combines their
    outputs proportionally.  This produces smooth visual transitions
    when a track sits between genres.

    When only one profile qualifies, it delegates directly to avoid
    the overhead of blending identity.

    Args:
        profiles: Mapping of profile name to BaseProfile instance.
        min_weight: Minimum genre_weight for a profile to be activated.
            Profiles below this threshold are skipped entirely to
            avoid wasted computation.
    """

    def __init__(
        self,
        profiles: dict[str, BaseProfile],
        min_weight: float = 0.1,
    ) -> None:
        self._profiles = dict(profiles)
        self._min_weight = min_weight

    @property
    def profile_names(self) -> list[str]:
        """Names of all registered profiles.

        Returns:
            Sorted list of profile name strings.
        """
        return sorted(self._profiles)

    def generate(self, state: MusicState) -> list[FixtureCommand]:
        """Generate blended fixture commands for the current music state.

        Steps:
        1. Filter profiles to those with genre_weight >= min_weight.
        2. Normalize the active weights to sum to 1.0.
        3. If only one profile is active, delegate directly (no blend).
        4. Otherwise, run each active profile and blend their outputs.

        Args:
            state: Current audio analysis frame with genre_weights.

        Returns:
            List of FixtureCommand, one per fixture in the map.
        """
        # Collect active profiles and their raw weights
        active: list[tuple[str, BaseProfile, float]] = []
        for name, profile in self._profiles.items():
            weight = state.genre_weights.get(name, 0.0)
            if weight >= self._min_weight:
                active.append((name, profile, weight))

        # If nothing qualifies, fall back to the highest-weighted profile
        # regardless of threshold
        if not active:
            if state.genre_weights and self._profiles:
                best_name = max(
                    (n for n in state.genre_weights if n in self._profiles),
                    key=lambda n: state.genre_weights[n],
                    default=None,
                )
                if best_name is not None:
                    logger.debug(
                        "No profile above min_weight=%.2f; "
                        "falling back to %s (weight=%.3f)",
                        self._min_weight,
                        best_name,
                        state.genre_weights[best_name],
                    )
                    return self._profiles[best_name].generate(state)

            # Truly empty — pick the first profile as last resort
            if self._profiles:
                fallback = next(iter(self._profiles.values()))
                logger.debug("No genre_weights; using fallback profile %s", fallback.name)
                return fallback.generate(state)
            return []

        # Single active profile — delegate directly, no blending overhead
        if len(active) == 1:
            _, profile, _ = active[0]
            return profile.generate(state)

        # Multiple active profiles — normalize weights and blend
        total_weight = sum(w for _, _, w in active)
        normalized: list[tuple[str, BaseProfile, float]] = [
            (name, profile, w / total_weight)
            for name, profile, w in active
        ]

        logger.debug(
            "Blending %d profiles: %s",
            len(normalized),
            ", ".join(f"{n}={w:.2f}" for n, _, w in normalized),
        )

        # Run each profile and collect outputs with weights
        sources: list[tuple[list[FixtureCommand], float]] = []
        fixture_count = 0
        for _name, profile, weight in normalized:
            commands = profile.generate(state)
            if commands:
                fixture_count = max(fixture_count, len(commands))
                sources.append((commands, weight))

        if not sources:
            return []

        return blend_commands(sources, fixture_count)
