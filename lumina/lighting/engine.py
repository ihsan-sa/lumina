"""Main lighting decision engine.

Receives MusicState at 60fps, selects the active profile(s) based on
genre_weights, runs them through the fixture map, and outputs a final
list of FixtureCommand — one per fixture.

Profile selection: the engine picks the highest-weighted registered
profile from genre_weights.  If no genre exceeds 0.3 weight, the
generic fallback profile is used to guarantee acceptable output.

LightingContext: maintained across frames to track recent patterns,
motif visual assignments, and section duration for contrast management.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field

from lumina.analysis.song_score import MotifAssignment
from lumina.audio.models import MusicState
from lumina.control.protocol import FixtureCommand
from lumina.lighting.fixture_map import FixtureMap, FixtureType
from lumina.lighting.profiles.base import BaseProfile, Color
from lumina.lighting.profiles.euro_alt import EuroAltProfile
from lumina.lighting.profiles.festival_edm import FestivalEdmProfile
from lumina.lighting.profiles.french_hard import FrenchHardProfile
from lumina.lighting.profiles.french_melodic import FrenchMelodicProfile
from lumina.lighting.profiles.generic import GenericProfile
from lumina.lighting.profiles.psych_rnb import PsychRnbProfile
from lumina.lighting.profiles.rage_trap import RageTrapProfile
from lumina.lighting.patterns import PATTERN_REGISTRY
from lumina.lighting.profiles.theatrical import TheatricalProfile
from lumina.lighting.profiles.uk_bass import UkBassProfile

logger = logging.getLogger(__name__)

# Minimum weight for a genre profile to be selected over the fallback.
_MIN_GENRE_WEIGHT = 0.3

# Context tracking constants
_RECENT_PATTERNS_MAXLEN = 16  # Track last N patterns for contrast


@dataclass
class LightingContext:
    """Cross-frame state for contrast management and motif tracking.

    Maintained by the engine across frames. Profiles read this to avoid
    repetition fatigue and to apply consistent visuals for motifs.

    Args:
        recent_patterns: Last N pattern names used (for contrast checking).
        motif_visual_map: motif_id -> assigned pattern name.
        motif_color_map: motif_id -> assigned color index.
        bars_in_section: How many bars spent in the current segment.
        recent_max_intensity: Max intensity in the last N bars.
        last_segment: Previous segment label for transition detection.
    """

    recent_patterns: deque[str] = field(
        default_factory=lambda: deque(maxlen=_RECENT_PATTERNS_MAXLEN)
    )
    motif_visual_map: dict[int, str] = field(default_factory=dict)
    motif_color_map: dict[int, int] = field(default_factory=dict)
    bars_in_section: int = 0
    recent_max_intensity: float = 0.0
    last_segment: str = ""

# Registry of profile name → class.  New profiles are added here.
_PROFILE_REGISTRY: dict[str, type[BaseProfile]] = {
    "rage_trap": RageTrapProfile,
    "psych_rnb": PsychRnbProfile,
    "festival_edm": FestivalEdmProfile,
    "french_melodic": FrenchMelodicProfile,
    "french_hard": FrenchHardProfile,
    "euro_alt": EuroAltProfile,
    "theatrical": TheatricalProfile,
    "uk_bass": UkBassProfile,
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
        self._context = LightingContext()
        self._motif_assignments: dict[int, MotifAssignment] = {}
        self._pattern_override: str | None = None
        self._genre_override: str | None = None

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

    @property
    def context(self) -> LightingContext:
        """The shared lighting context (cross-frame state)."""
        return self._context

    def set_motif_assignments(
        self, assignments: dict[int, MotifAssignment]
    ) -> None:
        """Load motif-to-visual assignments from the song score.

        Args:
            assignments: motif_id -> MotifAssignment mapping.
        """
        self._motif_assignments = dict(assignments)
        # Populate context maps
        self._context.motif_visual_map = {
            mid: a.pattern_name for mid, a in assignments.items()
        }
        self._context.motif_color_map = {
            mid: a.color_index for mid, a in assignments.items()
        }

    def set_genre_override(self, profile: str | None) -> None:
        """Force a specific genre profile, bypassing classifier output.

        Used by the control UI genre dropdown. Pass None to restore
        auto-detection from genre_weights.

        Args:
            profile: Profile name from _PROFILE_REGISTRY, or None to clear.
        """
        if profile is not None and profile not in self._profiles:
            logger.warning("Unknown genre profile: %s (ignoring)", profile)
            return
        self._genre_override = profile
        logger.info("Genre override: %s", profile or "(auto)")

    def set_pattern_override(self, name: str | None) -> None:
        """Force a specific pattern for all fixtures, bypassing profile logic.

        Used by the simulator's pattern showcase mode.  Pass None to restore
        normal profile-driven behaviour.

        Args:
            name: Pattern name from PATTERN_REGISTRY, or None to clear.
        """
        self._pattern_override = name
        logger.info("Pattern override: %s", name or "(auto)")

    def generate(self, state: MusicState) -> list[FixtureCommand]:
        """Generate fixture commands for a single frame.

        Selection logic:
        1. Find the registered profile with the highest genre_weight.
        2. If its weight >= 0.3, use it exclusively.
        3. Otherwise, use the generic fallback profile.

        The engine updates the LightingContext and passes it to profiles.

        Args:
            state: Current audio analysis frame.

        Returns:
            List of FixtureCommand, one per fixture in the map.
        """
        # Pattern override — bypass profile, run the named pattern on all fixtures
        if self._pattern_override is not None:
            import colorsys
            hue = (state.timestamp * 0.05) % 1.0  # full cycle every 20 s
            r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
            showcase_color = Color(r=r, g=g, b=b, w=0.0)
            pattern_fn = PATTERN_REGISTRY.get(self._pattern_override)
            if pattern_fn is not None:
                cmd_map: dict[int, FixtureCommand] = pattern_fn(
                    self._map.all, state, state.timestamp, showcase_color
                )

                # Patterns output RGBW color for all fixture types, but STROBE
                # fixtures render from strobe_intensity (not RGBW channels).
                # Post-process: convert max RGBW brightness → strobe_intensity DC mode.
                for f in self._map.all:
                    if f.fixture_type != FixtureType.STROBE:
                        continue
                    c = cmd_map.get(f.fixture_id)
                    if c is None or (c.strobe_rate > 0 or c.strobe_intensity > 0):
                        continue  # already has strobe channels set
                    brightness = max(c.red, c.green, c.blue, c.white)
                    if brightness > 0:
                        cmd_map[f.fixture_id] = FixtureCommand(
                            fixture_id=c.fixture_id,
                            red=c.red, green=c.green, blue=c.blue, white=c.white,
                            strobe_rate=0, strobe_intensity=brightness,
                            special=c.special,
                        )

                result: list[FixtureCommand] = []
                for f in self._map.all:
                    if f.fixture_id in cmd_map:
                        result.append(cmd_map[f.fixture_id])
                    else:
                        result.append(FixtureCommand(
                            fixture_id=f.fixture_id,
                            red=0, green=0, blue=0, white=0,
                            strobe_rate=0, strobe_intensity=0, special=0,
                        ))
                return result

        # Update context: track segment duration
        if state.segment != self._context.last_segment:
            self._context.bars_in_section = 0
            self._context.last_segment = state.segment
        if state.is_downbeat:
            self._context.bars_in_section += 1

        # Track recent max intensity for contrast
        self._context.recent_max_intensity = max(
            self._context.recent_max_intensity * 0.99,  # decay
            state.energy,
        )

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
        """Pick the active profile based on genre_weights or override.

        Falls back to generic when no registered genre exceeds
        ``_MIN_GENRE_WEIGHT`` (0.3).

        Args:
            state: Current music state with genre_weights.

        Returns:
            The selected BaseProfile instance.
        """
        # Genre override from UI takes priority
        if self._genre_override is not None and self._genre_override in self._profiles:
            return self._profiles[self._genre_override]

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
