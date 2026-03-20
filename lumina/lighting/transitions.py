"""Cross-genre transition engine.

Manages smooth transitions between lighting profiles when genre_weights
shift during playback. Instead of jarring palette jumps, the engine
cross-fades fixture commands over a configurable duration using easing
curves tuned per segment type.

Transition durations are segment-aware: drops snap instantly (0.1s),
breakdowns dissolve slowly (3.0s), and choruses use a moderate blend
(1.5s). The engine supports three easing curves: linear, cubic
ease-in-out, and equal-power crossfade (sqrt).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from lumina.control.protocol import FixtureCommand

logger = logging.getLogger(__name__)


# ---- Easing functions -------------------------------------------------------


def _linear(t: float) -> float:
    """Linear interpolation (no easing).

    Args:
        t: Normalised progress in [0.0, 1.0].

    Returns:
        The same value, clamped to [0.0, 1.0].
    """
    return max(0.0, min(1.0, t))


def _ease_in_out(t: float) -> float:
    """Cubic ease-in-out curve.

    Starts slow, accelerates through the midpoint, then decelerates.
    Feels natural for most lighting cross-fades.

    Args:
        t: Normalised progress in [0.0, 1.0].

    Returns:
        Eased value in [0.0, 1.0].
    """
    t = max(0.0, min(1.0, t))
    if t < 0.5:
        return 4.0 * t * t * t
    return 1.0 - (-2.0 * t + 2.0) ** 3 / 2.0


def _crossfade(t: float) -> float:
    """Equal-power crossfade (square-root curve).

    Preserves perceived brightness during the transition by using
    sqrt scaling, which compensates for the non-linear relationship
    between light intensity and human brightness perception.

    Args:
        t: Normalised progress in [0.0, 1.0].

    Returns:
        Eased value in [0.0, 1.0].
    """
    t = max(0.0, min(1.0, t))
    return math.sqrt(t)


# Map of curve name -> easing function.
_EASING_FUNCTIONS: dict[str, callable] = {  # type: ignore[type-arg]
    "linear": _linear,
    "ease_in_out": _ease_in_out,
    "crossfade": _crossfade,
}


# ---- TransitionState --------------------------------------------------------


@dataclass(slots=True)
class TransitionState:
    """Snapshot of an in-progress transition between two profiles.

    Args:
        from_profile: Name of the outgoing lighting profile.
        to_profile: Name of the incoming lighting profile.
        progress: Current blend position, 0.0 (fully *from*) to 1.0 (fully *to*).
        duration: Total transition time in seconds.
        start_time: Timestamp (seconds) when the transition began.
        curve: Name of the easing curve ("linear", "ease_in_out", "crossfade").
    """

    from_profile: str
    to_profile: str
    progress: float = 0.0
    duration: float = 2.0
    start_time: float = 0.0
    curve: str = "ease_in_out"


# ---- TransitionEngine -------------------------------------------------------


class TransitionEngine:
    """Manages smooth cross-fades between lighting profiles.

    Tracks the currently active profile and, when the dominant profile
    changes, creates a ``TransitionState`` that drives a timed blend
    between the old and new command sets. Segment type determines
    transition speed: drops snap almost instantly while breakdowns
    dissolve slowly.

    Args:
        default_duration: Fallback transition duration in seconds when
            no segment-specific override is defined.
        curve: Default easing curve name.
    """

    def __init__(
        self,
        default_duration: float = 2.0,
        curve: str = "ease_in_out",
    ) -> None:
        if curve not in _EASING_FUNCTIONS:
            logger.warning(
                "Unknown easing curve '%s', falling back to 'ease_in_out'",
                curve,
            )
            curve = "ease_in_out"

        self._default_duration = default_duration
        self._default_curve = curve
        self._active_transition: TransitionState | None = None
        self._last_profile: str = ""

        # Segment-type -> transition duration overrides.
        self._segment_transition_durations: dict[str, float] = {
            "drop": 0.1,
            "breakdown": 3.0,
            "chorus": 1.5,
        }

    # -- Public API -----------------------------------------------------------

    @property
    def active_transition(self) -> TransitionState | None:
        """The currently active transition, or None if idle."""
        return self._active_transition

    @property
    def last_profile(self) -> str:
        """Name of the most recently active profile."""
        return self._last_profile

    def update(
        self,
        current_profile: str,
        segment: str,
        timestamp: float,
    ) -> float | None:
        """Advance the transition state for this frame.

        Call once per frame with the dominant profile name, the current
        segment label, and the wall-clock timestamp.

        * If the profile has changed since the last call a new transition
          is started and the initial blend factor (close to 0.0) is returned.
        * If a transition is already in progress its progress is updated
          and the current blend factor is returned.
        * If no transition is active, returns ``None`` — the caller should
          use the single dominant profile without blending.

        Args:
            current_profile: Name of the profile that should be active
                right now (highest genre_weight).
            segment: Current song segment ("verse", "chorus", "drop", etc.).
            timestamp: Current time in seconds (monotonic or song-time).

        Returns:
            Blend factor in [0.0, 1.0] if a transition is active, or
            ``None`` when no blending is needed.
        """
        # Bootstrap: first call ever — just record the profile.
        if not self._last_profile:
            self._last_profile = current_profile
            return None

        # Profile changed — start a new transition.
        if current_profile != self._last_profile and (
            self._active_transition is None
            or self._active_transition.to_profile != current_profile
        ):
            duration = self._segment_transition_durations.get(
                segment, self._default_duration
            )
            self._active_transition = TransitionState(
                from_profile=self._last_profile,
                to_profile=current_profile,
                progress=0.0,
                duration=duration,
                start_time=timestamp,
                curve=self._default_curve,
            )
            logger.info(
                "Transition started: %s -> %s (%.2fs, %s, segment=%s)",
                self._last_profile,
                current_profile,
                duration,
                self._default_curve,
                segment,
            )

        # Update an in-progress transition.
        if self._active_transition is not None:
            factor = self._compute_progress(timestamp)

            if factor >= 1.0:
                # Transition complete — commit to the new profile.
                logger.info(
                    "Transition complete: now on '%s'",
                    self._active_transition.to_profile,
                )
                self._last_profile = self._active_transition.to_profile
                self._active_transition = None
                return None

            return factor

        return None

    def get_blend_factor(self, timestamp: float) -> float | None:
        """Return the current blend factor without advancing state.

        Useful when a caller needs the blend value outside the main
        ``update`` loop (e.g. for debug display).

        Args:
            timestamp: Current time in seconds.

        Returns:
            Blend factor in [0.0, 1.0], or ``None`` if no transition is
            active.
        """
        if self._active_transition is None:
            return None
        return self._compute_progress(timestamp)

    def blend_outputs(
        self,
        from_cmds: list[FixtureCommand],
        to_cmds: list[FixtureCommand],
        factor: float,
    ) -> list[FixtureCommand]:
        """Cross-fade between two sets of fixture commands.

        Each fixture is blended independently by interpolating RGBW
        channels and the special byte. Strobe fields follow the
        *incoming* profile once the blend factor exceeds 0.5 — strobes
        don't blend well visually, so a hard switch at the midpoint
        is cleaner.

        Both lists must be the same length and ordered by fixture_id.
        If lengths differ the shorter list is padded with dark commands.

        Args:
            from_cmds: Commands from the outgoing profile.
            to_cmds: Commands from the incoming profile.
            factor: Blend factor in [0.0, 1.0] (0 = fully *from*,
                1 = fully *to*).

        Returns:
            Blended list of FixtureCommand with the same length as the
            longer input list.
        """
        factor = max(0.0, min(1.0, factor))
        inv = 1.0 - factor

        # Pad shorter list with blackout commands.
        max_len = max(len(from_cmds), len(to_cmds))
        from_padded = _pad_commands(from_cmds, max_len)
        to_padded = _pad_commands(to_cmds, max_len)

        result: list[FixtureCommand] = []
        for fc, tc in zip(from_padded, to_padded, strict=True):
            # RGBW + special: smooth interpolation.
            red = _lerp_channel(fc.red, tc.red, factor, inv)
            green = _lerp_channel(fc.green, tc.green, factor, inv)
            blue = _lerp_channel(fc.blue, tc.blue, factor, inv)
            white = _lerp_channel(fc.white, tc.white, factor, inv)
            special = _lerp_channel(fc.special, tc.special, factor, inv)

            # Strobe: hard switch at midpoint — blending strobe rates
            # produces ugly visual artifacts.
            if factor >= 0.5:
                strobe_rate = tc.strobe_rate
                strobe_intensity = tc.strobe_intensity
            else:
                strobe_rate = fc.strobe_rate
                strobe_intensity = fc.strobe_intensity

            # Use the incoming fixture_id (should be the same fixture).
            result.append(
                FixtureCommand(
                    fixture_id=tc.fixture_id,
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

    # -- Internal helpers -----------------------------------------------------

    def _compute_progress(self, timestamp: float) -> float:
        """Compute eased blend factor for the active transition.

        Args:
            timestamp: Current time in seconds.

        Returns:
            Eased blend factor in [0.0, 1.0].
        """
        tr = self._active_transition
        if tr is None:
            return 0.0

        if tr.duration <= 0.0:
            return 1.0

        raw = (timestamp - tr.start_time) / tr.duration
        raw = max(0.0, min(1.0, raw))

        easing_fn = _EASING_FUNCTIONS.get(tr.curve, _ease_in_out)
        eased = easing_fn(raw)

        # Store raw progress on the state for external inspection.
        tr.progress = eased
        return eased


# ---- Module-level helpers ---------------------------------------------------


def _lerp_channel(a: int, b: int, factor: float, inv: float) -> int:
    """Linearly interpolate a single 0-255 channel value.

    Args:
        a: Value from the outgoing profile.
        b: Value from the incoming profile.
        factor: Blend weight toward *b* (0.0-1.0).
        inv: Pre-computed ``1.0 - factor`` for performance.

    Returns:
        Interpolated channel value clamped to 0-255.
    """
    return max(0, min(255, round(a * inv + b * factor)))


def _pad_commands(
    cmds: list[FixtureCommand], target_len: int
) -> list[FixtureCommand]:
    """Pad a command list to *target_len* with blackout commands.

    Padding commands use fixture_id=0 and all channels at zero.

    Args:
        cmds: Original command list.
        target_len: Desired length.

    Returns:
        List of length *target_len*.
    """
    if len(cmds) >= target_len:
        return cmds
    pad = [FixtureCommand() for _ in range(target_len - len(cmds))]
    return cmds + pad
