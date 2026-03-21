"""Safety limiter for LUMINA fixture commands.

Mandatory post-processing stage that prevents photosensitive seizure risks.
This module caps strobe rates, limits simultaneous strobe fixtures, and
enforces total brightness ceilings. It cannot be bypassed by profiles or
manual effects.

The limiter tracks a rolling 5-second window of strobe events per fixture
to enforce sustained and burst limits. Three safety levels are supported:

- STANDARD: Safe for general audiences (default).
- ACCESSIBLE: Extra conservative for photosensitive individuals.
- UNRESTRICTED: No limits applied (testing only, logged warning).
"""

from __future__ import annotations

import logging
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

from lumina.control.protocol import FixtureCommand

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────

# strobe_rate maps 0-255 to 0-25Hz linearly.
# 3Hz sustained max → 3/25 * 255 ≈ 30.6 → 30
_MAX_SUSTAINED_RATE_STANDARD = 30
# 10Hz burst max → 10/25 * 255 ≈ 102
_MAX_BURST_RATE_STANDARD = 102

_BURST_DURATION_S = 1.0  # Max burst duration in seconds
_BURST_COOLDOWN_S = 3.0  # Cooldown after burst before another burst allowed

_MAX_SIMULTANEOUS_STROBES_STANDARD = 8  # Out of 15 fixtures
_BRIGHTNESS_CEILING_STANDARD = 0.80  # 80% of theoretical max

# ACCESSIBLE level dividers / overrides
_ACCESSIBLE_RATE_DIVISOR = 2
_BRIGHTNESS_CEILING_ACCESSIBLE = 0.60

# Rolling window duration for tracking strobe history
_WINDOW_DURATION_S = 5.0


class SafetyLevel(Enum):
    """Safety level for the limiter."""

    STANDARD = "standard"
    ACCESSIBLE = "accessible"
    UNRESTRICTED = "unrestricted"


@dataclass
class _FixtureStrobeState:
    """Per-fixture strobe tracking state.

    Tracks when this fixture started strobing above the sustained limit
    to enforce burst duration and cooldown rules.
    """

    burst_start: float | None = None
    cooldown_until: float = 0.0
    # Rolling window of (timestamp, strobe_rate) samples
    history: deque[tuple[float, int]] = field(
        default_factory=lambda: deque(maxlen=300)  # 5s at 60fps
    )


class SafetyLimiter:
    """Mandatory post-processing safety limiter for fixture commands.

    Enforces strobe rate caps, simultaneous strobe limits, and total
    brightness ceilings to prevent photosensitive seizure risks.

    This limiter is applied after all profile and blending logic and
    cannot be bypassed.

    Args:
        safety_level: The safety enforcement level.
        fixture_count: Total number of fixtures in the venue.
        time_fn: Optional callable returning current time in seconds.
            Defaults to time.monotonic. Useful for testing.
    """

    def __init__(
        self,
        safety_level: SafetyLevel = SafetyLevel.STANDARD,
        fixture_count: int = 15,
        time_fn: Callable[[], float] | None = None,
    ) -> None:
        self._level = safety_level
        self._fixture_count = fixture_count
        self._time_fn = time_fn if time_fn is not None else time.monotonic
        self._fixture_states: dict[int, _FixtureStrobeState] = {}

        # Stats tracking
        self._stats = {
            "strobe_rate_caps": 0,
            "burst_cooldowns": 0,
            "simultaneous_strobe_caps": 0,
            "brightness_caps": 0,
            "frames_processed": 0,
        }

        if safety_level == SafetyLevel.UNRESTRICTED:
            logger.warning(
                "SafetyLimiter initialized in UNRESTRICTED mode. "
                "All safety limits are disabled. This is for testing only."
            )

        # Compute effective limits based on level
        if safety_level == SafetyLevel.ACCESSIBLE:
            self._max_sustained_rate = _MAX_SUSTAINED_RATE_STANDARD // 2
            self._max_burst_rate = _MAX_BURST_RATE_STANDARD // 2
            self._max_simultaneous = _MAX_SIMULTANEOUS_STROBES_STANDARD // 2
            self._brightness_ceiling = _BRIGHTNESS_CEILING_ACCESSIBLE
        else:
            self._max_sustained_rate = _MAX_SUSTAINED_RATE_STANDARD
            self._max_burst_rate = _MAX_BURST_RATE_STANDARD
            self._max_simultaneous = _MAX_SIMULTANEOUS_STROBES_STANDARD
            self._brightness_ceiling = _BRIGHTNESS_CEILING_STANDARD

    @property
    def safety_level(self) -> SafetyLevel:
        """The current safety level."""
        return self._level

    def process(self, commands: list[FixtureCommand]) -> list[FixtureCommand]:
        """Apply safety limits to fixture commands.

        This is the main entry point. All commands pass through here
        after profile generation and blending.

        Args:
            commands: Raw fixture commands from the lighting engine.

        Returns:
            Safety-limited fixture commands (new list, originals unchanged).
        """
        self._stats["frames_processed"] += 1

        if self._level == SafetyLevel.UNRESTRICTED:
            return list(commands)

        if not commands:
            return []

        now = self._time_fn()

        # Step 1: Apply per-fixture strobe rate limits (sustained + burst)
        limited = [self._limit_strobe_rate(cmd, now) for cmd in commands]

        # Step 2: Enforce simultaneous strobe limit
        limited = self._limit_simultaneous_strobes(limited)

        # Step 3: Enforce total brightness ceiling
        limited = self._limit_brightness(limited)

        return limited

    def get_stats(self) -> dict[str, int]:
        """Return current safety statistics.

        Returns:
            Dict with counts of limiting events and frames processed.
        """
        return dict(self._stats)

    def _get_fixture_state(self, fixture_id: int) -> _FixtureStrobeState:
        """Get or create per-fixture strobe tracking state.

        Args:
            fixture_id: The fixture identifier.

        Returns:
            The fixture's strobe state tracker.
        """
        if fixture_id not in self._fixture_states:
            self._fixture_states[fixture_id] = _FixtureStrobeState()
        return self._fixture_states[fixture_id]

    def _limit_strobe_rate(
        self, cmd: FixtureCommand, now: float
    ) -> FixtureCommand:
        """Apply per-fixture strobe rate limiting with burst tracking.

        Rules:
        - Sustained strobe rate is capped at max_sustained_rate.
        - Burst up to max_burst_rate is allowed for up to 1 second.
        - After a burst, a 3-second cooldown enforces sustained rate only.

        Args:
            cmd: The fixture command to limit.
            now: Current timestamp.

        Returns:
            A new FixtureCommand with limited strobe_rate if needed.
        """
        if cmd.strobe_rate == 0:
            # Not strobing — reset burst tracking
            state = self._get_fixture_state(cmd.fixture_id)
            if state.burst_start is not None:
                state.burst_start = None
            # Prune old history
            self._prune_history(state, now)
            state.history.append((now, 0))
            return cmd

        state = self._get_fixture_state(cmd.fixture_id)
        self._prune_history(state, now)
        state.history.append((now, cmd.strobe_rate))

        effective_rate = cmd.strobe_rate
        is_above_sustained = cmd.strobe_rate > self._max_sustained_rate

        if is_above_sustained:
            # Check if we are in cooldown
            if now < state.cooldown_until:
                # In cooldown: cap to sustained rate
                effective_rate = min(cmd.strobe_rate, self._max_sustained_rate)
                self._stats["burst_cooldowns"] += 1
                logger.debug(
                    "Fixture %d: burst cooldown active, capping to %d",
                    cmd.fixture_id,
                    effective_rate,
                )
            elif state.burst_start is None:
                # Start a new burst
                state.burst_start = now
                effective_rate = min(cmd.strobe_rate, self._max_burst_rate)
            else:
                # In an active burst — check duration
                burst_elapsed = now - state.burst_start
                if burst_elapsed <= _BURST_DURATION_S:
                    # Still within burst window
                    effective_rate = min(
                        cmd.strobe_rate, self._max_burst_rate
                    )
                else:
                    # Burst expired — enter cooldown
                    state.burst_start = None
                    state.cooldown_until = now + _BURST_COOLDOWN_S
                    effective_rate = min(
                        cmd.strobe_rate, self._max_sustained_rate
                    )
                    self._stats["burst_cooldowns"] += 1
                    logger.debug(
                        "Fixture %d: burst expired, entering cooldown",
                        cmd.fixture_id,
                    )
        else:
            # Below sustained limit — reset burst state
            if state.burst_start is not None:
                state.burst_start = None

        if effective_rate != cmd.strobe_rate:
            self._stats["strobe_rate_caps"] += 1
            logger.debug(
                "Fixture %d: strobe_rate capped %d -> %d",
                cmd.fixture_id,
                cmd.strobe_rate,
                effective_rate,
            )

        if effective_rate == cmd.strobe_rate:
            return cmd

        return FixtureCommand(
            fixture_id=cmd.fixture_id,
            red=cmd.red,
            green=cmd.green,
            blue=cmd.blue,
            white=cmd.white,
            strobe_rate=effective_rate,
            strobe_intensity=cmd.strobe_intensity,
            special=cmd.special,
        )

    def _prune_history(
        self, state: _FixtureStrobeState, now: float
    ) -> None:
        """Remove history entries older than the rolling window.

        Args:
            state: The fixture state to prune.
            now: Current timestamp.
        """
        cutoff = now - _WINDOW_DURATION_S
        while state.history and state.history[0][0] < cutoff:
            state.history.popleft()

    def _limit_simultaneous_strobes(
        self, commands: list[FixtureCommand]
    ) -> list[FixtureCommand]:
        """Limit how many fixtures can strobe simultaneously.

        When more than max_simultaneous fixtures have strobe_rate > 0,
        the lowest-rate fixtures are capped to strobe_rate=0.

        Args:
            commands: List of (possibly already rate-limited) commands.

        Returns:
            New list with simultaneous strobe count enforced.
        """
        strobing = [
            (i, cmd) for i, cmd in enumerate(commands) if cmd.strobe_rate > 0
        ]

        if len(strobing) <= self._max_simultaneous:
            return commands

        self._stats["simultaneous_strobe_caps"] += 1
        logger.debug(
            "Simultaneous strobe limit: %d fixtures strobing, max %d",
            len(strobing),
            self._max_simultaneous,
        )

        # Keep the fixtures with the highest strobe rates
        strobing.sort(key=lambda x: x[1].strobe_rate, reverse=True)
        kill_indices = {idx for idx, _ in strobing[self._max_simultaneous :]}

        result = []
        for i, cmd in enumerate(commands):
            if i in kill_indices:
                result.append(
                    FixtureCommand(
                        fixture_id=cmd.fixture_id,
                        red=cmd.red,
                        green=cmd.green,
                        blue=cmd.blue,
                        white=cmd.white,
                        strobe_rate=0,
                        strobe_intensity=0,
                        special=cmd.special,
                    )
                )
            else:
                result.append(cmd)
        return result

    def _limit_brightness(
        self, commands: list[FixtureCommand]
    ) -> list[FixtureCommand]:
        """Enforce total brightness ceiling across all fixtures.

        Sums all intensity channels (R+G+B+W+strobe_intensity+special)
        across all fixtures and scales down proportionally if the total
        exceeds the ceiling percentage of theoretical maximum.

        Args:
            commands: List of commands after strobe limiting.

        Returns:
            New list with brightness scaled down if needed.
        """
        if not commands:
            return commands

        # Theoretical max: each fixture has 6 brightness channels at 255
        max_total = len(commands) * 6 * 255
        ceiling = max_total * self._brightness_ceiling

        total = sum(
            cmd.red + cmd.green + cmd.blue + cmd.white
            + cmd.strobe_intensity + cmd.special
            for cmd in commands
        )

        if total <= ceiling:
            return commands

        self._stats["brightness_caps"] += 1
        scale = ceiling / total
        logger.debug(
            "Brightness ceiling: total %d exceeds ceiling %.0f, "
            "scaling by %.3f",
            total,
            ceiling,
            scale,
        )

        result = []
        for cmd in commands:
            result.append(
                FixtureCommand(
                    fixture_id=cmd.fixture_id,
                    red=int(cmd.red * scale),
                    green=int(cmd.green * scale),
                    blue=int(cmd.blue * scale),
                    white=int(cmd.white * scale),
                    strobe_rate=cmd.strobe_rate,
                    strobe_intensity=int(cmd.strobe_intensity * scale),
                    special=int(cmd.special * scale),
                )
            )
        return result
