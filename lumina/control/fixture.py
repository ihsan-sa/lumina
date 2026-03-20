"""Fixture abstraction layer — bridges layout metadata to the network control layer.

``FixtureState`` tracks the live output state of a single physical fixture,
combining its static ``FixtureInfo`` metadata (position, type, role) with
runtime state (online/offline, last received heartbeat, last sent command).

``FixtureRegistry`` owns the full set of known fixtures and is the single
source of truth for which fixtures are online at any given moment.  The
network manager writes commands into the registry via ``update_command``
and ``apply_commands``; the discovery/heartbeat subsystem calls
``mark_seen`` and ``check_timeouts``.

Typical call sequence (60 fps loop)::

    registry.apply_commands(commands)    # from lighting engine
    registry.check_timeouts()            # prune stale fixtures
    online = registry.online_fixtures()  # pass to network sender
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from lumina.control.protocol import FixtureCommand
from lumina.lighting.fixture_map import FixtureInfo, FixtureMap

logger = logging.getLogger(__name__)


# ─── FixtureState ─────────────────────────────────────────────────────────────


@dataclass
class FixtureState:
    """Runtime state of a single physical fixture.

    Combines static layout metadata (from ``FixtureInfo``) with live network
    status and the last command sent to this fixture.

    Args:
        fixture_id: Unique fixture address (1-255).  Mirrors
            ``info.fixture_id`` and kept at the top level for fast access.
        info: Static layout metadata (type, position, role, groups, name).
        last_command: Most recent ``FixtureCommand`` sent to this fixture,
            or ``None`` if no command has been sent yet.
        online: ``True`` if a heartbeat or discovery response has been
            received within the configured timeout window.
        last_seen: Monotonic timestamp (seconds) of the most recent
            heartbeat or discovery packet.  ``0.0`` means never seen.
        firmware_version: Firmware version integer reported in the last
            discovery response.  Defaults to ``0`` when unknown.
    """

    fixture_id: int
    info: FixtureInfo
    last_command: FixtureCommand | None = field(default=None)
    online: bool = False
    last_seen: float = 0.0
    firmware_version: int = 0

    def seconds_since_seen(self) -> float:
        """Return elapsed seconds since the last heartbeat or discovery response.

        Returns:
            Elapsed time in seconds using the monotonic clock, or
            ``float('inf')`` if the fixture has never been seen.
        """
        if self.last_seen == 0.0:
            return float("inf")
        return time.monotonic() - self.last_seen

    def is_dark(self) -> bool:
        """Return ``True`` if the last sent command has all output channels at zero.

        A fixture that has never received a command is also considered dark.

        Returns:
            ``True`` when the fixture is effectively producing no light output.
        """
        if self.last_command is None:
            return True
        cmd = self.last_command
        return (
            cmd.red == 0
            and cmd.green == 0
            and cmd.blue == 0
            and cmd.white == 0
            and cmd.strobe_rate == 0
            and cmd.strobe_intensity == 0
            and cmd.special == 0
        )

    def __repr__(self) -> str:
        status = "ONLINE" if self.online else "OFFLINE"
        return (
            f"<FixtureState id={self.fixture_id} name={self.info.name!r}"
            f" type={self.info.fixture_type.value} {status}"
            f" fw={self.firmware_version}>"
        )


# ─── FixtureRegistry ──────────────────────────────────────────────────────────


class FixtureRegistry:
    """Manages runtime state for all fixtures in the venue.

    The registry is the single source of truth for which fixtures exist,
    which are currently reachable, and what each fixture was last commanded
    to do.  It is constructed from a ``FixtureMap`` (static layout) and all
    fixtures begin in the ``offline`` state; they transition to ``online``
    when the discovery/heartbeat subsystem calls ``mark_seen``.

    The registry is **not** thread-safe.  All calls must originate from
    the same asyncio event loop that owns the network manager.

    Args:
        fixture_map: Static fixture layout from ``lumina.lighting.fixture_map``.
            One ``FixtureState`` entry is created for every fixture in the map.

    Example::

        fmap = FixtureMap()               # default 15-fixture layout
        registry = FixtureRegistry(fmap)

        # Discovery/heartbeat callback
        registry.mark_seen(fixture_id=1, firmware_version=3)

        # Lighting engine output -> registry -> network sender
        registry.apply_commands(commands)

        # Periodic health check (call once per second is sufficient)
        registry.check_timeouts(timeout=10.0)

        for state in registry.online_fixtures():
            send_udp(state.last_command)
    """

    def __init__(self, fixture_map: FixtureMap) -> None:
        self._states: dict[int, FixtureState] = {}
        for info in fixture_map.all:
            self._states[info.fixture_id] = FixtureState(
                fixture_id=info.fixture_id,
                info=info,
            )
        logger.info(
            "FixtureRegistry initialised with %d fixtures: IDs %s",
            len(self._states),
            sorted(self._states),
        )

    # ─── Read access ──────────────────────────────────────────────────

    def get(self, fixture_id: int) -> FixtureState | None:
        """Return the state for a given fixture ID, or ``None`` if unknown.

        Args:
            fixture_id: The fixture address to look up (1-255).

        Returns:
            ``FixtureState`` instance, or ``None`` if the ID is not in the
            registry.
        """
        return self._states.get(fixture_id)

    def all_states(self) -> list[FixtureState]:
        """Return all registered fixture states, sorted by fixture ID.

        Returns:
            List of every ``FixtureState`` in the registry, regardless of
            online/offline status, in ascending fixture-ID order.
        """
        return sorted(self._states.values(), key=lambda s: s.fixture_id)

    def online_fixtures(self) -> list[FixtureState]:
        """Return only the fixtures that are currently marked online.

        A fixture is online when ``mark_seen`` has been called for it within
        the last ``timeout`` seconds (as enforced by ``check_timeouts``).

        Returns:
            List of online ``FixtureState`` instances sorted by fixture ID.
        """
        return sorted(
            (s for s in self._states.values() if s.online),
            key=lambda s: s.fixture_id,
        )

    def __len__(self) -> int:
        """Return the total number of registered fixtures (online or offline)."""
        return len(self._states)

    # ─── State mutations ──────────────────────────────────────────────

    def update_command(self, cmd: FixtureCommand) -> None:
        """Record the most recently sent command for a single fixture.

        Called by the network manager after each UDP send so that the registry
        always reflects current output state.  Broadcast commands
        (``fixture_id == 0``) update every registered fixture.

        Args:
            cmd: The ``FixtureCommand`` that was (or is about to be) transmitted.
        """
        if cmd.fixture_id == 0:
            # Broadcast: apply to every fixture
            for state in self._states.values():
                state.last_command = cmd
            logger.debug("Broadcast command stored for all %d fixtures.", len(self._states))
        else:
            state = self._states.get(cmd.fixture_id)
            if state is None:
                logger.warning(
                    "update_command: fixture ID %d is not registered.", cmd.fixture_id
                )
                return
            state.last_command = cmd

    def mark_seen(self, fixture_id: int, firmware_version: int = 0) -> None:
        """Record that a heartbeat or discovery packet was received from a fixture.

        Marks the fixture as online and updates ``last_seen`` to the current
        monotonic clock value.  If the fixture ID is not in the registry a
        warning is logged and the call is silently ignored.

        Args:
            fixture_id: The fixture that sent the packet (1-255).
            firmware_version: Firmware version reported in the packet.
                Defaults to ``0`` when not available (e.g., plain heartbeat
                packets that carry no version field).
        """
        state = self._states.get(fixture_id)
        if state is None:
            logger.warning(
                "mark_seen: received packet from unknown fixture ID %d — "
                "fixture not in layout, ignoring.",
                fixture_id,
            )
            return

        was_offline = not state.online
        state.online = True
        state.last_seen = time.monotonic()
        state.firmware_version = firmware_version

        if was_offline:
            logger.info(
                "Fixture %d (%s) came online (firmware v%d).",
                fixture_id,
                state.info.name,
                firmware_version,
            )
        else:
            logger.debug(
                "Fixture %d (%s) heartbeat (firmware v%d).",
                fixture_id,
                state.info.name,
                firmware_version,
            )

    def check_timeouts(self, timeout: float = 10.0) -> None:
        """Mark fixtures offline if they have not been seen within *timeout* seconds.

        Should be called periodically (e.g., once per second) from a
        background asyncio task.  Fixtures that have never been seen
        (``last_seen == 0.0``) are skipped because they were never online.

        Args:
            timeout: Seconds without a heartbeat before a fixture is declared
                offline.  Default is ``10.0`` — two full heartbeat cycles at
                the standard 5-second fixture heartbeat interval.
        """
        now = time.monotonic()
        for state in self._states.values():
            if not state.online:
                continue
            elapsed = now - state.last_seen
            if elapsed > timeout:
                state.online = False
                logger.warning(
                    "Fixture %d (%s) timed out after %.1fs without heartbeat.",
                    state.fixture_id,
                    state.info.name,
                    elapsed,
                )

    def apply_commands(self, commands: list[FixtureCommand]) -> None:
        """Batch-store fixture commands from a single lighting engine frame.

        Iterates over *commands* and calls ``update_command`` for each entry.
        Commands targeting unregistered IDs are logged as warnings and skipped.
        Broadcast commands (``fixture_id == 0``) propagate to every fixture.

        Args:
            commands: List of ``FixtureCommand`` instances produced by the
                lighting engine for one 60 fps frame.
        """
        for cmd in commands:
            self.update_command(cmd)
        logger.debug("apply_commands: stored %d commands.", len(commands))

    # ─── Diagnostics ──────────────────────────────────────────────────

    def summary(self) -> str:
        """Return a human-readable status summary for logging or debug output.

        Returns:
            Multi-line string listing each fixture's ID, name, type, online
            status, and the most recently sent RGBW + strobe values.
        """
        lines: list[str] = [
            f"FixtureRegistry: {len(self._states)} fixtures, "
            f"{len(self.online_fixtures())} online"
        ]
        for state in self.all_states():
            status = "ONLINE " if state.online else "OFFLINE"
            last_cmd_info = ""
            if state.last_command is not None:
                cmd = state.last_command
                last_cmd_info = (
                    f" | last_cmd=RGB({cmd.red},{cmd.green},{cmd.blue})"
                    f" W={cmd.white} strobe={cmd.strobe_rate}"
                )
            lines.append(
                f"  [{status}] ID={state.fixture_id:3d} {state.info.name:<12s}"
                f" ({state.info.fixture_type.value})"
                f"{last_cmd_info}"
            )
        return "\n".join(lines)
