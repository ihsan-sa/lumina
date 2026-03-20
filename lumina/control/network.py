"""LUMINA fixture network manager — 60fps UDP command delivery.

Manages a non-blocking UDP socket and dispatches encoded fixture command packets
to physical fixtures over unicast and/or broadcast. The ``NetworkManager`` is the
sole owner of the outbound socket; callers interact only through the high-level
``send_commands`` coroutine.

Typical usage::

    manager = NetworkManager(target_fps=60)
    manager.add_fixture_target(1, "192.168.1.101")
    manager.add_fixture_target(2, "192.168.1.102")
    await manager.start()

    # Inside the 60fps loop:
    await manager.send_commands(commands, timestamp=current_time)

    await manager.stop()
"""

from __future__ import annotations

import asyncio
import logging
import socket
import time
from dataclasses import dataclass

from lumina.control.protocol import (
    MAX_FIXTURES_PER_PACKET,
    PROTOCOL_PORT,
    FixtureCommand,
    PacketType,
    encode_packet,
)

log = logging.getLogger(__name__)

# Exponential moving-average smoothing factor for latency tracking (0 < α ≤ 1).
# At 60fps each sample is ~16.7ms apart; α=0.1 gives a ~150ms smoothing window.
_EMA_ALPHA = 0.1

# Maximum sequence number before wrapping (uint16).
_SEQ_MAX = 0xFFFF


# ─── NetworkStats ─────────────────────────────────────────────────────────────


@dataclass
class NetworkStats:
    """Runtime send statistics for the network manager.

    Attributes:
        frames_sent: Total number of ``send_commands`` calls that resulted in at
            least one successful UDP send.
        bytes_sent: Cumulative bytes written to the socket.
        send_errors: Number of ``OSError`` exceptions caught during sends.
        last_send_time: ``time.monotonic()`` timestamp of the most recent send
            attempt (0.0 if no send has occurred yet).
        avg_send_latency_ms: Exponential moving average of per-call encode+send
            duration in milliseconds.
    """

    frames_sent: int = 0
    bytes_sent: int = 0
    send_errors: int = 0
    last_send_time: float = 0.0
    avg_send_latency_ms: float = 0.0


# ─── NetworkManager ───────────────────────────────────────────────────────────


class NetworkManager:
    """Async UDP network manager for LUMINA fixture control.

    Encodes ``FixtureCommand`` lists into the LUMINA binary packet format and
    dispatches them via UDP unicast (per-fixture) and/or broadcast (all-call)
    at up to ``target_fps`` frames per second.

    The manager does *not* run its own timer loop — the caller is responsible for
    invoking ``send_commands`` at the desired cadence (typically driven by the
    audio analysis loop). This keeps scheduling control in the caller and avoids
    double-buffering latency.

    Unicast vs. broadcast dispatch rules:
    - If one or more fixture targets are registered via ``add_fixture_target``,
      commands are grouped by destination IP and sent as per-destination packets.
      Commands for fixtures without a registered IP are dropped with a warning.
    - If a broadcast target is set via ``set_broadcast_target`` (and no unicast
      targets are registered), a single broadcast packet carrying all commands is
      sent to that address.
    - If *both* unicast targets and a broadcast target exist, unicast is used and
      the broadcast target is silently skipped. Set broadcast only when operating
      in discovery/setup mode without per-fixture IP assignments.

    Args:
        target_fps: Informational only — used to compute the nominal frame
            interval for logging. Does not drive any internal timer.
        port: Default UDP destination port (overridden per-target if needed).
    """

    def __init__(self, target_fps: int = 60, port: int = PROTOCOL_PORT) -> None:
        self._target_fps: int = target_fps
        self._default_port: int = port

        self._socket: socket.socket | None = None
        self._sequence: int = 0

        # fixture_id -> (ip, port)
        self._targets: dict[int, tuple[str, int]] = {}
        self._broadcast_addr: tuple[str, int] | None = None

        self._running: bool = False
        self._stats: NetworkStats = NetworkStats()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Create the UDP socket and mark the manager as running.

        The socket is set to non-blocking mode and configured for broadcast so
        that ``set_broadcast_target`` works without further socket options changes.

        Raises:
            RuntimeError: If ``start`` is called while already running.
        """
        if self._running:
            raise RuntimeError("NetworkManager is already running")

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        # Increase the send buffer so bursts of 20+ fixture packets don't drop.
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1 << 16)  # 64 KiB
        sock.setblocking(False)

        self._socket = sock
        self._running = True
        self._sequence = 0

        nominal_interval_ms = 1000.0 / self._target_fps
        log.info(
            "NetworkManager started — port=%d target_fps=%d (%.1fms interval)",
            self._default_port,
            self._target_fps,
            nominal_interval_ms,
        )

    async def stop(self) -> None:
        """Close the UDP socket and mark the manager as stopped.

        Safe to call multiple times; subsequent calls are no-ops.
        """
        if not self._running:
            return

        self._running = False

        if self._socket is not None:
            try:
                self._socket.close()
            except OSError as exc:
                log.warning("Error closing UDP socket: %s", exc)
            finally:
                self._socket = None

        log.info(
            "NetworkManager stopped — frames_sent=%d bytes_sent=%d errors=%d",
            self._stats.frames_sent,
            self._stats.bytes_sent,
            self._stats.send_errors,
        )

    # ── Target management ─────────────────────────────────────────────────────

    def set_broadcast_target(self, addr: str, port: int = PROTOCOL_PORT) -> None:
        """Set the UDP broadcast destination.

        Used when fixture IPs are not yet known (e.g., during initial discovery)
        or when an all-call command must reach every fixture on the subnet.

        Args:
            addr: Broadcast IP address (e.g., ``"192.168.1.255"`` or
                ``"255.255.255.255"``).
            port: UDP destination port. Defaults to ``PROTOCOL_PORT`` (5150).
        """
        self._broadcast_addr = (addr, port)
        log.debug("Broadcast target set to %s:%d", addr, port)

    def add_fixture_target(
        self, fixture_id: int, addr: str, port: int = PROTOCOL_PORT
    ) -> None:
        """Register a unicast UDP destination for a specific fixture.

        If the fixture ID is already registered, the address is updated.

        Args:
            fixture_id: Fixture identifier (1-255). 0 is reserved for broadcast
                commands and is not valid here.
            addr: IPv4 address of the fixture (e.g., ``"192.168.1.101"``).
            port: UDP destination port. Defaults to ``PROTOCOL_PORT`` (5150).

        Raises:
            ValueError: If ``fixture_id`` is 0 or outside the range 1-255.
        """
        if not (1 <= fixture_id <= 255):
            raise ValueError(f"fixture_id must be 1-255, got {fixture_id}")
        self._targets[fixture_id] = (addr, port)
        log.debug("Fixture %d registered at %s:%d", fixture_id, addr, port)

    def remove_fixture_target(self, fixture_id: int) -> None:
        """Remove a previously registered unicast fixture target.

        Silently ignores unknown fixture IDs.

        Args:
            fixture_id: Fixture identifier to remove.
        """
        if self._targets.pop(fixture_id, None) is not None:
            log.debug("Fixture %d removed from unicast targets", fixture_id)

    # ── Send interface ────────────────────────────────────────────────────────

    async def send_commands(
        self,
        commands: list[FixtureCommand],
        timestamp: float = 0.0,
    ) -> None:
        """Encode and send fixture commands over UDP.

        Commands are split into chunks of at most ``MAX_FIXTURES_PER_PACKET``
        (32) before encoding, so callers may pass arbitrarily long lists without
        worrying about packet size limits.

        Dispatch strategy (in priority order):
        1. **Unicast** — if ``_targets`` is non-empty, each command is routed to
           its fixture's registered IP. Commands for unregistered fixture IDs are
           logged and dropped. Multiple commands destined for the same IP are
           coalesced into a single packet (up to 32 per packet).
        2. **Broadcast** — if ``_targets`` is empty *and* ``_broadcast_addr`` is
           set, all commands are packed into broadcast packet(s) sent to the
           broadcast address.
        3. If neither is configured, a warning is logged and the frame is skipped.

        The sequence counter is incremented once per ``send_commands`` call (not
        once per UDP datagram), so all datagrams belonging to the same logical
        frame share the same sequence number.

        Args:
            commands: Fixture commands to send. May be empty (no-op).
            timestamp: Current audio/session time in seconds. Converted to
                milliseconds and stored as a uint16 (wraps at 65535ms ≈ 65s).
        """
        if not self._running:
            log.warning("send_commands called while NetworkManager is not running")
            return

        if not commands:
            return

        t_start = time.monotonic()
        timestamp_ms = int(timestamp * 1000) & 0xFFFF
        seq = self._sequence

        if self._targets:
            await self._send_unicast(commands, seq, timestamp_ms)
        elif self._broadcast_addr is not None:
            await self._send_broadcast(commands, seq, timestamp_ms)
        else:
            log.warning(
                "No unicast targets or broadcast address configured — "
                "fixture commands dropped (frame seq=%d)",
                seq,
            )
            return

        # Advance sequence after a successful dispatch attempt.
        self._sequence = (seq + 1) & _SEQ_MAX

        t_elapsed_ms = (time.monotonic() - t_start) * 1000.0
        self._stats.last_send_time = t_start
        self._stats.frames_sent += 1

        # Exponential moving average for latency.
        if self._stats.avg_send_latency_ms == 0.0:
            self._stats.avg_send_latency_ms = t_elapsed_ms
        else:
            self._stats.avg_send_latency_ms = (
                _EMA_ALPHA * t_elapsed_ms
                + (1.0 - _EMA_ALPHA) * self._stats.avg_send_latency_ms
            )

    async def send_raw(self, data: bytes, addr: tuple[str, int]) -> None:
        """Send a raw byte buffer to an arbitrary UDP destination.

        This is the lowest-level send primitive. It does not touch the sequence
        counter or stats (except ``send_errors`` on failure). Prefer
        ``send_commands`` for normal fixture control frames.

        Args:
            data: Raw bytes to transmit.
            addr: ``(host, port)`` destination tuple.

        Raises:
            RuntimeError: If called before ``start()``.
        """
        if self._socket is None:
            raise RuntimeError("Socket is not open — call start() first")

        loop = asyncio.get_running_loop()
        try:
            await loop.sock_sendto(self._socket, data, addr)
            self._stats.bytes_sent += len(data)
        except OSError as exc:
            self._stats.send_errors += 1
            log.error("UDP send to %s:%d failed: %s", addr[0], addr[1], exc)

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def stats(self) -> NetworkStats:
        """Current send statistics snapshot.

        Returns:
            A reference to the internal ``NetworkStats`` dataclass. Fields are
            updated in-place; callers that need a frozen snapshot should copy it.
        """
        return self._stats

    @property
    def is_running(self) -> bool:
        """True if the manager has been started and not yet stopped."""
        return self._running

    @property
    def fixture_count(self) -> int:
        """Number of registered unicast fixture targets."""
        return len(self._targets)

    # ── Internal helpers ──────────────────────────────────────────────────────

    async def _send_unicast(
        self,
        commands: list[FixtureCommand],
        seq: int,
        timestamp_ms: int,
    ) -> None:
        """Route each command to its registered fixture IP and send.

        Commands are grouped by destination address so that fixtures sharing an
        IP (unusual but valid in testing) receive a single coalesced packet.
        Commands for fixture IDs with no registered target are dropped.

        Args:
            commands: Full list of fixture commands for this frame.
            seq: Packet sequence number for this frame.
            timestamp_ms: Frame timestamp (uint16 milliseconds).
        """
        # Group commands by destination address.
        by_addr: dict[tuple[str, int], list[FixtureCommand]] = {}
        for cmd in commands:
            target = self._targets.get(cmd.fixture_id)
            if target is None:
                log.debug(
                    "No target registered for fixture %d — command dropped",
                    cmd.fixture_id,
                )
                continue
            by_addr.setdefault(target, []).append(cmd)

        for addr, addr_commands in by_addr.items():
            await self._send_chunked(addr_commands, seq, timestamp_ms, addr)

    async def _send_broadcast(
        self,
        commands: list[FixtureCommand],
        seq: int,
        timestamp_ms: int,
    ) -> None:
        """Send all commands as broadcast packet(s) to ``_broadcast_addr``.

        Args:
            commands: Full list of fixture commands for this frame.
            seq: Packet sequence number for this frame.
            timestamp_ms: Frame timestamp (uint16 milliseconds).
        """
        assert self._broadcast_addr is not None  # guarded by caller
        await self._send_chunked(commands, seq, timestamp_ms, self._broadcast_addr)

    async def _send_chunked(
        self,
        commands: list[FixtureCommand],
        seq: int,
        timestamp_ms: int,
        addr: tuple[str, int],
    ) -> None:
        """Split ``commands`` into max-32 chunks and send each as a UDP packet.

        All chunks share the same ``seq`` and ``timestamp_ms`` so the receiver
        can correlate them as a single logical frame.

        Args:
            commands: Commands to send (arbitrary length).
            seq: Sequence number shared across all chunks.
            timestamp_ms: Timestamp shared across all chunks.
            addr: ``(host, port)`` destination.
        """
        for offset in range(0, len(commands), MAX_FIXTURES_PER_PACKET):
            chunk = commands[offset : offset + MAX_FIXTURES_PER_PACKET]
            try:
                packet = encode_packet(
                    chunk,
                    sequence=seq,
                    timestamp_ms=timestamp_ms,
                    packet_type=PacketType.COMMAND,
                )
            except ValueError as exc:
                log.error("Failed to encode packet chunk: %s", exc)
                self._stats.send_errors += 1
                continue

            await self.send_raw(packet, addr)
