"""LUMINA fixture discovery and heartbeat service.

Implements the host-side discovery workflow:
  1. Host broadcasts a DISCOVER_REQUEST packet to 255.255.255.255:5150.
  2. Each fixture on the network replies with a DISCOVER_RESPONSE containing
     its fixture_id in the first FixtureCommand payload byte.
  3. The host collects responses for a configurable window (default 2 s) and
     returns the set of responding fixture IDs together with their source
     (IP, port) tuples.
  4. After discovery the host sends periodic HEARTBEAT packets so fixtures
     can detect a lost controller and fall back to a safe state.

Transport notes:
  - All packets use the binary format defined in `lumina.control.protocol`.
  - The socket is bound to INADDR_ANY so responses arrive on the same port
    regardless of which network interface the fixture uses.
  - ``SO_BROADCAST`` is required on the send socket to reach the broadcast
    address.  We re-use a single socket for both sending and receiving.
"""

from __future__ import annotations

import asyncio
import logging
import socket
import time
from collections.abc import Callable
from typing import Any

from lumina.control.protocol import (
    PROTOCOL_PORT,
    FixtureCommand,
    PacketType,
    decode_packet,
    encode_packet,
)

_log = logging.getLogger(__name__)

# How long (seconds) to wait for fixture responses before giving up.
_DEFAULT_DISCOVER_TIMEOUT: float = 2.0

# Sequence counter shared across all DiscoveryService instances in a process.
_seq: int = 0


def _next_seq() -> int:
    """Return the next global sequence number (wraps at 65535)."""
    global _seq
    _seq = (_seq + 1) & 0xFFFF
    return _seq


def _timestamp_ms() -> int:
    """Return milliseconds-since-epoch mod 65536 for the packet timestamp."""
    return int(time.monotonic() * 1000) & 0xFFFF


# ─── asyncio DatagramProtocol ─────────────────────────────────────────────────


class _LuminaDatagramProtocol(asyncio.DatagramProtocol):
    """Low-level asyncio protocol that receives UDP datagrams and enqueues them.

    Args:
        queue: asyncio Queue into which (data, addr) tuples are placed.
    """

    def __init__(self, queue: asyncio.Queue[tuple[bytes, tuple[str, int]]]) -> None:
        self._queue = queue
        self.transport: asyncio.DatagramTransport | None = None

    def connection_made(self, transport: asyncio.BaseTransport) -> None:
        """Store the transport reference once the socket is ready."""
        self.transport = transport  # type: ignore[assignment]

    def datagram_received(self, data: bytes, addr: tuple[str, int]) -> None:
        """Enqueue an incoming datagram for processing by the listen loop."""
        try:
            self._queue.put_nowait((data, addr))
        except asyncio.QueueFull:
            _log.warning("Discovery receive queue full — dropping datagram from %s:%d", *addr)

    def error_received(self, exc: Exception) -> None:
        """Log transport-level errors (e.g. ICMP port-unreachable)."""
        _log.warning("UDP transport error: %s", exc)

    def connection_lost(self, exc: Exception | None) -> None:
        """Called when the underlying socket is closed."""
        if exc is not None:
            _log.debug("UDP connection lost: %s", exc)


# ─── DiscoveryService ─────────────────────────────────────────────────────────


class DiscoveryService:
    """mDNS-style fixture discovery and heartbeat service.

    Broadcasts DISCOVER_REQUEST packets and listens for DISCOVER_RESPONSE
    replies from fixtures.  Maintains an internal registry of known fixtures
    and invokes an optional callback whenever a new (or updated) fixture is
    seen.

    Args:
        port: UDP port to bind and send on (default 5150).
        broadcast_addr: IPv4 broadcast address (default ``"255.255.255.255"``).
        queue_size: Maximum number of datagrams buffered in the receive queue.

    Example::

        service = DiscoveryService()
        await service.start()
        fixture_ids = await service.discover()
        print(f"Found fixtures: {fixture_ids}")
        await service.stop()
    """

    def __init__(
        self,
        port: int = PROTOCOL_PORT,
        broadcast_addr: str = "255.255.255.255",
        queue_size: int = 256,
    ) -> None:
        self._port = port
        self._broadcast_addr = broadcast_addr
        self._queue_size = queue_size

        # Fixture registry: fixture_id → (ip, port)
        self._discovered: dict[int, tuple[str, int]] = {}

        # Optional user callback; set before calling start() or at any time.
        self.on_discovery_response: Callable[[int, tuple[str, int]], None] | None = None

        # Internal asyncio state — populated by start()
        self._transport: asyncio.DatagramTransport | None = None
        self._protocol: _LuminaDatagramProtocol | None = None
        self._recv_queue: asyncio.Queue[tuple[bytes, tuple[str, int]]] | None = None
        self._listen_task: asyncio.Task[None] | None = None
        self._running: bool = False

    # ── Public properties ─────────────────────────────────────────────────────

    @property
    def discovered_fixtures(self) -> dict[int, tuple[str, int]]:
        """Map of fixture_id → (ip, port) for all fixtures seen so far.

        Returns a shallow copy so callers cannot accidentally mutate internal
        state.
        """
        return dict(self._discovered)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Bind the UDP socket and start the background listener task.

        Safe to call from any asyncio context.  Raises ``RuntimeError`` if
        the service is already running.

        Raises:
            RuntimeError: If called when the service is already started.
            OSError: If the socket cannot be bound (e.g. port in use).
        """
        if self._running:
            raise RuntimeError("DiscoveryService is already running")

        loop = asyncio.get_running_loop()

        self._recv_queue = asyncio.Queue(maxsize=self._queue_size)

        # Create a raw UDP socket with broadcast enabled, then hand it to
        # asyncio so we get non-blocking I/O without managing select() manually.
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        # SO_REUSEPORT lets multiple processes bind the same port during dev.
        if hasattr(socket, "SO_REUSEPORT"):
            try:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
            except OSError:
                pass  # Not available on all platforms (e.g. Windows)
        sock.bind(("", self._port))
        sock.setblocking(False)

        transport, protocol = await loop.create_datagram_endpoint(
            lambda: _LuminaDatagramProtocol(self._recv_queue),  # type: ignore[arg-type]
            sock=sock,
        )
        self._transport = transport  # type: ignore[assignment]
        self._protocol = protocol  # type: ignore[assignment]

        self._running = True
        self._listen_task = loop.create_task(
            self._listen_loop(), name="lumina-discovery-listener"
        )
        _log.info(
            "DiscoveryService started on port %d (broadcast → %s)",
            self._port,
            self._broadcast_addr,
        )

    async def stop(self) -> None:
        """Cancel the listener task and close the UDP socket.

        Idempotent — safe to call even if the service was never started or
        has already been stopped.
        """
        self._running = False

        if self._listen_task is not None and not self._listen_task.done():
            self._listen_task.cancel()
            try:
                await self._listen_task
            except asyncio.CancelledError:
                pass
            self._listen_task = None

        if self._transport is not None:
            self._transport.close()
            self._transport = None

        _log.info("DiscoveryService stopped")

    # ── High-level operations ─────────────────────────────────────────────────

    async def discover(self, timeout: float = _DEFAULT_DISCOVER_TIMEOUT) -> list[int]:
        """Broadcast a DISCOVER_REQUEST and collect responding fixture IDs.

        Sends a single broadcast and then waits ``timeout`` seconds for
        DISCOVER_RESPONSE packets.  Any fixture that replies within that
        window is added to ``discovered_fixtures`` and returned.

        Args:
            timeout: Seconds to wait for responses (default 2.0).

        Returns:
            Sorted list of fixture IDs that responded to this discovery round.

        Raises:
            RuntimeError: If the service has not been started.
        """
        if not self._running:
            raise RuntimeError("DiscoveryService must be started before calling discover()")

        # Snapshot the fixture set before the broadcast so we can report *new*
        # responders from this specific round.
        ids_before = set(self._discovered.keys())

        self._send_packet(PacketType.DISCOVER_REQUEST, [])
        _log.debug("Sent DISCOVER_REQUEST broadcast, waiting %.1f s", timeout)

        await asyncio.sleep(timeout)

        new_ids = sorted(set(self._discovered.keys()) - ids_before)
        all_ids = sorted(self._discovered.keys())
        _log.info(
            "Discovery complete: %d fixture(s) total (%d new): %s",
            len(all_ids),
            len(new_ids),
            all_ids,
        )
        return all_ids

    async def send_heartbeat(self) -> None:
        """Broadcast a HEARTBEAT packet to all fixtures.

        Fixtures use heartbeats to detect a lost controller and can fall back
        to a safe (off or idle) state if heartbeats stop arriving.

        Raises:
            RuntimeError: If the service has not been started.
        """
        if not self._running:
            raise RuntimeError("DiscoveryService must be started before calling send_heartbeat()")

        self._send_packet(PacketType.HEARTBEAT, [])
        _log.debug("Sent HEARTBEAT broadcast")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _send_packet(self, packet_type: PacketType, commands: list[FixtureCommand]) -> None:
        """Encode and send a packet to the broadcast address.

        Args:
            packet_type: Type of packet to send.
            commands: Fixture commands to include in the payload (may be empty).
        """
        if self._transport is None:
            _log.error("Cannot send packet: transport is not open")
            return

        data = encode_packet(
            commands,
            sequence=_next_seq(),
            timestamp_ms=_timestamp_ms(),
            packet_type=packet_type,
        )
        self._transport.sendto(data, (self._broadcast_addr, self._port))

    def _handle_datagram(self, data: bytes, addr: tuple[str, int]) -> None:
        """Parse one incoming datagram and update internal state.

        Silently ignores malformed packets and packets whose type is not
        DISCOVER_RESPONSE.  Logs a warning for decode errors so operators can
        spot rogue senders.

        Args:
            data: Raw UDP payload bytes.
            addr: Source address as (ip_string, port_int).
        """
        try:
            packet_type, sequence, timestamp_ms, commands = decode_packet(data)
        except ValueError as exc:
            _log.warning("Ignoring malformed packet from %s:%d — %s", *addr, exc)
            return

        if packet_type is not PacketType.DISCOVER_RESPONSE:
            # We may receive our own broadcasts reflected back on some OSes,
            # or HEARTBEAT ACKs in the future.  Just ignore them.
            _log.debug(
                "Ignoring packet type 0x%02X from %s:%d (seq=%d)",
                int(packet_type),
                addr[0],
                addr[1],
                sequence,
            )
            return

        if not commands:
            _log.warning(
                "DISCOVER_RESPONSE from %s:%d carried no fixture command payload", *addr
            )
            return

        # The fixture encodes its fixture_id in the first (and usually only)
        # FixtureCommand payload block.
        fixture_id = commands[0].fixture_id
        if fixture_id == 0:
            _log.warning(
                "DISCOVER_RESPONSE from %s:%d reported fixture_id=0 (broadcast ID invalid)",
                *addr,
            )
            return

        is_new = fixture_id not in self._discovered
        self._discovered[fixture_id] = addr

        if is_new:
            _log.info(
                "New fixture discovered: id=%d at %s:%d (seq=%d, ts=%d ms)",
                fixture_id,
                addr[0],
                addr[1],
                sequence,
                timestamp_ms,
            )
        else:
            _log.debug(
                "Known fixture re-announced: id=%d at %s:%d",
                fixture_id,
                addr[0],
                addr[1],
            )

        if self.on_discovery_response is not None:
            try:
                self.on_discovery_response(fixture_id, addr)
            except Exception:
                _log.exception(
                    "on_discovery_response callback raised for fixture_id=%d", fixture_id
                )

    async def _listen_loop(self) -> None:
        """Background task: drain the receive queue and dispatch datagrams.

        Runs until the service is stopped or the task is cancelled.  Uses a
        short ``asyncio.wait_for`` timeout on each queue.get() so the loop
        remains cancellation-responsive even when no packets arrive.
        """
        _log.debug("Discovery listener loop started")
        assert self._recv_queue is not None  # guaranteed by start()

        while self._running:
            try:
                data, addr = await asyncio.wait_for(self._recv_queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue  # Nothing in queue — check self._running and loop again
            except asyncio.CancelledError:
                break

            self._handle_datagram(data, addr)

        _log.debug("Discovery listener loop exited")


# ─── One-shot helper ──────────────────────────────────────────────────────────


async def run_discovery(
    port: int = PROTOCOL_PORT,
    timeout: float = _DEFAULT_DISCOVER_TIMEOUT,
    on_found: Callable[[int, tuple[str, int]], None] | None = None,
) -> dict[int, tuple[str, int]]:
    """One-shot fixture discovery utility.

    Creates a temporary ``DiscoveryService``, runs a single discovery round,
    and tears everything down cleanly before returning.

    This is the simplest way to enumerate fixtures without managing the full
    service lifecycle:

    Example::

        fixtures = await run_discovery(timeout=3.0)
        for fid, (ip, port) in fixtures.items():
            print(f"Fixture {fid} → {ip}:{port}")

    Args:
        port: UDP port to use (default 5150).
        timeout: Seconds to wait for responses (default 2.0).
        on_found: Optional callback invoked for each responding fixture
            with ``(fixture_id, (ip, port))`` arguments.

    Returns:
        Dict mapping fixture_id → (ip, port) for all discovered fixtures.

    Raises:
        OSError: If the UDP socket cannot be bound (e.g. port in use).
    """
    service = DiscoveryService(port=port)
    if on_found is not None:
        service.on_discovery_response = on_found

    await service.start()
    try:
        await service.discover(timeout=timeout)
    finally:
        await service.stop()

    return service.discovered_fixtures
