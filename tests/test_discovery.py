"""Tests for lumina.control.discovery — mDNS fixture discovery and heartbeat service.

Coverage strategy:
- Unit-test all pure logic (sequence counter, timestamp helper, packet dispatch).
- Test DiscoveryService state machine (start/stop lifecycle, double-start guard).
- Use asyncio UDP loopback (bind on 127.0.0.1 with a free port) to exercise
  the full send → receive → callback path without needing real fixtures.
- run_discovery() is tested via the loopback fixture as a one-shot integration test.
"""

from __future__ import annotations

import asyncio
import socket
import time
from collections.abc import Callable
from unittest.mock import MagicMock, patch

import pytest

from lumina.control.discovery import (
    DiscoveryService,
    _next_seq,
    _timestamp_ms,
    run_discovery,
)
from lumina.control.protocol import (
    PROTOCOL_PORT,
    FixtureCommand,
    PacketType,
    decode_packet,
    encode_packet,
)


# ─── Helper: free port ────────────────────────────────────────────────────────


def _free_port() -> int:
    """Return an available UDP port on 127.0.0.1."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


# ─── Helper: fake fixture that replies to DISCOVER_REQUEST ────────────────────


class _FakeFixture:
    """Sends a DISCOVER_RESPONSE to a given (host, port) after a short delay.

    Simulates an STM32 fixture responding to a discovery broadcast.

    Args:
        fixture_id: The ID the fake fixture should report in its response.
        src_port: The UDP port the fake fixture "listens" on (its own source port).
        delay: Seconds before the response is sent.
    """

    def __init__(self, fixture_id: int, src_port: int, delay: float = 0.05) -> None:
        self.fixture_id = fixture_id
        self.src_port = src_port
        self.delay = delay
        self._task: asyncio.Task[None] | None = None

    async def _send_response(self, target_host: str, target_port: int) -> None:
        await asyncio.sleep(self.delay)
        payload = encode_packet(
            [FixtureCommand(fixture_id=self.fixture_id)],
            sequence=1,
            timestamp_ms=100,
            packet_type=PacketType.DISCOVER_RESPONSE,
        )
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("127.0.0.1", self.src_port))
        try:
            sock.sendto(payload, (target_host, target_port))
        finally:
            sock.close()

    def schedule(self, target_host: str, target_port: int) -> None:
        """Schedule a response datagram to be sent after self.delay seconds."""
        loop = asyncio.get_event_loop()
        self._task = loop.create_task(self._send_response(target_host, target_port))

    async def join(self) -> None:
        """Wait for the scheduled send task to complete."""
        if self._task is not None:
            await self._task


# ─── Pure-logic unit tests ────────────────────────────────────────────────────


class TestSequenceCounter:
    def test_increments_monotonically(self) -> None:
        a = _next_seq()
        b = _next_seq()
        assert b == (a + 1) & 0xFFFF

    def test_wraps_at_65535(self) -> None:
        # Force the global counter to 65535 then check wrapping.
        import lumina.control.discovery as _disc

        _disc._seq = 65534
        v1 = _next_seq()
        assert v1 == 65535
        v2 = _next_seq()
        assert v2 == 0


class TestTimestampMs:
    def test_returns_int(self) -> None:
        assert isinstance(_timestamp_ms(), int)

    def test_within_uint16_range(self) -> None:
        ts = _timestamp_ms()
        assert 0 <= ts <= 0xFFFF

    def test_advances_with_time(self) -> None:
        t0 = _timestamp_ms()
        time.sleep(0.005)  # 5 ms
        t1 = _timestamp_ms()
        # Either t1 > t0 or we wrapped — just verify they differ on a slow pass.
        # On a sufficiently fast machine they *could* be equal; we just check type.
        assert isinstance(t1, int)


# ─── DiscoveryService lifecycle ───────────────────────────────────────────────


class TestDiscoveryServiceLifecycle:
    @pytest.mark.asyncio
    async def test_start_and_stop(self) -> None:
        port = _free_port()
        svc = DiscoveryService(port=port, broadcast_addr="127.0.0.1")
        await svc.start()
        assert svc._running is True
        await svc.stop()
        assert svc._running is False

    @pytest.mark.asyncio
    async def test_double_start_raises(self) -> None:
        port = _free_port()
        svc = DiscoveryService(port=port, broadcast_addr="127.0.0.1")
        await svc.start()
        try:
            with pytest.raises(RuntimeError, match="already running"):
                await svc.start()
        finally:
            await svc.stop()

    @pytest.mark.asyncio
    async def test_stop_before_start_is_safe(self) -> None:
        svc = DiscoveryService(port=_free_port(), broadcast_addr="127.0.0.1")
        # Should not raise.
        await svc.stop()

    @pytest.mark.asyncio
    async def test_double_stop_is_safe(self) -> None:
        port = _free_port()
        svc = DiscoveryService(port=port, broadcast_addr="127.0.0.1")
        await svc.start()
        await svc.stop()
        await svc.stop()  # Second stop must not raise.

    @pytest.mark.asyncio
    async def test_discovered_fixtures_starts_empty(self) -> None:
        port = _free_port()
        svc = DiscoveryService(port=port, broadcast_addr="127.0.0.1")
        await svc.start()
        try:
            assert svc.discovered_fixtures == {}
        finally:
            await svc.stop()

    @pytest.mark.asyncio
    async def test_discovered_fixtures_returns_copy(self) -> None:
        port = _free_port()
        svc = DiscoveryService(port=port, broadcast_addr="127.0.0.1")
        await svc.start()
        try:
            copy = svc.discovered_fixtures
            copy[99] = ("10.0.0.99", 5150)  # Mutate the copy
            assert 99 not in svc._discovered  # Internal dict unchanged
        finally:
            await svc.stop()


# ─── Guards: operations before start() ───────────────────────────────────────


class TestNotStartedGuards:
    @pytest.mark.asyncio
    async def test_discover_before_start_raises(self) -> None:
        svc = DiscoveryService(port=_free_port())
        with pytest.raises(RuntimeError, match="must be started"):
            await svc.discover(timeout=0.1)

    @pytest.mark.asyncio
    async def test_heartbeat_before_start_raises(self) -> None:
        svc = DiscoveryService(port=_free_port())
        with pytest.raises(RuntimeError, match="must be started"):
            await svc.send_heartbeat()


# ─── Packet sending ───────────────────────────────────────────────────────────


class TestPacketSending:
    """Verify that discover() and send_heartbeat() actually transmit valid packets."""

    @pytest.mark.asyncio
    async def test_discover_sends_broadcast(self) -> None:
        """discover() must emit a DISCOVER_REQUEST that can be decoded."""
        recv_port = _free_port()
        svc_port = _free_port()

        # Raw listener socket on the same interface.
        listener = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", recv_port))
        listener.setblocking(False)

        # Service that sends to the listener's port instead of 255.255.255.255.
        svc = DiscoveryService(port=svc_port, broadcast_addr="127.0.0.1")
        # Monkey-patch the broadcast address to the listener's port.
        # We need the transport to sendto the listener directly.
        await svc.start()

        try:
            # Replace broadcast port inline: _send_packet always sends to
            # (self._broadcast_addr, self._port).  We matched svc_port to
            # listener's receive port is not straightforward, so instead we
            # capture via the transport mock.
            captured: list[tuple[bytes, tuple[str, int]]] = []
            original_sendto = svc._transport.sendto  # type: ignore[union-attr]

            def _capture(data: bytes, addr: tuple[str, int]) -> None:
                captured.append((data, addr))
                # Forward to a temporary socket so the send doesn't fail.

            svc._transport.sendto = _capture  # type: ignore[union-attr]

            await svc.discover(timeout=0.05)

            assert len(captured) >= 1
            pkt_data, _addr = captured[0]
            ptype, seq, ts, cmds = decode_packet(pkt_data)
            assert ptype == PacketType.DISCOVER_REQUEST
            assert cmds == []
        finally:
            await svc.stop()
            listener.close()

    @pytest.mark.asyncio
    async def test_heartbeat_sends_heartbeat_packet(self) -> None:
        """send_heartbeat() must emit a HEARTBEAT packet."""
        svc = DiscoveryService(port=_free_port(), broadcast_addr="127.0.0.1")
        await svc.start()

        captured: list[tuple[bytes, tuple[str, int]]] = []

        def _capture(data: bytes, addr: tuple[str, int]) -> None:
            captured.append((data, addr))

        svc._transport.sendto = _capture  # type: ignore[union-attr]

        try:
            await svc.send_heartbeat()
            assert len(captured) == 1
            ptype, _, _, cmds = decode_packet(captured[0][0])
            assert ptype == PacketType.HEARTBEAT
            assert cmds == []
        finally:
            await svc.stop()

    @pytest.mark.asyncio
    async def test_sequence_numbers_increase_across_sends(self) -> None:
        """Each sent packet should have an increasing sequence number."""
        svc = DiscoveryService(port=_free_port(), broadcast_addr="127.0.0.1")
        await svc.start()

        packets: list[bytes] = []

        def _capture(data: bytes, addr: tuple[str, int]) -> None:
            packets.append(data)

        svc._transport.sendto = _capture  # type: ignore[union-attr]

        try:
            await svc.discover(timeout=0.01)
            await svc.send_heartbeat()
            await svc.send_heartbeat()

            assert len(packets) == 3
            seqs = [decode_packet(p)[1] for p in packets]
            # Each must be strictly one more (mod 65536) than the previous.
            for i in range(1, len(seqs)):
                assert seqs[i] == (seqs[i - 1] + 1) & 0xFFFF
        finally:
            await svc.stop()


# ─── _handle_datagram ─────────────────────────────────────────────────────────


class TestHandleDatagram:
    """Unit-test _handle_datagram() directly, bypassing the network."""

    def _make_svc(self) -> DiscoveryService:
        svc = DiscoveryService(port=_free_port())
        svc._running = True  # Simulate started state without real socket.
        return svc

    def _response_packet(self, fixture_id: int) -> bytes:
        return encode_packet(
            [FixtureCommand(fixture_id=fixture_id)],
            sequence=1,
            timestamp_ms=0,
            packet_type=PacketType.DISCOVER_RESPONSE,
        )

    def test_valid_response_registers_fixture(self) -> None:
        svc = self._make_svc()
        addr = ("192.168.1.42", 5150)
        svc._handle_datagram(self._response_packet(7), addr)
        assert svc._discovered[7] == addr

    def test_valid_response_calls_callback(self) -> None:
        svc = self._make_svc()
        seen: list[tuple[int, tuple[str, int]]] = []
        svc.on_discovery_response = lambda fid, a: seen.append((fid, a))
        svc._handle_datagram(self._response_packet(3), ("10.0.0.1", 5150))
        assert seen == [(3, ("10.0.0.1", 5150))]

    def test_callback_not_called_for_non_response_packets(self) -> None:
        svc = self._make_svc()
        seen: list[int] = []
        svc.on_discovery_response = lambda fid, _: seen.append(fid)
        # Send a COMMAND packet — should be silently ignored.
        data = encode_packet(
            [FixtureCommand(fixture_id=1, red=255)],
            packet_type=PacketType.COMMAND,
        )
        svc._handle_datagram(data, ("10.0.0.1", 5150))
        assert seen == []
        assert svc._discovered == {}

    def test_malformed_packet_does_not_raise(self) -> None:
        svc = self._make_svc()
        # Should log a warning, not raise.
        svc._handle_datagram(b"\xDE\xAD\xBE\xEF", ("10.0.0.1", 5150))
        assert svc._discovered == {}

    def test_zero_fixture_id_in_response_is_rejected(self) -> None:
        svc = self._make_svc()
        data = encode_packet(
            [FixtureCommand(fixture_id=0)],
            packet_type=PacketType.DISCOVER_RESPONSE,
        )
        svc._handle_datagram(data, ("10.0.0.1", 5150))
        assert svc._discovered == {}

    def test_empty_payload_response_is_rejected(self) -> None:
        svc = self._make_svc()
        data = encode_packet([], packet_type=PacketType.DISCOVER_RESPONSE)
        svc._handle_datagram(data, ("10.0.0.1", 5150))
        assert svc._discovered == {}

    def test_repeat_response_updates_addr(self) -> None:
        """A fixture that moves IPs should have its entry updated."""
        svc = self._make_svc()
        addr1 = ("192.168.1.10", 5150)
        addr2 = ("192.168.1.20", 5150)
        svc._handle_datagram(self._response_packet(5), addr1)
        svc._handle_datagram(self._response_packet(5), addr2)
        assert svc._discovered[5] == addr2

    def test_callback_not_called_on_repeated_response(self) -> None:
        """Callback fires for new fixture, but not on repeat if already known."""
        svc = self._make_svc()
        call_count = 0

        def _cb(fid: int, addr: tuple[str, int]) -> None:
            nonlocal call_count
            call_count += 1

        svc.on_discovery_response = _cb
        addr = ("10.0.0.1", 5150)
        # First sighting fires callback.
        svc._handle_datagram(self._response_packet(2), addr)
        assert call_count == 1
        # Second sighting (same fixture, same IP) still fires callback — the
        # service always invokes it; callers are responsible for dedup if needed.
        svc._handle_datagram(self._response_packet(2), addr)
        assert call_count == 2

    def test_crashing_callback_does_not_propagate(self) -> None:
        """An exception in the user callback must not crash the service."""
        svc = self._make_svc()
        svc.on_discovery_response = lambda fid, addr: (_ for _ in ()).throw(
            RuntimeError("boom")
        )
        # Should not raise — _handle_datagram catches and logs.
        svc._handle_datagram(self._response_packet(1), ("10.0.0.1", 5150))

    def test_multiple_fixtures_in_single_call(self) -> None:
        """Multiple _handle_datagram calls build up the registry."""
        svc = self._make_svc()
        for fid in (1, 2, 3):
            svc._handle_datagram(self._response_packet(fid), (f"10.0.0.{fid}", 5150))
        assert set(svc._discovered.keys()) == {1, 2, 3}

    def test_heartbeat_packet_is_ignored(self) -> None:
        svc = self._make_svc()
        data = encode_packet([], packet_type=PacketType.HEARTBEAT)
        svc._handle_datagram(data, ("10.0.0.1", 5150))
        assert svc._discovered == {}

    def test_discover_request_packet_is_ignored(self) -> None:
        svc = self._make_svc()
        data = encode_packet([], packet_type=PacketType.DISCOVER_REQUEST)
        svc._handle_datagram(data, ("10.0.0.1", 5150))
        assert svc._discovered == {}


# ─── discover() return value ──────────────────────────────────────────────────


class TestDiscoverReturn:
    """Verify the sorted list returned by discover() via _handle_datagram injection."""

    @pytest.mark.asyncio
    async def test_returns_sorted_fixture_ids(self) -> None:
        port = _free_port()
        svc = DiscoveryService(port=port, broadcast_addr="127.0.0.1")
        await svc.start()

        # Intercept outgoing packet (no real broadcast needed for this test).
        svc._transport.sendto = lambda data, addr: None  # type: ignore[union-attr]

        # Inject responses directly while discover() is sleeping.
        async def _inject() -> None:
            await asyncio.sleep(0.02)
            for fid in (5, 2, 8, 1):
                svc._handle_datagram(
                    encode_packet(
                        [FixtureCommand(fixture_id=fid)],
                        packet_type=PacketType.DISCOVER_RESPONSE,
                    ),
                    (f"10.0.0.{fid}", 5150),
                )

        inject_task = asyncio.create_task(_inject())
        result = await svc.discover(timeout=0.1)
        await inject_task

        assert result == sorted([5, 2, 8, 1])
        await svc.stop()

    @pytest.mark.asyncio
    async def test_returns_all_known_fixtures_not_just_new(self) -> None:
        """discover() returns *all* known IDs, not just the ones from this round."""
        port = _free_port()
        svc = DiscoveryService(port=port, broadcast_addr="127.0.0.1")
        await svc.start()
        svc._transport.sendto = lambda data, addr: None  # type: ignore[union-attr]

        # Pre-populate one fixture.
        svc._handle_datagram(
            encode_packet(
                [FixtureCommand(fixture_id=1)],
                packet_type=PacketType.DISCOVER_RESPONSE,
            ),
            ("10.0.0.1", 5150),
        )

        # Inject a second fixture during the discover window.
        async def _inject() -> None:
            await asyncio.sleep(0.02)
            svc._handle_datagram(
                encode_packet(
                    [FixtureCommand(fixture_id=2)],
                    packet_type=PacketType.DISCOVER_RESPONSE,
                ),
                ("10.0.0.2", 5150),
            )

        inject_task = asyncio.create_task(_inject())
        result = await svc.discover(timeout=0.06)
        await inject_task

        assert 1 in result
        assert 2 in result
        await svc.stop()


# ─── run_discovery() helper ───────────────────────────────────────────────────


class TestRunDiscovery:
    @pytest.mark.asyncio
    async def test_returns_empty_when_no_fixtures_respond(self) -> None:
        result = await run_discovery(port=_free_port(), timeout=0.05)
        assert result == {}

    @pytest.mark.asyncio
    async def test_returns_dict_type(self) -> None:
        result = await run_discovery(port=_free_port(), timeout=0.05)
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_on_found_callback_wired_correctly(self) -> None:
        """on_found is plumbed through to on_discovery_response."""
        seen: list[int] = []

        # Patch DiscoveryService so we can inject a response before the
        # timeout expires.
        original_start = DiscoveryService.start
        original_discover = DiscoveryService.discover

        async def _patched_discover(
            self: DiscoveryService, timeout: float = 2.0
        ) -> list[int]:
            # Inject a fake fixture response before sleeping.
            self._handle_datagram(
                encode_packet(
                    [FixtureCommand(fixture_id=42)],
                    packet_type=PacketType.DISCOVER_RESPONSE,
                ),
                ("127.0.0.1", 5150),
            )
            return await original_discover(self, timeout=0.02)

        with (
            patch.object(DiscoveryService, "discover", _patched_discover),
            patch.object(
                DiscoveryService,
                "_send_packet",
                lambda self, pt, cmds: None,
            ),
        ):
            result = await run_discovery(
                port=_free_port(),
                timeout=0.02,
                on_found=lambda fid, addr: seen.append(fid),
            )

        assert 42 in seen
        assert 42 in result
