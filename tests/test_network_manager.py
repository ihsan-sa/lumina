"""Tests for NetworkStats and NetworkManager in lumina.control.network."""

from __future__ import annotations

import asyncio
import socket
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lumina.control.network import NetworkManager, NetworkStats
from lumina.control.protocol import (
    PROTOCOL_PORT,
    FixtureCommand,
    PacketType,
    decode_packet,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _cmd(fixture_id: int, red: int = 0, green: int = 0, blue: int = 0) -> FixtureCommand:
    """Build a minimal FixtureCommand."""
    return FixtureCommand(fixture_id=fixture_id, red=red, green=green, blue=blue)


def _make_manager(fps: int = 60, port: int = PROTOCOL_PORT) -> NetworkManager:
    """Construct a NetworkManager without starting it."""
    return NetworkManager(target_fps=fps, port=port)


# ─── NetworkStats ─────────────────────────────────────────────────────────────


class TestNetworkStatsDefaults:
    def test_frames_sent_zero(self) -> None:
        stats = NetworkStats()
        assert stats.frames_sent == 0

    def test_bytes_sent_zero(self) -> None:
        stats = NetworkStats()
        assert stats.bytes_sent == 0

    def test_send_errors_zero(self) -> None:
        stats = NetworkStats()
        assert stats.send_errors == 0

    def test_last_send_time_zero(self) -> None:
        stats = NetworkStats()
        assert stats.last_send_time == 0.0

    def test_avg_send_latency_ms_zero(self) -> None:
        stats = NetworkStats()
        assert stats.avg_send_latency_ms == 0.0

    def test_can_be_mutated(self) -> None:
        stats = NetworkStats()
        stats.frames_sent = 10
        stats.bytes_sent = 500
        assert stats.frames_sent == 10
        assert stats.bytes_sent == 500


# ─── NetworkManager construction ──────────────────────────────────────────────


class TestNetworkManagerConstruction:
    def test_default_construction(self) -> None:
        manager = NetworkManager()
        assert manager.is_running is False
        assert manager.fixture_count == 0

    def test_custom_fps_and_port(self) -> None:
        manager = NetworkManager(target_fps=30, port=9999)
        assert not manager.is_running

    def test_stats_starts_at_defaults(self) -> None:
        manager = _make_manager()
        stats = manager.stats
        assert stats.frames_sent == 0
        assert stats.bytes_sent == 0
        assert stats.send_errors == 0
        assert stats.last_send_time == 0.0
        assert stats.avg_send_latency_ms == 0.0

    def test_fixture_count_initially_zero(self) -> None:
        manager = _make_manager()
        assert manager.fixture_count == 0

    def test_is_running_initially_false(self) -> None:
        manager = _make_manager()
        assert manager.is_running is False


# ─── set_broadcast_target() ───────────────────────────────────────────────────


class TestSetBroadcastTarget:
    def test_stores_address_and_default_port(self) -> None:
        manager = _make_manager()
        manager.set_broadcast_target("192.168.1.255")
        # The broadcast address is stored internally; verify indirectly by
        # checking send_commands uses it (tested via send path).
        assert manager._broadcast_addr == ("192.168.1.255", PROTOCOL_PORT)

    def test_stores_custom_port(self) -> None:
        manager = _make_manager()
        manager.set_broadcast_target("255.255.255.255", port=9000)
        assert manager._broadcast_addr == ("255.255.255.255", 9000)

    def test_overwrites_previous_broadcast_target(self) -> None:
        manager = _make_manager()
        manager.set_broadcast_target("192.168.1.255")
        manager.set_broadcast_target("10.0.0.255", port=8888)
        assert manager._broadcast_addr == ("10.0.0.255", 8888)

    def test_can_be_set_before_start(self) -> None:
        manager = _make_manager()
        manager.set_broadcast_target("192.168.1.255")  # Should not raise.


# ─── add_fixture_target() / remove_fixture_target() ──────────────────────────


class TestFixtureTargets:
    def test_add_fixture_target_increments_count(self) -> None:
        manager = _make_manager()
        manager.add_fixture_target(1, "192.168.1.101")
        assert manager.fixture_count == 1

    def test_add_multiple_fixture_targets(self) -> None:
        manager = _make_manager()
        manager.add_fixture_target(1, "192.168.1.101")
        manager.add_fixture_target(2, "192.168.1.102")
        manager.add_fixture_target(3, "192.168.1.103")
        assert manager.fixture_count == 3

    def test_add_uses_default_port(self) -> None:
        manager = _make_manager()
        manager.add_fixture_target(1, "192.168.1.101")
        assert manager._targets[1] == ("192.168.1.101", PROTOCOL_PORT)

    def test_add_uses_custom_port(self) -> None:
        manager = _make_manager()
        manager.add_fixture_target(5, "192.168.1.105", port=7000)
        assert manager._targets[5] == ("192.168.1.105", 7000)

    def test_add_updates_existing_entry(self) -> None:
        manager = _make_manager()
        manager.add_fixture_target(1, "192.168.1.101")
        manager.add_fixture_target(1, "10.0.0.1")
        assert manager._targets[1] == ("10.0.0.1", PROTOCOL_PORT)
        assert manager.fixture_count == 1  # Still one entry.

    def test_add_rejects_fixture_id_zero(self) -> None:
        manager = _make_manager()
        with pytest.raises(ValueError, match="fixture_id must be 1-255"):
            manager.add_fixture_target(0, "192.168.1.1")

    def test_add_rejects_fixture_id_256(self) -> None:
        manager = _make_manager()
        with pytest.raises(ValueError, match="fixture_id must be 1-255"):
            manager.add_fixture_target(256, "192.168.1.1")

    def test_add_rejects_negative_fixture_id(self) -> None:
        manager = _make_manager()
        with pytest.raises(ValueError, match="fixture_id must be 1-255"):
            manager.add_fixture_target(-1, "192.168.1.1")

    def test_add_accepts_boundary_fixture_ids(self) -> None:
        manager = _make_manager()
        manager.add_fixture_target(1, "192.168.1.1")
        manager.add_fixture_target(255, "192.168.1.255")
        assert manager.fixture_count == 2

    def test_remove_fixture_target_decrements_count(self) -> None:
        manager = _make_manager()
        manager.add_fixture_target(1, "192.168.1.101")
        manager.add_fixture_target(2, "192.168.1.102")
        manager.remove_fixture_target(1)
        assert manager.fixture_count == 1

    def test_remove_fixture_target_unknown_id_is_no_op(self) -> None:
        manager = _make_manager()
        manager.remove_fixture_target(99)  # Must not raise.
        assert manager.fixture_count == 0

    def test_remove_fixture_target_removes_correct_entry(self) -> None:
        manager = _make_manager()
        manager.add_fixture_target(1, "192.168.1.101")
        manager.add_fixture_target(2, "192.168.1.102")
        manager.remove_fixture_target(1)
        assert 1 not in manager._targets
        assert 2 in manager._targets

    def test_remove_all_fixture_targets(self) -> None:
        manager = _make_manager()
        for i in range(1, 6):
            manager.add_fixture_target(i, f"192.168.1.{100 + i}")
        for i in range(1, 6):
            manager.remove_fixture_target(i)
        assert manager.fixture_count == 0


# ─── start() / stop() lifecycle ───────────────────────────────────────────────


class TestNetworkManagerLifecycle:
    async def test_start_sets_is_running(self) -> None:
        manager = _make_manager()
        with patch("lumina.control.network.socket.socket") as mock_socket_cls:
            mock_sock = MagicMock()
            mock_socket_cls.return_value = mock_sock
            await manager.start()
            assert manager.is_running is True
            await manager.stop()

    async def test_stop_clears_is_running(self) -> None:
        manager = _make_manager()
        with patch("lumina.control.network.socket.socket") as mock_socket_cls:
            mock_sock = MagicMock()
            mock_socket_cls.return_value = mock_sock
            await manager.start()
            await manager.stop()
            assert manager.is_running is False

    async def test_start_resets_sequence(self) -> None:
        manager = _make_manager()
        with patch("lumina.control.network.socket.socket") as mock_socket_cls:
            mock_sock = MagicMock()
            mock_socket_cls.return_value = mock_sock
            await manager.start()
            assert manager._sequence == 0
            await manager.stop()

    async def test_start_twice_raises_runtime_error(self) -> None:
        manager = _make_manager()
        with patch("lumina.control.network.socket.socket") as mock_socket_cls:
            mock_sock = MagicMock()
            mock_socket_cls.return_value = mock_sock
            await manager.start()
            with pytest.raises(RuntimeError, match="already running"):
                await manager.start()
            await manager.stop()

    async def test_stop_when_not_running_is_no_op(self) -> None:
        manager = _make_manager()
        await manager.stop()  # Must not raise.
        assert manager.is_running is False

    async def test_stop_multiple_times_is_safe(self) -> None:
        manager = _make_manager()
        with patch("lumina.control.network.socket.socket") as mock_socket_cls:
            mock_sock = MagicMock()
            mock_socket_cls.return_value = mock_sock
            await manager.start()
            await manager.stop()
            await manager.stop()  # Second stop must not raise.

    async def test_start_creates_udp_socket(self) -> None:
        manager = _make_manager()
        with patch("lumina.control.network.socket.socket") as mock_socket_cls:
            mock_sock = MagicMock()
            mock_socket_cls.return_value = mock_sock
            await manager.start()
            mock_socket_cls.assert_called_once_with(socket.AF_INET, socket.SOCK_DGRAM)
            await manager.stop()

    async def test_start_enables_broadcast(self) -> None:
        manager = _make_manager()
        with patch("lumina.control.network.socket.socket") as mock_socket_cls:
            mock_sock = MagicMock()
            mock_socket_cls.return_value = mock_sock
            await manager.start()
            mock_sock.setsockopt.assert_any_call(
                socket.SOL_SOCKET, socket.SO_BROADCAST, 1
            )
            await manager.stop()

    async def test_start_sets_socket_non_blocking(self) -> None:
        manager = _make_manager()
        with patch("lumina.control.network.socket.socket") as mock_socket_cls:
            mock_sock = MagicMock()
            mock_socket_cls.return_value = mock_sock
            await manager.start()
            mock_sock.setblocking.assert_called_once_with(False)
            await manager.stop()

    async def test_stop_closes_socket(self) -> None:
        manager = _make_manager()
        with patch("lumina.control.network.socket.socket") as mock_socket_cls:
            mock_sock = MagicMock()
            mock_socket_cls.return_value = mock_sock
            await manager.start()
            await manager.stop()
            mock_sock.close.assert_called_once()


# ─── send_commands() — sequence increment ─────────────────────────────────────


class TestSendCommandsSequence:
    async def test_sequence_increments_per_call(self) -> None:
        manager = _make_manager()
        manager.add_fixture_target(1, "192.168.1.101")

        with patch("lumina.control.network.socket.socket") as mock_socket_cls:
            mock_sock = MagicMock()
            mock_socket_cls.return_value = mock_sock

            loop = asyncio.get_running_loop()
            # sock_sendto must be an awaitable; patch at loop level.
            loop_sendto_mock = AsyncMock(return_value=None)

            with patch.object(loop, "sock_sendto", loop_sendto_mock):
                await manager.start()

                assert manager._sequence == 0
                await manager.send_commands([_cmd(1, red=10)])
                assert manager._sequence == 1
                await manager.send_commands([_cmd(1, red=20)])
                assert manager._sequence == 2

                await manager.stop()

    async def test_sequence_wraps_at_65535(self) -> None:
        manager = _make_manager()
        manager.add_fixture_target(1, "192.168.1.101")

        with patch("lumina.control.network.socket.socket") as mock_socket_cls:
            mock_sock = MagicMock()
            mock_socket_cls.return_value = mock_sock

            loop = asyncio.get_running_loop()
            loop_sendto_mock = AsyncMock(return_value=None)

            with patch.object(loop, "sock_sendto", loop_sendto_mock):
                await manager.start()
                manager._sequence = 0xFFFF  # Prime to wrap.
                await manager.send_commands([_cmd(1)])
                assert manager._sequence == 0

                await manager.stop()

    async def test_sequence_not_incremented_when_no_targets(self) -> None:
        manager = _make_manager()
        # No targets, no broadcast — send should be a no-op.

        with patch("lumina.control.network.socket.socket") as mock_socket_cls:
            mock_sock = MagicMock()
            mock_socket_cls.return_value = mock_sock

            await manager.start()
            assert manager._sequence == 0
            await manager.send_commands([_cmd(1, red=255)])
            assert manager._sequence == 0  # No increment — nothing sent.

            await manager.stop()


# ─── send_commands() — dispatch rules ─────────────────────────────────────────


class TestSendCommandsDispatch:
    async def test_no_targets_no_broadcast_is_dropped(self) -> None:
        manager = _make_manager()

        with patch("lumina.control.network.socket.socket") as mock_socket_cls:
            mock_sock = MagicMock()
            mock_socket_cls.return_value = mock_sock

            loop = asyncio.get_running_loop()
            sendto_mock = AsyncMock(return_value=None)
            with patch.object(loop, "sock_sendto", sendto_mock):
                await manager.start()
                await manager.send_commands([_cmd(1)])
                sendto_mock.assert_not_called()
                await manager.stop()

    async def test_empty_command_list_is_no_op(self) -> None:
        manager = _make_manager()
        manager.add_fixture_target(1, "192.168.1.101")

        with patch("lumina.control.network.socket.socket") as mock_socket_cls:
            mock_sock = MagicMock()
            mock_socket_cls.return_value = mock_sock

            loop = asyncio.get_running_loop()
            sendto_mock = AsyncMock(return_value=None)
            with patch.object(loop, "sock_sendto", sendto_mock):
                await manager.start()
                await manager.send_commands([])
                sendto_mock.assert_not_called()
                await manager.stop()

    async def test_unicast_send_when_targets_registered(self) -> None:
        manager = _make_manager()
        manager.add_fixture_target(1, "192.168.1.101")

        with patch("lumina.control.network.socket.socket") as mock_socket_cls:
            mock_sock = MagicMock()
            mock_socket_cls.return_value = mock_sock

            loop = asyncio.get_running_loop()
            sendto_mock = AsyncMock(return_value=None)
            with patch.object(loop, "sock_sendto", sendto_mock):
                await manager.start()
                await manager.send_commands([_cmd(1, red=255)])
                sendto_mock.assert_called_once()
                # Verify destination is the registered IP:port.
                # loop.sock_sendto is called as (sock, data, addr).
                _sock, _data, addr = sendto_mock.call_args[0]
                assert addr == ("192.168.1.101", PROTOCOL_PORT)
                await manager.stop()

    async def test_broadcast_send_when_no_unicast_targets(self) -> None:
        manager = _make_manager()
        manager.set_broadcast_target("192.168.1.255")

        with patch("lumina.control.network.socket.socket") as mock_socket_cls:
            mock_sock = MagicMock()
            mock_socket_cls.return_value = mock_sock

            loop = asyncio.get_running_loop()
            sendto_mock = AsyncMock(return_value=None)
            with patch.object(loop, "sock_sendto", sendto_mock):
                await manager.start()
                await manager.send_commands([_cmd(1, red=128)])
                sendto_mock.assert_called_once()
                _sock, _data, addr = sendto_mock.call_args[0]
                assert addr == ("192.168.1.255", PROTOCOL_PORT)
                await manager.stop()

    async def test_unicast_takes_priority_over_broadcast(self) -> None:
        """When both unicast targets and a broadcast target exist, unicast is used."""
        manager = _make_manager()
        manager.add_fixture_target(1, "192.168.1.101")
        manager.set_broadcast_target("192.168.1.255")

        with patch("lumina.control.network.socket.socket") as mock_socket_cls:
            mock_sock = MagicMock()
            mock_socket_cls.return_value = mock_sock

            loop = asyncio.get_running_loop()
            sendto_mock = AsyncMock(return_value=None)
            with patch.object(loop, "sock_sendto", sendto_mock):
                await manager.start()
                await manager.send_commands([_cmd(1, red=1)])
                # Must be sent to unicast address, not broadcast.
                _sock, _data, addr = sendto_mock.call_args[0]
                assert addr == ("192.168.1.101", PROTOCOL_PORT)
                await manager.stop()

    async def test_command_for_unregistered_fixture_is_dropped(self) -> None:
        """Commands for fixtures with no unicast target are silently dropped."""
        manager = _make_manager()
        manager.add_fixture_target(1, "192.168.1.101")

        with patch("lumina.control.network.socket.socket") as mock_socket_cls:
            mock_sock = MagicMock()
            mock_socket_cls.return_value = mock_sock

            loop = asyncio.get_running_loop()
            sendto_mock = AsyncMock(return_value=None)
            with patch.object(loop, "sock_sendto", sendto_mock):
                await manager.start()
                # Fixture 99 has no registered target — only fixture 1 gets a packet.
                await manager.send_commands([_cmd(1, red=10), _cmd(99, red=200)])
                sendto_mock.assert_called_once()
                await manager.stop()

    async def test_send_while_not_running_is_a_no_op(self) -> None:
        manager = _make_manager()
        manager.add_fixture_target(1, "192.168.1.101")
        # manager.start() is intentionally not called.
        await manager.send_commands([_cmd(1)])  # Must not raise.


# ─── send_commands() — packet content ─────────────────────────────────────────


class TestSendCommandsPacketContent:
    async def test_packet_contains_correct_fixture_data(self) -> None:
        manager = _make_manager()
        manager.add_fixture_target(3, "192.168.1.103")

        with patch("lumina.control.network.socket.socket") as mock_socket_cls:
            mock_sock = MagicMock()
            mock_socket_cls.return_value = mock_sock

            loop = asyncio.get_running_loop()
            captured: list[bytes] = []

            async def capture_sendto(sock: object, data: bytes, addr: object) -> None:
                captured.append(data)

            with patch.object(loop, "sock_sendto", capture_sendto):
                await manager.start()
                cmd = FixtureCommand(
                    fixture_id=3, red=10, green=20, blue=30,
                    white=40, strobe_rate=50, strobe_intensity=60, special=70,
                )
                await manager.send_commands([cmd])
                await manager.stop()

        assert len(captured) == 1
        ptype, seq, ts, cmds = decode_packet(captured[0])
        assert ptype == PacketType.COMMAND
        assert len(cmds) == 1
        decoded = cmds[0]
        assert decoded.fixture_id == 3
        assert decoded.red == 10
        assert decoded.green == 20
        assert decoded.blue == 30
        assert decoded.white == 40
        assert decoded.strobe_rate == 50
        assert decoded.strobe_intensity == 60
        assert decoded.special == 70

    async def test_timestamp_encoded_in_packet(self) -> None:
        manager = _make_manager()
        manager.add_fixture_target(1, "192.168.1.101")

        with patch("lumina.control.network.socket.socket") as mock_socket_cls:
            mock_sock = MagicMock()
            mock_socket_cls.return_value = mock_sock

            loop = asyncio.get_running_loop()
            captured: list[bytes] = []

            async def capture_sendto(sock: object, data: bytes, addr: object) -> None:
                captured.append(data)

            with patch.object(loop, "sock_sendto", capture_sendto):
                await manager.start()
                # timestamp=1.5s → 1500ms
                await manager.send_commands([_cmd(1)], timestamp=1.5)
                await manager.stop()

        _, _, ts_ms, _ = decode_packet(captured[0])
        assert ts_ms == 1500

    async def test_sequence_number_in_packet_matches_manager_state(self) -> None:
        manager = _make_manager()
        manager.add_fixture_target(1, "192.168.1.101")

        with patch("lumina.control.network.socket.socket") as mock_socket_cls:
            mock_sock = MagicMock()
            mock_socket_cls.return_value = mock_sock

            loop = asyncio.get_running_loop()
            captured: list[bytes] = []

            async def capture_sendto(sock: object, data: bytes, addr: object) -> None:
                captured.append(data)

            with patch.object(loop, "sock_sendto", capture_sendto):
                await manager.start()
                manager._sequence = 42  # Prime to a known value.
                await manager.send_commands([_cmd(1)])
                await manager.stop()

        _, seq, _, _ = decode_packet(captured[0])
        assert seq == 42

    async def test_multiple_fixtures_coalesced_to_one_packet(self) -> None:
        """Multiple fixtures at the same IP are coalesced into a single UDP datagram."""
        manager = _make_manager()
        # Two fixtures pointing at the same IP.
        manager.add_fixture_target(1, "192.168.1.101")
        manager.add_fixture_target(2, "192.168.1.101")

        with patch("lumina.control.network.socket.socket") as mock_socket_cls:
            mock_sock = MagicMock()
            mock_socket_cls.return_value = mock_sock

            loop = asyncio.get_running_loop()
            captured: list[bytes] = []

            async def capture_sendto(sock: object, data: bytes, addr: object) -> None:
                captured.append(data)

            with patch.object(loop, "sock_sendto", capture_sendto):
                await manager.start()
                await manager.send_commands([_cmd(1, red=10), _cmd(2, green=20)])
                await manager.stop()

        assert len(captured) == 1
        _, _, _, cmds = decode_packet(captured[0])
        assert len(cmds) == 2

    async def test_commands_split_into_chunks_of_32(self) -> None:
        """33 commands for the same destination must produce 2 UDP packets."""
        manager = _make_manager()
        # Route all 33 commands to one IP by registering all fixture IDs there.
        for fid in range(1, 34):
            manager.add_fixture_target(fid, "192.168.1.101")

        with patch("lumina.control.network.socket.socket") as mock_socket_cls:
            mock_sock = MagicMock()
            mock_socket_cls.return_value = mock_sock

            loop = asyncio.get_running_loop()
            captured: list[bytes] = []

            async def capture_sendto(sock: object, data: bytes, addr: object) -> None:
                captured.append(data)

            with patch.object(loop, "sock_sendto", capture_sendto):
                await manager.start()
                commands = [_cmd(fid) for fid in range(1, 34)]
                await manager.send_commands(commands)
                await manager.stop()

        # 33 fixtures → first chunk of 32, second chunk of 1 → 2 packets.
        assert len(captured) == 2
        _, _, _, cmds0 = decode_packet(captured[0])
        _, _, _, cmds1 = decode_packet(captured[1])
        assert len(cmds0) == 32
        assert len(cmds1) == 1


# ─── send_commands() — stats tracking ────────────────────────────────────────


class TestSendCommandsStats:
    async def test_frames_sent_incremented(self) -> None:
        manager = _make_manager()
        manager.add_fixture_target(1, "192.168.1.101")

        with patch("lumina.control.network.socket.socket") as mock_socket_cls:
            mock_sock = MagicMock()
            mock_socket_cls.return_value = mock_sock

            loop = asyncio.get_running_loop()
            with patch.object(loop, "sock_sendto", AsyncMock(return_value=None)):
                await manager.start()
                await manager.send_commands([_cmd(1)])
                assert manager.stats.frames_sent == 1
                await manager.send_commands([_cmd(1)])
                assert manager.stats.frames_sent == 2
                await manager.stop()

    async def test_bytes_sent_accumulated(self) -> None:
        manager = _make_manager()
        manager.add_fixture_target(1, "192.168.1.101")

        with patch("lumina.control.network.socket.socket") as mock_socket_cls:
            mock_sock = MagicMock()
            mock_socket_cls.return_value = mock_sock

            loop = asyncio.get_running_loop()
            with patch.object(loop, "sock_sendto", AsyncMock(return_value=None)):
                await manager.start()
                await manager.send_commands([_cmd(1)])
                # One header (9 bytes) + one fixture (8 bytes) = 17 bytes.
                assert manager.stats.bytes_sent == 17
                await manager.stop()

    async def test_last_send_time_updated(self) -> None:
        import time

        manager = _make_manager()
        manager.add_fixture_target(1, "192.168.1.101")

        with patch("lumina.control.network.socket.socket") as mock_socket_cls:
            mock_sock = MagicMock()
            mock_socket_cls.return_value = mock_sock

            loop = asyncio.get_running_loop()
            with patch.object(loop, "sock_sendto", AsyncMock(return_value=None)):
                await manager.start()
                assert manager.stats.last_send_time == 0.0
                before = time.monotonic()
                await manager.send_commands([_cmd(1)])
                after = time.monotonic()
                assert before <= manager.stats.last_send_time <= after
                await manager.stop()

    async def test_avg_send_latency_initialized_on_first_send(self) -> None:
        manager = _make_manager()
        manager.add_fixture_target(1, "192.168.1.101")

        with patch("lumina.control.network.socket.socket") as mock_socket_cls:
            mock_sock = MagicMock()
            mock_socket_cls.return_value = mock_sock

            loop = asyncio.get_running_loop()
            with patch.object(loop, "sock_sendto", AsyncMock(return_value=None)):
                await manager.start()
                await manager.send_commands([_cmd(1)])
                # After first send the EMA must be seeded (>= 0.0).
                assert manager.stats.avg_send_latency_ms >= 0.0
                await manager.stop()

    async def test_send_errors_not_incremented_on_success(self) -> None:
        manager = _make_manager()
        manager.add_fixture_target(1, "192.168.1.101")

        with patch("lumina.control.network.socket.socket") as mock_socket_cls:
            mock_sock = MagicMock()
            mock_socket_cls.return_value = mock_sock

            loop = asyncio.get_running_loop()
            with patch.object(loop, "sock_sendto", AsyncMock(return_value=None)):
                await manager.start()
                await manager.send_commands([_cmd(1)])
                assert manager.stats.send_errors == 0
                await manager.stop()

    async def test_send_errors_incremented_on_oserror(self) -> None:
        manager = _make_manager()
        manager.add_fixture_target(1, "192.168.1.101")

        with patch("lumina.control.network.socket.socket") as mock_socket_cls:
            mock_sock = MagicMock()
            mock_socket_cls.return_value = mock_sock

            loop = asyncio.get_running_loop()
            error_sendto = AsyncMock(side_effect=OSError("Network unreachable"))
            with patch.object(loop, "sock_sendto", error_sendto):
                await manager.start()
                await manager.send_commands([_cmd(1)])
                assert manager.stats.send_errors >= 1
                await manager.stop()

    async def test_frames_sent_not_incremented_when_no_targets(self) -> None:
        manager = _make_manager()
        # No targets, no broadcast.

        with patch("lumina.control.network.socket.socket") as mock_socket_cls:
            mock_sock = MagicMock()
            mock_socket_cls.return_value = mock_sock

            await manager.start()
            await manager.send_commands([_cmd(1)])
            assert manager.stats.frames_sent == 0
            await manager.stop()


# ─── stats property ───────────────────────────────────────────────────────────


class TestNetworkManagerStatsProperty:
    def test_stats_returns_network_stats_instance(self) -> None:
        manager = _make_manager()
        assert isinstance(manager.stats, NetworkStats)

    def test_stats_is_same_object_across_calls(self) -> None:
        manager = _make_manager()
        assert manager.stats is manager.stats

    def test_fixture_count_property(self) -> None:
        manager = _make_manager()
        manager.add_fixture_target(1, "192.168.1.101")
        manager.add_fixture_target(2, "192.168.1.102")
        assert manager.fixture_count == 2

    def test_is_running_property_reflects_state(self) -> None:
        manager = _make_manager()
        assert manager.is_running is False


# ─── send_raw() ───────────────────────────────────────────────────────────────


class TestSendRaw:
    async def test_send_raw_before_start_raises(self) -> None:
        manager = _make_manager()
        with pytest.raises(RuntimeError, match="call start\\(\\) first"):
            await manager.send_raw(b"\x00", ("192.168.1.1", 5150))

    async def test_send_raw_increments_bytes_sent(self) -> None:
        manager = _make_manager()

        with patch("lumina.control.network.socket.socket") as mock_socket_cls:
            mock_sock = MagicMock()
            mock_socket_cls.return_value = mock_sock

            loop = asyncio.get_running_loop()
            with patch.object(loop, "sock_sendto", AsyncMock(return_value=None)):
                await manager.start()
                payload = b"\xDE\xAD\xBE\xEF"
                await manager.send_raw(payload, ("192.168.1.1", 5150))
                assert manager.stats.bytes_sent == len(payload)
                await manager.stop()

    async def test_send_raw_on_oserror_increments_errors(self) -> None:
        manager = _make_manager()

        with patch("lumina.control.network.socket.socket") as mock_socket_cls:
            mock_sock = MagicMock()
            mock_socket_cls.return_value = mock_sock

            loop = asyncio.get_running_loop()
            error_sendto = AsyncMock(side_effect=OSError("send failed"))
            with patch.object(loop, "sock_sendto", error_sendto):
                await manager.start()
                await manager.send_raw(b"\x00", ("192.168.1.1", 5150))
                assert manager.stats.send_errors == 1
                await manager.stop()

    async def test_send_raw_does_not_touch_sequence_or_frames_sent(self) -> None:
        manager = _make_manager()

        with patch("lumina.control.network.socket.socket") as mock_socket_cls:
            mock_sock = MagicMock()
            mock_socket_cls.return_value = mock_sock

            loop = asyncio.get_running_loop()
            with patch.object(loop, "sock_sendto", AsyncMock(return_value=None)):
                await manager.start()
                await manager.send_raw(b"\xAB\xCD", ("192.168.1.1", 5150))
                assert manager._sequence == 0          # Unchanged.
                assert manager.stats.frames_sent == 0  # Unchanged.
                await manager.stop()
