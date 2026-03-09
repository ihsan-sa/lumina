"""Tests for the LUMINA WebSocket bridge server."""

from __future__ import annotations

import asyncio
import json

import pytest
from httpx import ASGITransport, AsyncClient
from starlette.testclient import TestClient

from lumina.audio.models import MusicState
from lumina.control.protocol import FixtureCommand
from lumina.web.server import LuminaServer


@pytest.fixture
def server() -> LuminaServer:
    """Create a fresh LuminaServer for each test."""
    return LuminaServer(host="127.0.0.1", port=0)


# ── HTTP health endpoint ────────────────────────────────────────


@pytest.mark.asyncio
async def test_health_endpoint(server: LuminaServer) -> None:
    """GET /health returns 200 with status and client count."""
    transport = ASGITransport(app=server.app)  # type: ignore[arg-type]
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["clients"] == 0


# ── WebSocket connection ────────────────────────────────────────


def test_websocket_connect(server: LuminaServer) -> None:
    """Client can connect to /ws and server tracks it."""
    client = TestClient(server.app)
    with client.websocket_connect("/ws") as ws:
        assert server.client_count == 1
        # Send a valid message to confirm connection works
        ws.send_json({"type": "transport", "action": "play"})
    # After disconnect
    assert server.client_count == 0


# ── Broadcast ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_broadcast_fixture_commands(server: LuminaServer) -> None:
    """Putting state in queue sends fixture_commands and music_state to client."""
    await server.start_broadcast()

    client = TestClient(server.app)
    with client.websocket_connect("/ws") as ws:
        state = MusicState(timestamp=1.5, bpm=140.0, energy=0.8, segment="drop")
        commands = [
            FixtureCommand(fixture_id=1, red=255, green=0, blue=0, white=128, special=200),
            FixtureCommand(fixture_id=5, strobe_rate=250, strobe_intensity=255),
        ]
        server.state_queue.put_nowait((state, commands))

        # Allow broadcast loop to process
        await asyncio.sleep(0.1)

        # Should receive fixture_commands first
        cmd_msg = ws.receive_json()
        assert cmd_msg["type"] == "fixture_commands"
        assert cmd_msg["sequence"] == 1
        assert len(cmd_msg["commands"]) == 2
        assert cmd_msg["commands"][0]["fixture_id"] == 1
        assert cmd_msg["commands"][0]["red"] == 255
        assert cmd_msg["commands"][1]["strobe_rate"] == 250

        # Then music_state
        state_msg = ws.receive_json()
        assert state_msg["type"] == "music_state"
        assert state_msg["state"]["bpm"] == 140.0
        assert state_msg["state"]["energy"] == 0.8
        assert state_msg["state"]["segment"] == "drop"

    await server.stop()


# ── Transport messages from client ──────────────────────────────


@pytest.mark.asyncio
async def test_transport_command_received(server: LuminaServer) -> None:
    """Client transport messages appear in transport_queue."""
    client = TestClient(server.app)
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "transport", "action": "seek", "position": 42.5})
        # Give the server a moment to process
        await asyncio.sleep(0.05)

    msg = server.transport_queue.get_nowait()
    assert msg["type"] == "transport"
    assert msg["action"] == "seek"
    assert msg["position"] == 42.5


# ── Multiple clients ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_multiple_clients_receive_broadcast(server: LuminaServer) -> None:
    """Two clients both receive the same broadcast."""
    await server.start_broadcast()

    client = TestClient(server.app)
    with client.websocket_connect("/ws") as ws1, client.websocket_connect("/ws") as ws2:
        assert server.client_count == 2

        state = MusicState(timestamp=0.0, bpm=120.0)
        commands = [FixtureCommand(fixture_id=1, red=100)]
        server.state_queue.put_nowait((state, commands))

        await asyncio.sleep(0.1)

        msg1 = ws1.receive_json()
        msg2 = ws2.receive_json()
        assert msg1["type"] == "fixture_commands"
        assert msg2["type"] == "fixture_commands"
        assert msg1["commands"] == msg2["commands"]

    await server.stop()


# ── Client disconnect handling ──────────────────────────────────


@pytest.mark.asyncio
async def test_client_disconnect_handled(server: LuminaServer) -> None:
    """Disconnecting a client doesn't crash the server or broadcast loop."""
    await server.start_broadcast()

    client = TestClient(server.app)

    # Connect and disconnect first client
    with client.websocket_connect("/ws"):
        assert server.client_count == 1
    assert server.client_count == 0

    # Server should still be able to broadcast (no crash)
    state = MusicState(timestamp=0.0)
    commands = [FixtureCommand(fixture_id=1)]
    server.state_queue.put_nowait((state, commands))
    await asyncio.sleep(0.05)

    # Connect a new client — should work fine
    with client.websocket_connect("/ws") as ws:
        assert server.client_count == 1
        server.state_queue.put_nowait((state, commands))
        await asyncio.sleep(0.1)
        msg = ws.receive_json()
        assert msg["type"] == "fixture_commands"

    await server.stop()


# ── Serialization round-trip ────────────────────────────────────


def test_music_state_serialization() -> None:
    """MusicState round-trips correctly through JSON serialization."""
    state = MusicState(
        timestamp=5.0,
        bpm=128.0,
        beat_phase=0.5,
        bar_phase=0.25,
        is_beat=True,
        is_downbeat=False,
        energy=0.75,
        energy_derivative=0.1,
        segment="chorus",
        genre_weights={"festival_edm": 0.7, "uk_bass": 0.3},
        vocal_energy=0.6,
        spectral_centroid=2000.0,
        sub_bass_energy=0.4,
        onset_type="kick",
        drop_probability=0.2,
    )
    import dataclasses

    serialized = json.dumps(dataclasses.asdict(state))
    deserialized = json.loads(serialized)

    assert deserialized["bpm"] == 128.0
    assert deserialized["segment"] == "chorus"
    assert deserialized["genre_weights"]["festival_edm"] == 0.7
    assert deserialized["onset_type"] == "kick"
    assert deserialized["is_beat"] is True
