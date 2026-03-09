"""Tests for the LUMINA WebSocket bridge server."""

from __future__ import annotations

import asyncio
import json

import numpy as np
import pytest
import websockets
from httpx import ASGITransport, AsyncClient
from starlette.testclient import TestClient

from lumina.audio.models import MusicState
from lumina.control.protocol import FixtureCommand
from lumina.web.server import LuminaServer, _NumpyEncoder


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


# ── Numpy type serialization ────────────────────────────────────


def test_numpy_encoder_handles_all_types() -> None:
    """_NumpyEncoder converts numpy scalars to native Python types."""
    data = {
        "float64": np.float64(0.75),
        "float32": np.float32(0.5),
        "int64": np.int64(42),
        "int32": np.int32(7),
        "bool_true": np.bool_(True),
        "bool_false": np.bool_(False),
        "array": np.array([1.0, 2.0, 3.0]),
    }
    serialized = json.dumps(data, cls=_NumpyEncoder)
    deserialized = json.loads(serialized)

    assert deserialized["float64"] == 0.75
    assert isinstance(deserialized["float64"], float)
    assert deserialized["int64"] == 42
    assert isinstance(deserialized["int64"], int)
    assert deserialized["bool_true"] is True
    assert deserialized["bool_false"] is False
    assert deserialized["array"] == [1.0, 2.0, 3.0]


def test_numpy_encoder_without_numpy_types() -> None:
    """_NumpyEncoder still works for plain Python types."""
    data = {"a": 1, "b": 2.5, "c": True, "d": "hello", "e": None}
    serialized = json.dumps(data, cls=_NumpyEncoder)
    assert json.loads(serialized) == data


@pytest.mark.asyncio
async def test_broadcast_with_numpy_music_state(server: LuminaServer) -> None:
    """MusicState containing numpy types serializes and broadcasts correctly."""
    await server.start_broadcast()

    client = TestClient(server.app)
    with client.websocket_connect("/ws") as ws:
        # Simulate what the real audio pipeline produces: numpy scalars
        state = MusicState(
            timestamp=np.float64(1.5),
            bpm=np.float64(140.0),
            beat_phase=np.float64(0.3),
            bar_phase=np.float64(0.1),
            is_beat=np.bool_(True),
            is_downbeat=np.bool_(False),
            energy=np.float64(0.8),
            energy_derivative=np.float32(-0.1),
            segment="drop",
            genre_weights={"rage_trap": np.float64(0.7), "uk_bass": np.float64(0.3)},
            vocal_energy=np.float64(0.2),
            spectral_centroid=np.float64(3500.0),
            sub_bass_energy=np.float64(0.6),
            onset_type="kick",
            drop_probability=np.float64(0.9),
        )
        commands = [FixtureCommand(fixture_id=1, red=200, special=180)]
        server.state_queue.put_nowait((state, commands))

        await asyncio.sleep(0.1)

        # Should not crash — receive fixture_commands
        cmd_msg = ws.receive_json()
        assert cmd_msg["type"] == "fixture_commands"

        # Then music_state with all numpy types converted
        state_msg = ws.receive_json()
        assert state_msg["type"] == "music_state"
        s = state_msg["state"]
        assert s["bpm"] == 140.0
        assert isinstance(s["bpm"], float)
        assert s["is_beat"] is True
        assert isinstance(s["is_beat"], bool)
        assert s["energy"] == 0.8
        assert s["genre_weights"]["rage_trap"] == 0.7
        assert isinstance(s["genre_weights"]["rage_trap"], float)

    await server.stop()


# ── Uvicorn server startup ──────────────────────────────────────


@pytest.mark.asyncio
async def test_start_binds_port_and_accepts_websocket() -> None:
    """server.start() launches uvicorn so real WebSocket clients can connect."""
    srv = LuminaServer(host="127.0.0.1", port=0)
    # Port 0 lets the OS pick a free port — but uvicorn doesn't expose it easily,
    # so we pick a specific high port for the test.
    srv = LuminaServer(host="127.0.0.1", port=18765)
    await srv.start()

    try:
        async with websockets.connect("ws://127.0.0.1:18765/ws") as ws:
            assert srv.client_count == 1

            # Push a frame through
            state = MusicState(timestamp=0.0, bpm=120.0, energy=0.5)
            commands = [FixtureCommand(fixture_id=1, red=128)]
            srv.state_queue.put_nowait((state, commands))

            # Receive fixture_commands
            raw = await asyncio.wait_for(ws.recv(), timeout=2.0)
            msg = json.loads(raw)
            assert msg["type"] == "fixture_commands"
            assert msg["commands"][0]["red"] == 128

            # Receive music_state
            raw = await asyncio.wait_for(ws.recv(), timeout=2.0)
            msg = json.loads(raw)
            assert msg["type"] == "music_state"
            assert msg["state"]["bpm"] == 120.0
    finally:
        await srv.stop()
