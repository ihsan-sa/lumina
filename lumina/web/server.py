"""LUMINA WebSocket bridge — streams MusicState and FixtureCommands to browser clients.

Starlette ASGI app with:
- ``/ws`` — WebSocket endpoint for real-time fixture command + music state streaming
- ``/health`` — HTTP health check returning client count

The server is decoupled from the analysis pipeline via two asyncio queues:
- ``state_queue``: pipeline pushes ``(MusicState, list[FixtureCommand])``; broadcast
  loop sends to all connected clients.
- ``transport_queue``: clients send transport/config messages; the main app reads them.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from typing import Any

import numpy as np
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket, WebSocketDisconnect

from lumina.audio.models import MusicState
from lumina.control.protocol import FixtureCommand
from lumina.lighting.fixture_map import FixtureMap

logger = logging.getLogger(__name__)


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that converts numpy scalars to native Python types."""

    def default(self, o: object) -> Any:
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


class LuminaServer:
    """WebSocket bridge between the LUMINA analysis pipeline and browser clients.

    Args:
        host: Bind address.
        port: Bind port.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 8765) -> None:
        self._host = host
        self._port = port

        self._state_queue: asyncio.Queue[tuple[MusicState, list[FixtureCommand]]] = asyncio.Queue(
            maxsize=2
        )
        self._transport_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=64)

        self._clients: set[WebSocket] = set()
        self._broadcast_task: asyncio.Task[None] | None = None
        self._sequence = 0
        self._playback_info: dict[str, Any] | None = None
        self._audio_path: str | None = None
        self._current_timestamp: float = 0.0
        self._fixture_map = FixtureMap()

        self._app = Starlette(
            routes=[
                WebSocketRoute("/ws", self._ws_endpoint),
                Route("/health", self._health_endpoint),
                Route("/audio", self._serve_audio),
            ],
            middleware=[
                Middleware(
                    CORSMiddleware,
                    allow_origins=["*"],
                    allow_methods=["GET"],
                    allow_headers=["*"],
                ),
            ],
        )

    @property
    def state_queue(self) -> asyncio.Queue[tuple[MusicState, list[FixtureCommand]]]:
        """Queue for the pipeline to push (MusicState, commands) frames."""
        return self._state_queue

    @property
    def transport_queue(self) -> asyncio.Queue[dict[str, Any]]:
        """Queue where client transport/config messages arrive."""
        return self._transport_queue

    @property
    def app(self) -> Starlette:
        """The ASGI application (for uvicorn or testing)."""
        return self._app

    def set_audio_file(self, path: str) -> None:
        """Set the audio file path to serve via /audio endpoint.

        Args:
            path: Full path to the audio file.
        """
        self._audio_path = path

    def set_playback_info(self, filename: str, duration: float) -> None:
        """Set playback info to send to newly connected clients.

        Args:
            filename: Audio file name.
            duration: Track duration in seconds.
        """
        self._playback_info = {
            "type": "playback_start",
            "filename": filename,
            "duration": duration,
            "audio_url": "/audio",
            "start_timestamp": self._current_timestamp,
        }

    @property
    def client_count(self) -> int:
        """Number of connected WebSocket clients."""
        return len(self._clients)

    async def start(self) -> None:
        """Start the broadcast loop and uvicorn server.

        Launches uvicorn as a background task and waits until it is
        accepting connections before returning.  This allows the caller
        to ``await start()`` and then proceed with other work while the
        server keeps running.
        """
        import uvicorn

        self._broadcast_task = asyncio.create_task(self._broadcast_loop())

        config = uvicorn.Config(
            self._app,
            host=self._host,
            port=self._port,
            log_level="warning",
        )
        self._uvicorn_server = uvicorn.Server(config)
        self._serve_task = asyncio.create_task(self._uvicorn_server.serve())

        # Wait until uvicorn signals it has started
        while not self._uvicorn_server.started:
            await asyncio.sleep(0.05)

        logger.info("WebSocket server listening on %s:%d", self._host, self._port)

    async def start_broadcast(self) -> None:
        """Start only the broadcast loop (for testing without uvicorn)."""
        self._broadcast_task = asyncio.create_task(self._broadcast_loop())

    async def stop(self) -> None:
        """Shut down uvicorn, cancel the broadcast loop, and close all clients."""
        if hasattr(self, "_uvicorn_server"):
            self._uvicorn_server.should_exit = True
        if hasattr(self, "_serve_task"):
            with contextlib.suppress(asyncio.CancelledError):
                await self._serve_task
        if self._broadcast_task:
            self._broadcast_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._broadcast_task
            self._broadcast_task = None

    def _fixture_layout_msg(self) -> dict[str, Any]:
        """Generate a fixture_layout message from the current fixture map."""
        return {
            "type": "fixture_layout",
            "fixtures": [
                {
                    "fixture_id": f.fixture_id,
                    "fixture_type": f.fixture_type.value,
                    "position": list(f.position),
                    "role": f.role.value,
                    "name": f.name,
                }
                for f in self._fixture_map.all
            ],
        }

    # ── Endpoints ────────────────────────────────────────────────

    async def _ws_endpoint(self, websocket: WebSocket) -> None:
        """Handle a single WebSocket client connection."""
        await websocket.accept()
        self._clients.add(websocket)
        logger.info("Client connected (%d total)", len(self._clients))

        # Send fixture layout so the simulator knows fixture positions
        with contextlib.suppress(Exception):
            await websocket.send_text(json.dumps(self._fixture_layout_msg()))

        # Send playback info so the simulator can auto-start audio.
        # Inject the live timestamp so late-joining clients seek to the correct position.
        if self._playback_info is not None:
            with contextlib.suppress(Exception):
                msg = {**self._playback_info, "start_timestamp": self._current_timestamp}
                await websocket.send_text(json.dumps(msg))

        try:
            while True:
                text = await websocket.receive_text()
                try:
                    msg = json.loads(text)
                    if isinstance(msg, dict):
                        try:
                            self._transport_queue.put_nowait(msg)
                        except asyncio.QueueFull:
                            logger.warning("Transport queue full, dropping message")
                except json.JSONDecodeError:
                    logger.warning("Invalid JSON from client")
        except WebSocketDisconnect:
            pass
        finally:
            self._clients.discard(websocket)
            logger.info("Client disconnected (%d remaining)", len(self._clients))

    async def _health_endpoint(self, request: Request) -> JSONResponse:
        """HTTP health check."""
        return JSONResponse({"status": "ok", "clients": len(self._clients)})

    async def _serve_audio(self, request: Request) -> FileResponse | JSONResponse:
        """Serve the current audio file via HTTP for browser loading."""
        if self._audio_path is None:
            return JSONResponse({"error": "No audio loaded"}, status_code=404)
        import mimetypes
        mime_type = mimetypes.guess_type(self._audio_path)[0] or "application/octet-stream"
        return FileResponse(self._audio_path, media_type=mime_type)

    # ── Broadcast loop ───────────────────────────────────────────

    async def _broadcast_loop(self) -> None:
        """Read from state_queue and broadcast to all connected clients.

        Serialization uses manual dict construction instead of
        ``dataclasses.asdict()`` to avoid its per-field deep-copy overhead
        at 60fps.  When no clients are connected the frame is discarded
        without serialization.
        """
        while True:
            music_state, commands = await self._state_queue.get()

            # Track current playback position for late-joining clients
            self._current_timestamp = music_state.timestamp

            if not self._clients:
                continue  # nothing to send — skip serialization

            self._sequence = (self._sequence + 1) & 0xFFFF

            commands_msg = json.dumps(
                {
                    "type": "fixture_commands",
                    "sequence": self._sequence,
                    "timestamp_ms": int(music_state.timestamp * 1000) & 0xFFFF,
                    "commands": [
                        {
                            "fixture_id": c.fixture_id,
                            "red": c.red,
                            "green": c.green,
                            "blue": c.blue,
                            "white": c.white,
                            "strobe_rate": c.strobe_rate,
                            "strobe_intensity": c.strobe_intensity,
                            "special": c.special,
                        }
                        for c in commands
                    ],
                },
                cls=_NumpyEncoder,
            )

            state_msg = json.dumps(
                {
                    "type": "music_state",
                    "state": {
                        "timestamp": music_state.timestamp,
                        "bpm": music_state.bpm,
                        "beat_phase": music_state.beat_phase,
                        "bar_phase": music_state.bar_phase,
                        "is_beat": music_state.is_beat,
                        "is_downbeat": music_state.is_downbeat,
                        "energy": music_state.energy,
                        "energy_derivative": music_state.energy_derivative,
                        "segment": music_state.segment,
                        "genre_weights": dict(music_state.genre_weights),
                        "vocal_energy": music_state.vocal_energy,
                        "spectral_centroid": music_state.spectral_centroid,
                        "sub_bass_energy": music_state.sub_bass_energy,
                        "onset_type": music_state.onset_type,
                        "drop_probability": music_state.drop_probability,
                        "layer_count": music_state.layer_count,
                        "layer_mask": dict(music_state.layer_mask)
                        if music_state.layer_mask
                        else {},
                        "motif_id": music_state.motif_id,
                        "motif_repetition": music_state.motif_repetition,
                        "notes_per_beat": music_state.notes_per_beat,
                        "note_pattern_phase": music_state.note_pattern_phase,
                        "headroom": music_state.headroom,
                    },
                },
                cls=_NumpyEncoder,
            )

            dead: list[WebSocket] = []
            for client in list(self._clients):
                try:
                    await client.send_text(commands_msg)
                    await client.send_text(state_msg)
                except Exception:
                    dead.append(client)

            for client in dead:
                self._clients.discard(client)
                with contextlib.suppress(Exception):
                    await client.close()
                logger.info("Closed dead client (%d remaining)", len(self._clients))
