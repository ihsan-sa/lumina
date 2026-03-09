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
import dataclasses
import json
import logging
from typing import Any

import numpy as np
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket, WebSocketDisconnect

from lumina.audio.models import MusicState
from lumina.control.protocol import FixtureCommand

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

        self._app = Starlette(
            routes=[
                WebSocketRoute("/ws", self._ws_endpoint),
                Route("/health", self._health_endpoint),
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

    # ── Endpoints ────────────────────────────────────────────────

    async def _ws_endpoint(self, websocket: WebSocket) -> None:
        """Handle a single WebSocket client connection."""
        await websocket.accept()
        self._clients.add(websocket)
        logger.info("Client connected (%d total)", len(self._clients))

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

    # ── Broadcast loop ───────────────────────────────────────────

    async def _broadcast_loop(self) -> None:
        """Read from state_queue and broadcast to all connected clients."""
        while True:
            music_state, commands = await self._state_queue.get()
            self._sequence = (self._sequence + 1) & 0xFFFF

            commands_msg = json.dumps(
                {
                    "type": "fixture_commands",
                    "sequence": self._sequence,
                    "timestamp_ms": int(music_state.timestamp * 1000) & 0xFFFF,
                    "commands": [dataclasses.asdict(cmd) for cmd in commands],
                },
                cls=_NumpyEncoder,
            )

            state_msg = json.dumps(
                {
                    "type": "music_state",
                    "state": dataclasses.asdict(music_state),
                },
                cls=_NumpyEncoder,
            )

            dead: list[WebSocket] = []
            for client in self._clients:
                try:
                    await client.send_text(commands_msg)
                    await client.send_text(state_msg)
                except Exception:
                    dead.append(client)

            for client in dead:
                self._clients.discard(client)
