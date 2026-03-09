"""LUMINA main application entry point.

Orchestrates the audio analysis pipeline, generates lighting commands,
and streams results to the WebSocket server for the 3D simulator.

Usage::

    python -m lumina.app --mode file --file path/to/song.mp3
    python -m lumina.app --mode file --file song.mp3 --udp-target 192.168.1.100:5150
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from lumina.audio.beat_detector import BeatDetector, BeatInfo
from lumina.audio.drop_predictor import DropFrame, DropPredictor
from lumina.audio.energy_tracker import EnergyFrame, EnergyTracker
from lumina.audio.genre_classifier import GenreClassifier, GenreFrame
from lumina.audio.models import MusicState
from lumina.audio.onset_detector import OnsetDetector, OnsetEvent
from lumina.audio.segment_classifier import SegmentClassifier, SegmentFrame
from lumina.audio.vocal_detector import VocalDetector, VocalFrame
from lumina.control.protocol import encode_packet
from lumina.lighting.engine import LightingEngine
from lumina.web.server import LuminaServer

logger = logging.getLogger(__name__)


@dataclass
class AppConfig:
    """Application configuration.

    Args:
        mode: Operating mode — "file" for offline analysis, "live" for real-time.
        file_path: Path to audio file (required for file mode).
        host: WebSocket server bind address.
        port: WebSocket server bind port.
        fps: Target output frame rate.
        sr: Sample rate for audio analysis.
        udp_target: Optional "IP:PORT" for physical fixture UDP output.
    """

    mode: str = "file"
    file_path: Path | None = None
    host: str = "0.0.0.0"
    port: int = 8765
    fps: int = 60
    sr: int = 44100
    udp_target: str | None = None


def _assemble_music_state(
    timestamp: float,
    beat: BeatInfo,
    energy: EnergyFrame,
    onset: OnsetEvent | None,
    vocal: VocalFrame,
    segment: SegmentFrame,
    genre: GenreFrame,
    drop: DropFrame,
) -> MusicState:
    """Assemble a MusicState from individual analyzer frame outputs.

    Args:
        timestamp: Current time in seconds.
        beat: Beat tracking frame.
        energy: Energy analysis frame.
        onset: Onset event (None if no onset this frame).
        vocal: Vocal detection frame.
        segment: Segment classification frame.
        genre: Genre classification frame.
        drop: Drop prediction frame.

    Returns:
        Complete MusicState for this time frame.
    """
    return MusicState(
        timestamp=timestamp,
        bpm=beat.bpm,
        beat_phase=beat.beat_phase,
        bar_phase=beat.bar_phase,
        is_beat=beat.is_beat,
        is_downbeat=beat.is_downbeat,
        energy=energy.energy,
        energy_derivative=energy.energy_derivative,
        segment=segment.segment,
        genre_weights=dict(genre.genre_weights),
        vocal_energy=vocal.vocal_energy,
        spectral_centroid=energy.spectral_centroid,
        sub_bass_energy=energy.sub_bass_energy,
        onset_type=onset.onset_type if onset is not None else None,
        drop_probability=drop.drop_probability,
    )


class LuminaApp:
    """Main LUMINA application.

    Args:
        config: Application configuration.
    """

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._server = LuminaServer(host=config.host, port=config.port)
        self._engine = LightingEngine()
        self._music_states: list[MusicState] = []
        self._frame_index = 0
        self._playing = True
        self._udp_socket: socket.socket | None = None
        self._udp_addr: tuple[str, int] | None = None

        if config.udp_target:
            host, port_str = config.udp_target.rsplit(":", 1)
            self._udp_addr = (host, int(port_str))
            self._udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    async def run(self) -> None:
        """Main entry point — run analysis and start output loop."""
        if self._config.mode == "file":
            await self._run_file_mode()
        else:
            logger.error("Live mode not yet implemented")

    async def _run_file_mode(self) -> None:
        """Offline analysis + playback at fps rate."""
        if not self._config.file_path:
            logger.error("No file path specified for file mode")
            return

        logger.info("Loading audio: %s", self._config.file_path)

        import librosa  # type: ignore[import-untyped]

        loop = asyncio.get_event_loop()
        audio, sr = await loop.run_in_executor(
            None,
            lambda: librosa.load(
                str(self._config.file_path),
                sr=self._config.sr,
                mono=True,
            ),
        )
        audio = np.asarray(audio, dtype=np.float32)
        duration = len(audio) / sr
        logger.info("Loaded %.1fs of audio at %dHz", duration, sr)

        fps = self._config.fps

        # Create analyzers
        beat_detector = BeatDetector(sr=sr, fps=fps)
        energy_tracker = EnergyTracker(sr=sr, fps=fps)
        onset_detector = OnsetDetector(sr=sr, fps=fps)
        vocal_detector = VocalDetector(sr=sr, fps=fps)
        segment_classifier = SegmentClassifier(fps=fps)
        genre_classifier = GenreClassifier(fps=fps)
        drop_predictor = DropPredictor(fps=fps)

        # Run analysis (beat_detector uses madmom — heavy, run in executor)
        logger.info("Analyzing audio...")
        beat_results = await loop.run_in_executor(None, beat_detector.analyze_offline, audio)
        energy_results = energy_tracker.analyze_offline(audio)
        onset_results = onset_detector.analyze_offline(audio)
        vocal_results = vocal_detector.analyze_offline(audio)

        # Align to shortest result set
        n = min(len(beat_results), len(energy_results), len(onset_results), len(vocal_results))
        if n == 0:
            logger.error("No frames produced by analyzers")
            return

        # Extract per-frame feature lists for derived classifiers
        energies = [energy_results[i].energy for i in range(n)]
        derivs = [energy_results[i].energy_derivative for i in range(n)]
        centroids = [energy_results[i].spectral_centroid for i in range(n)]
        basses = [energy_results[i].sub_bass_energy for i in range(n)]
        vocals = [vocal_results[i].vocal_energy for i in range(n)]
        onset_bools = [onset_results[i] is not None for i in range(n)]

        # Derived classifiers
        segment_classifier.set_track_duration(duration)
        drop_results = drop_predictor.process_features(
            energies, derivs, centroids, basses, vocals, onset_bools
        )
        segment_results = segment_classifier.classify_offline(
            energies, derivs, centroids, basses, vocals, onset_bools
        )

        drop_probs = [drop_results[i].drop_probability for i in range(n)]
        genre_results = genre_classifier.classify_offline(
            energies, centroids, basses, onset_bools, vocals, drop_probs
        )

        # Assemble per-frame MusicState
        frame_interval = 1.0 / fps
        self._music_states = [
            _assemble_music_state(
                timestamp=i * frame_interval,
                beat=beat_results[i],
                energy=energy_results[i],
                onset=onset_results[i],
                vocal=vocal_results[i],
                segment=segment_results[i],
                genre=genre_results[i],
                drop=drop_results[i],
            )
            for i in range(n)
        ]
        logger.info("Analysis complete: %d frames (%.1fs)", n, n * frame_interval)

        # Start WebSocket server (uvicorn + broadcast loop), then output loop
        await self._server.start()

        transport_task = asyncio.create_task(self._handle_transport())
        try:
            await self._output_loop()
        finally:
            transport_task.cancel()
            await self._server.stop()

    async def _output_loop(self) -> None:
        """Play back pre-computed MusicState list at fps rate."""
        frame_interval = 1.0 / self._config.fps
        n = len(self._music_states)
        seq = 0

        while self._frame_index < n:
            t_start = time.monotonic()

            if self._playing:
                state = self._music_states[self._frame_index]
                commands = self._engine.generate(state)

                # Push to WebSocket server (drop if full)
                with contextlib.suppress(asyncio.QueueFull):
                    self._server.state_queue.put_nowait((state, commands))

                # Optional UDP to physical fixtures
                if self._udp_socket and self._udp_addr:
                    seq = (seq + 1) & 0xFFFF
                    ts_ms = int(state.timestamp * 1000) & 0xFFFF
                    packet = encode_packet(commands, sequence=seq, timestamp_ms=ts_ms)
                    self._udp_socket.sendto(packet, self._udp_addr)

                self._frame_index += 1

            elapsed = time.monotonic() - t_start
            await asyncio.sleep(max(0.0, frame_interval - elapsed))

        logger.info("Playback complete")

    async def _handle_transport(self) -> None:
        """Process transport commands from WebSocket clients."""
        fps = self._config.fps
        n = len(self._music_states)

        while True:
            msg: dict[str, Any] = await self._server.transport_queue.get()
            action = msg.get("type")

            if action == "transport":
                cmd = msg.get("action")
                if cmd == "play":
                    self._playing = True
                    logger.info("Transport: play")
                elif cmd == "pause":
                    self._playing = False
                    logger.info("Transport: pause")
                elif cmd == "seek":
                    position = float(msg.get("position", 0.0))
                    self._frame_index = min(int(position * fps), n - 1)
                    self._frame_index = max(0, self._frame_index)
                    logger.info("Transport: seek to %.1fs (frame %d)", position, self._frame_index)


def parse_args(argv: list[str] | None = None) -> AppConfig:
    """Parse CLI arguments into AppConfig.

    Args:
        argv: Argument list (defaults to sys.argv).

    Returns:
        Parsed AppConfig.
    """
    parser = argparse.ArgumentParser(
        prog="lumina",
        description="LUMINA — AI-Powered Intelligent Light Show System",
    )
    parser.add_argument(
        "--mode",
        choices=["file", "live"],
        default="file",
        help="Operating mode (default: file)",
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=None,
        help="Path to audio file (required for file mode)",
    )
    parser.add_argument("--host", default="0.0.0.0", help="WebSocket server host")
    parser.add_argument("--port", type=int, default=8765, help="WebSocket server port")
    parser.add_argument("--fps", type=int, default=60, help="Output frame rate")
    parser.add_argument("--sr", type=int, default=44100, help="Audio sample rate")
    parser.add_argument(
        "--udp-target",
        default=None,
        help="UDP target for physical fixtures (IP:PORT)",
    )
    args = parser.parse_args(argv)

    return AppConfig(
        mode=args.mode,
        file_path=args.file,
        host=args.host,
        port=args.port,
        fps=args.fps,
        sr=args.sr,
        udp_target=args.udp_target,
    )


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    config = parse_args()
    app = LuminaApp(config)
    asyncio.run(app.run())


if __name__ == "__main__":
    main()
