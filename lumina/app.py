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
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from lumina.analysis.arc_planner import ArcPlanner
from lumina.analysis.layer_tracker import LayerTracker
from lumina.analysis.motif_detector import MotifDetector
from lumina.analysis.song_score import ScoreFrame, SongScore
from lumina.audio.beat_detector import BeatDetector, BeatInfo
from lumina.audio.drop_predictor import DropFrame, DropPredictor
from lumina.audio.energy_tracker import EnergyFrame, EnergyTracker
from lumina.audio.genre_classifier import GenreClassifier, GenreFrame
from lumina.audio.models import MusicState
from lumina.audio.onset_detector import OnsetDetector, OnsetEvent
from lumina.audio.segment_classifier import SegmentClassifier, SegmentFrame
from lumina.audio.source_separator import SourceSeparator
from lumina.audio.structural_analyzer import StructuralAnalyzer
from lumina.audio.vocal_detector import VocalDetector, VocalFrame
from lumina.control.network import NetworkManager
from lumina.control.protocol import FixtureCommand
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
    debug: bool = False
    genre_override: str | None = None


def _assemble_music_state(
    timestamp: float,
    beat: BeatInfo,
    energy: EnergyFrame,
    onset: OnsetEvent | None,
    vocal: VocalFrame,
    segment: SegmentFrame,
    genre: GenreFrame,
    drop: DropFrame,
    score: ScoreFrame | None = None,
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
        score: Song score frame with layer/motif/arc data (None in live mode).

    Returns:
        Complete MusicState for this time frame.
    """
    state = MusicState(
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

    if score is not None:
        state.layer_count = score.layer_count
        state.layer_mask = dict(score.layer_mask)
        state.motif_id = score.motif_id
        state.motif_repetition = score.motif_repetition
        state.notes_per_beat = score.notes_per_beat
        state.note_pattern_phase = score.note_pattern_phase
        state.headroom = score.headroom

    return state


class LuminaApp:
    """Main LUMINA application.

    Args:
        config: Application configuration.
    """

    # Maximum number of audio reconnection attempts before giving up.
    _MAX_AUDIO_RETRIES: int = 5
    # Seconds to wait between audio reconnection attempts.
    _AUDIO_RETRY_DELAY: float = 2.0

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._server = LuminaServer(host=config.host, port=config.port)
        self._engine = LightingEngine()
        self._music_states: list[MusicState] = []
        self._frame_index = 0
        self._playing = True
        self._network: NetworkManager | None = None
        self._shutdown_event: asyncio.Event = asyncio.Event()
        self._global_intensity: float = 0.8
        self._manual_effect: str | None = None
        self._manual_effect_start: float = 0.0
        self._manual_effect_duration: float = 0.5

        if config.udp_target:
            host, port_str = config.udp_target.rsplit(":", 1)
            self._network = NetworkManager(
                target_fps=config.fps, port=int(port_str),
            )
            self._network.set_broadcast_target(host, int(port_str))

    async def run(self) -> None:
        """Main entry point — run analysis and start output loop.

        Installs signal handlers for SIGINT/SIGTERM to trigger graceful
        shutdown across all operating modes.
        """
        loop = asyncio.get_running_loop()

        def _signal_handler() -> None:
            logger.info("Shutdown signal received — initiating graceful shutdown")
            self._shutdown_event.set()

        for sig in (signal.SIGINT, signal.SIGTERM):
            with contextlib.suppress(NotImplementedError):
                loop.add_signal_handler(sig, _signal_handler)

        # Start NetworkManager if configured
        if self._network is not None:
            await self._network.start()

        try:
            if self._config.mode == "file":
                await self._run_file_mode()
            elif self._config.mode == "showcase":
                await self._run_showcase_mode()
            elif self._config.mode == "live":
                await self._run_live_mode()
        finally:
            await self._shutdown()

    async def _shutdown(self) -> None:
        """Graceful shutdown: stop audio, flush network, close servers.

        Cleanup sequence:
        1. Signal shutdown event (stops loops)
        2. Flush pending commands via NetworkManager
        3. Stop NetworkManager (closes UDP socket)
        4. Stop WebSocket server
        """
        logger.info("Running graceful shutdown sequence...")
        self._shutdown_event.set()

        if self._network is not None:
            await self._network.stop()
            logger.info("NetworkManager stopped")

        await self._server.stop()
        logger.info("WebSocket server stopped")
        logger.info("Shutdown complete")

    async def _run_showcase_mode(self) -> None:
        """Start server immediately for pattern showcase — no audio analysis."""
        await self._server.start()
        transport_task = asyncio.create_task(self._handle_transport())
        try:
            await self._idle_loop()
        finally:
            transport_task.cancel()

    async def _run_live_mode(self) -> None:
        """Live audio capture and real-time analysis (Mode A).

        Captures system audio via sounddevice loopback, runs streaming
        analyzers on each audio chunk, generates lighting commands per
        frame, and pushes them to the WebSocket server in real time.

        Includes error recovery: if the audio device is lost, the system
        retries up to ``_MAX_AUDIO_RETRIES`` times with a
        ``_AUDIO_RETRY_DELAY`` second delay between attempts.

        Latency target: ~100-200ms (capture block + analysis).
        """
        import sounddevice as sd

        sr = self._config.sr
        fps = self._config.fps
        block_size = sr // fps  # One block per frame
        frame_interval = 1.0 / fps

        # Apply genre override at engine level if configured
        if self._config.genre_override:
            self._engine.set_genre_override(self._config.genre_override)

        # Start WebSocket server
        await self._server.start()
        transport_task = asyncio.create_task(self._handle_transport())

        try:
            await self._live_capture_loop(sd, sr, fps, block_size, frame_interval)
        finally:
            transport_task.cancel()

    async def _live_capture_loop(
        self,
        sd: Any,
        sr: int,
        fps: int,
        block_size: int,
        frame_interval: float,
    ) -> None:
        """Run the live audio capture loop with automatic reconnection.

        Args:
            sd: The ``sounddevice`` module (imported lazily by caller).
            sr: Sample rate in Hz.
            fps: Target frames per second.
            block_size: Audio samples per frame.
            frame_interval: Seconds per frame (1/fps).
        """
        retries = 0

        while retries <= self._MAX_AUDIO_RETRIES and not self._shutdown_event.is_set():
            # Create fresh streaming analyzers on each connection attempt
            beat_detector = BeatDetector(sr=sr, fps=fps)
            energy_tracker = EnergyTracker(sr=sr, fps=fps)
            onset_detector = OnsetDetector(sr=sr, fps=fps)
            vocal_detector = VocalDetector(sr=sr, fps=fps)
            genre_classifier = GenreClassifier(fps=fps)
            drop_predictor = DropPredictor(fps=fps)
            segment_classifier = SegmentClassifier(fps=fps)

            audio_queue: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=16)
            loop = asyncio.get_running_loop()

            def _make_audio_callback(
                ev_loop: asyncio.AbstractEventLoop,
                queue: asyncio.Queue[np.ndarray],
            ) -> Any:
                def _cb(
                    indata: np.ndarray,
                    frames: int,
                    time_info: object,
                    status: Any,
                ) -> None:
                    try:
                        if status:
                            logger.warning("Audio callback status: %s", status)
                        mono = (
                            indata[:, 0].copy()
                            if indata.ndim > 1
                            else indata.copy().flatten()
                        )
                        with contextlib.suppress(asyncio.QueueFull):
                            ev_loop.call_soon_threadsafe(
                                queue.put_nowait, mono,
                            )
                    except Exception:
                        logger.exception("Error in audio callback")
                return _cb

            audio_callback = _make_audio_callback(loop, audio_queue)

            stream = None
            try:
                logger.info(
                    "Starting live audio capture (sr=%d, block=%d, attempt=%d)",
                    sr, block_size, retries + 1,
                )
                stream = sd.InputStream(
                    samplerate=sr,
                    blocksize=block_size,
                    channels=1,
                    dtype="float32",
                    callback=audio_callback,
                )
                stream.start()
                logger.info("Live audio stream started")

                # Reset retry counter on successful start
                retries = 0
                timestamp = 0.0

                await self._live_process_frames(
                    audio_queue, sr, fps, frame_interval, timestamp,
                    beat_detector, energy_tracker, onset_detector,
                    vocal_detector, genre_classifier, drop_predictor,
                    segment_classifier,
                )
                # If _live_process_frames returns normally, shutdown was requested
                break

            except Exception:
                retries += 1
                logger.exception(
                    "Live audio error (attempt %d/%d)",
                    retries, self._MAX_AUDIO_RETRIES,
                )
                if retries > self._MAX_AUDIO_RETRIES:
                    logger.error(
                        "Max audio reconnection attempts (%d) exceeded — "
                        "giving up",
                        self._MAX_AUDIO_RETRIES,
                    )
                    break
                logger.info(
                    "Retrying audio capture in %.1fs...",
                    self._AUDIO_RETRY_DELAY,
                )
                await asyncio.sleep(self._AUDIO_RETRY_DELAY)
            finally:
                if stream is not None:
                    try:
                        stream.stop()
                        stream.close()
                    except Exception:
                        logger.exception("Error closing audio stream")

    async def _live_process_frames(
        self,
        audio_queue: asyncio.Queue[np.ndarray],
        sr: int,
        fps: int,
        frame_interval: float,
        timestamp: float,
        beat_detector: BeatDetector,
        energy_tracker: EnergyTracker,
        onset_detector: OnsetDetector,
        vocal_detector: VocalDetector,
        genre_classifier: GenreClassifier,
        drop_predictor: DropPredictor,
        segment_classifier: SegmentClassifier,
    ) -> None:
        """Process audio frames from the queue until shutdown.

        Args:
            audio_queue: Queue receiving audio chunks from the callback.
            sr: Sample rate.
            fps: Target frames per second.
            frame_interval: Seconds per frame.
            timestamp: Starting timestamp.
            beat_detector: Beat detection analyzer.
            energy_tracker: Energy analysis analyzer.
            onset_detector: Onset detection analyzer.
            vocal_detector: Vocal detection analyzer.
            genre_classifier: Genre classification analyzer.
            drop_predictor: Drop prediction analyzer.
            segment_classifier: Segment classification analyzer.
        """
        while not self._shutdown_event.is_set():
            # Use wait_for so we can check shutdown periodically
            try:
                chunk = await asyncio.wait_for(
                    audio_queue.get(), timeout=1.0,
                )
            except TimeoutError:
                continue

            t0 = time.monotonic()
            timestamp += len(chunk) / sr

            # Run streaming analyzers
            beat_frames = beat_detector.process_chunk(chunk)
            energy_frames = energy_tracker.process_chunk(chunk)
            onset_frames = onset_detector.process_chunk(chunk)
            vocal_frames = vocal_detector.process_chunk(chunk)

            # Align frame counts (take minimum)
            n = min(
                len(beat_frames),
                len(energy_frames),
                len(onset_frames),
                len(vocal_frames),
            )
            if n == 0:
                continue

            for i in range(n):
                beat = beat_frames[i]
                energy = energy_frames[i]
                onset = onset_frames[i]
                vocal = vocal_frames[i]
                has_onset = onset is not None

                # Frame-level classifiers
                drop = drop_predictor.process_frame(
                    energy.energy,
                    energy.energy_derivative,
                    energy.spectral_centroid,
                    energy.sub_bass_energy,
                    vocal.vocal_energy,
                    has_onset,
                )
                genre = genre_classifier.process_frame(
                    energy.energy,
                    energy.spectral_centroid,
                    energy.sub_bass_energy,
                    has_onset,
                    vocal.vocal_energy,
                    drop.drop_probability,
                )
                segment = segment_classifier.process_frame(
                    energy.energy,
                    energy.energy_derivative,
                    energy.spectral_centroid,
                    energy.sub_bass_energy,
                    vocal.vocal_energy,
                    has_onset,
                )

                frame_time = timestamp - (n - i - 1) * frame_interval
                state = _assemble_music_state(
                    timestamp=frame_time,
                    beat=beat,
                    energy=energy,
                    onset=onset,
                    vocal=vocal,
                    segment=segment,
                    genre=genre,
                    drop=drop,
                )

                commands = self._engine.generate(state)
                commands = self._apply_effects(commands, state)

                with contextlib.suppress(asyncio.QueueFull):
                    self._server.state_queue.put_nowait((state, commands))

                # Optional UDP output via NetworkManager
                if self._network is not None:
                    await self._network.send_commands(
                        commands, timestamp=frame_time,
                    )

            # Debug logging (once per second)
            if self._config.debug:
                gen_ms = (time.monotonic() - t0) * 1000
                if int(timestamp) != int(timestamp - len(chunk) / sr):
                    logger.info(
                        "LIVE t=%.1f n=%d gen=%.1fms qsize=%d",
                        timestamp, n, gen_ms, audio_queue.qsize(),
                    )

    async def _run_file_mode(self) -> None:
        """Offline analysis + playback at fps rate.

        Pipeline:
        1. Load audio
        2. Source separation (demucs)
        3. Beat/onset on drums stem, vocals on vocal stem
        4. Energy from full mix + bass stem for sub-bass
        5. Structural analysis (replaces frame-by-frame segment classifier)
        6. Genre classification (locked for entire track)
        """
        if not self._config.file_path:
            logger.error("No file path specified for file mode")
            return

        logger.info("Loading audio: %s", self._config.file_path)

        import librosa  # type: ignore[import-untyped]

        loop = asyncio.get_running_loop()
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

        # Source separation (demucs — GPU heavy, run in executor)
        logger.info("Running source separation (demucs)...")
        separator = SourceSeparator()
        stems = await loop.run_in_executor(None, separator.separate, audio, sr)
        logger.info("Source separation complete")

        # Create analyzers
        beat_detector = BeatDetector(sr=sr, fps=fps)
        energy_tracker = EnergyTracker(sr=sr, fps=fps)
        onset_detector = OnsetDetector(sr=sr, fps=fps)
        vocal_detector = VocalDetector(sr=sr, fps=fps)
        genre_classifier = GenreClassifier(fps=fps)
        drop_predictor = DropPredictor(fps=fps)
        structural_analyzer = StructuralAnalyzer(sr=sr, fps=fps)

        # Run analysis with stem routing:
        # - BeatDetector, OnsetDetector → drums stem
        # - VocalDetector → vocals stem
        # - EnergyTracker → full mix + bass stem
        logger.info("Analyzing audio...")
        beat_results = await loop.run_in_executor(
            None, beat_detector.analyze_offline, stems.drums
        )
        onset_results = onset_detector.analyze_offline(stems.drums)
        vocal_results = vocal_detector.analyze_offline(stems.vocals)
        energy_results = energy_tracker.analyze_offline_with_bass_stem(audio, stems.bass)

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

        # Drop predictor
        drop_results = drop_predictor.process_features(
            energies, derivs, centroids, basses, vocals, onset_bools
        )

        # Genre classification — runs before structural analysis so family
        # can be passed to the EDM structural pass
        drop_probs = [drop_results[i].drop_probability for i in range(n)]
        genre_results = genre_classifier.classify_file(
            energies, centroids, basses, onset_bools, vocals, drop_probs,
            stems=stems,
            genre_override=self._config.genre_override,
        )

        # Determine top genre family and profile for structural analysis
        first_genre = genre_results[0]
        genre_family = first_genre.family
        genre_profile = max(
            first_genre.genre_weights, key=first_genre.genre_weights.__getitem__
        )

        # Structural analysis (replaces SegmentClassifier in file mode)
        # Passes genre_family so electronic tracks use the EDM structural pass
        logger.info("Running structural analysis (genre_family=%s)...", genre_family)
        structural_map = structural_analyzer.analyze(
            audio, stems, beat_results, energy_results, onset_results, vocal_results,
            genre_family=genre_family,
            genre_profile=genre_profile,
            drop_results=drop_results,
        )
        segment_results = structural_analyzer.map_to_frames(structural_map, n, fps)

        # Log section distribution when debug is on
        if self._config.debug:
            dist: dict[str, int] = {}
            for sec in structural_map.sections:
                dist[sec.segment_type] = dist.get(sec.segment_type, 0) + 1
            parts = ", ".join(
                f"{count} {stype}" for stype, count in sorted(dist.items())
            )
            logger.info("Sections: %s", parts)

        # ─── New analysis: layers, motifs, arc, song score ───────────
        logger.info("Running layer/motif/arc analysis...")

        # Layer tracker: count active stems per frame
        layer_tracker = LayerTracker(sr=sr)
        raw_layer_frames = await loop.run_in_executor(
            None, layer_tracker.analyze, stems,
        )
        layer_frames = layer_tracker.resample_to_fps(raw_layer_frames, n, fps)

        # Motif detector: bar-level repetition + note-level patterns
        motif_detector = MotifDetector(sr=sr, fps=fps)
        motif_timeline = await loop.run_in_executor(
            None, motif_detector.detect_macro_motifs, audio, beat_results,
        )
        note_patterns = await loop.run_in_executor(
            None, motif_detector.detect_micro_patterns, stems.other, beat_results,
        )

        # Arc planner: headroom budgeting
        arc_planner = ArcPlanner(fps=fps)
        arc_frames = arc_planner.plan(energy_results, layer_frames, structural_map)

        # Song score: aggregate into per-frame ScoreFrames
        # Get pattern preferences from the active profile
        active_profile = self._engine.get_profile(genre_profile)
        pattern_prefs = (
            active_profile.motif_pattern_preferences
            if active_profile and hasattr(active_profile, "motif_pattern_preferences")
            else None
        )
        song_score = SongScore(fps=fps)
        score_frames = song_score.build(
            layer_frames, note_patterns, arc_frames, motif_timeline,
            n_frames=n, pattern_preferences=pattern_prefs,
        )

        # Pass motif assignments to the lighting engine
        self._engine.set_motif_assignments(song_score.motif_assignments)

        logger.info("Layer/motif/arc analysis complete")

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
                score=score_frames[i] if i < len(score_frames) else None,
            )
            for i in range(n)
        ]
        logger.info("Analysis complete: %d frames (%.1fs)", n, n * frame_interval)

        # Start WebSocket server (uvicorn + broadcast loop), then output loop
        await self._server.start()

        # Tell connecting clients about the audio file for auto-play
        if self._config.file_path:
            self._server.set_audio_file(str(self._config.file_path))
            self._server.set_playback_info(
                filename=self._config.file_path.name,
                duration=duration,
            )

        transport_task = asyncio.create_task(self._handle_transport())
        try:
            await self._output_loop()
            await self._idle_loop()
        finally:
            transport_task.cancel()

    async def _output_loop(self) -> None:
        """Play back pre-computed MusicState list at fps rate.

        Uses absolute time tracking: each frame has a target wall-clock
        time (base + frame_index * interval).  If a frame runs late the
        next sleep is shortened to catch up, keeping average playback at
        1.0x real-time.
        """
        frame_interval = 1.0 / self._config.fps
        n = len(self._music_states)

        # Absolute time reference — frame 0 plays at base_time
        base_time = time.monotonic()

        while self._frame_index < n and not self._shutdown_event.is_set():
            if not self._playing:
                await asyncio.sleep(0.05)
                # Re-anchor so we don't fast-forward after unpause
                base_time = time.monotonic() - self._frame_index * frame_interval
                continue

            t0 = time.monotonic()
            state = self._music_states[self._frame_index]
            commands = self._engine.generate(state)
            commands = self._apply_effects(commands, state)
            t_gen = time.monotonic()

            # Debug: timing + MusicState once per second
            if self._config.debug and self._frame_index % self._config.fps == 0:
                gen_ms = (t_gen - t0) * 1000
                wall_elapsed = t_gen - base_time
                ratio = state.timestamp / wall_elapsed if wall_elapsed > 0.1 else 0.0
                top_genre = max(
                    state.genre_weights,
                    key=state.genre_weights.get,
                    default="?",  # type: ignore[arg-type]
                )
                logger.info(
                    "DBG t=%.1f seg=%-10s e=%.2f de=%+.2f onset=%-5s "
                    "drop=%.2f genre=%s | gen=%.1fms speed=%.2fx",
                    state.timestamp,
                    state.segment,
                    state.energy,
                    state.energy_derivative,
                    state.onset_type or "\u2014",
                    state.drop_probability,
                    top_genre,
                    gen_ms,
                    ratio,
                )

                # LIT line: lighting engine debug summary
                di = self._engine.last_debug_info
                patterns = di.get("patterns", [])
                pattern_str = "+".join(patterns) if patterns else "—"
                type_counts: dict[str, tuple[int, int]] = di.get("type_counts", {})  # type: ignore[assignment]
                pars = type_counts.get("par", (0, 0))
                strobes = type_counts.get("strobe", (0, 0))
                bars = type_counts.get("led_bar", (0, 0))
                laser = type_counts.get("laser", (0, 0))
                laser_str = "ON" if laser[0] > 0 else "OFF"
                colors: list[str] = di.get("colors", [])  # type: ignore[assignment]
                color_str = "[" + ",".join(colors) + "]" if colors else "[]"
                logger.info(
                    "LIT t=%.1f profile=%s seg=%s pattern=%s "
                    "active=%d/%d pars=%d/%d strobes=%d/%d bars=%d/%d laser=%s "
                    "colors=%s strobe_rate=%d",
                    state.timestamp,
                    di.get("profile", "?"),
                    di.get("segment", "?"),
                    pattern_str,
                    di.get("active", 0),
                    di.get("total", 0),
                    pars[0], pars[1],
                    strobes[0], strobes[1],
                    bars[0], bars[1],
                    laser_str,
                    color_str,
                    di.get("strobe_rate_max", 0),
                )

            # Push to WebSocket server (drop if full)
            with contextlib.suppress(asyncio.QueueFull):
                self._server.state_queue.put_nowait((state, commands))

            # Optional UDP to physical fixtures via NetworkManager
            if self._network is not None:
                await self._network.send_commands(
                    commands, timestamp=state.timestamp,
                )

            self._frame_index += 1

            # Sleep until this frame's absolute target time
            target = base_time + self._frame_index * frame_interval
            sleep_for = target - time.monotonic()
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)
            # else: behind schedule — skip sleep, catch up immediately

        logger.info("Playback complete")

    async def _idle_loop(self) -> None:
        """Keep the server alive after playback for pattern showcase and transport commands.

        Generates synthetic frames at fps rate so that pattern_override commands
        sent from the simulator continue to produce visible output even after the
        song has finished.  Uses absolute wall-clock timing to avoid drift.
        """
        logger.info("Entering idle mode — server still accepting connections")
        frame_interval = 1.0 / self._config.fps
        base_offset = self._music_states[-1].timestamp if self._music_states else 0.0
        frame_count = 0

        while not self._shutdown_event.is_set():
            frame_count += 1
            timestamp = base_offset + frame_count * frame_interval

            # Create a new MusicState each frame to avoid race with broadcast loop
            idle_state = MusicState(
                timestamp=timestamp,
                bpm=120.0,
                beat_phase=0.0,
                bar_phase=0.0,
                is_beat=False,
                is_downbeat=False,
                energy=0.0,
                energy_derivative=0.0,
                segment="verse",
                genre_weights={},
                vocal_energy=0.0,
                spectral_centroid=0.5,
                sub_bass_energy=0.0,
                onset_type=None,
                drop_probability=0.0,
            )

            commands = self._engine.generate(idle_state)
            commands = self._apply_effects(commands, idle_state)
            with contextlib.suppress(asyncio.QueueFull):
                self._server.state_queue.put_nowait((idle_state, commands))

            await asyncio.sleep(frame_interval)

    def _apply_effects(
        self, commands: list[FixtureCommand], state: MusicState
    ) -> list[FixtureCommand]:
        """Apply global intensity and manual effects to fixture commands.

        Args:
            commands: Raw commands from the lighting engine.
            state: Current music state (for timestamp).

        Returns:
            Modified commands with intensity/effects applied.
        """
        # Check if a manual effect is active
        if self._manual_effect is not None:
            elapsed = time.monotonic() - self._manual_effect_start
            if elapsed < self._manual_effect_duration:
                return self._generate_manual_effect(commands, self._manual_effect)
            self._manual_effect = None

        # Apply global intensity multiplier
        if self._global_intensity < 1.0:
            scale = self._global_intensity
            result: list[FixtureCommand] = []
            for c in commands:
                result.append(FixtureCommand(
                    fixture_id=c.fixture_id,
                    red=int(c.red * scale),
                    green=int(c.green * scale),
                    blue=int(c.blue * scale),
                    white=int(c.white * scale),
                    strobe_rate=c.strobe_rate,
                    strobe_intensity=int(c.strobe_intensity * scale),
                    special=int(c.special * scale),
                ))
            return result

        return commands

    def _generate_manual_effect(
        self, commands: list[FixtureCommand], effect: str
    ) -> list[FixtureCommand]:
        """Generate commands for a manual effect override.

        Args:
            commands: Original commands (used for fixture IDs).
            effect: Effect name — "blackout", "strobe_burst", or "uv_flash".

        Returns:
            Overridden fixture commands for the effect.
        """
        result: list[FixtureCommand] = []
        for c in commands:
            if effect == "blackout":
                result.append(FixtureCommand(
                    fixture_id=c.fixture_id,
                    red=0, green=0, blue=0, white=0,
                    strobe_rate=0, strobe_intensity=0, special=0,
                ))
            elif effect == "strobe_burst":
                result.append(FixtureCommand(
                    fixture_id=c.fixture_id,
                    red=255, green=255, blue=255, white=255,
                    strobe_rate=255, strobe_intensity=255, special=255,
                ))
            elif effect == "uv_flash":
                result.append(FixtureCommand(
                    fixture_id=c.fixture_id,
                    red=50, green=0, blue=100, white=0,
                    strobe_rate=0, strobe_intensity=0, special=255,
                ))
            else:
                result.append(c)
        return result

    async def _handle_transport(self) -> None:
        """Process transport and control commands from WebSocket clients."""
        fps = self._config.fps

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
                    n = len(self._music_states)
                    position = float(msg.get("position", 0.0))
                    self._frame_index = max(0, min(int(position * fps), max(n - 1, 0)))
                    logger.info("Transport: seek to %.1fs (frame %d)", position, self._frame_index)

            elif action == "pattern_override":
                pattern = msg.get("pattern")
                if isinstance(pattern, str) or pattern is None:
                    self._engine.set_pattern_override(pattern if isinstance(pattern, str) else None)

            elif action == "genre_override":
                profile = msg.get("profile")
                if profile is None:
                    self._engine.set_genre_override(None)
                    logger.info("Genre override: auto")
                elif isinstance(profile, str):
                    self._engine.set_genre_override(profile)
                    logger.info("Genre override: %s", profile)

            elif action == "intensity":
                value = msg.get("value")
                if isinstance(value, (int, float)):
                    self._global_intensity = max(0.0, min(1.0, float(value) / 100.0))
                    logger.info("Global intensity: %.0f%%", self._global_intensity * 100)

            elif action == "manual_effect":
                effect = msg.get("effect")
                if isinstance(effect, str) and effect in ("blackout", "strobe_burst", "uv_flash"):
                    self._manual_effect = effect
                    self._manual_effect_start = time.monotonic()
                    logger.info("Manual effect: %s", effect)

            elif action == "audio_loaded":
                logger.info(
                    "Client loaded audio: %s (%.1fs)",
                    msg.get("filename", "?"),
                    msg.get("duration", 0.0),
                )


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
        choices=["file", "live", "showcase"],
        default="file",
        help="Operating mode (default: file). 'showcase' starts server "
             "immediately for pattern testing with no audio analysis.",
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
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print MusicState once per second for pipeline debugging",
    )
    parser.add_argument(
        "--genre",
        default=None,
        help="Override genre classification with a fixed profile (e.g. rage_trap)",
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
        debug=args.debug,
        genre_override=args.genre,
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
