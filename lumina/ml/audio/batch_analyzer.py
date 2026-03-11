"""Batch audio analysis runner for the ML training pipeline.

Runs LUMINA's existing audio analysis pipeline on extracted video audio
files (WAV) and generates MusicState timelines at 10fps (matching the
video frame extraction rate). Results are saved as Parquet files for
efficient loading during model training.

Output directory: ``data/features/audio/{video_id}.parquet``

Each Parquet file contains one row per frame (at 10fps) with all
MusicState fields serialized as columns.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from lumina.audio.beat_detector import BeatDetector
from lumina.audio.drop_predictor import DropPredictor
from lumina.audio.energy_tracker import EnergyTracker
from lumina.audio.genre_classifier import GenreClassifier
from lumina.audio.models import MusicState
from lumina.audio.onset_detector import OnsetDetector
from lumina.audio.segment_classifier import SegmentClassifier
from lumina.audio.vocal_detector import VocalDetector

logger = logging.getLogger(__name__)

# Default sample rate and analysis frame rate
_DEFAULT_SR = 44100
_DEFAULT_FPS = 10  # Match video frame extraction rate

# Default output directory (relative to project root)
_DEFAULT_OUTPUT_DIR = Path("data/features/audio")


def _load_audio(audio_path: Path, sr: int) -> np.ndarray:
    """Load an audio file as a mono float32 numpy array.

    Args:
        audio_path: Path to the audio file (WAV, MP3, FLAC, etc.).
        sr: Target sample rate. Audio is resampled if necessary.

    Returns:
        Mono float32 audio normalized to [-1, 1].

    Raises:
        FileNotFoundError: If the audio file does not exist.
        RuntimeError: If the audio file cannot be loaded.
    """
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    import librosa

    audio, _ = librosa.load(str(audio_path), sr=sr, mono=True)
    return audio.astype(np.float32)


def analyze_video_audio(
    audio_path: Path,
    sr: int = _DEFAULT_SR,
    fps: int = _DEFAULT_FPS,
) -> list[MusicState]:
    """Run full LUMINA audio analysis on extracted video audio.

    Runs the same pipeline as the main application, but at 10fps to
    match the video frame extraction rate. Uses offline analysis mode
    for each module to get the best possible accuracy on pre-recorded
    audio.

    Args:
        audio_path: Path to the extracted audio file (WAV preferred).
        sr: Sample rate in Hz.
        fps: Output frame rate. Default 10fps to match video extraction.

    Returns:
        List of MusicState, one per output frame at the specified fps.
    """
    logger.info("Analyzing audio: %s (sr=%d, fps=%d)", audio_path, sr, fps)

    audio = _load_audio(audio_path, sr)
    duration = len(audio) / sr
    logger.info("Audio loaded: %.1fs, %d samples", duration, len(audio))

    # Initialize all analyzers at the target fps
    beat_detector = BeatDetector(sr=sr, fps=fps)
    energy_tracker = EnergyTracker(sr=sr, fps=fps)
    onset_detector = OnsetDetector(sr=sr, fps=fps)
    vocal_detector = VocalDetector(sr=sr, fps=fps)
    segment_classifier = SegmentClassifier(fps=fps)
    genre_classifier = GenreClassifier(fps=fps)
    drop_predictor = DropPredictor(fps=fps)

    # Run offline analysis for each module
    logger.info("Running beat detection...")
    beat_frames = beat_detector.analyze_offline(audio)

    logger.info("Running energy tracking...")
    energy_frames = energy_tracker.analyze_offline(audio)

    logger.info("Running onset detection...")
    onset_frames = onset_detector.analyze_offline(audio)

    logger.info("Running vocal detection...")
    vocal_frames = vocal_detector.analyze_offline(audio)

    # Determine the minimum frame count across all analyzers
    n_frames = min(
        len(beat_frames),
        len(energy_frames),
        len(onset_frames),
        len(vocal_frames),
    )

    if n_frames == 0:
        logger.warning("No frames produced by audio analysis")
        return []

    logger.info("Base analysis complete: %d frames", n_frames)

    # Run drop predictor frame-by-frame (depends on other analyzers)
    drop_frames = []
    for i in range(n_frames):
        ef = energy_frames[i]
        of = onset_frames[i]
        vf = vocal_frames[i]
        bf = beat_frames[i]

        # Update BPM for drop predictor
        if i == 0 or bf.bpm != beat_frames[i - 1].bpm:
            drop_predictor.update_bpm(bf.bpm)

        drop_frame = drop_predictor.process_frame(
            energy=ef.energy,
            energy_derivative=ef.energy_derivative,
            spectral_centroid=ef.spectral_centroid,
            sub_bass_energy=ef.sub_bass_energy,
            vocal_energy=vf.vocal_energy,
            has_onset=of is not None,
        )
        drop_frames.append(drop_frame)

    # Run segment classifier (depends on energy, onsets, vocals)
    segment_frames = segment_classifier.classify_offline(
        energies=[energy_frames[i].energy for i in range(n_frames)],
        energy_derivatives=[
            energy_frames[i].energy_derivative for i in range(n_frames)
        ],
        spectral_centroids=[
            energy_frames[i].spectral_centroid for i in range(n_frames)
        ],
        sub_bass_energies=[
            energy_frames[i].sub_bass_energy for i in range(n_frames)
        ],
        vocal_energies=[
            vocal_frames[i].vocal_energy for i in range(n_frames)
        ],
        has_onsets=[onset_frames[i] is not None for i in range(n_frames)],
    )

    # Run genre classifier (depends on energy, onsets, vocals, drops)
    genre_frames = genre_classifier.classify_offline(
        energies=[energy_frames[i].energy for i in range(n_frames)],
        spectral_centroids=[
            energy_frames[i].spectral_centroid for i in range(n_frames)
        ],
        sub_bass_energies=[
            energy_frames[i].sub_bass_energy for i in range(n_frames)
        ],
        has_onsets=[onset_frames[i] is not None for i in range(n_frames)],
        vocal_energies=[
            vocal_frames[i].vocal_energy for i in range(n_frames)
        ],
        drop_probabilities=[
            drop_frames[i].drop_probability for i in range(n_frames)
        ],
    )

    # Assemble MusicState for each frame
    states: list[MusicState] = []
    for i in range(n_frames):
        bf = beat_frames[i]
        ef = energy_frames[i]
        of = onset_frames[i]
        vf = vocal_frames[i]
        sf = segment_frames[i] if i < len(segment_frames) else None
        gf = genre_frames[i] if i < len(genre_frames) else None
        df = drop_frames[i]

        state = MusicState(
            timestamp=float(i) / fps,
            bpm=bf.bpm,
            beat_phase=bf.beat_phase,
            bar_phase=bf.bar_phase,
            is_beat=bf.is_beat,
            is_downbeat=bf.is_downbeat,
            energy=ef.energy,
            energy_derivative=ef.energy_derivative,
            segment=sf.segment if sf else "verse",
            genre_weights=gf.genre_weights if gf else {},
            vocal_energy=vf.vocal_energy,
            spectral_centroid=ef.spectral_centroid,
            sub_bass_energy=ef.sub_bass_energy,
            onset_type=of.onset_type if of else None,
            drop_probability=df.drop_probability,
        )
        states.append(state)

    logger.info(
        "Audio analysis complete: %d MusicState frames (%.1fs at %dfps)",
        len(states),
        len(states) / fps,
        fps,
    )

    return states


def _music_state_to_row(state: MusicState) -> dict:
    """Convert a MusicState to a flat dictionary for Parquet serialization.

    Flattens nested dicts (genre_weights, layer_mask) into prefixed
    columns for tabular storage.

    Args:
        state: MusicState to convert.

    Returns:
        Flat dictionary suitable for DataFrame row construction.
    """
    row = asdict(state)

    # Flatten genre_weights dict into separate columns
    genre_weights = row.pop("genre_weights", {})
    for profile_name, weight in genre_weights.items():
        row[f"genre_weight_{profile_name}"] = weight

    # Flatten layer_mask dict into separate columns
    layer_mask = row.pop("layer_mask", {})
    for stem_name, level in layer_mask.items():
        row[f"layer_mask_{stem_name}"] = level

    return row


def save_as_parquet(
    states: list[MusicState],
    output_path: Path,
) -> Path:
    """Save a MusicState timeline as a Parquet file.

    Args:
        states: List of MusicState frames to save.
        output_path: Path to the output Parquet file.

    Returns:
        The output path (for chaining convenience).
    """
    if not states:
        logger.warning("No states to save, skipping: %s", output_path)
        return output_path

    rows = [_music_state_to_row(s) for s in states]
    df = pd.DataFrame(rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, engine="pyarrow", index=False)

    logger.info(
        "Saved %d frames to %s (%.1f KB)",
        len(df),
        output_path,
        output_path.stat().st_size / 1024,
    )

    return output_path


def analyze_and_save(
    audio_path: Path,
    video_id: str,
    output_dir: Path = _DEFAULT_OUTPUT_DIR,
    sr: int = _DEFAULT_SR,
    fps: int = _DEFAULT_FPS,
) -> Path:
    """Run audio analysis and save results as Parquet.

    Convenience function that combines ``analyze_video_audio`` and
    ``save_as_parquet``. Output is saved to
    ``{output_dir}/{video_id}.parquet``.

    Args:
        audio_path: Path to the extracted audio file.
        video_id: Unique video identifier (used as filename).
        output_dir: Directory for output Parquet files.
        sr: Sample rate in Hz.
        fps: Output frame rate.

    Returns:
        Path to the saved Parquet file.
    """
    states = analyze_video_audio(audio_path, sr=sr, fps=fps)
    output_path = output_dir / f"{video_id}.parquet"
    return save_as_parquet(states, output_path)


def batch_analyze(
    audio_files: dict[str, Path],
    output_dir: Path = _DEFAULT_OUTPUT_DIR,
    sr: int = _DEFAULT_SR,
    fps: int = _DEFAULT_FPS,
    skip_existing: bool = True,
) -> dict[str, Path]:
    """Run audio analysis on multiple video audio files.

    Processes each audio file through the full LUMINA pipeline and
    saves MusicState timelines as Parquet files.

    Args:
        audio_files: Mapping of video_id to audio file path.
        output_dir: Directory for output Parquet files.
        sr: Sample rate in Hz.
        fps: Output frame rate (default 10fps matching video).
        skip_existing: If True, skip videos that already have a
            Parquet file in the output directory.

    Returns:
        Mapping of video_id to output Parquet file path.
    """
    results: dict[str, Path] = {}
    total = len(audio_files)

    for idx, (video_id, audio_path) in enumerate(audio_files.items(), 1):
        output_path = output_dir / f"{video_id}.parquet"

        if skip_existing and output_path.exists():
            logger.info(
                "[%d/%d] Skipping %s (already exists)", idx, total, video_id
            )
            results[video_id] = output_path
            continue

        logger.info("[%d/%d] Analyzing %s", idx, total, video_id)

        try:
            result_path = analyze_and_save(
                audio_path=audio_path,
                video_id=video_id,
                output_dir=output_dir,
                sr=sr,
                fps=fps,
            )
            results[video_id] = result_path
        except Exception:
            logger.exception(
                "[%d/%d] Failed to analyze %s", idx, total, video_id
            )

    logger.info(
        "Batch analysis complete: %d/%d succeeded",
        len(results),
        total,
    )

    return results
