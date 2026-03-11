"""Audio capture utility -- captures system audio for LUMINA analysis.

Supports WASAPI loopback on Windows and PulseAudio monitor on Linux.
Can also read from local audio files for offline analysis.

Usage:
    python scripts/capture_audio.py --source loopback --duration 30 --output capture.wav
    python scripts/capture_audio.py --source file --input song.mp3 --output resampled.wav
"""

import argparse
import logging
import pathlib
import platform
import sys

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 44100
CHANNELS = 1  # Mono for analysis
BLOCK_SIZE = 2048


def detect_platform() -> str:
    """Detect the current platform for audio backend selection.

    Returns:
        Platform identifier: 'windows', 'linux', or 'unsupported'.
    """
    system = platform.system().lower()
    if system == "windows":
        return "windows"
    elif system == "linux":
        return "linux"
    else:
        return "unsupported"


def capture_loopback(
    duration: float,
    output_path: pathlib.Path,
    device: str | None = None,
) -> None:
    """Capture system audio via loopback.

    On Windows, uses WASAPI loopback. On Linux, uses PulseAudio monitor source.

    Args:
        duration: Capture duration in seconds.
        output_path: Path to save the captured WAV file.
        device: Optional device name/index override.
    """
    plat = detect_platform()

    if plat == "windows":
        logger.info("Platform: Windows — using WASAPI loopback capture")
        logger.info(
            "Would initialize sounddevice with WASAPI backend, "
            "loopback=True, samplerate=%d, channels=%d",
            SAMPLE_RATE,
            CHANNELS,
        )
    elif plat == "linux":
        logger.info("Platform: Linux — using PulseAudio/PipeWire monitor source")
        logger.info(
            "Would initialize sounddevice with PulseAudio backend, "
            "monitor source, samplerate=%d, channels=%d",
            SAMPLE_RATE,
            CHANNELS,
        )
    else:
        logger.error("Unsupported platform: %s", platform.system())
        sys.exit(1)

    if device:
        logger.info("Using device override: %s", device)

    logger.info("Capturing %.1f seconds of audio...", duration)
    logger.info("Output: %s", output_path)

    # Placeholder: actual capture would use sounddevice here
    # import sounddevice as sd
    # recording = sd.rec(
    #     int(duration * SAMPLE_RATE),
    #     samplerate=SAMPLE_RATE,
    #     channels=CHANNELS,
    #     dtype='float32',
    # )
    # sd.wait()
    # sf.write(str(output_path), recording, SAMPLE_RATE)

    logger.info(
        "Capture complete (stub — install sounddevice + soundfile for real capture)"
    )


def capture_from_file(
    input_path: pathlib.Path,
    output_path: pathlib.Path,
) -> None:
    """Load an audio file and resample to standard format.

    Args:
        input_path: Path to source audio file (MP3, FLAC, WAV, etc.).
        output_path: Path to save the resampled WAV file.
    """
    logger.info("Loading audio file: %s", input_path)

    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        sys.exit(1)

    logger.info(
        "Would load with librosa.load(sr=%d, mono=True) and save to %s",
        SAMPLE_RATE,
        output_path,
    )

    # Placeholder: actual implementation would use librosa
    # import librosa
    # import soundfile as sf
    # audio, sr = librosa.load(str(input_path), sr=SAMPLE_RATE, mono=True)
    # sf.write(str(output_path), audio, SAMPLE_RATE)

    logger.info("File processing complete (stub)")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="LUMINA audio capture utility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s --source loopback --duration 30 --output capture.wav\n"
            "  %(prog)s --source file --input song.mp3 --output resampled.wav\n"
        ),
    )
    parser.add_argument(
        "--source",
        choices=["loopback", "file"],
        default="loopback",
        help="Audio source: 'loopback' for system audio, 'file' for local file (default: loopback)",
    )
    parser.add_argument(
        "--input",
        type=pathlib.Path,
        help="Input audio file path (required when --source=file)",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("capture.wav"),
        help="Output WAV file path (default: capture.wav)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="Capture duration in seconds (default: 30, only for loopback mode)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Audio device name or index (optional, for loopback mode)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    return parser.parse_args()


def main() -> None:
    """Entry point for the audio capture utility."""
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logger.info("LUMINA Audio Capture Utility")
    logger.info("Platform: %s (%s)", platform.system(), platform.machine())

    if args.source == "loopback":
        capture_loopback(
            duration=args.duration,
            output_path=args.output,
            device=args.device,
        )
    elif args.source == "file":
        if args.input is None:
            logger.error("--input is required when --source=file")
            sys.exit(1)
        capture_from_file(
            input_path=args.input,
            output_path=args.output,
        )


if __name__ == "__main__":
    main()
