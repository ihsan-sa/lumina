"""ShowRecorder — records fixture commands frame-by-frame to a compact binary format.

Binary format (.lrec), little-endian:

    Header (42 bytes):
        magic:         4 bytes  ("LREC")
        version:       uint8    (1)
        audio_hash:    32 bytes (SHA256 of audio file)
        fps:           uint8    (e.g. 60)
        fixture_count: uint8    (e.g. 15)
        duration_ms:   uint32   (total duration in milliseconds)
        frame_count:   uint32   (total number of recorded frames)

    Frame (4 + fixture_count x 8 bytes each):
        timestamp_ms:  uint32
        commands:      fixture_count x 8 bytes
                       (fixture_id, R, G, B, W, strobe_rate, strobe_intensity, special)

File size estimate: 60fps x 15 fixtures x 8 bytes + 4 bytes timestamp = ~7.2 KB/s
= ~2.2 MB per 5-minute song (uncompressed).
"""

from __future__ import annotations

import hashlib
import io
import logging
import struct
from pathlib import Path

from lumina.control.protocol import FIXTURE_SIZE, FixtureCommand

logger = logging.getLogger(__name__)

# ─── Format constants ─────────────────────────────────────────────────────────

LREC_MAGIC = b"LREC"
LREC_VERSION = 1

# Header layout (little-endian):
#   4s magic | B version | 32s audio_hash | B fps | B fixture_count | I duration_ms | I frame_count
HEADER_FORMAT = "<4sB32sBBII"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)  # 42 bytes

FRAME_TIMESTAMP_FORMAT = "<I"  # uint32 little-endian
FRAME_TIMESTAMP_SIZE = struct.calcsize(FRAME_TIMESTAMP_FORMAT)  # 4 bytes


# ─── Utility ──────────────────────────────────────────────────────────────────


def hash_audio_file(path: Path) -> bytes:
    """Compute the SHA256 hash of an audio file.

    Reads the file in 64 KiB chunks to keep memory usage bounded for large files.

    Args:
        path: Path to the audio file.

    Returns:
        32-byte SHA256 digest.

    Raises:
        FileNotFoundError: If the path does not exist.
        IsADirectoryError: If the path is a directory.
    """
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            hasher.update(chunk)
    digest = hasher.digest()
    logger.debug("SHA256 of %s: %s", path.name, digest.hex())
    return digest


# ─── ShowRecorder ─────────────────────────────────────────────────────────────


class ShowRecorder:
    """Records fixture commands frame-by-frame to a compact binary .lrec file.

    Usage::

        recorder = ShowRecorder(audio_hash=hash_audio_file(audio_path))
        for timestamp_ms, commands in my_show_generator():
            recorder.record_frame(timestamp_ms, commands)
        recorder.save(Path("output.lrec"))

    Args:
        audio_hash: 32-byte SHA256 digest of the associated audio file.
        fps: Nominal frame rate (informational; does not enforce timing).
        fixture_count: Number of fixture commands expected per frame.

    Raises:
        ValueError: If audio_hash is not exactly 32 bytes, fps is 0, or
            fixture_count is outside 1-255.
    """

    def __init__(
        self,
        audio_hash: bytes,
        fps: int = 60,
        fixture_count: int = 15,
    ) -> None:
        if len(audio_hash) != 32:
            msg = f"audio_hash must be exactly 32 bytes, got {len(audio_hash)}"
            raise ValueError(msg)
        if fps < 1 or fps > 255:
            msg = f"fps must be 1-255, got {fps}"
            raise ValueError(msg)
        if fixture_count < 1 or fixture_count > 255:
            msg = f"fixture_count must be 1-255, got {fixture_count}"
            raise ValueError(msg)

        self._audio_hash: bytes = audio_hash
        self._fps: int = fps
        self._fixture_count: int = fixture_count

        # Buffer accumulates raw frame bytes for zero-copy final write.
        self._buffer: io.BytesIO = io.BytesIO()
        self._frame_count: int = 0
        self._last_timestamp_ms: int = 0

        logger.debug(
            "ShowRecorder initialised: fps=%d fixture_count=%d hash=%s",
            fps,
            fixture_count,
            audio_hash.hex()[:16] + "...",
        )

    # ── Recording ─────────────────────────────────────────────────────────────

    def record_frame(self, timestamp_ms: int, commands: list[FixtureCommand]) -> None:
        """Record a single frame of fixture commands.

        The number of commands must equal the fixture_count supplied at construction.
        Extra commands are silently truncated; missing commands are zero-padded.

        Args:
            timestamp_ms: Frame timestamp in milliseconds (uint32, max 4 294 967 295).
            commands: Fixture commands for this frame.

        Raises:
            ValueError: If timestamp_ms is negative or exceeds uint32 range.
        """
        if timestamp_ms < 0 or timestamp_ms > 0xFFFFFFFF:
            msg = f"timestamp_ms must be 0-4294967295, got {timestamp_ms}"
            raise ValueError(msg)

        # Write timestamp.
        self._buffer.write(struct.pack(FRAME_TIMESTAMP_FORMAT, timestamp_ms))

        # Write exactly fixture_count command blocks.
        written = 0
        for cmd in commands:
            if written >= self._fixture_count:
                break
            self._buffer.write(cmd.to_bytes())
            written += 1

        # Zero-pad if fewer commands were supplied.
        pad_blocks = self._fixture_count - written
        if pad_blocks > 0:
            self._buffer.write(b"\x00" * (FIXTURE_SIZE * pad_blocks))
            logger.warning(
                "Frame %d: only %d commands supplied, expected %d — zero-padded",
                self._frame_count,
                written,
                self._fixture_count,
            )

        self._frame_count += 1
        self._last_timestamp_ms = timestamp_ms

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        """Save the recording to a .lrec file.

        The file is written atomically: data is first assembled in memory, then
        written to the destination path in a single call.

        Args:
            path: Destination file path. Parent directory must exist.

        Raises:
            OSError: If the file cannot be written.
        """
        path = Path(path)  # Accept str for convenience
        header = struct.pack(
            HEADER_FORMAT,
            LREC_MAGIC,
            LREC_VERSION,
            self._audio_hash,
            self._fps,
            self._fixture_count,
            self._last_timestamp_ms,  # duration_ms = timestamp of last frame
            self._frame_count,
        )

        frame_data = self._buffer.getvalue()

        # Try zstd compression if available; otherwise write raw bytes.
        suffix = path.suffix.lower()
        use_zstd = suffix == ".zst"
        if not use_zstd:
            # Honour explicit .lrec.zst double-extension too.
            use_zstd = path.name.lower().endswith(".lrec.zst")

        payload = header + frame_data
        if use_zstd:
            payload = _compress_zstd(payload)

        path.write_bytes(payload)
        logger.info(
            "Saved recording to %s: %d frames, %.1f s, %d bytes",
            path,
            self._frame_count,
            self._last_timestamp_ms / 1000.0,
            len(payload),
        )

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def frame_count(self) -> int:
        """Number of frames recorded so far."""
        return self._frame_count

    @property
    def duration_ms(self) -> int:
        """Timestamp of the last recorded frame in milliseconds (0 if no frames)."""
        return self._last_timestamp_ms

    @property
    def fps(self) -> int:
        """Nominal frame rate."""
        return self._fps

    @property
    def fixture_count(self) -> int:
        """Number of fixture commands per frame."""
        return self._fixture_count

    @property
    def audio_hash(self) -> bytes:
        """32-byte SHA256 digest of the associated audio file."""
        return self._audio_hash


# ─── Optional zstd compression ────────────────────────────────────────────────


def _compress_zstd(data: bytes) -> bytes:
    """Compress bytes with zstd if the zstandard package is available.

    Falls back to returning the original data unmodified if zstandard is not
    installed, so the dependency remains optional.

    Args:
        data: Raw bytes to compress.

    Returns:
        Compressed bytes, or the original bytes if zstandard is unavailable.
    """
    try:
        import zstandard as zstd  # type: ignore[import-untyped]

        cctx = zstd.ZstdCompressor(level=3)
        compressed: bytes = cctx.compress(data)
        logger.debug(
            "zstd compressed %d → %d bytes (%.1f%%)",
            len(data),
            len(compressed),
            100.0 * len(compressed) / len(data),
        )
        return compressed
    except ImportError:
        logger.warning(
            "zstandard not installed — saving uncompressed despite .zst extension"
        )
        return data


def _decompress_zstd(data: bytes) -> bytes:
    """Decompress zstd-compressed bytes.

    Args:
        data: Compressed bytes.

    Returns:
        Decompressed bytes.

    Raises:
        ImportError: If zstandard is not installed.
        zstandard.ZstdError: If the data is not valid zstd.
    """
    import zstandard as zstd  # type: ignore[import-untyped]

    dctx = zstd.ZstdDecompressor()
    decompressed: bytes = dctx.decompress(data)
    return decompressed
