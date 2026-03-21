"""ShowPlayer — loads a .lrec recording and replays fixture commands.

Supports both sequential iteration and random-access by frame index or timestamp.
Optionally verifies that the audio file used during recording matches the loaded
recording via SHA256 hash comparison.
"""

from __future__ import annotations

import logging
import struct
from collections.abc import Iterator
from pathlib import Path

from lumina.control.protocol import FIXTURE_SIZE, FixtureCommand
from lumina.recording.recorder import (
    FRAME_TIMESTAMP_FORMAT,
    FRAME_TIMESTAMP_SIZE,
    HEADER_FORMAT,
    HEADER_SIZE,
    LREC_MAGIC,
    LREC_VERSION,
    _decompress_zstd,
    hash_audio_file,
)

logger = logging.getLogger(__name__)


class ShowPlayer:
    """Loads a .lrec recording file and replays fixture commands.

    The entire file is read into memory on construction, making all random-access
    operations O(1) after loading. For a 5-minute show at 60fps with 15 fixtures
    the in-memory footprint is ~2.2 MB — well within normal constraints.

    Usage::

        player = ShowPlayer(Path("show.lrec"))
        for timestamp_ms, commands in player.frames():
            send_to_network(commands)

    Args:
        path: Path to the .lrec (or .lrec.zst) file.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If the file has an invalid magic, unsupported version, or
            is otherwise malformed.
    """

    def __init__(self, path: Path) -> None:
        path = Path(path)
        if not path.exists():
            msg = f"Recording file not found: {path}"
            raise FileNotFoundError(msg)

        raw = path.read_bytes()

        # Decompress if needed.
        name_lower = path.name.lower()
        if name_lower.endswith(".zst"):
            raw = _decompress_zstd(raw)

        self._parse(raw, path)
        logger.info(
            "Loaded recording from %s: %d frames, %.1f s, %d fixtures @ %d fps",
            path,
            self._frame_count,
            self._duration_ms / 1000.0,
            self._fixture_count,
            self._fps,
        )

    # ── Parsing ───────────────────────────────────────────────────────────────

    def _parse(self, data: bytes, path: Path) -> None:
        """Parse raw (possibly decompressed) recording bytes.

        Args:
            data: Full file contents.
            path: Original path (used in error messages only).

        Raises:
            ValueError: On any structural problem with the data.
        """
        if len(data) < HEADER_SIZE:
            msg = f"File too short to contain a valid header: {len(data)} bytes"
            raise ValueError(msg)

        magic, version, audio_hash, fps, fixture_count, duration_ms, frame_count = (
            struct.unpack(HEADER_FORMAT, data[:HEADER_SIZE])
        )

        if magic != LREC_MAGIC:
            msg = f"Invalid magic: {magic!r} (expected {LREC_MAGIC!r})"
            raise ValueError(msg)

        if version != LREC_VERSION:
            msg = f"Unsupported recording version: {version} (expected {LREC_VERSION})"
            raise ValueError(msg)

        if fps == 0:
            msg = "fps cannot be 0"
            raise ValueError(msg)

        if fixture_count == 0:
            msg = "fixture_count cannot be 0"
            raise ValueError(msg)

        frame_size = FRAME_TIMESTAMP_SIZE + fixture_count * FIXTURE_SIZE
        payload = data[HEADER_SIZE:]
        expected_payload = frame_count * frame_size

        if len(payload) < expected_payload:
            msg = (
                f"Payload too short: {len(payload)} bytes for {frame_count} frames "
                f"x {frame_size} bytes/frame = {expected_payload} bytes expected"
            )
            raise ValueError(msg)

        self._audio_hash: bytes = bytes(audio_hash)
        self._fps: int = fps
        self._fixture_count: int = fixture_count
        self._duration_ms: int = duration_ms
        self._frame_count: int = frame_count
        self._frame_size: int = frame_size
        self._payload: bytes = payload

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def audio_hash(self) -> bytes:
        """32-byte SHA256 digest stored in the recording header."""
        return self._audio_hash

    @property
    def fps(self) -> int:
        """Nominal frame rate of the recording."""
        return self._fps

    @property
    def duration_ms(self) -> int:
        """Timestamp of the last frame in milliseconds."""
        return self._duration_ms

    @property
    def frame_count(self) -> int:
        """Total number of frames in the recording."""
        return self._frame_count

    @property
    def fixture_count(self) -> int:
        """Number of fixture commands per frame."""
        return self._fixture_count

    # ── Iteration ─────────────────────────────────────────────────────────────

    def frames(self) -> Iterator[tuple[int, list[FixtureCommand]]]:
        """Iterate over every frame in recording order.

        Yields:
            Tuple of (timestamp_ms, commands) for each frame.
        """
        for index in range(self._frame_count):
            yield self.frame_at(index)

    # ── Random access ─────────────────────────────────────────────────────────

    def frame_at(self, index: int) -> tuple[int, list[FixtureCommand]]:
        """Return a specific frame by zero-based index.

        Args:
            index: Frame index (0 to frame_count - 1).

        Returns:
            Tuple of (timestamp_ms, commands).

        Raises:
            IndexError: If index is out of range.
        """
        if index < 0 or index >= self._frame_count:
            msg = f"Frame index {index} out of range [0, {self._frame_count})"
            raise IndexError(msg)

        offset = index * self._frame_size

        # Unpack timestamp.
        (timestamp_ms,) = struct.unpack_from(FRAME_TIMESTAMP_FORMAT, self._payload, offset)
        offset += FRAME_TIMESTAMP_SIZE

        # Unpack fixture commands.
        commands: list[FixtureCommand] = []
        for _ in range(self._fixture_count):
            cmd = FixtureCommand.from_bytes(self._payload[offset : offset + FIXTURE_SIZE])
            commands.append(cmd)
            offset += FIXTURE_SIZE

        return timestamp_ms, commands

    def seek(self, timestamp_ms: int) -> int:
        """Return the index and actual timestamp of the frame nearest to timestamp_ms.

        Uses binary search over stored timestamps for O(log n) performance.

        Args:
            timestamp_ms: Target timestamp in milliseconds.

        Returns:
            Actual timestamp_ms of the nearest frame. Use ``frame_at`` with the
            returned value to retrieve the frame data, or combine both calls::

                actual_ts = player.seek(target_ms)
                # The frame index is available via _seek_index if needed; prefer
                # iterating from a known index.

        Raises:
            ValueError: If the recording contains no frames.
        """
        if self._frame_count == 0:
            msg = "Cannot seek in an empty recording"
            raise ValueError(msg)

        # Binary search: find the frame index whose timestamp is closest.
        lo, hi = 0, self._frame_count - 1
        while lo < hi:
            mid = (lo + hi) // 2
            mid_ts = self._timestamp_at(mid)
            if mid_ts < timestamp_ms:
                lo = mid + 1
            else:
                hi = mid

        # lo is now the first frame with timestamp >= target.
        # Check both lo and lo-1 to find the nearest.
        best_index = lo
        if lo > 0:
            prev_ts = self._timestamp_at(lo - 1)
            curr_ts = self._timestamp_at(lo)
            if abs(prev_ts - timestamp_ms) <= abs(curr_ts - timestamp_ms):
                best_index = lo - 1

        actual_ts = self._timestamp_at(best_index)
        self._last_seek_index: int = best_index
        logger.debug("seek(%d ms) → frame %d @ %d ms", timestamp_ms, best_index, actual_ts)
        return actual_ts

    def seek_frame_index(self, timestamp_ms: int) -> int:
        """Return the index of the frame nearest to timestamp_ms.

        Convenience wrapper around :meth:`seek` that exposes the frame index
        directly.

        Args:
            timestamp_ms: Target timestamp in milliseconds.

        Returns:
            Zero-based frame index of the nearest frame.

        Raises:
            ValueError: If the recording contains no frames.
        """
        self.seek(timestamp_ms)
        return self._last_seek_index

    # ── Audio verification ────────────────────────────────────────────────────

    def verify_audio(self, audio_path: Path) -> bool:
        """Verify that an audio file matches the hash stored in the recording.

        Args:
            audio_path: Path to the audio file to check.

        Returns:
            True if the SHA256 hash matches; False otherwise.

        Raises:
            FileNotFoundError: If audio_path does not exist.
        """
        actual_hash = hash_audio_file(audio_path)
        match = actual_hash == self._audio_hash
        if match:
            logger.debug("Audio hash verified: %s", audio_path.name)
        else:
            logger.warning(
                "Audio hash mismatch for %s: expected %s, got %s",
                audio_path.name,
                self._audio_hash.hex(),
                actual_hash.hex(),
            )
        return match

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _timestamp_at(self, index: int) -> int:
        """Read the timestamp of frame at index without unpacking commands.

        Args:
            index: Frame index.

        Returns:
            Timestamp in milliseconds.
        """
        offset = index * self._frame_size
        (ts,) = struct.unpack_from(FRAME_TIMESTAMP_FORMAT, self._payload, offset)
        return int(ts)
