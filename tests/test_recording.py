"""Tests for lumina.recording — ShowRecorder and ShowPlayer roundtrip, seeking,
audio verification, and edge cases.
"""

from __future__ import annotations

import hashlib
import struct
import tempfile
from pathlib import Path

import pytest

from lumina.control.protocol import FixtureCommand
from lumina.recording import ShowPlayer, ShowRecorder, hash_audio_file
from lumina.recording.recorder import (
    HEADER_FORMAT,
    HEADER_SIZE,
    LREC_MAGIC,
    LREC_VERSION,
)


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _make_command(fixture_id: int, seed: int = 0) -> FixtureCommand:
    """Build a deterministic FixtureCommand from a seed value."""
    return FixtureCommand(
        fixture_id=fixture_id,
        red=(seed * 13 + fixture_id) % 256,
        green=(seed * 17 + fixture_id * 3) % 256,
        blue=(seed * 19 + fixture_id * 7) % 256,
        white=(seed * 23 + fixture_id * 11) % 256,
        strobe_rate=(seed * 5) % 256,
        strobe_intensity=(seed * 7) % 256,
        special=(seed * 11) % 256,
    )


def _make_frame(fixture_count: int, frame_index: int) -> list[FixtureCommand]:
    """Build a list of fixture_count deterministic commands for frame_index."""
    return [_make_command(i + 1, seed=frame_index) for i in range(fixture_count)]


def _fake_audio_hash() -> bytes:
    """Return a fake but valid 32-byte SHA256-like hash."""
    return hashlib.sha256(b"test audio content").digest()


# ─── Roundtrip tests ─────────────────────────────────────────────────────────


class TestRoundtrip:
    """Record frames → save → load → verify identical frames."""

    def test_basic_roundtrip(self, tmp_path: Path) -> None:
        """Single-frame roundtrip produces identical output."""
        audio_hash = _fake_audio_hash()
        recorder = ShowRecorder(audio_hash=audio_hash, fps=60, fixture_count=3)

        commands_in = _make_frame(3, frame_index=0)
        recorder.record_frame(timestamp_ms=0, commands=commands_in)

        out_path = tmp_path / "show.lrec"
        recorder.save(out_path)

        player = ShowPlayer(out_path)
        assert player.frame_count == 1
        ts, commands_out = player.frame_at(0)
        assert ts == 0
        assert commands_out == commands_in

    def test_multi_frame_roundtrip(self, tmp_path: Path) -> None:
        """100 frames with varied commands roundtrip correctly."""
        fixture_count = 5
        fps = 60
        audio_hash = _fake_audio_hash()
        recorder = ShowRecorder(audio_hash=audio_hash, fps=fps, fixture_count=fixture_count)

        frames_in: list[tuple[int, list[FixtureCommand]]] = []
        for i in range(100):
            ts = i * (1000 // fps)
            cmds = _make_frame(fixture_count, frame_index=i)
            recorder.record_frame(ts, cmds)
            frames_in.append((ts, cmds))

        out_path = tmp_path / "show.lrec"
        recorder.save(out_path)

        player = ShowPlayer(out_path)
        assert player.frame_count == 100

        for i, (expected_ts, expected_cmds) in enumerate(frames_in):
            actual_ts, actual_cmds = player.frame_at(i)
            assert actual_ts == expected_ts, f"Frame {i}: timestamp mismatch"
            assert actual_cmds == expected_cmds, f"Frame {i}: commands mismatch"

    def test_frames_iterator_roundtrip(self, tmp_path: Path) -> None:
        """frames() iterator produces the same sequence as individual frame_at calls."""
        fixture_count = 4
        audio_hash = _fake_audio_hash()
        recorder = ShowRecorder(audio_hash=audio_hash, fps=30, fixture_count=fixture_count)

        n_frames = 50
        for i in range(n_frames):
            recorder.record_frame(i * 33, _make_frame(fixture_count, i))

        out_path = tmp_path / "show.lrec"
        recorder.save(out_path)
        player = ShowPlayer(out_path)

        frames_iter = list(player.frames())
        assert len(frames_iter) == n_frames

        for idx, (ts_iter, cmds_iter) in enumerate(frames_iter):
            ts_direct, cmds_direct = player.frame_at(idx)
            assert ts_iter == ts_direct
            assert cmds_iter == cmds_direct

    def test_all_channel_values_preserved(self, tmp_path: Path) -> None:
        """Extreme channel values (0 and 255) are preserved exactly."""
        audio_hash = _fake_audio_hash()
        recorder = ShowRecorder(audio_hash=audio_hash, fps=60, fixture_count=2)

        all_zero = FixtureCommand(0, 0, 0, 0, 0, 0, 0, 0)
        all_max = FixtureCommand(255, 255, 255, 255, 255, 255, 255, 255)
        recorder.record_frame(0, [all_zero, all_max])

        out_path = tmp_path / "show.lrec"
        recorder.save(out_path)

        player = ShowPlayer(out_path)
        _, cmds = player.frame_at(0)
        assert cmds[0] == all_zero
        assert cmds[1] == all_max


# ─── Header parsing tests ─────────────────────────────────────────────────────


class TestHeaderParsing:
    """Verify header fields are written and read correctly."""

    def test_magic_bytes(self, tmp_path: Path) -> None:
        """Magic bytes are 'LREC'."""
        audio_hash = _fake_audio_hash()
        recorder = ShowRecorder(audio_hash=audio_hash)
        recorder.record_frame(0, _make_frame(15, 0))
        out_path = tmp_path / "show.lrec"
        recorder.save(out_path)

        raw = out_path.read_bytes()
        assert raw[:4] == LREC_MAGIC

    def test_version_field(self, tmp_path: Path) -> None:
        """Version byte is LREC_VERSION (1)."""
        audio_hash = _fake_audio_hash()
        recorder = ShowRecorder(audio_hash=audio_hash)
        recorder.record_frame(0, _make_frame(15, 0))
        out_path = tmp_path / "show.lrec"
        recorder.save(out_path)

        raw = out_path.read_bytes()
        fields = struct.unpack(HEADER_FORMAT, raw[:HEADER_SIZE])
        assert fields[1] == LREC_VERSION

    def test_audio_hash_round_trips(self, tmp_path: Path) -> None:
        """Audio hash stored in header matches the one supplied at construction."""
        audio_hash = hashlib.sha256(b"specific content").digest()
        recorder = ShowRecorder(audio_hash=audio_hash, fps=60, fixture_count=1)
        recorder.record_frame(0, [FixtureCommand(1, 10, 20, 30)])
        out_path = tmp_path / "show.lrec"
        recorder.save(out_path)

        player = ShowPlayer(out_path)
        assert player.audio_hash == audio_hash

    def test_fps_field(self, tmp_path: Path) -> None:
        """fps field round-trips for non-default value."""
        recorder = ShowRecorder(audio_hash=_fake_audio_hash(), fps=30, fixture_count=5)
        recorder.record_frame(0, _make_frame(5, 0))
        out_path = tmp_path / "show.lrec"
        recorder.save(out_path)

        player = ShowPlayer(out_path)
        assert player.fps == 30

    def test_fixture_count_field(self, tmp_path: Path) -> None:
        """fixture_count field round-trips."""
        recorder = ShowRecorder(audio_hash=_fake_audio_hash(), fps=60, fixture_count=7)
        recorder.record_frame(0, _make_frame(7, 0))
        out_path = tmp_path / "show.lrec"
        recorder.save(out_path)

        player = ShowPlayer(out_path)
        assert player.fixture_count == 7

    def test_duration_ms_field(self, tmp_path: Path) -> None:
        """duration_ms reflects the last recorded frame's timestamp."""
        recorder = ShowRecorder(audio_hash=_fake_audio_hash(), fps=60, fixture_count=2)
        recorder.record_frame(0, _make_frame(2, 0))
        recorder.record_frame(16, _make_frame(2, 1))
        recorder.record_frame(33, _make_frame(2, 2))
        out_path = tmp_path / "show.lrec"
        recorder.save(out_path)

        player = ShowPlayer(out_path)
        assert player.duration_ms == 33

    def test_frame_count_field(self, tmp_path: Path) -> None:
        """frame_count in header matches the number of recorded frames."""
        n = 42
        recorder = ShowRecorder(audio_hash=_fake_audio_hash(), fps=60, fixture_count=3)
        for i in range(n):
            recorder.record_frame(i * 16, _make_frame(3, i))
        out_path = tmp_path / "show.lrec"
        recorder.save(out_path)

        player = ShowPlayer(out_path)
        assert player.frame_count == n


# ─── Seek tests ───────────────────────────────────────────────────────────────


class TestSeek:
    """Test seek() and seek_frame_index()."""

    def _make_player(self, tmp_path: Path, n_frames: int = 200, fps: int = 60) -> ShowPlayer:
        recorder = ShowRecorder(audio_hash=_fake_audio_hash(), fps=fps, fixture_count=3)
        interval = 1000 // fps
        for i in range(n_frames):
            recorder.record_frame(i * interval, _make_frame(3, i))
        out_path = tmp_path / "seek_test.lrec"
        recorder.save(out_path)
        return ShowPlayer(out_path)

    def test_seek_exact_timestamp(self, tmp_path: Path) -> None:
        """Seeking to an exact frame timestamp returns that timestamp."""
        player = self._make_player(tmp_path)
        ts, _ = player.frame_at(10)
        result = player.seek(ts)
        assert result == ts

    def test_seek_first_frame(self, tmp_path: Path) -> None:
        """Seeking to 0 returns the first frame."""
        player = self._make_player(tmp_path)
        result = player.seek(0)
        first_ts, _ = player.frame_at(0)
        assert result == first_ts

    def test_seek_beyond_end(self, tmp_path: Path) -> None:
        """Seeking past the last frame returns the last frame's timestamp."""
        player = self._make_player(tmp_path, n_frames=10)
        last_ts, _ = player.frame_at(player.frame_count - 1)
        result = player.seek(last_ts + 100_000)
        assert result == last_ts

    def test_seek_nearest_frame(self, tmp_path: Path) -> None:
        """Seeking between two frames returns the nearest one."""
        fps = 10
        player = self._make_player(tmp_path, n_frames=10, fps=fps)
        # Interval is 100ms; seek to 145ms — should land on frame 1 (100ms) since
        # |145-100| = 45 < |145-200| = 55.
        result = player.seek(145)
        assert result == 100

    def test_seek_frame_index_correctness(self, tmp_path: Path) -> None:
        """seek_frame_index returns a valid index whose timestamp matches seek."""
        player = self._make_player(tmp_path)
        target = 500
        actual_ts = player.seek(target)
        idx = player.seek_frame_index(target)
        frame_ts, _ = player.frame_at(idx)
        assert frame_ts == actual_ts

    def test_seek_empty_recording_raises(self, tmp_path: Path) -> None:
        """Seeking in an empty recording raises ValueError."""
        recorder = ShowRecorder(audio_hash=_fake_audio_hash(), fps=60, fixture_count=3)
        out_path = tmp_path / "empty.lrec"
        recorder.save(out_path)

        player = ShowPlayer(out_path)
        with pytest.raises(ValueError, match="empty recording"):
            player.seek(1000)


# ─── frame_at tests ───────────────────────────────────────────────────────────


class TestFrameAt:
    """Test frame_at random access."""

    def test_first_frame(self, tmp_path: Path) -> None:
        """First frame (index 0) is retrievable."""
        recorder = ShowRecorder(audio_hash=_fake_audio_hash(), fps=60, fixture_count=2)
        cmds = _make_frame(2, 0)
        recorder.record_frame(0, cmds)
        recorder.record_frame(16, _make_frame(2, 1))
        out_path = tmp_path / "show.lrec"
        recorder.save(out_path)

        player = ShowPlayer(out_path)
        ts, result = player.frame_at(0)
        assert ts == 0
        assert result == cmds

    def test_last_frame(self, tmp_path: Path) -> None:
        """Last frame is retrievable by index frame_count - 1."""
        n = 50
        fixture_count = 4
        recorder = ShowRecorder(audio_hash=_fake_audio_hash(), fps=60, fixture_count=fixture_count)
        last_cmds = _make_frame(fixture_count, n - 1)
        for i in range(n):
            cmds = _make_frame(fixture_count, i)
            recorder.record_frame(i * 16, cmds)
        out_path = tmp_path / "show.lrec"
        recorder.save(out_path)

        player = ShowPlayer(out_path)
        ts, result = player.frame_at(n - 1)
        assert ts == (n - 1) * 16
        assert result == last_cmds

    def test_negative_index_raises(self, tmp_path: Path) -> None:
        """Negative index raises IndexError."""
        recorder = ShowRecorder(audio_hash=_fake_audio_hash(), fps=60, fixture_count=2)
        recorder.record_frame(0, _make_frame(2, 0))
        out_path = tmp_path / "show.lrec"
        recorder.save(out_path)

        player = ShowPlayer(out_path)
        with pytest.raises(IndexError):
            player.frame_at(-1)

    def test_out_of_range_index_raises(self, tmp_path: Path) -> None:
        """Index >= frame_count raises IndexError."""
        recorder = ShowRecorder(audio_hash=_fake_audio_hash(), fps=60, fixture_count=2)
        recorder.record_frame(0, _make_frame(2, 0))
        out_path = tmp_path / "show.lrec"
        recorder.save(out_path)

        player = ShowPlayer(out_path)
        with pytest.raises(IndexError):
            player.frame_at(1)  # Only frame 0 exists.

    def test_random_access_consistency(self, tmp_path: Path) -> None:
        """Random-access reads are consistent with sequential reads."""
        import random

        random.seed(42)
        n = 120
        fixture_count = 6
        recorder = ShowRecorder(audio_hash=_fake_audio_hash(), fps=60, fixture_count=fixture_count)
        frames_in = []
        for i in range(n):
            cmds = _make_frame(fixture_count, i)
            recorder.record_frame(i * 16, cmds)
            frames_in.append((i * 16, cmds))
        out_path = tmp_path / "show.lrec"
        recorder.save(out_path)

        player = ShowPlayer(out_path)
        indices = random.sample(range(n), 30)
        for idx in indices:
            ts, cmds = player.frame_at(idx)
            assert ts == frames_in[idx][0]
            assert cmds == frames_in[idx][1]


# ─── verify_audio tests ───────────────────────────────────────────────────────


class TestVerifyAudio:
    """Test verify_audio() with matching and non-matching hashes."""

    def test_matching_hash_returns_true(self, tmp_path: Path) -> None:
        """verify_audio returns True when the file matches the stored hash."""
        audio_file = tmp_path / "audio.mp3"
        audio_file.write_bytes(b"fake mp3 content for testing")

        audio_hash = hash_audio_file(audio_file)
        recorder = ShowRecorder(audio_hash=audio_hash, fps=60, fixture_count=2)
        recorder.record_frame(0, _make_frame(2, 0))
        out_path = tmp_path / "show.lrec"
        recorder.save(out_path)

        player = ShowPlayer(out_path)
        assert player.verify_audio(audio_file) is True

    def test_non_matching_hash_returns_false(self, tmp_path: Path) -> None:
        """verify_audio returns False when the file has a different hash."""
        audio_file = tmp_path / "audio.mp3"
        audio_file.write_bytes(b"original content")

        audio_hash = hash_audio_file(audio_file)
        recorder = ShowRecorder(audio_hash=audio_hash, fps=60, fixture_count=2)
        recorder.record_frame(0, _make_frame(2, 0))
        out_path = tmp_path / "show.lrec"
        recorder.save(out_path)

        # Overwrite the audio file with different content.
        audio_file.write_bytes(b"completely different content")

        player = ShowPlayer(out_path)
        assert player.verify_audio(audio_file) is False

    def test_missing_audio_raises(self, tmp_path: Path) -> None:
        """verify_audio raises FileNotFoundError if the audio file doesn't exist."""
        recorder = ShowRecorder(audio_hash=_fake_audio_hash(), fps=60, fixture_count=2)
        recorder.record_frame(0, _make_frame(2, 0))
        out_path = tmp_path / "show.lrec"
        recorder.save(out_path)

        player = ShowPlayer(out_path)
        with pytest.raises(FileNotFoundError):
            player.verify_audio(tmp_path / "nonexistent.mp3")

    def test_hash_audio_file_deterministic(self, tmp_path: Path) -> None:
        """hash_audio_file returns the same hash for identical content."""
        audio_file = tmp_path / "audio.flac"
        audio_file.write_bytes(b"deterministic content" * 1000)

        h1 = hash_audio_file(audio_file)
        h2 = hash_audio_file(audio_file)
        assert h1 == h2
        assert len(h1) == 32

    def test_hash_audio_file_differs_for_different_content(self, tmp_path: Path) -> None:
        """hash_audio_file returns different hashes for different content."""
        f1 = tmp_path / "a.mp3"
        f2 = tmp_path / "b.mp3"
        f1.write_bytes(b"content A")
        f2.write_bytes(b"content B")

        assert hash_audio_file(f1) != hash_audio_file(f2)


# ─── Edge cases ───────────────────────────────────────────────────────────────


class TestEdgeCases:
    """Empty recordings, single frames, large recordings, and errors."""

    def test_empty_recording_save_and_load(self, tmp_path: Path) -> None:
        """Empty recording (0 frames) saves and loads cleanly."""
        recorder = ShowRecorder(audio_hash=_fake_audio_hash(), fps=60, fixture_count=5)
        out_path = tmp_path / "empty.lrec"
        recorder.save(out_path)

        player = ShowPlayer(out_path)
        assert player.frame_count == 0
        assert player.duration_ms == 0
        assert list(player.frames()) == []

    def test_single_frame_recording(self, tmp_path: Path) -> None:
        """A recording with exactly one frame works correctly."""
        audio_hash = _fake_audio_hash()
        recorder = ShowRecorder(audio_hash=audio_hash, fps=60, fixture_count=1)
        cmd = FixtureCommand(5, 100, 150, 200, 50, 30, 40, 255)
        recorder.record_frame(500, [cmd])

        out_path = tmp_path / "single.lrec"
        recorder.save(out_path)

        player = ShowPlayer(out_path)
        assert player.frame_count == 1
        ts, cmds = player.frame_at(0)
        assert ts == 500
        assert cmds == [cmd]

    def test_large_recording(self, tmp_path: Path) -> None:
        """1000-frame recording encodes and decodes without error."""
        n = 1000
        fixture_count = 15
        recorder = ShowRecorder(
            audio_hash=_fake_audio_hash(), fps=60, fixture_count=fixture_count
        )
        for i in range(n):
            recorder.record_frame(i * 16, _make_frame(fixture_count, i))
        out_path = tmp_path / "large.lrec"
        recorder.save(out_path)

        player = ShowPlayer(out_path)
        assert player.frame_count == n

        # Spot-check first, last, and middle frames.
        for idx in (0, n // 2, n - 1):
            ts, cmds = player.frame_at(idx)
            assert ts == idx * 16
            assert cmds == _make_frame(fixture_count, idx)

    def test_frame_count_property_updates(self) -> None:
        """recorder.frame_count increments on each record_frame call."""
        recorder = ShowRecorder(audio_hash=_fake_audio_hash(), fps=60, fixture_count=2)
        assert recorder.frame_count == 0
        for i in range(5):
            recorder.record_frame(i * 16, _make_frame(2, i))
            assert recorder.frame_count == i + 1

    def test_duration_ms_property_updates(self) -> None:
        """recorder.duration_ms reflects the latest recorded timestamp."""
        recorder = ShowRecorder(audio_hash=_fake_audio_hash(), fps=60, fixture_count=2)
        assert recorder.duration_ms == 0
        recorder.record_frame(16, _make_frame(2, 0))
        assert recorder.duration_ms == 16
        recorder.record_frame(32, _make_frame(2, 1))
        assert recorder.duration_ms == 32


# ─── Invalid file detection ───────────────────────────────────────────────────


class TestInvalidFiles:
    """Bad magic, wrong version, truncated data, missing file."""

    def test_bad_magic_raises(self, tmp_path: Path) -> None:
        """File with wrong magic raises ValueError."""
        bad_file = tmp_path / "bad.lrec"
        # Write a header with wrong magic.
        data = struct.pack(
            HEADER_FORMAT,
            b"XXXX",      # bad magic
            LREC_VERSION,
            _fake_audio_hash(),
            60,
            15,
            0,
            0,
        )
        bad_file.write_bytes(data)

        with pytest.raises(ValueError, match="magic"):
            ShowPlayer(bad_file)

    def test_wrong_version_raises(self, tmp_path: Path) -> None:
        """File with unsupported version raises ValueError."""
        bad_file = tmp_path / "v99.lrec"
        data = struct.pack(
            HEADER_FORMAT,
            LREC_MAGIC,
            99,            # unsupported version
            _fake_audio_hash(),
            60,
            15,
            0,
            0,
        )
        bad_file.write_bytes(data)

        with pytest.raises(ValueError, match="[Vv]ersion"):
            ShowPlayer(bad_file)

    def test_truncated_file_raises(self, tmp_path: Path) -> None:
        """File shorter than the header raises ValueError."""
        bad_file = tmp_path / "truncated.lrec"
        bad_file.write_bytes(b"LREC")  # Only 4 bytes.

        with pytest.raises(ValueError):
            ShowPlayer(bad_file)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        """Loading a non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            ShowPlayer(tmp_path / "does_not_exist.lrec")

    def test_payload_too_short_raises(self, tmp_path: Path) -> None:
        """Header claims frames exist but payload is empty — raises ValueError."""
        bad_file = tmp_path / "short_payload.lrec"
        # Header says 10 frames, but we write no frame data.
        data = struct.pack(
            HEADER_FORMAT,
            LREC_MAGIC,
            LREC_VERSION,
            _fake_audio_hash(),
            60,
            3,   # fixture_count = 3
            160, # duration_ms
            10,  # frame_count = 10 (but no payload follows)
        )
        bad_file.write_bytes(data)

        with pytest.raises(ValueError, match="[Pp]ayload|short"):
            ShowPlayer(bad_file)


# ─── Constructor validation tests ─────────────────────────────────────────────


class TestConstructorValidation:
    """ShowRecorder validates its constructor arguments."""

    def test_wrong_hash_length_raises(self) -> None:
        """audio_hash that is not 32 bytes raises ValueError."""
        with pytest.raises(ValueError, match="32 bytes"):
            ShowRecorder(audio_hash=b"too short")

    def test_zero_fps_raises(self) -> None:
        """fps=0 raises ValueError."""
        with pytest.raises(ValueError, match="fps"):
            ShowRecorder(audio_hash=_fake_audio_hash(), fps=0)

    def test_zero_fixture_count_raises(self) -> None:
        """fixture_count=0 raises ValueError."""
        with pytest.raises(ValueError, match="fixture_count"):
            ShowRecorder(audio_hash=_fake_audio_hash(), fps=60, fixture_count=0)

    def test_invalid_timestamp_raises(self) -> None:
        """Negative timestamp raises ValueError."""
        recorder = ShowRecorder(audio_hash=_fake_audio_hash(), fps=60, fixture_count=2)
        with pytest.raises(ValueError):
            recorder.record_frame(-1, _make_frame(2, 0))

    def test_timestamp_overflow_raises(self) -> None:
        """timestamp_ms > 0xFFFFFFFF raises ValueError."""
        recorder = ShowRecorder(audio_hash=_fake_audio_hash(), fps=60, fixture_count=2)
        with pytest.raises(ValueError):
            recorder.record_frame(0xFFFFFFFF + 1, _make_frame(2, 0))


# ─── Padding behaviour ────────────────────────────────────────────────────────


class TestPaddingBehaviour:
    """Commands shorter than fixture_count are zero-padded."""

    def test_too_few_commands_zero_padded(self, tmp_path: Path) -> None:
        """Providing fewer commands than fixture_count pads with zeroes."""
        fixture_count = 4
        recorder = ShowRecorder(
            audio_hash=_fake_audio_hash(), fps=60, fixture_count=fixture_count
        )
        # Provide only 2 of the 4 expected commands.
        recorder.record_frame(0, _make_frame(2, 0))

        out_path = tmp_path / "padded.lrec"
        recorder.save(out_path)

        player = ShowPlayer(out_path)
        _, cmds = player.frame_at(0)
        assert len(cmds) == fixture_count
        # Last two commands should be all-zero.
        zero_cmd = FixtureCommand(0, 0, 0, 0, 0, 0, 0, 0)
        assert cmds[2] == zero_cmd
        assert cmds[3] == zero_cmd

    def test_extra_commands_truncated(self, tmp_path: Path) -> None:
        """Providing more commands than fixture_count silently truncates."""
        fixture_count = 2
        recorder = ShowRecorder(
            audio_hash=_fake_audio_hash(), fps=60, fixture_count=fixture_count
        )
        # Provide 5 commands; only first 2 should be stored.
        all_cmds = _make_frame(5, 0)
        recorder.record_frame(0, all_cmds)

        out_path = tmp_path / "truncated_cmds.lrec"
        recorder.save(out_path)

        player = ShowPlayer(out_path)
        _, cmds = player.frame_at(0)
        assert len(cmds) == fixture_count
        assert cmds[0] == all_cmds[0]
        assert cmds[1] == all_cmds[1]


# ─── Recorder property tests ──────────────────────────────────────────────────


class TestRecorderProperties:
    """Verify recorder exposes correct metadata properties."""

    def test_audio_hash_property(self) -> None:
        h = _fake_audio_hash()
        recorder = ShowRecorder(audio_hash=h, fps=60, fixture_count=15)
        assert recorder.audio_hash == h

    def test_fps_property(self) -> None:
        recorder = ShowRecorder(audio_hash=_fake_audio_hash(), fps=30, fixture_count=5)
        assert recorder.fps == 30

    def test_fixture_count_property(self) -> None:
        recorder = ShowRecorder(audio_hash=_fake_audio_hash(), fps=60, fixture_count=8)
        assert recorder.fixture_count == 8
