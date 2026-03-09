"""Tests for the LUMINA fixture control protocol."""

from __future__ import annotations

import struct

import pytest

from lumina.control.protocol import (
    FIXTURE_SIZE,
    HEADER_SIZE,
    MAX_FIXTURES_PER_PACKET,
    PROTOCOL_MAGIC,
    PROTOCOL_PORT,
    PROTOCOL_VERSION,
    FixtureCommand,
    PacketType,
    decode_packet,
    encode_packet,
)


# ─── Protocol constants ──────────────────────────────────────────────


class TestConstants:
    def test_header_size_is_9(self) -> None:
        assert HEADER_SIZE == 9

    def test_fixture_size_is_8(self) -> None:
        assert FIXTURE_SIZE == 8

    def test_magic_is_lu(self) -> None:
        # "LU" in little-endian uint16
        assert PROTOCOL_MAGIC == 0x4C55

    def test_version_is_1(self) -> None:
        assert PROTOCOL_VERSION == 1

    def test_port_defined(self) -> None:
        assert PROTOCOL_PORT == 5150

    def test_max_fixtures_per_packet(self) -> None:
        assert MAX_FIXTURES_PER_PACKET == 32


# ─── FixtureCommand ──────────────────────────────────────────────────


class TestFixtureCommand:
    def test_default_values(self) -> None:
        cmd = FixtureCommand()
        assert cmd.fixture_id == 0
        assert cmd.red == 0
        assert cmd.green == 0
        assert cmd.blue == 0
        assert cmd.white == 0
        assert cmd.strobe_rate == 0
        assert cmd.strobe_intensity == 0
        assert cmd.special == 0

    def test_all_fields(self) -> None:
        cmd = FixtureCommand(
            fixture_id=1,
            red=255,
            green=128,
            blue=64,
            white=32,
            strobe_rate=200,
            strobe_intensity=180,
            special=100,
        )
        assert cmd.fixture_id == 1
        assert cmd.red == 255
        assert cmd.green == 128
        assert cmd.blue == 64
        assert cmd.white == 32
        assert cmd.strobe_rate == 200
        assert cmd.strobe_intensity == 180
        assert cmd.special == 100

    @pytest.mark.parametrize("field", [
        "fixture_id", "red", "green", "blue", "white",
        "strobe_rate", "strobe_intensity", "special",
    ])
    def test_rejects_negative_values(self, field: str) -> None:
        with pytest.raises(ValueError, match=f"{field} must be 0-255"):
            FixtureCommand(**{field: -1})

    @pytest.mark.parametrize("field", [
        "fixture_id", "red", "green", "blue", "white",
        "strobe_rate", "strobe_intensity", "special",
    ])
    def test_rejects_values_over_255(self, field: str) -> None:
        with pytest.raises(ValueError, match=f"{field} must be 0-255"):
            FixtureCommand(**{field: 256})

    def test_boundary_values_accepted(self) -> None:
        cmd = FixtureCommand(
            fixture_id=0, red=0, green=0, blue=0,
            white=0, strobe_rate=0, strobe_intensity=0, special=0,
        )
        assert cmd.fixture_id == 0

        cmd = FixtureCommand(
            fixture_id=255, red=255, green=255, blue=255,
            white=255, strobe_rate=255, strobe_intensity=255, special=255,
        )
        assert cmd.fixture_id == 255

    def test_to_bytes_length(self) -> None:
        cmd = FixtureCommand(fixture_id=1, red=255)
        assert len(cmd.to_bytes()) == FIXTURE_SIZE

    def test_to_bytes_content(self) -> None:
        cmd = FixtureCommand(
            fixture_id=42, red=10, green=20, blue=30,
            white=40, strobe_rate=50, strobe_intensity=60, special=70,
        )
        raw = cmd.to_bytes()
        assert raw == bytes([42, 10, 20, 30, 40, 50, 60, 70])

    def test_from_bytes_roundtrip(self) -> None:
        original = FixtureCommand(
            fixture_id=7, red=100, green=200, blue=50,
            white=150, strobe_rate=30, strobe_intensity=90, special=12,
        )
        restored = FixtureCommand.from_bytes(original.to_bytes())
        assert restored == original

    def test_from_bytes_wrong_length(self) -> None:
        with pytest.raises(ValueError, match="Expected 8 bytes"):
            FixtureCommand.from_bytes(b"\x00" * 7)

        with pytest.raises(ValueError, match="Expected 8 bytes"):
            FixtureCommand.from_bytes(b"\x00" * 9)


# ─── Packet encode / decode ──────────────────────────────────────────


class TestEncodePacket:
    def test_empty_packet(self) -> None:
        raw = encode_packet([], sequence=0, timestamp_ms=0)
        assert len(raw) == HEADER_SIZE

    def test_single_fixture(self) -> None:
        cmd = FixtureCommand(fixture_id=1, red=255)
        raw = encode_packet([cmd], sequence=1, timestamp_ms=100)
        assert len(raw) == HEADER_SIZE + FIXTURE_SIZE

    def test_max_fixtures(self) -> None:
        cmds = [FixtureCommand(fixture_id=i + 1) for i in range(MAX_FIXTURES_PER_PACKET)]
        raw = encode_packet(cmds)
        assert len(raw) == HEADER_SIZE + MAX_FIXTURES_PER_PACKET * FIXTURE_SIZE

    def test_exceeds_max_fixtures(self) -> None:
        cmds = [FixtureCommand(fixture_id=1)] * (MAX_FIXTURES_PER_PACKET + 1)
        with pytest.raises(ValueError, match="Max 32"):
            encode_packet(cmds)

    def test_header_magic_and_version(self) -> None:
        raw = encode_packet([], sequence=0, timestamp_ms=0)
        magic, version = struct.unpack_from("<HB", raw, 0)
        assert magic == PROTOCOL_MAGIC
        assert version == PROTOCOL_VERSION

    def test_packet_type_in_header(self) -> None:
        raw = encode_packet([], packet_type=PacketType.HEARTBEAT)
        ptype = raw[3]
        assert ptype == PacketType.HEARTBEAT

    def test_sequence_wraps(self) -> None:
        raw = encode_packet([], sequence=70000)
        _, _, _, seq, _, _ = struct.unpack_from("<HBBHHB", raw, 0)
        assert seq == 70000 & 0xFFFF

    def test_timestamp_wraps(self) -> None:
        raw = encode_packet([], timestamp_ms=100000)
        _, _, _, _, ts, _ = struct.unpack_from("<HBBHHB", raw, 0)
        assert ts == 100000 & 0xFFFF


class TestDecodePacket:
    def test_roundtrip_empty(self) -> None:
        raw = encode_packet([], sequence=42, timestamp_ms=1234)
        ptype, seq, ts, cmds = decode_packet(raw)
        assert ptype == PacketType.COMMAND
        assert seq == 42
        assert ts == 1234
        assert cmds == []

    def test_roundtrip_single_fixture(self) -> None:
        original = FixtureCommand(
            fixture_id=5, red=100, green=200, blue=50,
            white=150, strobe_rate=30, strobe_intensity=90, special=12,
        )
        raw = encode_packet([original], sequence=99, timestamp_ms=5000)
        ptype, seq, ts, cmds = decode_packet(raw)
        assert ptype == PacketType.COMMAND
        assert seq == 99
        assert ts == 5000
        assert len(cmds) == 1
        assert cmds[0] == original

    def test_roundtrip_multiple_fixtures(self) -> None:
        originals = [
            FixtureCommand(fixture_id=1, red=255, green=0, blue=0),
            FixtureCommand(fixture_id=2, red=0, green=255, blue=0),
            FixtureCommand(fixture_id=3, red=0, green=0, blue=255),
        ]
        raw = encode_packet(originals, sequence=7, timestamp_ms=999)
        ptype, seq, ts, cmds = decode_packet(raw)
        assert len(cmds) == 3
        for orig, decoded in zip(originals, cmds):
            assert orig == decoded

    def test_roundtrip_max_fixtures(self) -> None:
        originals = [
            FixtureCommand(
                fixture_id=i + 1,
                red=i * 8 % 256,
                green=(i * 13) % 256,
                blue=(i * 17) % 256,
                white=(i * 23) % 256,
                strobe_rate=(i * 7) % 256,
                strobe_intensity=(i * 11) % 256,
                special=(i * 3) % 256,
            )
            for i in range(MAX_FIXTURES_PER_PACKET)
        ]
        raw = encode_packet(originals, sequence=65535, timestamp_ms=65535)
        ptype, seq, ts, cmds = decode_packet(raw)
        assert seq == 65535
        assert ts == 65535
        assert len(cmds) == MAX_FIXTURES_PER_PACKET
        for orig, decoded in zip(originals, cmds):
            assert orig == decoded

    def test_roundtrip_all_packet_types(self) -> None:
        for pt in PacketType:
            raw = encode_packet([], packet_type=pt, sequence=1)
            ptype, _, _, _ = decode_packet(raw)
            assert ptype == pt

    def test_roundtrip_broadcast_fixture(self) -> None:
        cmd = FixtureCommand(fixture_id=0, red=128, green=128, blue=128, white=128)
        raw = encode_packet([cmd])
        _, _, _, cmds = decode_packet(raw)
        assert cmds[0].fixture_id == 0

    def test_roundtrip_all_max_values(self) -> None:
        cmd = FixtureCommand(
            fixture_id=255, red=255, green=255, blue=255,
            white=255, strobe_rate=255, strobe_intensity=255, special=255,
        )
        raw = encode_packet([cmd])
        _, _, _, cmds = decode_packet(raw)
        assert cmds[0] == cmd

    def test_roundtrip_all_zero_values(self) -> None:
        cmd = FixtureCommand()
        raw = encode_packet([cmd])
        _, _, _, cmds = decode_packet(raw)
        assert cmds[0] == cmd


class TestDecodeErrors:
    def test_packet_too_short(self) -> None:
        with pytest.raises(ValueError, match="Packet too short"):
            decode_packet(b"\x00" * (HEADER_SIZE - 1))

    def test_bad_magic(self) -> None:
        # Build a valid packet then corrupt the magic bytes
        raw = bytearray(encode_packet([]))
        raw[0] = 0xFF
        raw[1] = 0xFF
        with pytest.raises(ValueError, match="Invalid magic"):
            decode_packet(bytes(raw))

    def test_bad_version(self) -> None:
        raw = bytearray(encode_packet([]))
        raw[2] = 99  # Wrong version
        with pytest.raises(ValueError, match="Unsupported protocol version"):
            decode_packet(bytes(raw))

    def test_truncated_payload(self) -> None:
        cmd = FixtureCommand(fixture_id=1, red=255)
        raw = encode_packet([cmd])
        # Chop off last byte of the fixture data
        with pytest.raises(ValueError, match="Packet too short for"):
            decode_packet(raw[:-1])

    def test_fixture_count_exceeds_max(self) -> None:
        raw = bytearray(encode_packet([]))
        raw[8] = MAX_FIXTURES_PER_PACKET + 1  # count byte
        with pytest.raises(ValueError, match="exceeds max"):
            decode_packet(bytes(raw))

    def test_extra_trailing_bytes_ignored(self) -> None:
        """Extra data after valid payload should be silently ignored."""
        cmd = FixtureCommand(fixture_id=1, red=128)
        raw = encode_packet([cmd]) + b"\xDE\xAD"
        ptype, seq, ts, cmds = decode_packet(raw)
        assert len(cmds) == 1
        assert cmds[0] == cmd
