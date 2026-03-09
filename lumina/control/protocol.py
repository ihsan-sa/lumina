"""LUMINA fixture control protocol — UDP packet encoding and decoding.

Packet format (little-endian):
    Header (9 bytes):
        - magic:         2 bytes  (0x4C55 = "LU")
        - version:       1 byte   (protocol version, currently 1)
        - packet_type:   1 byte   (see PacketType enum)
        - sequence:      2 bytes  (uint16, wraps at 65535)
        - timestamp_ms:  2 bytes  (uint16, wraps at 65535 — milliseconds mod 65536)
        - fixture_count: 1 byte   (number of fixture commands, 0-32)

    Payload (fixture_count × 8 bytes each):
        - fixture_id:       1 byte (1-255 unicast, 0 = broadcast)
        - red:              1 byte (0-255)
        - green:            1 byte (0-255)
        - blue:             1 byte (0-255)
        - white:            1 byte (0-255)
        - strobe_rate:      1 byte (0 = off, 255 = max ~25Hz)
        - strobe_intensity: 1 byte (0-255)
        - special:          1 byte (fixture-type-specific)

    Max packet size: 9 + 32×8 = 265 bytes.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from enum import IntEnum

# ─── Protocol constants ──────────────────────────────────────────────

PROTOCOL_MAGIC = 0x4C55  # "LU" in ASCII
PROTOCOL_VERSION = 1
PROTOCOL_PORT = 5150
MAX_FIXTURES_PER_PACKET = 32

HEADER_FORMAT = "<HBBHHB"  # magic(H) version(B) type(B) seq(H) ts(H) count(B)
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)  # 9 bytes
FIXTURE_FORMAT = "<8B"  # 8 unsigned bytes
FIXTURE_SIZE = struct.calcsize(FIXTURE_FORMAT)  # 8 bytes


class PacketType(IntEnum):
    """Packet type identifiers."""

    COMMAND = 0x01  # Fixture command (normal frame)
    DISCOVER_REQUEST = 0x10  # Host → broadcast: discover fixtures
    DISCOVER_RESPONSE = 0x11  # Fixture → host: announce presence
    HEARTBEAT = 0x20  # Host → fixtures: keepalive
    CONFIG = 0x30  # Host → fixture: configuration update


# ─── FixtureCommand ──────────────────────────────────────────────────


@dataclass(slots=True)
class FixtureCommand:
    """Command for a single fixture.

    All fixture types receive the same 8-byte payload. Each type
    interprets the fields according to its capabilities (see CLAUDE.md
    fixture type table).

    Args:
        fixture_id: Target fixture (1-255 unicast, 0 = broadcast).
        red: Red channel (0-255).
        green: Green channel (0-255).
        blue: Blue channel (0-255).
        white: White channel (0-255).
        strobe_rate: Strobe frequency (0 = off, 255 = max ~25Hz).
        strobe_intensity: Strobe flash brightness (0-255).
        special: Fixture-type-specific byte.
    """

    fixture_id: int = 0
    red: int = 0
    green: int = 0
    blue: int = 0
    white: int = 0
    strobe_rate: int = 0
    strobe_intensity: int = 0
    special: int = 0

    def __post_init__(self) -> None:
        for name in (
            "fixture_id",
            "red",
            "green",
            "blue",
            "white",
            "strobe_rate",
            "strobe_intensity",
            "special",
        ):
            val = getattr(self, name)
            if not (0 <= val <= 255):
                msg = f"{name} must be 0-255, got {val}"
                raise ValueError(msg)

    def to_bytes(self) -> bytes:
        """Serialize this command to 8 bytes."""
        return struct.pack(
            FIXTURE_FORMAT,
            self.fixture_id,
            self.red,
            self.green,
            self.blue,
            self.white,
            self.strobe_rate,
            self.strobe_intensity,
            self.special,
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> FixtureCommand:
        """Deserialize a FixtureCommand from 8 bytes.

        Args:
            data: Exactly 8 bytes of fixture command data.

        Returns:
            Parsed FixtureCommand instance.

        Raises:
            ValueError: If data is not exactly 8 bytes.
        """
        if len(data) != FIXTURE_SIZE:
            msg = f"Expected {FIXTURE_SIZE} bytes, got {len(data)}"
            raise ValueError(msg)
        fields = struct.unpack(FIXTURE_FORMAT, data)
        return cls(
            fixture_id=fields[0],
            red=fields[1],
            green=fields[2],
            blue=fields[3],
            white=fields[4],
            strobe_rate=fields[5],
            strobe_intensity=fields[6],
            special=fields[7],
        )


# ─── Packet-level encode / decode ────────────────────────────────────


def encode_packet(
    commands: list[FixtureCommand],
    sequence: int = 0,
    timestamp_ms: int = 0,
    packet_type: PacketType = PacketType.COMMAND,
) -> bytes:
    """Encode a list of fixture commands into a single UDP packet.

    Args:
        commands: Up to 32 fixture commands to include.
        sequence: Packet sequence number (uint16, wraps).
        timestamp_ms: Timestamp in milliseconds (uint16, wraps).
        packet_type: Type of packet (default: COMMAND).

    Returns:
        Raw bytes ready to send over UDP.

    Raises:
        ValueError: If more than 32 commands are provided.
    """
    if len(commands) > MAX_FIXTURES_PER_PACKET:
        msg = f"Max {MAX_FIXTURES_PER_PACKET} fixtures per packet, got {len(commands)}"
        raise ValueError(msg)

    header = struct.pack(
        HEADER_FORMAT,
        PROTOCOL_MAGIC,
        PROTOCOL_VERSION,
        int(packet_type),
        sequence & 0xFFFF,
        timestamp_ms & 0xFFFF,
        len(commands),
    )

    payload = b"".join(cmd.to_bytes() for cmd in commands)
    return header + payload


def decode_packet(
    data: bytes,
) -> tuple[PacketType, int, int, list[FixtureCommand]]:
    """Decode a raw UDP packet into its components.

    Args:
        data: Raw bytes received from UDP socket.

    Returns:
        Tuple of (packet_type, sequence, timestamp_ms, commands).

    Raises:
        ValueError: If packet is malformed (bad magic, version, or size).
    """
    if len(data) < HEADER_SIZE:
        msg = f"Packet too short: {len(data)} bytes (need at least {HEADER_SIZE})"
        raise ValueError(msg)

    magic, version, ptype, sequence, timestamp_ms, count = struct.unpack(
        HEADER_FORMAT, data[:HEADER_SIZE]
    )

    if magic != PROTOCOL_MAGIC:
        msg = f"Invalid magic: 0x{magic:04X} (expected 0x{PROTOCOL_MAGIC:04X})"
        raise ValueError(msg)

    if version != PROTOCOL_VERSION:
        msg = f"Unsupported protocol version: {version} (expected {PROTOCOL_VERSION})"
        raise ValueError(msg)

    if count > MAX_FIXTURES_PER_PACKET:
        msg = f"Fixture count {count} exceeds max {MAX_FIXTURES_PER_PACKET}"
        raise ValueError(msg)

    expected_size = HEADER_SIZE + count * FIXTURE_SIZE
    if len(data) < expected_size:
        msg = f"Packet too short for {count} fixtures: {len(data)} bytes (need {expected_size})"
        raise ValueError(msg)

    packet_type = PacketType(ptype)
    commands: list[FixtureCommand] = []
    for i in range(count):
        offset = HEADER_SIZE + i * FIXTURE_SIZE
        cmd = FixtureCommand.from_bytes(data[offset : offset + FIXTURE_SIZE])
        commands.append(cmd)

    return packet_type, sequence, timestamp_ms, commands
