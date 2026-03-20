"""LUMINA fixture control — protocol encoding and network management."""

from lumina.control.discovery import DiscoveryService, run_discovery
from lumina.control.fixture import FixtureRegistry, FixtureState
from lumina.control.network import NetworkManager, NetworkStats
from lumina.control.protocol import (
    FixtureCommand,
    PacketType,
    decode_packet,
    encode_packet,
)

__all__ = [
    "DiscoveryService",
    "FixtureCommand",
    "FixtureRegistry",
    "FixtureState",
    "NetworkManager",
    "NetworkStats",
    "PacketType",
    "decode_packet",
    "encode_packet",
    "run_discovery",
]
