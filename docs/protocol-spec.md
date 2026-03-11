# LUMINA UDP Fixture Control Protocol Specification

Version 1.0

## Overview

LUMINA uses a custom lightweight UDP protocol for fixture control, designed for minimum
latency (<1ms) and simplicity. This is intentionally NOT DMX or ArtNet — those protocols
carry unnecessary overhead for this use case. ArtNet compatibility can be added as a bridge
layer later.

## Protocol Constants

| Constant | Value | Description |
|----------|-------|-------------|
| Magic | `0x4C55` | "LU" in ASCII, little-endian |
| Version | `1` | Protocol version |
| Port | `5150` | UDP port for all fixture communication |
| Max fixtures/packet | `32` | Maximum fixture commands per packet |
| Max packet size | `265` bytes | 9-byte header + 32 x 8-byte commands |
| Target refresh rate | `60 fps` | 16.7ms interval between command packets |

## Packet Types

| Type | Value | Direction | Description |
|------|-------|-----------|-------------|
| COMMAND | `0x01` | Host -> Fixture | Lighting commands |
| DISCOVER_REQ | `0x10` | Host -> Broadcast | Fixture discovery request |
| DISCOVER_RESP | `0x11` | Fixture -> Host | Fixture discovery response |
| HEARTBEAT | `0x20` | Bidirectional | Keep-alive / health check |
| CONFIG | `0x30` | Host -> Fixture | Configuration update |

## Packet Format

### Header (9 bytes, little-endian)

```
Offset  Size    Field           Description
0       2       magic           0x4C55 ("LU")
2       1       version         Protocol version (1)
3       1       packet_type     See packet types table
4       2       sequence        Packet sequence number (wraps at 65535)
6       2       timestamp_ms    Millisecond timestamp (wraps at 65535)
8       1       fixture_count   Number of fixture commands (0-32)
```

### Payload (fixture_count x 8 bytes each)

Each fixture command is exactly 8 bytes:

```
Offset  Size    Field               Range       Description
0       1       fixture_id          1-255       Target fixture (0 = broadcast)
1       1       red                 0-255       Red channel
2       1       green               0-255       Green channel
3       1       blue                0-255       Blue channel
4       1       white               0-255       White channel
5       1       strobe_rate         0-255       Strobe frequency (0=off, 255=~25Hz)
6       1       strobe_intensity    0-255       Strobe flash brightness
7       1       special             0-255       Fixture-type-specific channel
```

## Fixture Type Interpretation

All fixture types receive the same 8-byte command. Each type interprets fields differently:

| Fixture Type | R,G,B,W | strobe_rate | strobe_intensity | special |
|-------------|---------|-------------|-----------------|---------|
| RGB Strobe | Base color | 0-255 (off to ~25Hz) | Flash brightness | Unused (0) |
| RGBW Par | Color wash | Ignored (0) | Ignored (0) | Master dimmer (0-255) |
| UV Bar | Ignored | Ignored | Ignored | UV intensity (0-255) |
| Laser Module | Ignored | Ignored | Ignored | Pattern ID (0-255) |
| Moving Head | Color output | Ignored | Ignored | Gobo ID (pan/tilt via extended packet) |

## Broadcast vs Unicast

- **fixture_id = 0**: Broadcast — all fixtures apply the command
- **fixture_id = 1-255**: Unicast — only the addressed fixture responds

## Sequence Number

The `sequence` field increments with each packet and wraps at 65535. Fixtures use this to:
- Detect dropped packets (gaps in sequence)
- Ignore duplicate packets (same sequence number)
- Maintain ordering when packets arrive out of order

## Discovery Protocol

1. Host sends `DISCOVER_REQ` (broadcast to port 5150)
2. Each fixture responds with `DISCOVER_RESP` containing its fixture_id and type
3. Host builds fixture map from responses
4. Discovery runs on startup and periodically (every 30 seconds)

## Heartbeat

- Host sends `HEARTBEAT` every 5 seconds to all fixtures
- Fixtures respond with `HEARTBEAT` within 1 second
- If no heartbeat response after 3 attempts, fixture is marked offline
- Fixtures that receive no commands or heartbeats for 10 seconds enter safe mode (all outputs off)

## Implementation

Reference implementation: `lumina/control/protocol.py`

```python
# Encoding example
from lumina.control.protocol import encode_command_packet, FixtureCommand

commands = [
    FixtureCommand(fixture_id=1, red=255, green=0, blue=0, white=0,
                   strobe_rate=0, strobe_intensity=0, special=200),
]
packet = encode_command_packet(commands, sequence=42)
# Send packet via UDP to port 5150
```
