"""Send test commands to fixtures via UDP for hardware testing.

Sends LUMINA protocol packets to a target fixture or broadcast address.
Useful for verifying fixture connectivity, color output, and strobe behavior.

Usage:
    python scripts/fixture_tester.py --target 192.168.1.100 --fixture-id 1 --color 255 0 0
    python scripts/fixture_tester.py --target 192.168.1.255 --fixture-id 0 --strobe 200 255
    python scripts/fixture_tester.py --target 192.168.1.100 --fixture-id 1 --cycle
"""

import argparse
import logging
import socket
import struct
import sys
import time

logger = logging.getLogger(__name__)

# Protocol constants (must match lumina/control/protocol.py)
MAGIC = 0x4C55
VERSION = 1
PACKET_TYPE_COMMAND = 0x01
DEFAULT_PORT = 5150


def build_command_packet(
    fixture_id: int,
    red: int,
    green: int,
    blue: int,
    white: int,
    strobe_rate: int,
    strobe_intensity: int,
    special: int,
    sequence: int = 0,
) -> bytes:
    """Build a LUMINA command packet for a single fixture.

    Args:
        fixture_id: Target fixture ID (0 for broadcast, 1-255 for specific).
        red: Red channel (0-255).
        green: Green channel (0-255).
        blue: Blue channel (0-255).
        white: White channel (0-255).
        strobe_rate: Strobe frequency (0=off, 255=max ~25Hz).
        strobe_intensity: Strobe flash brightness (0-255).
        special: Fixture-type-specific channel (0-255).
        sequence: Packet sequence number.

    Returns:
        Encoded UDP packet bytes.
    """
    timestamp_ms = int(time.time() * 1000) & 0xFFFF

    # Header: magic(2) + version(1) + type(1) + sequence(2) + timestamp(2) + count(1)
    header = struct.pack(
        "<HBBHHB",
        MAGIC,
        VERSION,
        PACKET_TYPE_COMMAND,
        sequence & 0xFFFF,
        timestamp_ms,
        1,  # fixture_count
    )

    # Payload: fixture_id(1) + r(1) + g(1) + b(1) + w(1) + strobe_rate(1) + strobe_int(1) + special(1)
    payload = struct.pack(
        "BBBBBBBB",
        fixture_id,
        red,
        green,
        blue,
        white,
        strobe_rate,
        strobe_intensity,
        special,
    )

    return header + payload


def send_command(
    target: str,
    port: int,
    packet: bytes,
) -> None:
    """Send a UDP packet to the target address.

    Args:
        target: Target IP address.
        port: Target UDP port.
        packet: Encoded packet bytes.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.sendto(packet, (target, port))
        logger.info("Sent %d bytes to %s:%d", len(packet), target, port)


def run_color_test(
    target: str,
    port: int,
    fixture_id: int,
    red: int,
    green: int,
    blue: int,
    white: int,
) -> None:
    """Send a static color command to a fixture.

    Args:
        target: Target IP address.
        port: Target UDP port.
        fixture_id: Target fixture ID.
        red: Red channel value.
        green: Green channel value.
        blue: Blue channel value.
        white: White channel value.
    """
    logger.info(
        "Sending color (R=%d G=%d B=%d W=%d) to fixture %d at %s:%d",
        red, green, blue, white, fixture_id, target, port,
    )
    packet = build_command_packet(
        fixture_id=fixture_id,
        red=red, green=green, blue=blue, white=white,
        strobe_rate=0, strobe_intensity=0, special=0,
    )
    send_command(target, port, packet)


def run_strobe_test(
    target: str,
    port: int,
    fixture_id: int,
    rate: int,
    intensity: int,
) -> None:
    """Send a strobe command to a fixture.

    Args:
        target: Target IP address.
        port: Target UDP port.
        fixture_id: Target fixture ID.
        rate: Strobe rate (0-255).
        intensity: Strobe intensity (0-255).
    """
    logger.info(
        "Sending strobe (rate=%d intensity=%d) to fixture %d at %s:%d",
        rate, intensity, fixture_id, target, port,
    )
    packet = build_command_packet(
        fixture_id=fixture_id,
        red=255, green=255, blue=255, white=255,
        strobe_rate=rate, strobe_intensity=intensity, special=0,
    )
    send_command(target, port, packet)


def run_cycle_test(
    target: str,
    port: int,
    fixture_id: int,
    interval: float = 1.0,
) -> None:
    """Cycle through primary colors on a fixture.

    Args:
        target: Target IP address.
        port: Target UDP port.
        fixture_id: Target fixture ID.
        interval: Seconds between color changes.
    """
    colors = [
        ("Red", 255, 0, 0),
        ("Green", 0, 255, 0),
        ("Blue", 0, 0, 255),
        ("White", 0, 0, 0),  # white channel used
        ("Yellow", 255, 255, 0),
        ("Magenta", 255, 0, 255),
        ("Cyan", 0, 255, 255),
        ("Off", 0, 0, 0),
    ]

    logger.info("Cycling colors on fixture %d (%.1fs interval, Ctrl+C to stop)", fixture_id, interval)
    seq = 0
    try:
        while True:
            for name, r, g, b in colors:
                w = 255 if name == "White" else 0
                logger.info("  %s", name)
                packet = build_command_packet(
                    fixture_id=fixture_id,
                    red=r, green=g, blue=b, white=w,
                    strobe_rate=0, strobe_intensity=0, special=0,
                    sequence=seq,
                )
                send_command(target, port, packet)
                seq += 1
                time.sleep(interval)
    except KeyboardInterrupt:
        logger.info("Cycle test stopped")
        # Send all-off
        packet = build_command_packet(
            fixture_id=fixture_id,
            red=0, green=0, blue=0, white=0,
            strobe_rate=0, strobe_intensity=0, special=0,
            sequence=seq,
        )
        send_command(target, port, packet)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="LUMINA fixture test utility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        help="Target IP address (e.g., 192.168.1.100 or 192.168.1.255 for broadcast)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Target UDP port (default: {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--fixture-id",
        type=int,
        default=0,
        help="Fixture ID (0 for broadcast, 1-255 for specific fixture, default: 0)",
    )
    parser.add_argument(
        "--color",
        type=int,
        nargs=3,
        metavar=("R", "G", "B"),
        help="Send RGB color (0-255 each)",
    )
    parser.add_argument(
        "--white",
        type=int,
        default=0,
        help="White channel value (0-255, default: 0)",
    )
    parser.add_argument(
        "--strobe",
        type=int,
        nargs=2,
        metavar=("RATE", "INTENSITY"),
        help="Send strobe command (rate 0-255, intensity 0-255)",
    )
    parser.add_argument(
        "--cycle",
        action="store_true",
        help="Cycle through primary colors (1s interval)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    return parser.parse_args()


def main() -> None:
    """Entry point for the fixture test utility."""
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if args.cycle:
        run_cycle_test(args.target, args.port, args.fixture_id)
    elif args.strobe:
        run_strobe_test(args.target, args.port, args.fixture_id, args.strobe[0], args.strobe[1])
    elif args.color:
        run_color_test(
            args.target, args.port, args.fixture_id,
            args.color[0], args.color[1], args.color[2], args.white,
        )
    else:
        logger.error("Specify one of: --color R G B, --strobe RATE INTENSITY, or --cycle")
        sys.exit(1)


if __name__ == "__main__":
    main()
