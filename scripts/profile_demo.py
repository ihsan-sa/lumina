"""Demo each lighting profile in the simulator.

Connects to the LUMINA backend and cycles through all 8 genre lighting profiles,
demonstrating characteristic patterns for each. Useful for visual tuning and
profile comparison.

Usage:
    python scripts/profile_demo.py
    python scripts/profile_demo.py --profile rage_trap --duration 30
    python scripts/profile_demo.py --host localhost --port 8765 --all
"""

import argparse
import asyncio
import json
import logging
import sys

logger = logging.getLogger(__name__)

# All 8 genre profiles in presentation order
PROFILES = [
    "rage_trap",
    "psych_rnb",
    "french_melodic",
    "french_hard",
    "euro_alt",
    "theatrical",
    "festival_edm",
    "uk_bass",
]

PROFILE_DESCRIPTIONS = {
    "rage_trap": "Rage / Experimental Trap (Playboi Carti, Travis Scott)",
    "psych_rnb": "Psychedelic Trap / Dark R&B (Don Toliver, The Weeknd)",
    "french_melodic": "French Rap Melodic (Ninho, Jul)",
    "french_hard": "French Rap Hard (Kaaris)",
    "euro_alt": "European Alt Hip-Hop (AyVe, Exetra Archive)",
    "theatrical": "Theatrical Electronic (Stromae)",
    "festival_edm": "Festival EDM / Trance (Guetta, Armin, Edward Maya)",
    "uk_bass": "UK Bass / Dubstep / Grime (Fred again..)",
}


async def send_profile_override(
    host: str,
    port: int,
    profile: str,
) -> None:
    """Send a genre override command to the backend via WebSocket.

    Args:
        host: WebSocket server hostname.
        port: WebSocket server port.
        profile: Genre profile ID to activate.
    """
    uri = f"ws://{host}:{port}"
    logger.info("Connecting to %s", uri)

    # Placeholder: actual implementation would use websockets library
    # import websockets
    # async with websockets.connect(uri) as ws:
    #     message = json.dumps({
    #         "type": "genre_override",
    #         "profile": profile,
    #     })
    #     await ws.send(message)
    #     logger.info("Sent genre_override: %s", profile)

    logger.info(
        "Would send genre_override message: %s",
        json.dumps({"type": "genre_override", "profile": profile}),
    )


async def demo_single_profile(
    host: str,
    port: int,
    profile: str,
    duration: float,
) -> None:
    """Demo a single lighting profile for the specified duration.

    Args:
        host: WebSocket server hostname.
        port: WebSocket server port.
        profile: Genre profile ID to demo.
        duration: Duration in seconds to hold this profile.
    """
    description = PROFILE_DESCRIPTIONS.get(profile, profile)
    logger.info("=" * 60)
    logger.info("Profile: %s", description)
    logger.info("=" * 60)

    await send_profile_override(host, port, profile)

    logger.info("Holding for %.1f seconds...", duration)
    await asyncio.sleep(duration)


async def demo_all_profiles(
    host: str,
    port: int,
    duration: float,
) -> None:
    """Cycle through all 8 genre profiles.

    Args:
        host: WebSocket server hostname.
        port: WebSocket server port.
        duration: Duration in seconds per profile.
    """
    logger.info("LUMINA Profile Demo — cycling through all %d profiles", len(PROFILES))
    logger.info("Duration per profile: %.1f seconds", duration)
    logger.info("Total duration: %.1f seconds", duration * len(PROFILES))
    logger.info("")

    for i, profile in enumerate(PROFILES, 1):
        logger.info("[%d/%d]", i, len(PROFILES))
        await demo_single_profile(host, port, profile, duration)

    logger.info("")
    logger.info("Demo complete!")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="LUMINA lighting profile demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="WebSocket server hostname (default: localhost)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="WebSocket server port (default: 8765)",
    )
    parser.add_argument(
        "--profile",
        type=str,
        choices=PROFILES,
        default=None,
        help="Demo a single profile (default: cycle through all)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=15.0,
        help="Duration per profile in seconds (default: 15)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Explicitly demo all profiles (default behavior if --profile not set)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    return parser.parse_args()


def main() -> None:
    """Entry point for the profile demo utility."""
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    logger.info("LUMINA Profile Demo Utility")
    logger.info(
        "NOTE: This is a skeleton — connect to a running LUMINA backend "
        "with audio playing for full visual demo."
    )

    if args.profile:
        asyncio.run(
            demo_single_profile(args.host, args.port, args.profile, args.duration)
        )
    else:
        asyncio.run(
            demo_all_profiles(args.host, args.port, args.duration)
        )


if __name__ == "__main__":
    main()
