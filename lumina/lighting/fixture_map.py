"""Spatial fixture registry for the LUMINA lighting engine.

Manages fixture metadata — ID, type, position, role, and group
memberships — so that lighting profiles can generate spatially-aware
commands (sweeps, chases, corner isolation, etc.).

Default layout: 15 fixtures in a 5m x 7m x 2.5m room.
  IDs 1-4:   RGBW Par, left wall (evenly spaced along 7m wall)
  IDs 5-8:   RGBW Par, right wall (mirroring left)
  IDs 9-12:  Strobe, four corners (mounted high)
  IDs 13-14: LED Bar, ceiling-mounted overhead (running lengthwise)
  ID 15:     Laser, rear wall center (mounted high)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class FixtureType(Enum):
    """Physical fixture type."""

    PAR = "par"
    STROBE = "strobe"
    UV = "uv"
    LED_BAR = "led_bar"
    LASER = "laser"


class FixtureRole(Enum):
    """Spatial role describing where the fixture is aimed or positioned."""

    FRONT_LEFT = "front_left"
    FRONT_RIGHT = "front_right"
    BACK_LEFT = "back_left"
    BACK_RIGHT = "back_right"
    CENTER = "center"
    LEFT = "left"
    RIGHT = "right"
    FRONT = "front"
    BACK = "back"


@dataclass(slots=True)
class FixtureInfo:
    """Metadata for a single fixture.

    Args:
        fixture_id: Unique fixture address (1-255).
        fixture_type: Physical type (par, strobe, uv, led_bar, laser).
        position: (x, y, z) in meters.  Origin is floor-level front-left corner.
        role: Spatial role for pattern generation.
        groups: Set of group names this fixture belongs to.
        name: Human-readable name for this fixture.
    """

    fixture_id: int
    fixture_type: FixtureType
    position: tuple[float, float, float]
    role: FixtureRole
    groups: set[str] = field(default_factory=set)
    name: str = ""


# ─── Room dimensions ────────────────────────────────────────────────

ROOM_WIDTH = 5.0  # x-axis (meters)
ROOM_DEPTH = 7.0  # y-axis (meters)
ROOM_HEIGHT = 2.5  # z-axis (meters)

# Margin from walls for corner-mounted fixtures
_CORNER_MARGIN = 0.3


class FixtureMap:
    """Registry of fixtures with spatial query helpers.

    Args:
        fixtures: Optional list of FixtureInfo to initialize with.
            If None, the default 15-fixture layout is used.
    """

    def __init__(self, fixtures: list[FixtureInfo] | None = None) -> None:
        if fixtures is None:
            fixtures = _default_fixtures()
        self._fixtures: dict[int, FixtureInfo] = {f.fixture_id: f for f in fixtures}

    # ─── Serialization ────────────────────────────────────────────

    @classmethod
    def load_from_json(cls, path: Path) -> FixtureMap:
        """Load a fixture layout from a JSON file.

        Falls back to the default 15-fixture layout if the file cannot be
        read or parsed.

        Args:
            path: Path to the JSON file.

        Returns:
            A FixtureMap populated from the JSON data.
        """
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "Failed to load fixture layout from %s (%s) — "
                "using default layout",
                path, exc,
            )
            return cls()

        fixtures: list[FixtureInfo] = []
        for entry in data.get("fixtures", []):
            try:
                fixtures.append(_fixture_from_dict(entry))
            except (KeyError, ValueError) as exc:
                logger.warning(
                    "Skipping invalid fixture entry %r: %s", entry, exc,
                )

        if not fixtures:
            logger.warning(
                "No valid fixtures in %s — using default layout", path,
            )
            return cls()

        logger.info("Loaded %d fixtures from %s", len(fixtures), path)
        return cls(fixtures=fixtures)

    def save_to_json(self, path: Path) -> None:
        """Export the current fixture layout to a JSON file.

        Args:
            path: Destination file path. Parent directories are created
                if they do not exist.
        """
        entries: list[dict[str, Any]] = []
        for f in self.all:
            entries.append({
                "fixture_id": f.fixture_id,
                "fixture_type": f.fixture_type.value,
                "position": list(f.position),
                "role": f.role.value,
                "groups": sorted(f.groups),
                "name": f.name,
            })

        data = {
            "room": {
                "width": ROOM_WIDTH,
                "depth": ROOM_DEPTH,
                "height": ROOM_HEIGHT,
            },
            "fixtures": entries,
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(data, indent=2) + "\n", encoding="utf-8",
        )
        logger.info("Saved %d fixtures to %s", len(entries), path)

    # ─── Access ──────────────────────────────────────────────────

    @property
    def all(self) -> list[FixtureInfo]:
        """All fixtures sorted by ID."""
        return sorted(self._fixtures.values(), key=lambda f: f.fixture_id)

    @property
    def ids(self) -> list[int]:
        """All fixture IDs sorted."""
        return sorted(self._fixtures)

    def get(self, fixture_id: int) -> FixtureInfo:
        """Get fixture by ID.

        Args:
            fixture_id: The fixture ID to look up.

        Returns:
            FixtureInfo for the requested fixture.

        Raises:
            KeyError: If fixture_id is not in the map.
        """
        return self._fixtures[fixture_id]

    def __len__(self) -> int:
        return len(self._fixtures)

    # ─── Spatial queries ─────────────────────────────────────────

    def by_type(self, fixture_type: FixtureType) -> list[FixtureInfo]:
        """Get all fixtures of a given type, sorted by ID.

        Args:
            fixture_type: Type to filter by.

        Returns:
            List of matching FixtureInfo.
        """
        return sorted(
            (f for f in self._fixtures.values() if f.fixture_type == fixture_type),
            key=lambda f: f.fixture_id,
        )

    def by_role(self, role: FixtureRole) -> list[FixtureInfo]:
        """Get all fixtures with a given role, sorted by ID.

        Args:
            role: Role to filter by.

        Returns:
            List of matching FixtureInfo.
        """
        return sorted(
            (f for f in self._fixtures.values() if f.role == role),
            key=lambda f: f.fixture_id,
        )

    def by_group(self, group: str) -> list[FixtureInfo]:
        """Get all fixtures in a named group, sorted by ID.

        Args:
            group: Group name to filter by.

        Returns:
            List of matching FixtureInfo.
        """
        return sorted(
            (f for f in self._fixtures.values() if group in f.groups),
            key=lambda f: f.fixture_id,
        )

    def sorted_by_x(self, fixtures: list[FixtureInfo] | None = None) -> list[FixtureInfo]:
        """Sort fixtures left-to-right (ascending x).

        Args:
            fixtures: Subset to sort; defaults to all fixtures.

        Returns:
            Sorted list.
        """
        src = fixtures if fixtures is not None else self.all
        return sorted(src, key=lambda f: f.position[0])

    def sorted_by_y(self, fixtures: list[FixtureInfo] | None = None) -> list[FixtureInfo]:
        """Sort fixtures front-to-back (ascending y).

        Args:
            fixtures: Subset to sort; defaults to all fixtures.

        Returns:
            Sorted list.
        """
        src = fixtures if fixtures is not None else self.all
        return sorted(src, key=lambda f: f.position[1])

    def left_side(self) -> list[FixtureInfo]:
        """Fixtures on the left half of the room (x < midpoint)."""
        mid = ROOM_WIDTH / 2
        return sorted(
            (f for f in self._fixtures.values() if f.position[0] < mid),
            key=lambda f: f.fixture_id,
        )

    def right_side(self) -> list[FixtureInfo]:
        """Fixtures on the right half of the room (x >= midpoint)."""
        mid = ROOM_WIDTH / 2
        return sorted(
            (f for f in self._fixtures.values() if f.position[0] >= mid),
            key=lambda f: f.fixture_id,
        )

    def front_half(self) -> list[FixtureInfo]:
        """Fixtures in the front half of the room (y < midpoint)."""
        mid = ROOM_DEPTH / 2
        return sorted(
            (f for f in self._fixtures.values() if f.position[1] < mid),
            key=lambda f: f.fixture_id,
        )

    def back_half(self) -> list[FixtureInfo]:
        """Fixtures in the back half of the room (y >= midpoint)."""
        mid = ROOM_DEPTH / 2
        return sorted(
            (f for f in self._fixtures.values() if f.position[1] >= mid),
            key=lambda f: f.fixture_id,
        )

    # ─── Convenience aliases ─────────────────────────────────────

    def get_by_type(self, fixture_type: FixtureType) -> list[FixtureInfo]:
        """Get all fixtures of a given type (alias for by_type).

        Args:
            fixture_type: Type to filter by.

        Returns:
            List of matching FixtureInfo sorted by ID.
        """
        return self.by_type(fixture_type)

    def get_by_group(self, group: str) -> list[FixtureInfo]:
        """Get all fixtures in a named group (alias for by_group).

        Args:
            group: Group name to filter by.

        Returns:
            List of matching FixtureInfo sorted by ID.
        """
        return self.by_group(group)

    def get_left(self) -> list[FixtureInfo]:
        """Fixtures in the 'left' group, sorted by y position.

        Returns:
            Left-wall fixtures sorted front-to-back.
        """
        return self.sorted_by_y(self.by_group("left"))

    def get_right(self) -> list[FixtureInfo]:
        """Fixtures in the 'right' group, sorted by y position.

        Returns:
            Right-wall fixtures sorted front-to-back.
        """
        return self.sorted_by_y(self.by_group("right"))

    def get_by_spatial_order(self) -> list[FixtureInfo]:
        """All fixtures sorted left-to-right (by x position) for chase patterns.

        Returns:
            All fixtures sorted by ascending x, then by y for ties.
        """
        return sorted(self.all, key=lambda f: (f.position[0], f.position[1]))


def _default_fixtures() -> list[FixtureInfo]:
    """Create the default 15-fixture layout for a 5m x 7m x 2.5m room.

    Layout:
        IDs 1-4:   RGBW Par — left wall, evenly spaced along 7m wall
        IDs 5-8:   RGBW Par — right wall, mirroring left
        IDs 9-12:  Strobe — four corners, mounted high
        IDs 13-14: LED Bar — ceiling center-line, running lengthwise
        ID 15:     Laser — rear wall center, mounted high

    Returns:
        List of 15 FixtureInfo instances.
    """
    m = _CORNER_MARGIN

    return [
        # ── RGBW Pars: left wall (x=0, evenly spaced along y) ──
        FixtureInfo(
            fixture_id=1,
            fixture_type=FixtureType.PAR,
            position=(0.0, 1.4, 2.0),
            role=FixtureRole.LEFT,
            groups={"par_left", "par_all", "left"},
            name="Par L1",
        ),
        FixtureInfo(
            fixture_id=2,
            fixture_type=FixtureType.PAR,
            position=(0.0, 2.8, 2.1),
            role=FixtureRole.LEFT,
            groups={"par_left", "par_all", "left"},
            name="Par L2",
        ),
        FixtureInfo(
            fixture_id=3,
            fixture_type=FixtureType.PAR,
            position=(0.0, 4.2, 2.2),
            role=FixtureRole.LEFT,
            groups={"par_left", "par_all", "left"},
            name="Par L3",
        ),
        FixtureInfo(
            fixture_id=4,
            fixture_type=FixtureType.PAR,
            position=(0.0, 5.6, 2.3),
            role=FixtureRole.LEFT,
            groups={"par_left", "par_all", "left"},
            name="Par L4",
        ),
        # ── RGBW Pars: right wall (x=5.0, mirroring left) ──
        FixtureInfo(
            fixture_id=5,
            fixture_type=FixtureType.PAR,
            position=(5.0, 1.4, 2.0),
            role=FixtureRole.RIGHT,
            groups={"par_right", "par_all", "right"},
            name="Par R1",
        ),
        FixtureInfo(
            fixture_id=6,
            fixture_type=FixtureType.PAR,
            position=(5.0, 2.8, 2.1),
            role=FixtureRole.RIGHT,
            groups={"par_right", "par_all", "right"},
            name="Par R2",
        ),
        FixtureInfo(
            fixture_id=7,
            fixture_type=FixtureType.PAR,
            position=(5.0, 4.2, 2.2),
            role=FixtureRole.RIGHT,
            groups={"par_right", "par_all", "right"},
            name="Par R3",
        ),
        FixtureInfo(
            fixture_id=8,
            fixture_type=FixtureType.PAR,
            position=(5.0, 5.6, 2.3),
            role=FixtureRole.RIGHT,
            groups={"par_right", "par_all", "right"},
            name="Par R4",
        ),
        # ── Strobes: four corners, mounted at 2.4m ──
        FixtureInfo(
            fixture_id=9,
            fixture_type=FixtureType.STROBE,
            position=(m, m, 2.4),
            role=FixtureRole.FRONT_LEFT,
            groups={"strobe_corners", "strobe_left"},
            name="Strobe FL",
        ),
        FixtureInfo(
            fixture_id=10,
            fixture_type=FixtureType.STROBE,
            position=(ROOM_WIDTH - m, m, 2.4),
            role=FixtureRole.FRONT_RIGHT,
            groups={"strobe_corners", "strobe_right"},
            name="Strobe FR",
        ),
        FixtureInfo(
            fixture_id=11,
            fixture_type=FixtureType.STROBE,
            position=(m, ROOM_DEPTH - m, 2.4),
            role=FixtureRole.BACK_LEFT,
            groups={"strobe_corners", "strobe_left"},
            name="Strobe BL",
        ),
        FixtureInfo(
            fixture_id=12,
            fixture_type=FixtureType.STROBE,
            position=(ROOM_WIDTH - m, ROOM_DEPTH - m, 2.4),
            role=FixtureRole.BACK_RIGHT,
            groups={"strobe_corners", "strobe_right"},
            name="Strobe BR",
        ),
        # ── LED Bars: ceiling center-line, running lengthwise ──
        FixtureInfo(
            fixture_id=13,
            fixture_type=FixtureType.LED_BAR,
            position=(ROOM_WIDTH / 2, ROOM_DEPTH / 3, ROOM_HEIGHT),
            role=FixtureRole.CENTER,
            groups={"overhead", "center"},
            name="Bar Front",
        ),
        FixtureInfo(
            fixture_id=14,
            fixture_type=FixtureType.LED_BAR,
            position=(ROOM_WIDTH / 2, ROOM_DEPTH * 2 / 3, ROOM_HEIGHT),
            role=FixtureRole.CENTER,
            groups={"overhead", "center"},
            name="Bar Rear",
        ),
        # ── Laser: rear wall center, mounted high ──
        FixtureInfo(
            fixture_id=15,
            fixture_type=FixtureType.LASER,
            position=(ROOM_WIDTH / 2, ROOM_DEPTH, 2.4),
            role=FixtureRole.BACK,
            groups={"laser", "back"},
            name="Laser",
        ),
    ]


def _fixture_from_dict(d: dict[str, Any]) -> FixtureInfo:
    """Deserialize a single fixture entry from a JSON dict.

    Args:
        d: Dictionary with keys matching FixtureInfo fields.
            ``fixture_type`` and ``role`` are strings matching enum values.

    Returns:
        Parsed FixtureInfo.

    Raises:
        KeyError: If a required key is missing.
        ValueError: If an enum value is invalid.
    """
    pos = d["position"]
    return FixtureInfo(
        fixture_id=int(d["fixture_id"]),
        fixture_type=FixtureType(d["fixture_type"]),
        position=(float(pos[0]), float(pos[1]), float(pos[2])),
        role=FixtureRole(d["role"]),
        groups=set(d.get("groups", [])),
        name=str(d.get("name", "")),
    )
