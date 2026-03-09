"""Spatial fixture registry for the LUMINA lighting engine.

Manages fixture metadata — ID, type, position, role, and group
memberships — so that lighting profiles can generate spatially-aware
commands (sweeps, chases, corner isolation, etc.).

Default layout: 4 RGBW pars in ceiling corners, 2 strobes on
center-line, 2 UV bars on side walls.  Room: 5m × 7m × 2.5m.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class FixtureType(Enum):
    """Physical fixture type."""

    PAR = "par"
    STROBE = "strobe"
    UV = "uv"


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
        fixture_type: Physical type (par, strobe, uv).
        position: (x, y, z) in meters.  Origin is floor-level back-left corner.
        role: Spatial role for pattern generation.
        groups: Set of group names this fixture belongs to.
    """

    fixture_id: int
    fixture_type: FixtureType
    position: tuple[float, float, float]
    role: FixtureRole
    groups: set[str] = field(default_factory=set)


# ─── Room dimensions ────────────────────────────────────────────────

ROOM_WIDTH = 5.0  # x-axis (meters)
ROOM_DEPTH = 7.0  # y-axis (meters)
ROOM_HEIGHT = 2.5  # z-axis (meters)

# Margin from walls for ceiling-mounted fixtures
_CORNER_MARGIN = 0.3


class FixtureMap:
    """Registry of fixtures with spatial query helpers.

    Args:
        fixtures: Optional list of FixtureInfo to initialize with.
            If None, the default 8-fixture layout is used.
    """

    def __init__(self, fixtures: list[FixtureInfo] | None = None) -> None:
        if fixtures is None:
            fixtures = _default_fixtures()
        self._fixtures: dict[int, FixtureInfo] = {f.fixture_id: f for f in fixtures}

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


def _default_fixtures() -> list[FixtureInfo]:
    """Create the default 8-fixture layout for a 5m×7m×2.5m room.

    Layout:
        ID 1: RGBW Par — front-left ceiling corner
        ID 2: RGBW Par — front-right ceiling corner
        ID 3: RGBW Par — back-left ceiling corner
        ID 4: RGBW Par — back-right ceiling corner
        ID 5: Strobe  — ceiling center-line, front third
        ID 6: Strobe  — ceiling center-line, back third
        ID 7: UV Bar  — left wall, mid-height
        ID 8: UV Bar  — right wall, mid-height

    Returns:
        List of 8 FixtureInfo instances.
    """
    cx = ROOM_WIDTH / 2  # center x
    m = _CORNER_MARGIN
    ceil = ROOM_HEIGHT
    uv_h = 2.0  # UV bars mounted at 2m

    return [
        # ── RGBW Pars (ceiling corners) ──
        FixtureInfo(
            fixture_id=1,
            fixture_type=FixtureType.PAR,
            position=(m, m, ceil),
            role=FixtureRole.FRONT_LEFT,
            groups={"pars", "corners", "front", "left"},
        ),
        FixtureInfo(
            fixture_id=2,
            fixture_type=FixtureType.PAR,
            position=(ROOM_WIDTH - m, m, ceil),
            role=FixtureRole.FRONT_RIGHT,
            groups={"pars", "corners", "front", "right"},
        ),
        FixtureInfo(
            fixture_id=3,
            fixture_type=FixtureType.PAR,
            position=(m, ROOM_DEPTH - m, ceil),
            role=FixtureRole.BACK_LEFT,
            groups={"pars", "corners", "back", "left"},
        ),
        FixtureInfo(
            fixture_id=4,
            fixture_type=FixtureType.PAR,
            position=(ROOM_WIDTH - m, ROOM_DEPTH - m, ceil),
            role=FixtureRole.BACK_RIGHT,
            groups={"pars", "corners", "back", "right"},
        ),
        # ── Strobes (ceiling center-line) ──
        FixtureInfo(
            fixture_id=5,
            fixture_type=FixtureType.STROBE,
            position=(cx, ROOM_DEPTH * 0.33, ceil),
            role=FixtureRole.CENTER,
            groups={"strobes", "center", "front"},
        ),
        FixtureInfo(
            fixture_id=6,
            fixture_type=FixtureType.STROBE,
            position=(cx, ROOM_DEPTH * 0.67, ceil),
            role=FixtureRole.CENTER,
            groups={"strobes", "center", "back"},
        ),
        # ── UV Bars (side walls) ──
        FixtureInfo(
            fixture_id=7,
            fixture_type=FixtureType.UV,
            position=(0.0, ROOM_DEPTH / 2, uv_h),
            role=FixtureRole.LEFT,
            groups={"uv", "walls", "left"},
        ),
        FixtureInfo(
            fixture_id=8,
            fixture_type=FixtureType.UV,
            position=(ROOM_WIDTH, ROOM_DEPTH / 2, uv_h),
            role=FixtureRole.RIGHT,
            groups={"uv", "walls", "right"},
        ),
    ]
