"""Tests for lumina.lighting.fixture_map."""

from __future__ import annotations

from lumina.lighting.fixture_map import (
    ROOM_DEPTH,
    ROOM_HEIGHT,
    ROOM_WIDTH,
    FixtureInfo,
    FixtureMap,
    FixtureRole,
    FixtureType,
)


class TestFixtureMapDefaults:
    """Default 8-fixture layout."""

    def test_default_has_eight_fixtures(self) -> None:
        fm = FixtureMap()
        assert len(fm) == 8

    def test_ids_sequential(self) -> None:
        fm = FixtureMap()
        assert fm.ids == [1, 2, 3, 4, 5, 6, 7, 8]

    def test_four_pars(self) -> None:
        fm = FixtureMap()
        pars = fm.by_type(FixtureType.PAR)
        assert len(pars) == 4
        assert all(f.fixture_type == FixtureType.PAR for f in pars)

    def test_two_strobes(self) -> None:
        fm = FixtureMap()
        strobes = fm.by_type(FixtureType.STROBE)
        assert len(strobes) == 2

    def test_two_uv(self) -> None:
        fm = FixtureMap()
        uv = fm.by_type(FixtureType.UV)
        assert len(uv) == 2

    def test_get_by_id(self) -> None:
        fm = FixtureMap()
        f = fm.get(1)
        assert f.fixture_id == 1
        assert f.fixture_type == FixtureType.PAR

    def test_get_missing_raises(self) -> None:
        fm = FixtureMap()
        try:
            fm.get(99)
            assert False, "Should have raised KeyError"
        except KeyError:
            pass

    def test_positions_within_room(self) -> None:
        fm = FixtureMap()
        for f in fm.all:
            x, y, z = f.position
            assert 0.0 <= x <= ROOM_WIDTH
            assert 0.0 <= y <= ROOM_DEPTH
            assert 0.0 <= z <= ROOM_HEIGHT


class TestSpatialQueries:
    """Spatial filtering and sorting."""

    def test_by_role(self) -> None:
        fm = FixtureMap()
        fl = fm.by_role(FixtureRole.FRONT_LEFT)
        assert len(fl) == 1
        assert fl[0].fixture_id == 1

    def test_by_group(self) -> None:
        fm = FixtureMap()
        corners = fm.by_group("corners")
        assert len(corners) == 4
        assert all(f.fixture_type == FixtureType.PAR for f in corners)

    def test_left_right_partition(self) -> None:
        fm = FixtureMap()
        left = fm.left_side()
        right = fm.right_side()
        all_ids = {f.fixture_id for f in left} | {f.fixture_id for f in right}
        assert all_ids == set(fm.ids)

    def test_front_back_partition(self) -> None:
        fm = FixtureMap()
        front = fm.front_half()
        back = fm.back_half()
        all_ids = {f.fixture_id for f in front} | {f.fixture_id for f in back}
        assert all_ids == set(fm.ids)

    def test_sorted_by_x(self) -> None:
        fm = FixtureMap()
        ordered = fm.sorted_by_x()
        xs = [f.position[0] for f in ordered]
        assert xs == sorted(xs)

    def test_sorted_by_y(self) -> None:
        fm = FixtureMap()
        ordered = fm.sorted_by_y()
        ys = [f.position[1] for f in ordered]
        assert ys == sorted(ys)


class TestCustomMap:
    """Custom fixture layouts."""

    def test_custom_fixtures(self) -> None:
        custom = [
            FixtureInfo(
                fixture_id=10,
                fixture_type=FixtureType.STROBE,
                position=(1.0, 2.0, 2.5),
                role=FixtureRole.CENTER,
                groups={"test"},
            ),
        ]
        fm = FixtureMap(fixtures=custom)
        assert len(fm) == 1
        assert fm.get(10).fixture_type == FixtureType.STROBE

    def test_empty_map(self) -> None:
        fm = FixtureMap(fixtures=[])
        assert len(fm) == 0
        assert fm.ids == []
