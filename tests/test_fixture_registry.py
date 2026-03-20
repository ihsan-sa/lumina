"""Tests for FixtureState and FixtureRegistry in lumina.control.fixture."""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from lumina.control.fixture import FixtureRegistry, FixtureState
from lumina.control.protocol import FixtureCommand
from lumina.lighting.fixture_map import (
    FixtureInfo,
    FixtureMap,
    FixtureRole,
    FixtureType,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _make_info(fixture_id: int, name: str = "") -> FixtureInfo:
    """Build a minimal FixtureInfo for test fixtures."""
    return FixtureInfo(
        fixture_id=fixture_id,
        fixture_type=FixtureType.PAR,
        position=(0.0, 0.0, 0.0),
        role=FixtureRole.LEFT,
        groups={"test"},
        name=name or f"Fixture {fixture_id}",
    )


def _make_map(*ids: int) -> FixtureMap:
    """Build a FixtureMap containing fixtures with the given IDs."""
    return FixtureMap(fixtures=[_make_info(fid) for fid in ids])


def _make_registry(*ids: int) -> FixtureRegistry:
    """Build a FixtureRegistry with fixtures for each given ID."""
    return FixtureRegistry(_make_map(*ids))


def _cmd(fixture_id: int, red: int = 0, green: int = 0, blue: int = 0) -> FixtureCommand:
    """Build a minimal FixtureCommand."""
    return FixtureCommand(fixture_id=fixture_id, red=red, green=green, blue=blue)


# ─── FixtureState ─────────────────────────────────────────────────────────────


class TestFixtureStateDefaults:
    def test_fixture_id_stored(self) -> None:
        info = _make_info(7)
        state = FixtureState(fixture_id=7, info=info)
        assert state.fixture_id == 7

    def test_info_stored(self) -> None:
        info = _make_info(3, "My Fixture")
        state = FixtureState(fixture_id=3, info=info)
        assert state.info is info
        assert state.info.name == "My Fixture"

    def test_last_command_defaults_to_none(self) -> None:
        info = _make_info(1)
        state = FixtureState(fixture_id=1, info=info)
        assert state.last_command is None

    def test_online_defaults_to_false(self) -> None:
        info = _make_info(1)
        state = FixtureState(fixture_id=1, info=info)
        assert state.online is False

    def test_last_seen_defaults_to_zero(self) -> None:
        info = _make_info(1)
        state = FixtureState(fixture_id=1, info=info)
        assert state.last_seen == 0.0

    def test_firmware_version_defaults_to_zero(self) -> None:
        info = _make_info(1)
        state = FixtureState(fixture_id=1, info=info)
        assert state.firmware_version == 0


class TestFixtureStateSecondsSinceSeen:
    def test_never_seen_returns_inf(self) -> None:
        state = FixtureState(fixture_id=1, info=_make_info(1))
        assert state.seconds_since_seen() == float("inf")

    def test_recently_seen_is_small(self) -> None:
        state = FixtureState(fixture_id=1, info=_make_info(1))
        state.last_seen = time.monotonic()
        elapsed = state.seconds_since_seen()
        assert 0.0 <= elapsed < 0.5

    def test_old_timestamp_is_large(self) -> None:
        state = FixtureState(fixture_id=1, info=_make_info(1))
        # Simulate a fixture seen 30 seconds ago.
        state.last_seen = time.monotonic() - 30.0
        assert state.seconds_since_seen() >= 29.9

    def test_returns_zero_for_current_timestamp(self) -> None:
        state = FixtureState(fixture_id=1, info=_make_info(1))
        fake_now = 1000.0
        state.last_seen = fake_now
        with patch("lumina.control.fixture.time.monotonic", return_value=fake_now):
            assert state.seconds_since_seen() == pytest.approx(0.0)

    def test_returns_exact_elapsed_when_patched(self) -> None:
        state = FixtureState(fixture_id=1, info=_make_info(1))
        state.last_seen = 900.0
        with patch("lumina.control.fixture.time.monotonic", return_value=903.5):
            assert state.seconds_since_seen() == pytest.approx(3.5)


class TestFixtureStateIsDark:
    def test_no_command_is_dark(self) -> None:
        state = FixtureState(fixture_id=1, info=_make_info(1))
        assert state.is_dark() is True

    def test_all_zero_command_is_dark(self) -> None:
        state = FixtureState(fixture_id=1, info=_make_info(1))
        state.last_command = FixtureCommand()
        assert state.is_dark() is True

    @pytest.mark.parametrize(
        "field,value",
        [
            ("red", 1),
            ("green", 128),
            ("blue", 255),
            ("white", 10),
            ("strobe_rate", 50),
            ("strobe_intensity", 200),
            ("special", 1),
        ],
    )
    def test_nonzero_channel_is_not_dark(self, field: str, value: int) -> None:
        state = FixtureState(fixture_id=1, info=_make_info(1))
        state.last_command = FixtureCommand(**{field: value})
        assert state.is_dark() is False

    def test_all_max_channels_is_not_dark(self) -> None:
        state = FixtureState(fixture_id=1, info=_make_info(1))
        state.last_command = FixtureCommand(
            red=255, green=255, blue=255, white=255,
            strobe_rate=255, strobe_intensity=255, special=255,
        )
        assert state.is_dark() is False


class TestFixtureStateRepr:
    def test_repr_contains_id_and_status(self) -> None:
        info = _make_info(42, "Par L1")
        state = FixtureState(fixture_id=42, info=info)
        r = repr(state)
        assert "42" in r
        assert "OFFLINE" in r

    def test_repr_shows_online_when_online(self) -> None:
        info = _make_info(5)
        state = FixtureState(fixture_id=5, info=info, online=True)
        assert "ONLINE" in repr(state)

    def test_repr_contains_fixture_name(self) -> None:
        info = _make_info(3, "Strobe FR")
        state = FixtureState(fixture_id=3, info=info)
        assert "Strobe FR" in repr(state)


# ─── FixtureRegistry construction ─────────────────────────────────────────────


class TestFixtureRegistryConstruction:
    def test_length_matches_fixture_count(self) -> None:
        registry = _make_registry(1, 2, 3)
        assert len(registry) == 3

    def test_default_map_has_15_fixtures(self) -> None:
        registry = FixtureRegistry(FixtureMap())
        assert len(registry) == 15

    def test_single_fixture_registry(self) -> None:
        registry = _make_registry(7)
        assert len(registry) == 1

    def test_custom_fixture_map(self) -> None:
        info = FixtureInfo(
            fixture_id=42,
            fixture_type=FixtureType.STROBE,
            position=(1.0, 2.0, 3.0),
            role=FixtureRole.CENTER,
            groups={"strobe_all"},
            name="Test Strobe",
        )
        fmap = FixtureMap(fixtures=[info])
        registry = FixtureRegistry(fmap)
        assert len(registry) == 1
        state = registry.get(42)
        assert state is not None
        assert state.info.name == "Test Strobe"

    def test_all_fixtures_start_offline(self) -> None:
        registry = _make_registry(1, 2, 3)
        assert all(not s.online for s in registry.all_states())

    def test_all_fixtures_start_with_no_command(self) -> None:
        registry = _make_registry(1, 2, 3)
        assert all(s.last_command is None for s in registry.all_states())

    def test_fixture_ids_preserved_from_map(self) -> None:
        ids = [3, 7, 12]
        registry = _make_registry(*ids)
        registered = [s.fixture_id for s in registry.all_states()]
        assert sorted(registered) == sorted(ids)


# ─── FixtureRegistry.get() ────────────────────────────────────────────────────


class TestFixtureRegistryGet:
    def test_get_known_id_returns_state(self) -> None:
        registry = _make_registry(1, 2, 3)
        state = registry.get(2)
        assert state is not None
        assert state.fixture_id == 2

    def test_get_unknown_id_returns_none(self) -> None:
        registry = _make_registry(1, 2, 3)
        assert registry.get(99) is None

    def test_get_zero_returns_none(self) -> None:
        # 0 is the broadcast address, never a valid registry entry.
        registry = _make_registry(1, 2, 3)
        assert registry.get(0) is None

    def test_get_returns_correct_info(self) -> None:
        info = _make_info(5, "Special Fixture")
        fmap = FixtureMap(fixtures=[info])
        registry = FixtureRegistry(fmap)
        state = registry.get(5)
        assert state is not None
        assert state.info.name == "Special Fixture"
        assert state.info.fixture_id == 5

    def test_all_registered_ids_are_retrievable(self) -> None:
        ids = [1, 5, 10, 50, 100, 255]
        registry = _make_registry(*ids)
        for fid in ids:
            assert registry.get(fid) is not None

    def test_get_returns_same_object_on_repeated_calls(self) -> None:
        registry = _make_registry(4)
        assert registry.get(4) is registry.get(4)


# ─── FixtureRegistry.all_states() ─────────────────────────────────────────────


class TestFixtureRegistryAllStates:
    def test_returns_all_fixtures(self) -> None:
        registry = _make_registry(3, 1, 2)
        states = registry.all_states()
        assert len(states) == 3

    def test_sorted_by_fixture_id(self) -> None:
        registry = _make_registry(10, 3, 7, 1)
        states = registry.all_states()
        ids = [s.fixture_id for s in states]
        assert ids == sorted(ids)

    def test_includes_offline_fixtures(self) -> None:
        registry = _make_registry(1, 2, 3)
        # All start offline; all_states must still return them.
        states = registry.all_states()
        assert all(not s.online for s in states)
        assert len(states) == 3

    def test_includes_online_fixtures(self) -> None:
        registry = _make_registry(1, 2, 3)
        registry.mark_seen(2)
        states = registry.all_states()
        assert len(states) == 3  # all three, regardless of online status

    def test_returns_list_type(self) -> None:
        registry = _make_registry(1)
        assert isinstance(registry.all_states(), list)


# ─── FixtureRegistry.online_fixtures() ───────────────────────────────────────


class TestFixtureRegistryOnlineFixtures:
    def test_initially_empty(self) -> None:
        registry = _make_registry(1, 2, 3)
        assert registry.online_fixtures() == []

    def test_returns_only_online_fixtures(self) -> None:
        registry = _make_registry(1, 2, 3)
        registry.mark_seen(2)
        online = registry.online_fixtures()
        assert len(online) == 1
        assert online[0].fixture_id == 2

    def test_multiple_online_fixtures(self) -> None:
        registry = _make_registry(1, 2, 3, 4)
        registry.mark_seen(1)
        registry.mark_seen(3)
        online = registry.online_fixtures()
        assert len(online) == 2
        ids = {s.fixture_id for s in online}
        assert ids == {1, 3}

    def test_sorted_by_fixture_id(self) -> None:
        registry = _make_registry(5, 1, 3)
        registry.mark_seen(5)
        registry.mark_seen(1)
        registry.mark_seen(3)
        ids = [s.fixture_id for s in registry.online_fixtures()]
        assert ids == [1, 3, 5]

    def test_all_online(self) -> None:
        registry = _make_registry(1, 2, 3)
        for fid in [1, 2, 3]:
            registry.mark_seen(fid)
        assert len(registry.online_fixtures()) == 3

    def test_does_not_include_offline_after_timeout(self) -> None:
        registry = _make_registry(1)
        registry.mark_seen(1)
        state = registry.get(1)
        assert state is not None
        state.online = False  # Simulate manual offline transition.
        assert registry.online_fixtures() == []


# ─── FixtureRegistry.mark_seen() ─────────────────────────────────────────────


class TestFixtureRegistryMarkSeen:
    def test_mark_seen_sets_online(self) -> None:
        registry = _make_registry(1)
        state = registry.get(1)
        assert state is not None
        assert state.online is False
        registry.mark_seen(1)
        assert state.online is True

    def test_mark_seen_updates_last_seen(self) -> None:
        registry = _make_registry(1)
        state = registry.get(1)
        assert state is not None
        assert state.last_seen == 0.0
        before = time.monotonic()
        registry.mark_seen(1)
        after = time.monotonic()
        assert before <= state.last_seen <= after

    def test_mark_seen_stores_firmware_version(self) -> None:
        registry = _make_registry(1)
        registry.mark_seen(1, firmware_version=7)
        state = registry.get(1)
        assert state is not None
        assert state.firmware_version == 7

    def test_mark_seen_default_firmware_version_zero(self) -> None:
        registry = _make_registry(1)
        registry.mark_seen(1)
        state = registry.get(1)
        assert state is not None
        assert state.firmware_version == 0

    def test_mark_seen_unknown_id_does_not_raise(self) -> None:
        registry = _make_registry(1)
        # Should log a warning but not raise.
        registry.mark_seen(99)

    def test_mark_seen_unknown_id_does_not_add_fixture(self) -> None:
        registry = _make_registry(1)
        registry.mark_seen(99)
        assert len(registry) == 1
        assert registry.get(99) is None

    def test_mark_seen_repeated_updates_last_seen(self) -> None:
        registry = _make_registry(1)
        registry.mark_seen(1)
        state = registry.get(1)
        assert state is not None
        first_seen = state.last_seen
        with patch("lumina.control.fixture.time.monotonic", return_value=first_seen + 5.0):
            registry.mark_seen(1)
        assert state.last_seen == first_seen + 5.0

    def test_mark_seen_updates_firmware_version_on_repeat(self) -> None:
        registry = _make_registry(1)
        registry.mark_seen(1, firmware_version=1)
        registry.mark_seen(1, firmware_version=5)
        state = registry.get(1)
        assert state is not None
        assert state.firmware_version == 5

    def test_mark_seen_appears_in_online_fixtures(self) -> None:
        registry = _make_registry(1, 2)
        registry.mark_seen(2)
        online_ids = {s.fixture_id for s in registry.online_fixtures()}
        assert online_ids == {2}


# ─── FixtureRegistry.check_timeouts() ────────────────────────────────────────


class TestFixtureRegistryCheckTimeouts:
    def test_fixtures_never_seen_stay_offline(self) -> None:
        registry = _make_registry(1, 2)
        registry.check_timeouts(timeout=10.0)
        assert registry.online_fixtures() == []

    def test_recently_seen_fixtures_stay_online(self) -> None:
        registry = _make_registry(1)
        registry.mark_seen(1)
        registry.check_timeouts(timeout=10.0)
        assert len(registry.online_fixtures()) == 1

    def test_stale_fixture_goes_offline(self) -> None:
        registry = _make_registry(1)
        registry.mark_seen(1)
        state = registry.get(1)
        assert state is not None
        # Backdate last_seen so it exceeds the timeout.
        state.last_seen = time.monotonic() - 20.0
        registry.check_timeouts(timeout=10.0)
        assert state.online is False
        assert registry.online_fixtures() == []

    def test_only_stale_fixture_goes_offline(self) -> None:
        registry = _make_registry(1, 2)
        registry.mark_seen(1)
        registry.mark_seen(2)

        # Backdate only fixture 1.
        s1 = registry.get(1)
        assert s1 is not None
        s1.last_seen = time.monotonic() - 30.0

        registry.check_timeouts(timeout=10.0)

        assert s1.online is False
        s2 = registry.get(2)
        assert s2 is not None
        assert s2.online is True

    def test_custom_timeout_respected(self) -> None:
        registry = _make_registry(1)
        registry.mark_seen(1)
        state = registry.get(1)
        assert state is not None
        # 3 seconds ago — within a 5-second timeout, outside a 2-second timeout.
        state.last_seen = time.monotonic() - 3.0

        registry.check_timeouts(timeout=5.0)
        assert state.online is True  # 3s < 5s timeout

        registry.check_timeouts(timeout=2.0)
        assert state.online is False  # 3s > 2s timeout

    def test_already_offline_fixture_is_not_re_offlined(self) -> None:
        """check_timeouts must not error or double-trigger on already-offline fixtures."""
        registry = _make_registry(1)
        # Fixture was never seen; check_timeouts must be a no-op for it.
        registry.check_timeouts(timeout=1.0)
        state = registry.get(1)
        assert state is not None
        assert state.online is False

    def test_exact_timeout_boundary_stays_online(self) -> None:
        """A fixture seen exactly at the timeout boundary (elapsed == timeout) stays online
        because check_timeouts uses strict > comparison."""
        registry = _make_registry(1)
        registry.mark_seen(1)
        state = registry.get(1)
        assert state is not None

        fake_seen = 1000.0
        state.last_seen = fake_seen

        # elapsed == timeout exactly: not strictly greater, so stays online.
        with patch("lumina.control.fixture.time.monotonic", return_value=fake_seen + 10.0):
            registry.check_timeouts(timeout=10.0)

        assert state.online is True

    def test_multiple_fixtures_mixed_timeouts(self) -> None:
        registry = _make_registry(1, 2, 3)
        for fid in [1, 2, 3]:
            registry.mark_seen(fid)

        now = time.monotonic()
        s1 = registry.get(1)
        s2 = registry.get(2)
        s3 = registry.get(3)
        assert s1 and s2 and s3

        s1.last_seen = now - 5.0   # fresh (5s < 10s timeout)
        s2.last_seen = now - 15.0  # stale  (15s > 10s timeout)
        s3.last_seen = now - 25.0  # stale  (25s > 10s timeout)

        registry.check_timeouts(timeout=10.0)

        assert s1.online is True
        assert s2.online is False
        assert s3.online is False


# ─── FixtureRegistry.update_command() ────────────────────────────────────────


class TestFixtureRegistryUpdateCommand:
    def test_update_command_stores_on_fixture(self) -> None:
        registry = _make_registry(1)
        cmd = _cmd(1, red=100, green=50, blue=25)
        registry.update_command(cmd)
        state = registry.get(1)
        assert state is not None
        assert state.last_command is cmd

    def test_update_command_overwrites_previous(self) -> None:
        registry = _make_registry(1)
        first = _cmd(1, red=10)
        second = _cmd(1, red=200)
        registry.update_command(first)
        registry.update_command(second)
        state = registry.get(1)
        assert state is not None
        assert state.last_command is second

    def test_update_command_unknown_id_does_not_raise(self) -> None:
        registry = _make_registry(1)
        cmd = _cmd(99, red=255)
        registry.update_command(cmd)  # Should warn and return silently.

    def test_update_command_unknown_id_does_not_change_known_fixture(self) -> None:
        registry = _make_registry(1)
        registry.update_command(_cmd(99, red=255))
        state = registry.get(1)
        assert state is not None
        assert state.last_command is None

    def test_update_command_broadcast_applies_to_all(self) -> None:
        registry = _make_registry(1, 2, 3)
        broadcast = FixtureCommand(fixture_id=0, red=128, green=128, blue=128)
        registry.update_command(broadcast)
        for fid in [1, 2, 3]:
            state = registry.get(fid)
            assert state is not None
            assert state.last_command is broadcast

    def test_update_command_broadcast_only_touches_registered_fixtures(self) -> None:
        """Broadcast applies only to registered fixtures; no errors for unregistered."""
        registry = _make_registry(1)
        broadcast = FixtureCommand(fixture_id=0, red=255)
        registry.update_command(broadcast)
        state = registry.get(1)
        assert state is not None
        assert state.last_command is broadcast

    def test_update_command_does_not_affect_other_fixtures(self) -> None:
        registry = _make_registry(1, 2)
        registry.update_command(_cmd(1, red=99))
        state2 = registry.get(2)
        assert state2 is not None
        assert state2.last_command is None


# ─── FixtureRegistry.apply_commands() ────────────────────────────────────────


class TestFixtureRegistryApplyCommands:
    def test_apply_stores_last_command_on_each_fixture(self) -> None:
        registry = _make_registry(1, 2, 3)
        commands = [
            _cmd(1, red=10),
            _cmd(2, green=20),
            _cmd(3, blue=30),
        ]
        registry.apply_commands(commands)
        assert registry.get(1) is not None and registry.get(1).last_command == commands[0]  # type: ignore[union-attr]
        assert registry.get(2) is not None and registry.get(2).last_command == commands[1]  # type: ignore[union-attr]
        assert registry.get(3) is not None and registry.get(3).last_command == commands[2]  # type: ignore[union-attr]

    def test_apply_empty_list_is_no_op(self) -> None:
        registry = _make_registry(1)
        registry.apply_commands([])
        state = registry.get(1)
        assert state is not None
        assert state.last_command is None

    def test_apply_partial_fixture_set(self) -> None:
        """Only commanded fixtures get updated; others stay None."""
        registry = _make_registry(1, 2, 3)
        registry.apply_commands([_cmd(2, red=99)])
        assert registry.get(1) is not None and registry.get(1).last_command is None  # type: ignore[union-attr]
        assert registry.get(2) is not None and registry.get(2).last_command is not None  # type: ignore[union-attr]
        assert registry.get(3) is not None and registry.get(3).last_command is None  # type: ignore[union-attr]

    def test_apply_broadcast_command(self) -> None:
        registry = _make_registry(1, 2, 3)
        broadcast = FixtureCommand(fixture_id=0, white=255)
        registry.apply_commands([broadcast])
        for fid in [1, 2, 3]:
            state = registry.get(fid)
            assert state is not None
            assert state.last_command is broadcast

    def test_apply_preserves_command_values(self) -> None:
        registry = _make_registry(5)
        cmd = FixtureCommand(
            fixture_id=5,
            red=10,
            green=20,
            blue=30,
            white=40,
            strobe_rate=50,
            strobe_intensity=60,
            special=70,
        )
        registry.apply_commands([cmd])
        state = registry.get(5)
        assert state is not None
        lc = state.last_command
        assert lc is not None
        assert lc.red == 10
        assert lc.green == 20
        assert lc.blue == 30
        assert lc.white == 40
        assert lc.strobe_rate == 50
        assert lc.strobe_intensity == 60
        assert lc.special == 70

    def test_apply_unknown_fixture_id_does_not_raise(self) -> None:
        registry = _make_registry(1)
        registry.apply_commands([_cmd(99, red=128)])
        # Fixture 1 should be untouched.
        state = registry.get(1)
        assert state is not None
        assert state.last_command is None

    def test_apply_last_command_wins_for_duplicate_fixture(self) -> None:
        """When two commands target the same fixture in one batch, the last wins."""
        registry = _make_registry(1)
        cmd_a = _cmd(1, red=10)
        cmd_b = _cmd(1, red=200)
        registry.apply_commands([cmd_a, cmd_b])
        state = registry.get(1)
        assert state is not None
        assert state.last_command is cmd_b

    def test_apply_20_fixture_frame(self) -> None:
        """Simulate a realistic 20-fixture lighting frame."""
        ids = list(range(1, 21))
        registry = _make_registry(*ids)
        commands = [_cmd(fid, red=fid * 12 % 256) for fid in ids]
        registry.apply_commands(commands)
        for cmd in commands:
            state = registry.get(cmd.fixture_id)
            assert state is not None
            assert state.last_command is cmd


# ─── FixtureRegistry.__len__ ──────────────────────────────────────────────────


class TestFixtureRegistryLen:
    def test_len_reflects_fixture_count(self) -> None:
        assert len(_make_registry(1, 2, 3)) == 3
        assert len(_make_registry(10)) == 1
        assert len(FixtureRegistry(FixtureMap())) == 15


# ─── FixtureRegistry.summary() ───────────────────────────────────────────────


class TestFixtureRegistrySummary:
    def test_summary_returns_string(self) -> None:
        registry = _make_registry(1, 2)
        s = registry.summary()
        assert isinstance(s, str)

    def test_summary_includes_fixture_count(self) -> None:
        registry = _make_registry(1, 2, 3)
        s = registry.summary()
        assert "3" in s

    def test_summary_shows_online_count(self) -> None:
        registry = _make_registry(1, 2, 3)
        registry.mark_seen(1)
        s = registry.summary()
        assert "1 online" in s

    def test_summary_includes_fixture_id(self) -> None:
        registry = _make_registry(42)
        s = registry.summary()
        assert "42" in s

    def test_summary_includes_command_info_when_set(self) -> None:
        registry = _make_registry(1)
        registry.apply_commands([FixtureCommand(fixture_id=1, red=200, green=100, blue=50)])
        s = registry.summary()
        assert "200" in s
        assert "100" in s
        assert "50" in s

    def test_summary_non_empty(self) -> None:
        registry = _make_registry(1)
        assert len(registry.summary()) > 0
