"""Tests for lumina.lighting.safety — photosensitive seizure prevention."""

from __future__ import annotations

from lumina.control.protocol import FixtureCommand
from lumina.lighting.safety import SafetyLevel, SafetyLimiter


def _cmd(
    fixture_id: int = 1,
    red: int = 0,
    green: int = 0,
    blue: int = 0,
    white: int = 0,
    strobe_rate: int = 0,
    strobe_intensity: int = 0,
    special: int = 0,
) -> FixtureCommand:
    """Helper to create a FixtureCommand with defaults."""
    return FixtureCommand(
        fixture_id=fixture_id,
        red=red,
        green=green,
        blue=blue,
        white=white,
        strobe_rate=strobe_rate,
        strobe_intensity=strobe_intensity,
        special=special,
    )


class _FakeClock:
    """Deterministic clock for testing time-dependent safety logic."""

    def __init__(self, start: float = 0.0) -> None:
        self._now = start

    def __call__(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


class TestStrobeRateCapping:
    """Test sustained strobe rate limits."""

    def test_rate_below_limit_unchanged(self) -> None:
        clock = _FakeClock()
        limiter = SafetyLimiter(SafetyLevel.STANDARD, time_fn=clock)
        cmd = _cmd(strobe_rate=20, strobe_intensity=200)
        result = limiter.process([cmd])
        assert result[0].strobe_rate == 20

    def test_rate_above_sustained_capped(self) -> None:
        """Strobe rate above 3Hz sustained limit (30) gets capped."""
        clock = _FakeClock()
        limiter = SafetyLimiter(SafetyLevel.STANDARD, time_fn=clock)
        # Rate 200 is way above sustained limit of 30
        # But first frame starts a burst, so it's capped to burst limit (102)
        cmd = _cmd(strobe_rate=200, strobe_intensity=255)
        result = limiter.process([cmd])
        assert result[0].strobe_rate == 102  # burst limit

    def test_sustained_rate_at_limit_unchanged(self) -> None:
        clock = _FakeClock()
        limiter = SafetyLimiter(SafetyLevel.STANDARD, time_fn=clock)
        cmd = _cmd(strobe_rate=30, strobe_intensity=200)
        result = limiter.process([cmd])
        assert result[0].strobe_rate == 30

    def test_zero_strobe_rate_unchanged(self) -> None:
        clock = _FakeClock()
        limiter = SafetyLimiter(SafetyLevel.STANDARD, time_fn=clock)
        cmd = _cmd(strobe_rate=0, strobe_intensity=0)
        result = limiter.process([cmd])
        assert result[0].strobe_rate == 0


class TestBurstMode:
    """Test burst strobe: 10Hz for max 1s, then 3s cooldown."""

    def test_burst_allowed_within_1_second(self) -> None:
        clock = _FakeClock()
        limiter = SafetyLimiter(SafetyLevel.STANDARD, time_fn=clock)
        cmd = _cmd(strobe_rate=100, strobe_intensity=255)

        # First call starts burst
        result = limiter.process([cmd])
        assert result[0].strobe_rate == 100  # within burst limit (102)

        # Still within 1s
        clock.advance(0.5)
        result = limiter.process([cmd])
        assert result[0].strobe_rate == 100

    def test_burst_expires_after_1_second(self) -> None:
        clock = _FakeClock()
        limiter = SafetyLimiter(SafetyLevel.STANDARD, time_fn=clock)
        cmd = _cmd(strobe_rate=100, strobe_intensity=255)

        # Start burst
        limiter.process([cmd])

        # Advance past 1s burst window
        clock.advance(1.1)
        result = limiter.process([cmd])
        # Now capped to sustained limit (30)
        assert result[0].strobe_rate == 30

    def test_cooldown_after_burst(self) -> None:
        clock = _FakeClock()
        limiter = SafetyLimiter(SafetyLevel.STANDARD, time_fn=clock)
        cmd = _cmd(strobe_rate=100, strobe_intensity=255)

        # Start and exhaust burst
        limiter.process([cmd])
        clock.advance(1.1)
        limiter.process([cmd])  # triggers cooldown

        # During cooldown (3s), high rate is capped to sustained
        clock.advance(1.0)
        result = limiter.process([cmd])
        assert result[0].strobe_rate == 30

    def test_burst_available_after_cooldown(self) -> None:
        clock = _FakeClock()
        limiter = SafetyLimiter(SafetyLevel.STANDARD, time_fn=clock)
        cmd = _cmd(strobe_rate=100, strobe_intensity=255)

        # Start burst
        limiter.process([cmd])
        # Exhaust burst
        clock.advance(1.1)
        limiter.process([cmd])  # enters cooldown at t=1.1
        # Wait out cooldown (3s from t=1.1)
        clock.advance(3.1)

        # Now a new burst should be allowed
        result = limiter.process([cmd])
        assert result[0].strobe_rate == 100

    def test_burst_rate_capped_at_burst_limit(self) -> None:
        clock = _FakeClock()
        limiter = SafetyLimiter(SafetyLevel.STANDARD, time_fn=clock)
        # Rate 255 exceeds burst limit of 102
        cmd = _cmd(strobe_rate=255, strobe_intensity=255)
        result = limiter.process([cmd])
        assert result[0].strobe_rate == 102


class TestSimultaneousStrobeLimit:
    """Test that max 8 fixtures can strobe simultaneously (STANDARD)."""

    def test_within_limit_unchanged(self) -> None:
        clock = _FakeClock()
        limiter = SafetyLimiter(SafetyLevel.STANDARD, time_fn=clock)
        commands = [
            _cmd(fixture_id=i, strobe_rate=20, strobe_intensity=200)
            for i in range(1, 9)  # 8 fixtures
        ]
        result = limiter.process(commands)
        strobing = [c for c in result if c.strobe_rate > 0]
        assert len(strobing) == 8

    def test_exceeding_limit_kills_lowest_rate(self) -> None:
        clock = _FakeClock()
        limiter = SafetyLimiter(SafetyLevel.STANDARD, time_fn=clock)
        # 10 fixtures strobing at different rates
        commands = [
            _cmd(fixture_id=i, strobe_rate=i * 3, strobe_intensity=200)
            for i in range(1, 11)
        ]
        result = limiter.process(commands)
        strobing = [c for c in result if c.strobe_rate > 0]
        assert len(strobing) == 8

        # The 2 lowest-rate fixtures should have been killed
        killed = [c for c in result if c.strobe_rate == 0]
        assert len(killed) == 2
        # Fixture IDs 1 and 2 had the lowest rates (3 and 6)
        killed_ids = {c.fixture_id for c in killed}
        assert killed_ids == {1, 2}

    def test_non_strobing_fixtures_unaffected(self) -> None:
        clock = _FakeClock()
        limiter = SafetyLimiter(SafetyLevel.STANDARD, time_fn=clock)
        commands = [
            _cmd(fixture_id=i, strobe_rate=20, strobe_intensity=200)
            for i in range(1, 11)  # 10 strobing
        ]
        # Add 5 non-strobing fixtures
        commands.extend(
            _cmd(fixture_id=i, red=255, green=100, blue=50, special=200)
            for i in range(11, 16)
        )
        result = limiter.process(commands)

        # Non-strobing fixtures should keep their colors
        for cmd in result:
            if cmd.fixture_id >= 11:
                assert cmd.red == 255
                assert cmd.green == 100
                assert cmd.blue == 50


class TestBrightnessCeiling:
    """Test total brightness ceiling enforcement."""

    def test_below_ceiling_unchanged(self) -> None:
        clock = _FakeClock()
        limiter = SafetyLimiter(SafetyLevel.STANDARD, time_fn=clock)
        # Low brightness
        commands = [
            _cmd(fixture_id=i, red=50, green=50, blue=50)
            for i in range(1, 6)
        ]
        result = limiter.process(commands)
        for orig, lim in zip(commands, result):
            assert lim.red == orig.red
            assert lim.green == orig.green
            assert lim.blue == orig.blue

    def test_above_ceiling_scaled_down(self) -> None:
        clock = _FakeClock()
        limiter = SafetyLimiter(SafetyLevel.STANDARD, time_fn=clock)
        # All channels at max for 15 fixtures
        commands = [
            _cmd(
                fixture_id=i,
                red=255,
                green=255,
                blue=255,
                white=255,
                strobe_intensity=255,
                special=255,
            )
            for i in range(1, 16)
        ]
        result = limiter.process(commands)

        # Total should be at or below 80% of max
        total = sum(
            c.red + c.green + c.blue + c.white + c.strobe_intensity + c.special
            for c in result
        )
        max_total = 15 * 6 * 255
        assert total <= max_total * 0.80 + 1  # +1 for rounding tolerance

    def test_brightness_ceiling_preserves_strobe_rate(self) -> None:
        clock = _FakeClock()
        limiter = SafetyLimiter(SafetyLevel.STANDARD, time_fn=clock)
        # Use only 8 strobing fixtures (within simultaneous limit)
        commands = [
            _cmd(
                fixture_id=i,
                red=255,
                green=255,
                blue=255,
                white=255,
                strobe_rate=20,
                strobe_intensity=255,
                special=255,
            )
            for i in range(1, 9)
        ]
        result = limiter.process(commands)
        # strobe_rate should NOT be scaled by brightness limiter
        for cmd in result:
            assert cmd.strobe_rate == 20


class TestAccessibleLevel:
    """Test ACCESSIBLE safety level (stricter limits)."""

    def test_strobe_rate_halved(self) -> None:
        clock = _FakeClock()
        limiter = SafetyLimiter(SafetyLevel.ACCESSIBLE, time_fn=clock)
        # Sustained limit is 30 // 2 = 15
        cmd = _cmd(strobe_rate=20, strobe_intensity=200)
        result = limiter.process([cmd])
        # 20 > 15 sustained, starts a burst. Burst limit = 102 // 2 = 51
        assert result[0].strobe_rate == 20  # within burst limit

    def test_accessible_sustained_limit(self) -> None:
        clock = _FakeClock()
        limiter = SafetyLimiter(SafetyLevel.ACCESSIBLE, time_fn=clock)
        # Sustained limit is 15. Rate 15 should pass.
        cmd = _cmd(strobe_rate=15, strobe_intensity=200)
        result = limiter.process([cmd])
        assert result[0].strobe_rate == 15

    def test_accessible_simultaneous_limit(self) -> None:
        clock = _FakeClock()
        limiter = SafetyLimiter(SafetyLevel.ACCESSIBLE, time_fn=clock)
        # Max simultaneous is 8 // 2 = 4
        commands = [
            _cmd(fixture_id=i, strobe_rate=10, strobe_intensity=200)
            for i in range(1, 7)  # 6 strobing
        ]
        result = limiter.process(commands)
        strobing = [c for c in result if c.strobe_rate > 0]
        assert len(strobing) == 4

    def test_accessible_brightness_ceiling(self) -> None:
        clock = _FakeClock()
        limiter = SafetyLimiter(SafetyLevel.ACCESSIBLE, time_fn=clock)
        commands = [
            _cmd(
                fixture_id=i,
                red=255,
                green=255,
                blue=255,
                white=255,
                strobe_intensity=255,
                special=255,
            )
            for i in range(1, 16)
        ]
        result = limiter.process(commands)
        total = sum(
            c.red + c.green + c.blue + c.white + c.strobe_intensity + c.special
            for c in result
        )
        max_total = 15 * 6 * 255
        assert total <= max_total * 0.60 + 1


class TestUnrestrictedLevel:
    """Test UNRESTRICTED safety level (no limiting)."""

    def test_no_strobe_rate_limiting(self) -> None:
        clock = _FakeClock()
        limiter = SafetyLimiter(SafetyLevel.UNRESTRICTED, time_fn=clock)
        cmd = _cmd(strobe_rate=255, strobe_intensity=255)
        result = limiter.process([cmd])
        assert result[0].strobe_rate == 255

    def test_no_simultaneous_limiting(self) -> None:
        clock = _FakeClock()
        limiter = SafetyLimiter(SafetyLevel.UNRESTRICTED, time_fn=clock)
        commands = [
            _cmd(fixture_id=i, strobe_rate=255, strobe_intensity=255)
            for i in range(1, 16)
        ]
        result = limiter.process(commands)
        strobing = [c for c in result if c.strobe_rate > 0]
        assert len(strobing) == 15

    def test_no_brightness_limiting(self) -> None:
        clock = _FakeClock()
        limiter = SafetyLimiter(SafetyLevel.UNRESTRICTED, time_fn=clock)
        commands = [
            _cmd(
                fixture_id=i,
                red=255,
                green=255,
                blue=255,
                white=255,
                strobe_intensity=255,
                special=255,
            )
            for i in range(1, 16)
        ]
        result = limiter.process(commands)
        total = sum(
            c.red + c.green + c.blue + c.white + c.strobe_intensity + c.special
            for c in result
        )
        assert total == 15 * 6 * 255


class TestStats:
    """Test that safety statistics are tracked correctly."""

    def test_frames_counted(self) -> None:
        clock = _FakeClock()
        limiter = SafetyLimiter(SafetyLevel.STANDARD, time_fn=clock)
        limiter.process([_cmd()])
        limiter.process([_cmd()])
        assert limiter.get_stats()["frames_processed"] == 2

    def test_strobe_rate_caps_counted(self) -> None:
        clock = _FakeClock()
        limiter = SafetyLimiter(SafetyLevel.STANDARD, time_fn=clock)
        # Rate 200 will be capped to burst limit
        limiter.process([_cmd(strobe_rate=200, strobe_intensity=255)])
        assert limiter.get_stats()["strobe_rate_caps"] >= 1

    def test_simultaneous_caps_counted(self) -> None:
        clock = _FakeClock()
        limiter = SafetyLimiter(SafetyLevel.STANDARD, time_fn=clock)
        commands = [
            _cmd(fixture_id=i, strobe_rate=20, strobe_intensity=200)
            for i in range(1, 12)  # 11 > 8 limit
        ]
        limiter.process(commands)
        assert limiter.get_stats()["simultaneous_strobe_caps"] >= 1

    def test_brightness_caps_counted(self) -> None:
        clock = _FakeClock()
        limiter = SafetyLimiter(SafetyLevel.STANDARD, time_fn=clock)
        commands = [
            _cmd(
                fixture_id=i,
                red=255,
                green=255,
                blue=255,
                white=255,
                strobe_intensity=255,
                special=255,
            )
            for i in range(1, 16)
        ]
        limiter.process(commands)
        assert limiter.get_stats()["brightness_caps"] >= 1

    def test_stats_returns_copy(self) -> None:
        limiter = SafetyLimiter(SafetyLevel.STANDARD)
        stats1 = limiter.get_stats()
        stats1["frames_processed"] = 999
        assert limiter.get_stats()["frames_processed"] == 0


class TestEdgeCases:
    """Test edge cases: empty commands, single fixture, etc."""

    def test_empty_commands(self) -> None:
        limiter = SafetyLimiter(SafetyLevel.STANDARD)
        result = limiter.process([])
        assert result == []

    def test_single_fixture(self) -> None:
        clock = _FakeClock()
        limiter = SafetyLimiter(SafetyLevel.STANDARD, time_fn=clock)
        cmd = _cmd(fixture_id=1, red=128, green=64, blue=32, special=100)
        result = limiter.process([cmd])
        assert len(result) == 1
        assert result[0].red == 128

    def test_all_fixtures_at_max_strobe(self) -> None:
        clock = _FakeClock()
        limiter = SafetyLimiter(SafetyLevel.STANDARD, time_fn=clock)
        commands = [
            _cmd(fixture_id=i, strobe_rate=255, strobe_intensity=255)
            for i in range(1, 16)
        ]
        result = limiter.process(commands)
        # At most 8 should be strobing
        strobing = [c for c in result if c.strobe_rate > 0]
        assert len(strobing) <= 8
        # All strobing fixtures should be capped to burst limit
        for cmd in strobing:
            assert cmd.strobe_rate <= 102

    def test_fixture_id_preserved(self) -> None:
        clock = _FakeClock()
        limiter = SafetyLimiter(SafetyLevel.STANDARD, time_fn=clock)
        commands = [_cmd(fixture_id=42, red=200, strobe_rate=200)]
        result = limiter.process(commands)
        assert result[0].fixture_id == 42

    def test_strobe_intensity_zeroed_when_strobe_killed(self) -> None:
        """When simultaneous limit kills a strobe, intensity should be 0."""
        clock = _FakeClock()
        limiter = SafetyLimiter(SafetyLevel.STANDARD, time_fn=clock)
        commands = [
            _cmd(fixture_id=i, strobe_rate=20, strobe_intensity=200)
            for i in range(1, 12)  # 11 fixtures, limit is 8
        ]
        result = limiter.process(commands)
        for cmd in result:
            if cmd.strobe_rate == 0:
                assert cmd.strobe_intensity == 0
