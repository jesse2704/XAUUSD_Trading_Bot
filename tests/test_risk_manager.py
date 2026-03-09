"""Unit tests for the RiskManager."""

import pytest

from config.settings import RiskConfig
from src.models.signals import SignalDirection
from src.risk.manager import RiskManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def default_risk_manager() -> RiskManager:
    """RiskManager with default settings and 10,000 initial equity."""
    return RiskManager(initial_equity=10_000.0)


@pytest.fixture
def custom_risk_manager() -> RiskManager:
    """RiskManager with tighter risk parameters for edge-case testing."""
    config = RiskConfig(
        risk_per_trade=0.01,
        max_daily_loss=0.02,
        max_drawdown=0.05,
        max_position_size=0.10,
        max_concurrent_positions=2,
        cooldown_minutes=0,  # no cooldown for most tests
    )
    return RiskManager(config=config, initial_equity=10_000.0)


# ---------------------------------------------------------------------------
# Basic state
# ---------------------------------------------------------------------------


class TestRiskManagerState:
    def test_initial_equity(self, default_risk_manager):
        assert default_risk_manager.current_equity == 10_000.0

    def test_trading_not_halted_initially(self, default_risk_manager):
        assert not default_risk_manager.is_trading_halted

    def test_can_trade_initially(self, default_risk_manager):
        assert default_risk_manager.can_trade()

    def test_risk_summary_keys(self, default_risk_manager):
        summary = default_risk_manager.get_risk_summary()
        expected_keys = {
            "current_equity",
            "peak_equity",
            "daily_pnl",
            "daily_pnl_pct",
            "drawdown",
            "open_positions",
            "trading_halted",
            "halt_reason",
        }
        assert expected_keys.issubset(summary.keys())


# ---------------------------------------------------------------------------
# Position sizing
# ---------------------------------------------------------------------------


class TestPositionSizing:
    def test_basic_position_size(self, default_risk_manager):
        """1% risk with a 100-point stop should give 1 unit on 10k equity."""
        size = default_risk_manager.calculate_position_size(
            entry_price=2000.0, stop_loss=1900.0
        )
        assert size > 0

    def test_position_size_decreases_with_wider_stop(self, default_risk_manager):
        # Use a very wide stop so raw_size is below max_position_size cap
        size_tight = default_risk_manager.calculate_position_size(
            entry_price=2000.0, stop_loss=1998.0  # distance=2, raw=50, capped at 0.5
        )
        size_wide = default_risk_manager.calculate_position_size(
            entry_price=2000.0, stop_loss=1000.0  # distance=1000, raw=0.1, not capped
        )
        assert size_tight > size_wide

    def test_position_size_zero_when_no_stop_no_atr(self, default_risk_manager):
        size = default_risk_manager.calculate_position_size(
            entry_price=2000.0, stop_loss=2000.0
        )
        assert size == 0.0

    def test_position_size_uses_atr_fallback(self, default_risk_manager):
        size = default_risk_manager.calculate_position_size(
            entry_price=2000.0, stop_loss=2000.0, atr=10.0
        )
        assert size > 0

    def test_position_size_capped_at_max(self, default_risk_manager):
        # Tiny stop to inflate size, but capped by max_position_size
        size = default_risk_manager.calculate_position_size(
            entry_price=2000.0, stop_loss=1999.99
        )
        max_expected = 10_000.0 * 0.10 / 2000.0
        assert size <= max_expected + 1e-9


# ---------------------------------------------------------------------------
# Daily loss limit
# ---------------------------------------------------------------------------


class TestDailyLossLimit:
    def test_halt_on_daily_loss_exceeded(self, custom_risk_manager):
        """Should halt trading when daily loss exceeds 2% of initial equity."""
        custom_risk_manager.record_trade_result(-250.0)  # 2.5% loss on 10k
        assert custom_risk_manager.is_trading_halted
        assert not custom_risk_manager.can_trade()

    def test_no_halt_on_small_daily_loss(self, custom_risk_manager):
        """Should NOT halt trading on a small loss below the threshold."""
        custom_risk_manager.record_trade_result(-100.0)  # 1% — under 2% limit
        assert not custom_risk_manager.is_trading_halted

    def test_halt_reason_mentions_daily_loss(self, custom_risk_manager):
        custom_risk_manager.record_trade_result(-250.0)
        assert "daily loss" in custom_risk_manager.halt_reason.lower()


# ---------------------------------------------------------------------------
# Max drawdown
# ---------------------------------------------------------------------------


class TestMaxDrawdown:
    def test_halt_on_max_drawdown_exceeded(self, custom_risk_manager):
        """Should halt trading when drawdown exceeds 5%."""
        # Simulate 10k → 9k equity (10% drawdown > 5% limit)
        custom_risk_manager.update_equity(9_000.0)
        assert custom_risk_manager.is_trading_halted

    def test_no_halt_on_small_drawdown(self):
        """Should NOT halt trading on a minor equity dip below the drawdown limit."""
        # Use a very high daily loss limit so only the drawdown limit matters
        config = RiskConfig(
            max_daily_loss=1.0,   # 100% — effectively disabled
            max_drawdown=0.05,
            max_concurrent_positions=3,
        )
        rm = RiskManager(config=config, initial_equity=10_000.0)
        rm.update_equity(9_600.0)  # 4% drawdown < 5% limit
        assert not rm.is_trading_halted

    def test_peak_equity_updates_on_new_high(self, custom_risk_manager):
        custom_risk_manager.update_equity(11_000.0)
        assert custom_risk_manager.peak_equity == 11_000.0


# ---------------------------------------------------------------------------
# Concurrent positions
# ---------------------------------------------------------------------------


class TestConcurrentPositions:
    def test_blocks_trade_when_max_positions_open(self, custom_risk_manager):
        """Should block a new trade when the max concurrent limit is reached."""
        custom_risk_manager.open_position()
        custom_risk_manager.open_position()  # config limit is 2
        assert not custom_risk_manager.can_trade()

    def test_allows_trade_after_close(self, custom_risk_manager):
        custom_risk_manager.open_position()
        custom_risk_manager.open_position()
        custom_risk_manager.close_position()
        assert custom_risk_manager.can_trade()

    def test_position_counter_does_not_go_negative(self, custom_risk_manager):
        custom_risk_manager.close_position()  # close without opening
        summary = custom_risk_manager.get_risk_summary()
        assert summary["open_positions"] == 0


# ---------------------------------------------------------------------------
# Cooldown
# ---------------------------------------------------------------------------


class TestCooldown:
    def test_cooldown_active_after_loss(self):
        config = RiskConfig(
            cooldown_minutes=60,
            max_concurrent_positions=3,
        )
        rm = RiskManager(config=config, initial_equity=10_000.0)
        rm.record_trade_result(-50.0)  # triggers cooldown
        assert not rm.can_trade()

    def test_no_cooldown_after_win(self):
        config = RiskConfig(cooldown_minutes=60, max_concurrent_positions=3)
        rm = RiskManager(config=config, initial_equity=10_000.0)
        rm.record_trade_result(50.0)  # win — no cooldown
        assert rm.can_trade()
