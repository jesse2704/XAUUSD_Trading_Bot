"""Unit tests for all trading strategies."""

import numpy as np
import pandas as pd
import pytest

from src.models.signals import SignalDirection
from src.strategies.breakout import BreakoutStrategy
from src.strategies.confluence import ConfluenceStrategy
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.momentum import MomentumStrategy
from src.strategies.vwap import VWAPStrategy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ohlcv(
    n: int = 300,
    base_price: float = 2000.0,
    trend: float = 0.0,
    noise: float = 5.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic OHLCV data with an optional trend component."""
    rng = np.random.default_rng(seed)
    closes = base_price + trend * np.arange(n) + rng.normal(0, noise, n)
    highs = closes + rng.uniform(1, 3, n)
    lows = closes - rng.uniform(1, 3, n)
    opens = closes + rng.normal(0, 1, n)
    volumes = rng.uniform(1000, 5000, n)

    index = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes},
        index=index,
    )


def _make_uptrend_df(n: int = 300) -> pd.DataFrame:
    """DataFrame with a clear uptrend for momentum signals."""
    return _make_ohlcv(n=n, trend=0.5, noise=0.5)


def _make_downtrend_df(n: int = 300) -> pd.DataFrame:
    """DataFrame with a clear downtrend for momentum signals."""
    return _make_ohlcv(n=n, trend=-0.5, noise=0.5)


def _make_flat_df(n: int = 300) -> pd.DataFrame:
    """DataFrame with no trend (flat / mean-reverting)."""
    return _make_ohlcv(n=n, trend=0.0, noise=0.3)


# ---------------------------------------------------------------------------
# MomentumStrategy
# ---------------------------------------------------------------------------


class TestMomentumStrategy:
    def test_returns_signal(self):
        strategy = MomentumStrategy()
        df = _make_uptrend_df()
        sig = strategy.generate_signal(df)
        assert sig is not None
        assert isinstance(sig.direction, SignalDirection)

    def test_neutral_on_insufficient_data(self):
        strategy = MomentumStrategy()
        df = _make_ohlcv(n=5)
        sig = strategy.generate_signal(df)
        assert sig.direction == SignalDirection.NEUTRAL

    def test_signal_has_stop_loss_when_not_neutral(self):
        strategy = MomentumStrategy()
        df = _make_uptrend_df(n=300)
        sig = strategy.generate_signal(df)
        if sig.direction != SignalDirection.NEUTRAL:
            assert sig.stop_loss is not None
            assert sig.take_profit is not None

    def test_strength_in_valid_range(self):
        strategy = MomentumStrategy()
        df = _make_uptrend_df(n=300)
        sig = strategy.generate_signal(df)
        assert -1.0 <= sig.strength <= 1.0

    def test_last_signal_cached(self):
        strategy = MomentumStrategy()
        df = _make_uptrend_df(n=300)
        sig = strategy.generate_signal(df)
        assert strategy.last_signal is sig

    def test_metadata_contains_indicators(self):
        strategy = MomentumStrategy()
        df = _make_uptrend_df(n=300)
        sig = strategy.generate_signal(df)
        if sig.direction != SignalDirection.NEUTRAL:
            assert "rsi" in sig.metadata
            assert "adx" in sig.metadata
            assert "atr" in sig.metadata


# ---------------------------------------------------------------------------
# MeanReversionStrategy
# ---------------------------------------------------------------------------


class TestMeanReversionStrategy:
    def test_returns_signal(self):
        strategy = MeanReversionStrategy()
        df = _make_flat_df()
        sig = strategy.generate_signal(df)
        assert sig is not None
        assert isinstance(sig.direction, SignalDirection)

    def test_neutral_on_insufficient_data(self):
        strategy = MeanReversionStrategy()
        df = _make_ohlcv(n=5)
        sig = strategy.generate_signal(df)
        assert sig.direction == SignalDirection.NEUTRAL

    def test_signal_strength_bounded(self):
        strategy = MeanReversionStrategy()
        df = _make_flat_df(n=200)
        sig = strategy.generate_signal(df)
        assert 0.0 <= sig.strength <= 1.0

    def test_long_signal_stop_below_entry(self):
        """Long stop-loss must be below entry price."""
        strategy = MeanReversionStrategy()
        df = _make_flat_df(n=200)
        sig = strategy.generate_signal(df)
        if sig.direction == SignalDirection.LONG:
            assert sig.stop_loss < sig.entry_price

    def test_short_signal_stop_above_entry(self):
        """Short stop-loss must be above entry price."""
        strategy = MeanReversionStrategy()
        df = _make_flat_df(n=200)
        sig = strategy.generate_signal(df)
        if sig.direction == SignalDirection.SHORT:
            assert sig.stop_loss > sig.entry_price


# ---------------------------------------------------------------------------
# BreakoutStrategy
# ---------------------------------------------------------------------------


class TestBreakoutStrategy:
    def test_returns_signal(self):
        strategy = BreakoutStrategy()
        df = _make_ohlcv(n=300)
        sig = strategy.generate_signal(df)
        assert sig is not None

    def test_neutral_on_insufficient_data(self):
        strategy = BreakoutStrategy()
        df = _make_ohlcv(n=10)
        sig = strategy.generate_signal(df)
        assert sig.direction == SignalDirection.NEUTRAL

    def test_breakout_long_when_price_above_upper(self):
        """Inject a strong upside breakout and verify a LONG signal is possible."""
        strategy = BreakoutStrategy()
        df = _make_ohlcv(n=250, base_price=2000.0, noise=0.1)
        # Force the last bar to be a massive spike above the channel
        df.iloc[-1, df.columns.get_loc("close")] = 2500.0
        df.iloc[-1, df.columns.get_loc("high")] = 2500.0
        df.iloc[-1, df.columns.get_loc("volume")] = 1e9  # massive volume
        sig = strategy.generate_signal(df)
        assert sig.direction == SignalDirection.LONG

    def test_breakout_short_when_price_below_lower(self):
        """Inject a strong downside breakout and verify a SHORT signal is possible."""
        strategy = BreakoutStrategy()
        df = _make_ohlcv(n=250, base_price=2000.0, noise=0.1)
        df.iloc[-1, df.columns.get_loc("close")] = 1500.0
        df.iloc[-1, df.columns.get_loc("low")] = 1500.0
        df.iloc[-1, df.columns.get_loc("volume")] = 1e9
        sig = strategy.generate_signal(df)
        assert sig.direction == SignalDirection.SHORT

    def test_metadata_has_volume_spike(self):
        strategy = BreakoutStrategy()
        df = _make_ohlcv(n=250)
        sig = strategy.generate_signal(df)
        if sig.direction != SignalDirection.NEUTRAL:
            assert "volume_spike" in sig.metadata


# ---------------------------------------------------------------------------
# VWAPStrategy
# ---------------------------------------------------------------------------


class TestVWAPStrategy:
    def test_returns_signal(self):
        strategy = VWAPStrategy()
        df = _make_ohlcv(n=100)
        sig = strategy.generate_signal(df)
        assert sig is not None

    def test_neutral_on_insufficient_data(self):
        strategy = VWAPStrategy()
        df = _make_ohlcv(n=5)
        sig = strategy.generate_signal(df)
        assert sig.direction == SignalDirection.NEUTRAL

    def test_neutral_outside_trading_session(self):
        """Signal should be NEUTRAL outside London/NY sessions."""
        strategy = VWAPStrategy()
        df = _make_ohlcv(n=100)
        # Move all timestamps to 3 AM UTC (outside both sessions)
        df.index = pd.date_range("2024-01-02 03:00", periods=100, freq="1h", tz="UTC")
        sig = strategy.generate_signal(df)
        assert sig.direction == SignalDirection.NEUTRAL

    def test_signal_during_london_session(self):
        """Strategy may (but is not required to) signal during London hours."""
        strategy = VWAPStrategy()
        df = _make_ohlcv(n=100)
        df.index = pd.date_range("2024-01-02 09:00", periods=100, freq="1h", tz="UTC")
        sig = strategy.generate_signal(df)
        # We just verify it does not raise an exception
        assert isinstance(sig.direction, SignalDirection)


# ---------------------------------------------------------------------------
# ConfluenceStrategy
# ---------------------------------------------------------------------------


class TestConfluenceStrategy:
    def test_returns_signal(self):
        strategy = ConfluenceStrategy()
        df = _make_ohlcv(n=300)
        sig = strategy.generate_signal(df)
        assert sig is not None

    def test_neutral_on_insufficient_data(self):
        strategy = ConfluenceStrategy()
        df = _make_ohlcv(n=5)
        sig = strategy.generate_signal(df)
        assert sig.direction == SignalDirection.NEUTRAL

    def test_metadata_contains_votes(self):
        strategy = ConfluenceStrategy()
        df = _make_ohlcv(n=300)
        sig = strategy.generate_signal(df)
        assert "votes" in sig.metadata
        assert "weighted_score" in sig.metadata

    def test_weights_start_equal(self):
        strategy = ConfluenceStrategy()
        assert all(w == 1.0 for w in strategy.weights)

    def test_update_weights_long(self):
        strategy = ConfluenceStrategy()
        df = _make_ohlcv(n=300)
        strategy.generate_signal(df)  # populate last_signal on sub-strategies
        strategy.update_weights(SignalDirection.LONG)
        # Weights should still be normalised (mean ≈ 1.0)
        assert abs(sum(strategy.weights) / len(strategy.weights) - 1.0) < 1e-9

    def test_sub_strategies_count(self):
        strategy = ConfluenceStrategy()
        assert len(strategy.sub_strategies) == 4

    def test_votes_list_length_matches_sub_strategies(self):
        strategy = ConfluenceStrategy()
        df = _make_ohlcv(n=300)
        sig = strategy.generate_signal(df)
        assert len(sig.metadata.get("votes", [])) == len(strategy.sub_strategies)
