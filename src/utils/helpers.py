"""Utility helper functions for the XAUUSD Trading Bot."""

from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd


def utc_now() -> datetime:
    """Return the current UTC datetime."""
    return datetime.now(tz=timezone.utc)


def round_to_tick(value: float, tick_size: float) -> float:
    """
    Round a price to the nearest tick size.

    Args:
        value: The raw price value.
        tick_size: The instrument's minimum price increment.

    Returns:
        Price rounded to the nearest tick.
    """
    if tick_size <= 0:
        return value
    return round(round(value / tick_size) * tick_size, 10)


def calculate_pnl(
    entry_price: float,
    exit_price: float,
    quantity: float,
    direction: str,
) -> float:
    """
    Calculate the profit / loss for a closed trade.

    Args:
        entry_price: Price at which the trade was entered.
        exit_price: Price at which the trade was closed.
        quantity: Number of units traded.
        direction: 'long' or 'short'.

    Returns:
        Signed P&L value (positive = profit, negative = loss).
    """
    if direction.lower() == "long":
        return (exit_price - entry_price) * quantity
    return (entry_price - exit_price) * quantity


def calculate_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Calculate the Average True Range (ATR).

    Args:
        high: High price series.
        low: Low price series.
        close: Close price series.
        period: Look-back period (default 14).

    Returns:
        ATR series.
    """
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def is_trading_session(
    dt: Optional[datetime] = None,
    london_open: int = 8,
    london_close: int = 17,
    ny_open: int = 13,
    ny_close: int = 22,
) -> bool:
    """
    Check whether a given UTC datetime falls within the London or New York trading
    sessions (or their overlap).

    Args:
        dt: UTC datetime to check. Defaults to current UTC time.
        london_open: London session open hour (UTC, default 8).
        london_close: London session close hour (UTC, default 17).
        ny_open: New York session open hour (UTC, default 13).
        ny_close: New York session close hour (UTC, default 22).

    Returns:
        True if within an active trading session, False otherwise.
    """
    if dt is None:
        dt = utc_now()
    hour = dt.hour
    in_london = london_open <= hour < london_close
    in_ny = ny_open <= hour < ny_close
    return in_london or in_ny


def compute_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Compute the annualised Sharpe ratio for a returns series.

    Args:
        returns: Series of period returns (not cumulative).
        risk_free_rate: Annualised risk-free rate (default 0).
        periods_per_year: Number of trading periods per year (default 252 for daily).

    Returns:
        Annualised Sharpe ratio. Returns 0.0 if standard deviation is zero.
    """
    excess = returns - risk_free_rate / periods_per_year
    std = excess.std()
    if std == 0 or np.isnan(std):
        return 0.0
    return float(excess.mean() / std * np.sqrt(periods_per_year))


def compute_max_drawdown(equity_curve: pd.Series) -> float:
    """
    Compute the maximum drawdown from peak to trough.

    Args:
        equity_curve: Series of equity values over time.

    Returns:
        Maximum drawdown as a positive fraction (0.0 to 1.0).
    """
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max
    return float(drawdown.min()) * -1  # return positive value


def candles_to_dataframe(candles: list) -> pd.DataFrame:
    """
    Convert a list of raw OHLCV candle dicts to a pandas DataFrame.

    Expected dict keys: 'timestamp', 'open', 'high', 'low', 'close', 'volume'

    Args:
        candles: List of OHLCV dictionaries.

    Returns:
        DataFrame with a DatetimeIndex and typed OHLCV columns.
    """
    df = pd.DataFrame(candles)
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = df[col].astype(float)
    return df
