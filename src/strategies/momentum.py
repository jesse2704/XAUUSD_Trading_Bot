"""Strategy 1: Momentum / Trend Following strategy."""

import pandas as pd
import ta

from config.settings import MomentumStrategyConfig
from src.models.signals import Signal, SignalDirection
from src.strategies.base import BaseStrategy
from src.utils.helpers import calculate_atr


class MomentumStrategy(BaseStrategy):
    """
    Momentum / Trend Following strategy using EMA crossovers, RSI, and ADX.

    Entry logic
    -----------
    *  **Long**: fast EMA crosses above slow EMA **and** RSI > 50 **and**
       ADX > threshold (strong trend) **and** price is above the long-term EMAs.
    *  **Short**: fast EMA crosses below slow EMA **and** RSI < 50 **and**
       ADX > threshold **and** price is below the long-term EMAs.

    Exit is handled externally via stop-loss and take-profit levels that are
    set on each signal (ATR-based).

    Args:
        config: Strategy configuration object.
        symbol: Instrument symbol.
    """

    def __init__(
        self,
        config: MomentumStrategyConfig | None = None,
        symbol: str = "XAUUSD",
    ) -> None:
        super().__init__("Momentum", symbol)
        self.config = config or MomentumStrategyConfig()

    def generate_signal(self, df: pd.DataFrame) -> Signal:
        """
        Compute indicators and return a directional signal.

        Args:
            df: OHLCV DataFrame (requires at least ``ema_long_slow`` + 1 rows).

        Returns:
            Signal with LONG, SHORT, or NEUTRAL direction.
        """
        min_rows = self.config.ema_long_slow + self.config.rsi_period + 1
        if len(df) < min_rows:
            return self._neutral_signal(f"Insufficient data: need {min_rows} rows")

        df = df.copy()
        close = df["close"]

        # EMAs
        ema_fast = ta.trend.EMAIndicator(close, self.config.ema_fast).ema_indicator()
        ema_slow = ta.trend.EMAIndicator(close, self.config.ema_slow).ema_indicator()
        ema_long_fast = ta.trend.EMAIndicator(
            close, self.config.ema_long_fast
        ).ema_indicator()
        ema_long_slow = ta.trend.EMAIndicator(
            close, self.config.ema_long_slow
        ).ema_indicator()

        # RSI
        rsi = ta.momentum.RSIIndicator(close, self.config.rsi_period).rsi()

        # ADX
        adx = ta.trend.ADXIndicator(
            df["high"], df["low"], close, self.config.adx_period
        ).adx()

        # ATR for stop-loss sizing
        atr = calculate_atr(df["high"], df["low"], close, self.config.atr_period)

        current_price = float(close.iloc[-1])
        current_rsi = float(rsi.iloc[-1])
        current_adx = float(adx.iloc[-1])
        current_atr = float(atr.iloc[-1])

        prev_ema_fast = float(ema_fast.iloc[-2])
        curr_ema_fast = float(ema_fast.iloc[-1])
        prev_ema_slow = float(ema_slow.iloc[-2])
        curr_ema_slow = float(ema_slow.iloc[-1])

        curr_ema_long_fast = float(ema_long_fast.iloc[-1])
        curr_ema_long_slow = float(ema_long_slow.iloc[-1])

        # Detect crossover
        bullish_cross = prev_ema_fast < prev_ema_slow and curr_ema_fast > curr_ema_slow
        bearish_cross = prev_ema_fast > prev_ema_slow and curr_ema_fast < curr_ema_slow

        strong_trend = current_adx > self.config.adx_threshold

        metadata = {
            "ema_fast": curr_ema_fast,
            "ema_slow": curr_ema_slow,
            "rsi": current_rsi,
            "adx": current_adx,
            "atr": current_atr,
        }

        if (
            bullish_cross
            and current_rsi > 50
            and strong_trend
            and current_price > curr_ema_long_fast > curr_ema_long_slow
        ):
            stop_loss = current_price - self.config.atr_multiplier * current_atr
            take_profit = current_price + 2 * self.config.atr_multiplier * current_atr
            return self._build_signal(
                SignalDirection.LONG,
                strength=min((current_rsi - 50) / 50, 1.0),
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                metadata=metadata,
            )

        if (
            bearish_cross
            and current_rsi < 50
            and strong_trend
            and current_price < curr_ema_long_fast < curr_ema_long_slow
        ):
            stop_loss = current_price + self.config.atr_multiplier * current_atr
            take_profit = current_price - 2 * self.config.atr_multiplier * current_atr
            return self._build_signal(
                SignalDirection.SHORT,
                strength=min((50 - current_rsi) / 50, 1.0),
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                metadata=metadata,
            )

        return self._neutral_signal("No crossover or confirmation")
