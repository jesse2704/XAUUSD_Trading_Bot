"""Strategy 3: Breakout strategy using Donchian Channels."""

import pandas as pd
import ta

from config.settings import BreakoutConfig
from src.models.signals import Signal, SignalDirection
from src.strategies.base import BaseStrategy
from src.utils.helpers import calculate_atr


class BreakoutStrategy(BaseStrategy):
    """
    Breakout strategy based on Donchian Channels with volume confirmation.

    Entry logic
    -----------
    *  **Long**: close breaks above the *upper* Donchian Channel **and**
       volume exceeds the rolling average by the configured multiplier.
    *  **Short**: close breaks below the *lower* Donchian Channel **and**
       volume exceeds the rolling average.

    Take-profit is set at ``atr_tp_multiplier × ATR`` from entry.
    Stop-loss is set at ``atr_sl_multiplier × ATR`` from entry.

    Args:
        config: Strategy configuration.
        symbol: Instrument symbol.
    """

    def __init__(
        self,
        config: BreakoutConfig | None = None,
        symbol: str = "XAUUSD",
    ) -> None:
        super().__init__("Breakout", symbol)
        self.config = config or BreakoutConfig()

    def generate_signal(self, df: pd.DataFrame) -> Signal:
        """
        Compute indicators and return a directional signal.

        Args:
            df: OHLCV DataFrame.

        Returns:
            Signal with LONG, SHORT, or NEUTRAL direction.
        """
        min_rows = self.config.donchian_period + 20 + 1  # +20 for volume MA
        if len(df) < min_rows:
            return self._neutral_signal(f"Insufficient data: need {min_rows} rows")

        df = df.copy()
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df.get("volume", pd.Series(1, index=df.index))

        # Donchian Channel (use previous period — shift by 1 to avoid lookahead)
        upper_channel = high.rolling(self.config.donchian_period).max().shift(1)
        lower_channel = low.rolling(self.config.donchian_period).min().shift(1)

        # Volume confirmation
        volume_ma = volume.rolling(20).mean()

        # ATR for take-profit / stop-loss
        atr = calculate_atr(high, low, close, self.config.atr_period)

        current_price = float(close.iloc[-1])
        current_upper = float(upper_channel.iloc[-1])
        current_lower = float(lower_channel.iloc[-1])
        current_volume = float(volume.iloc[-1])
        current_volume_ma = float(volume_ma.iloc[-1])
        current_atr = float(atr.iloc[-1])

        volume_spike = (
            current_volume_ma > 0
            and current_volume > self.config.volume_multiplier * current_volume_ma
        )

        metadata = {
            "upper_channel": current_upper,
            "lower_channel": current_lower,
            "volume": current_volume,
            "volume_ma": current_volume_ma,
            "volume_spike": volume_spike,
            "atr": current_atr,
        }

        # Upside breakout
        if current_price > current_upper and volume_spike:
            stop_loss = (
                current_price - self.config.atr_sl_multiplier * current_atr
            )
            take_profit = (
                current_price + self.config.atr_tp_multiplier * current_atr
            )
            return self._build_signal(
                SignalDirection.LONG,
                strength=0.8,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                metadata=metadata,
            )

        # Downside breakout
        if current_price < current_lower and volume_spike:
            stop_loss = (
                current_price + self.config.atr_sl_multiplier * current_atr
            )
            take_profit = (
                current_price - self.config.atr_tp_multiplier * current_atr
            )
            return self._build_signal(
                SignalDirection.SHORT,
                strength=0.8,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                metadata=metadata,
            )

        return self._neutral_signal("No confirmed breakout")
