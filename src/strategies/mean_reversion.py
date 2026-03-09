"""Strategy 2: Mean Reversion (Bollinger Band Bounce)."""

import pandas as pd
import ta

from config.settings import MeanReversionConfig
from src.models.signals import Signal, SignalDirection
from src.strategies.base import BaseStrategy
from src.utils.helpers import calculate_atr


class MeanReversionStrategy(BaseStrategy):
    """
    Mean Reversion strategy using Bollinger Bands and RSI.

    Entry logic
    -----------
    *  **Long**: close touches or crosses the lower Bollinger Band **and**
       RSI < oversold threshold.  This signals a potential bounce back to
       the middle band (SMA).
    *  **Short**: close touches or crosses the upper Bollinger Band **and**
       RSI > overbought threshold.  Price is expected to revert to the SMA.

    Take-profit is set at the middle band (20-period SMA).
    Stop-loss is set at ``atr_multiplier × ATR`` beyond the entry.

    Args:
        config: Strategy configuration.
        symbol: Instrument symbol.
    """

    def __init__(
        self,
        config: MeanReversionConfig | None = None,
        symbol: str = "XAUUSD",
    ) -> None:
        super().__init__("MeanReversion", symbol)
        self.config = config or MeanReversionConfig()

    def generate_signal(self, df: pd.DataFrame) -> Signal:
        """
        Compute indicators and return a directional signal.

        Args:
            df: OHLCV DataFrame.

        Returns:
            Signal with LONG, SHORT, or NEUTRAL direction.
        """
        min_rows = self.config.bb_period + self.config.rsi_period + 1
        if len(df) < min_rows:
            return self._neutral_signal(f"Insufficient data: need {min_rows} rows")

        df = df.copy()
        close = df["close"]

        # Bollinger Bands
        bb = ta.volatility.BollingerBands(
            close, self.config.bb_period, self.config.bb_std
        )
        upper_band = bb.bollinger_hband()
        lower_band = bb.bollinger_lband()
        middle_band = bb.bollinger_mavg()

        # RSI
        rsi = ta.momentum.RSIIndicator(close, self.config.rsi_period).rsi()

        # ATR for stop-loss
        atr = calculate_atr(df["high"], df["low"], close, self.config.atr_period)

        current_price = float(close.iloc[-1])
        current_rsi = float(rsi.iloc[-1])
        current_atr = float(atr.iloc[-1])
        current_lower = float(lower_band.iloc[-1])
        current_upper = float(upper_band.iloc[-1])
        current_middle = float(middle_band.iloc[-1])

        metadata = {
            "upper_band": current_upper,
            "middle_band": current_middle,
            "lower_band": current_lower,
            "rsi": current_rsi,
            "atr": current_atr,
        }

        # Long: price at/below lower band + RSI oversold
        if (
            current_price <= current_lower
            and current_rsi < self.config.rsi_oversold
        ):
            stop_loss = current_price - self.config.atr_multiplier * current_atr
            take_profit = current_middle
            strength = min(
                (self.config.rsi_oversold - current_rsi) / self.config.rsi_oversold,
                1.0,
            )
            return self._build_signal(
                SignalDirection.LONG,
                strength=strength,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                metadata=metadata,
            )

        # Short: price at/above upper band + RSI overbought
        if (
            current_price >= current_upper
            and current_rsi > self.config.rsi_overbought
        ):
            stop_loss = current_price + self.config.atr_multiplier * current_atr
            take_profit = current_middle
            strength = min(
                (current_rsi - self.config.rsi_overbought)
                / (100 - self.config.rsi_overbought),
                1.0,
            )
            return self._build_signal(
                SignalDirection.SHORT,
                strength=strength,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                metadata=metadata,
            )

        return self._neutral_signal("No Bollinger Band touch with RSI confirmation")
