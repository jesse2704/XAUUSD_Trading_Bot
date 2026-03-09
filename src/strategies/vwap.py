"""Strategy 4: VWAP + Order Flow strategy with session timing filters."""

from datetime import timezone

import pandas as pd
import ta

from config.settings import VWAPConfig
from src.data.market_data import MarketDataFetcher
from src.models.signals import Signal, SignalDirection
from src.strategies.base import BaseStrategy
from src.utils.helpers import calculate_atr, is_trading_session


class VWAPStrategy(BaseStrategy):
    """
    VWAP + Order Flow strategy with London / New York session filters.

    Entry logic
    -----------
    *  **Long**: price is below VWAP by more than ``vwap_deviation`` **and**
       RSI is oversold **and** MACD is bullish (MACD line > signal line) **and**
       current time is within the London or New York session.
    *  **Short**: price is above VWAP by more than ``vwap_deviation`` **and**
       RSI is overbought **and** MACD is bearish **and** within an active session.

    Stop-loss and take-profit are ATR-based.

    Args:
        config: Strategy configuration.
        symbol: Instrument symbol.
    """

    def __init__(
        self,
        config: VWAPConfig | None = None,
        symbol: str = "XAUUSD",
    ) -> None:
        super().__init__("VWAP", symbol)
        self.config = config or VWAPConfig()

    def generate_signal(self, df: pd.DataFrame) -> Signal:
        """
        Compute indicators and return a directional signal.

        Args:
            df: OHLCV DataFrame with a timezone-aware DatetimeIndex.

        Returns:
            Signal with LONG, SHORT, or NEUTRAL direction.
        """
        min_rows = self.config.macd_slow + self.config.macd_signal + 1
        if len(df) < min_rows:
            return self._neutral_signal(f"Insufficient data: need {min_rows} rows")

        # Session filter — use the timestamp of the last candle
        last_ts = df.index[-1]
        if hasattr(last_ts, "tzinfo") and last_ts.tzinfo is not None:
            last_ts_utc = last_ts.astimezone(timezone.utc)
        else:
            last_ts_utc = last_ts.replace(tzinfo=timezone.utc)

        if not is_trading_session(
            last_ts_utc,
            london_open=self.config.london_open_hour,
            london_close=self.config.london_close_hour,
            ny_open=self.config.ny_open_hour,
            ny_close=self.config.ny_close_hour,
        ):
            return self._neutral_signal("Outside active trading session")

        df = df.copy()
        close = df["close"]

        # VWAP
        vwap = MarketDataFetcher.calculate_vwap(df)

        # RSI
        rsi = ta.momentum.RSIIndicator(close, self.config.rsi_period).rsi()

        # MACD
        macd_obj = ta.trend.MACD(
            close,
            self.config.macd_fast,
            self.config.macd_slow,
            self.config.macd_signal,
        )
        macd_line = macd_obj.macd()
        macd_signal_line = macd_obj.macd_signal()

        # ATR
        atr = calculate_atr(
            df["high"], df["low"], close, self.config.atr_period
        )

        current_price = float(close.iloc[-1])
        current_vwap = float(vwap.iloc[-1])
        current_rsi = float(rsi.iloc[-1])
        current_macd = float(macd_line.iloc[-1])
        current_signal = float(macd_signal_line.iloc[-1])
        current_atr = float(atr.iloc[-1])

        deviation = (current_price - current_vwap) / current_vwap

        metadata = {
            "vwap": current_vwap,
            "deviation_pct": deviation,
            "rsi": current_rsi,
            "macd": current_macd,
            "macd_signal": current_signal,
            "atr": current_atr,
        }

        bullish_macd = current_macd > current_signal
        bearish_macd = current_macd < current_signal

        # Long: price below VWAP, oversold RSI, bullish MACD
        if (
            deviation < -self.config.vwap_deviation
            and current_rsi < 50
            and bullish_macd
        ):
            stop_loss = current_price - self.config.atr_multiplier * current_atr
            take_profit = current_vwap  # target: revert to VWAP
            return self._build_signal(
                SignalDirection.LONG,
                strength=min(abs(deviation) / (self.config.vwap_deviation * 2), 1.0),
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                metadata=metadata,
            )

        # Short: price above VWAP, overbought RSI, bearish MACD
        if (
            deviation > self.config.vwap_deviation
            and current_rsi > 50
            and bearish_macd
        ):
            stop_loss = current_price + self.config.atr_multiplier * current_atr
            take_profit = current_vwap  # target: revert to VWAP
            return self._build_signal(
                SignalDirection.SHORT,
                strength=min(abs(deviation) / (self.config.vwap_deviation * 2), 1.0),
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                metadata=metadata,
            )

        return self._neutral_signal("No VWAP deviation signal with confirmation")
