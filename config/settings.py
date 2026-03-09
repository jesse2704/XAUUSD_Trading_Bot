"""Configuration management for the XAUUSD Trading Bot."""

import os
from dataclasses import dataclass, field
from typing import List

from dotenv import load_dotenv

load_dotenv()


def _get_env(key: str, default: str) -> str:
    """Get environment variable with a default value."""
    return os.getenv(key, default)


def _get_env_float(key: str, default: float) -> float:
    """Get environment variable as float."""
    return float(os.getenv(key, str(default)))


def _get_env_int(key: str, default: int) -> int:
    """Get environment variable as int."""
    return int(os.getenv(key, str(default)))


@dataclass
class APIConfig:
    """Trading212 API configuration."""

    api_key: str = field(default_factory=lambda: _get_env("TRADING212_API_KEY", ""))
    environment: str = field(
        default_factory=lambda: _get_env("TRADING212_ENVIRONMENT", "demo")
    )

    @property
    def base_url(self) -> str:
        """Get the base URL based on the environment."""
        if self.environment.lower() == "live":
            return "https://live.trading212.com/api/v0"
        return "https://demo.trading212.com/api/v0"


@dataclass
class InstrumentConfig:
    """Instrument configuration."""

    symbol: str = field(default_factory=lambda: _get_env("INSTRUMENT", "XAUUSD"))
    timeframe: str = field(default_factory=lambda: _get_env("TIMEFRAME", "1h"))


@dataclass
class RiskConfig:
    """Risk management configuration."""

    risk_per_trade: float = field(
        default_factory=lambda: _get_env_float("RISK_PER_TRADE", 0.01)
    )
    max_daily_loss: float = field(
        default_factory=lambda: _get_env_float("MAX_DAILY_LOSS", 0.02)
    )
    max_drawdown: float = field(
        default_factory=lambda: _get_env_float("MAX_DRAWDOWN", 0.05)
    )
    max_position_size: float = field(
        default_factory=lambda: _get_env_float("MAX_POSITION_SIZE", 0.10)
    )
    max_concurrent_positions: int = field(
        default_factory=lambda: _get_env_int("MAX_CONCURRENT_POSITIONS", 3)
    )
    cooldown_minutes: int = field(
        default_factory=lambda: _get_env_int("COOLDOWN_MINUTES", 30)
    )


@dataclass
class MomentumStrategyConfig:
    """Configuration for the Momentum / Trend Following strategy."""

    ema_fast: int = field(default_factory=lambda: _get_env_int("EMA_FAST", 9))
    ema_slow: int = field(default_factory=lambda: _get_env_int("EMA_SLOW", 21))
    ema_long_fast: int = field(
        default_factory=lambda: _get_env_int("EMA_LONG_FAST", 50)
    )
    ema_long_slow: int = field(
        default_factory=lambda: _get_env_int("EMA_LONG_SLOW", 200)
    )
    rsi_period: int = field(default_factory=lambda: _get_env_int("RSI_PERIOD", 14))
    rsi_overbought: float = field(
        default_factory=lambda: _get_env_float("RSI_OVERBOUGHT", 70.0)
    )
    rsi_oversold: float = field(
        default_factory=lambda: _get_env_float("RSI_OVERSOLD", 30.0)
    )
    adx_period: int = field(default_factory=lambda: _get_env_int("ADX_PERIOD", 14))
    adx_threshold: float = field(
        default_factory=lambda: _get_env_float("ADX_THRESHOLD", 25.0)
    )
    atr_period: int = 14
    atr_multiplier: float = 2.0


@dataclass
class MeanReversionConfig:
    """Configuration for the Mean Reversion strategy."""

    bb_period: int = field(default_factory=lambda: _get_env_int("BB_PERIOD", 20))
    bb_std: float = field(default_factory=lambda: _get_env_float("BB_STD", 2.0))
    rsi_period: int = field(default_factory=lambda: _get_env_int("RSI_PERIOD", 14))
    rsi_overbought: float = field(
        default_factory=lambda: _get_env_float("RSI_OVERBOUGHT", 70.0)
    )
    rsi_oversold: float = field(
        default_factory=lambda: _get_env_float("RSI_OVERSOLD", 30.0)
    )
    atr_period: int = 14
    atr_multiplier: float = 1.5


@dataclass
class BreakoutConfig:
    """Configuration for the Breakout strategy."""

    donchian_period: int = field(
        default_factory=lambda: _get_env_int("DONCHIAN_PERIOD", 20)
    )
    volume_multiplier: float = field(
        default_factory=lambda: _get_env_float("VOLUME_MULTIPLIER", 1.5)
    )
    atr_period: int = 14
    atr_tp_multiplier: float = 2.0
    atr_sl_multiplier: float = 1.0
    max_candles_in_trade: int = 10


@dataclass
class VWAPConfig:
    """Configuration for the VWAP + Order Flow strategy."""

    vwap_deviation: float = field(
        default_factory=lambda: _get_env_float("VWAP_DEVIATION", 0.005)
    )
    rsi_period: int = field(default_factory=lambda: _get_env_int("RSI_PERIOD", 14))
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    atr_period: int = 14
    atr_multiplier: float = 1.5
    london_open_hour: int = 8   # UTC
    london_close_hour: int = 17  # UTC
    ny_open_hour: int = 13       # UTC
    ny_close_hour: int = 22      # UTC


@dataclass
class ConfluenceConfig:
    """Configuration for the Multi-Indicator Confluence strategy."""

    min_confluence_score: int = field(
        default_factory=lambda: _get_env_int("MIN_CONFLUENCE_SCORE", 3)
    )
    performance_window: int = 20   # bars to evaluate strategy performance
    weight_decay: float = 0.9      # decay factor for older performance


@dataclass
class LoggingConfig:
    """Logging configuration."""

    level: str = field(default_factory=lambda: _get_env("LOG_LEVEL", "INFO"))
    log_file: str = field(
        default_factory=lambda: _get_env("LOG_FILE", "logs/trading_bot.log")
    )
    max_bytes: int = 10 * 1024 * 1024   # 10 MB
    backup_count: int = 5


@dataclass
class BacktestConfig:
    """Backtesting configuration."""

    start_date: str = field(
        default_factory=lambda: _get_env("BACKTEST_START_DATE", "2024-01-01")
    )
    end_date: str = field(
        default_factory=lambda: _get_env("BACKTEST_END_DATE", "2024-12-31")
    )
    initial_capital: float = field(
        default_factory=lambda: _get_env_float("INITIAL_CAPITAL", 10000.0)
    )
    commission_per_trade: float = 0.0001  # 0.01%
    slippage: float = 0.0001              # 0.01%


@dataclass
class Settings:
    """Master settings container that aggregates all configuration sections."""

    api: APIConfig = field(default_factory=APIConfig)
    instrument: InstrumentConfig = field(default_factory=InstrumentConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    momentum: MomentumStrategyConfig = field(default_factory=MomentumStrategyConfig)
    mean_reversion: MeanReversionConfig = field(default_factory=MeanReversionConfig)
    breakout: BreakoutConfig = field(default_factory=BreakoutConfig)
    vwap: VWAPConfig = field(default_factory=VWAPConfig)
    confluence: ConfluenceConfig = field(default_factory=ConfluenceConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)


# Global settings singleton
settings = Settings()
