"""Script to run backtests for all strategies or a selected strategy."""

import argparse
import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from backtest.backtester import Backtester
from config.settings import settings
from src.strategies.breakout import BreakoutStrategy
from src.strategies.confluence import ConfluenceStrategy
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.momentum import MomentumStrategy
from src.strategies.vwap import VWAPStrategy
from src.utils.logger import get_logger

logger = get_logger(__name__)

AVAILABLE_STRATEGIES = {
    "momentum": MomentumStrategy,
    "mean_reversion": MeanReversionStrategy,
    "breakout": BreakoutStrategy,
    "vwap": VWAPStrategy,
    "confluence": ConfluenceStrategy,
}


def load_csv_data(path: str) -> pd.DataFrame:
    """
    Load OHLCV data from a CSV file.

    The CSV must have columns: timestamp, open, high, low, close, volume.

    Args:
        path: Path to the CSV file.

    Returns:
        OHLCV DataFrame with a DatetimeIndex.
    """
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = df[col].astype(float)
    return df


def print_result(result) -> None:
    """Pretty-print a BacktestResult to stdout."""
    print(f"\n{'='*60}")
    print(f"  Strategy : {result.strategy_name}")
    print(f"{'='*60}")
    print(f"  Initial capital  : ${result.initial_capital:,.2f}")
    print(f"  Final capital    : ${result.final_capital:,.2f}")
    print(f"  Total return     : {result.total_return:.2%}")
    print(f"  Annualised return: {result.annualised_return:.2%}")
    print(f"  Sharpe ratio     : {result.sharpe_ratio:.2f}")
    print(f"  Max drawdown     : {result.max_drawdown:.2%}")
    print(f"  Total trades     : {result.total_trades}")
    print(f"  Win rate         : {result.win_rate:.2%}")
    print(f"  Avg profit       : ${result.avg_profit:.2f}")
    print(f"  Avg loss         : ${result.avg_loss:.2f}")
    print(f"  Profit factor    : {result.profit_factor:.2f}")
    print(f"{'='*60}\n")


def main() -> None:
    """CLI entry point for the backtesting script."""
    parser = argparse.ArgumentParser(description="XAUUSD Trading Bot — Backtester")
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to OHLCV CSV file (columns: timestamp, open, high, low, close, volume)",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="all",
        choices=list(AVAILABLE_STRATEGIES.keys()) + ["all"],
        help="Strategy to backtest (default: all)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=200,
        help="Number of warm-up bars before trading starts (default: 200)",
    )
    args = parser.parse_args()

    logger.info("Loading data from %s …", args.data)
    df = load_csv_data(args.data)
    logger.info("Loaded %d bars from %s to %s", len(df), df.index[0], df.index[-1])

    backtester = Backtester(config=settings.backtest)

    strategies_to_run = (
        list(AVAILABLE_STRATEGIES.keys())
        if args.strategy == "all"
        else [args.strategy]
    )

    for name in strategies_to_run:
        strategy_cls = AVAILABLE_STRATEGIES[name]
        strategy = strategy_cls(symbol=settings.instrument.symbol)
        logger.info("Running backtest for strategy: %s", name)
        result = backtester.run(strategy, df, warm_up_bars=args.warmup)
        print_result(result)


if __name__ == "__main__":
    main()
