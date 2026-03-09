"""Simple event-driven backtesting engine for the XAUUSD Trading Bot."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Type

import numpy as np
import pandas as pd

from config.settings import BacktestConfig
from src.models.signals import OrderSide, SignalDirection
from src.strategies.base import BaseStrategy
from src.utils.helpers import compute_max_drawdown, compute_sharpe_ratio
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class BacktestTrade:
    """Record of a single simulated trade."""

    strategy_name: str
    side: OrderSide
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    entry_bar: int
    exit_bar: int
    exit_reason: str = ""


@dataclass
class BacktestResult:
    """Aggregated results from a backtest run."""

    strategy_name: str
    initial_capital: float
    final_capital: float
    total_return: float
    annualised_return: float
    sharpe_ratio: float
    max_drawdown: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_profit: float
    avg_loss: float
    profit_factor: float
    trades: List[BacktestTrade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)


class Backtester:
    """
    Event-driven backtesting engine.

    Iterates over OHLCV candles bar-by-bar, feeding a window of history
    into the strategy at each step and simulating order execution at the
    *next* bar's open price to avoid look-ahead bias.

    Args:
        config: Backtesting parameters (capital, commission, slippage).
    """

    def __init__(self, config: BacktestConfig | None = None) -> None:
        self.config = config or BacktestConfig()

    def run(
        self,
        strategy: BaseStrategy,
        df: pd.DataFrame,
        warm_up_bars: int = 200,
    ) -> BacktestResult:
        """
        Run a single-strategy backtest over the provided DataFrame.

        Args:
            strategy: An instantiated strategy to test.
            df: Full OHLCV DataFrame (ideally several hundred bars or more).
            warm_up_bars: Number of bars at the start used only for
                          indicator warm-up; no trades are opened.

        Returns:
            BacktestResult with performance metrics and trade log.
        """
        capital = self.config.initial_capital
        equity_curve: List[float] = [capital]
        trades: List[BacktestTrade] = []

        # Current open position state
        open_side: Optional[OrderSide] = None
        open_price: float = 0.0
        open_qty: float = 0.0
        open_sl: Optional[float] = None
        open_tp: Optional[float] = None
        open_bar: int = 0
        strategy_name: str = strategy.name

        total_bars = len(df)

        for i in range(warm_up_bars, total_bars - 1):
            # Feed history up to bar i to the strategy
            window = df.iloc[: i + 1]
            signal_obj = strategy.generate_signal(window)

            current_bar = df.iloc[i]
            next_bar = df.iloc[i + 1]
            next_open = float(next_bar["open"])
            next_high = float(next_bar["high"])
            next_low = float(next_bar["low"])

            # ---- Check stop-loss / take-profit for open position ----
            if open_side is not None:
                exit_price: Optional[float] = None
                exit_reason = ""

                if open_side == OrderSide.BUY:
                    if open_sl is not None and next_low <= open_sl:
                        exit_price = open_sl
                        exit_reason = "stop_loss"
                    elif open_tp is not None and next_high >= open_tp:
                        exit_price = open_tp
                        exit_reason = "take_profit"
                else:
                    if open_sl is not None and next_high >= open_sl:
                        exit_price = open_sl
                        exit_reason = "stop_loss"
                    elif open_tp is not None and next_low <= open_tp:
                        exit_price = open_tp
                        exit_reason = "take_profit"

                # Opposite signal closes the position
                if exit_price is None:
                    if (
                        open_side == OrderSide.BUY
                        and signal_obj.direction == SignalDirection.SHORT
                    ):
                        exit_price = next_open
                        exit_reason = "reverse_signal"
                    elif (
                        open_side == OrderSide.SELL
                        and signal_obj.direction == SignalDirection.LONG
                    ):
                        exit_price = next_open
                        exit_reason = "reverse_signal"

                if exit_price is not None:
                    pnl = self._calculate_pnl(
                        open_side, open_price, exit_price, open_qty
                    )
                    capital += pnl
                    trades.append(
                        BacktestTrade(
                            strategy_name=strategy_name,
                            side=open_side,
                            entry_price=open_price,
                            exit_price=exit_price,
                            quantity=open_qty,
                            pnl=pnl,
                            entry_bar=open_bar,
                            exit_bar=i + 1,
                            exit_reason=exit_reason,
                        )
                    )
                    open_side = None

            # ---- Open new position if no current position ----
            if (
                open_side is None
                and signal_obj.direction != SignalDirection.NEUTRAL
            ):
                entry_price = next_open * (
                    1 + self.config.slippage
                    if signal_obj.direction == SignalDirection.LONG
                    else 1 - self.config.slippage
                )
                commission = capital * self.config.commission_per_trade
                capital -= commission

                # Risk 1% of equity per trade
                risk_amount = capital * 0.01
                stop_dist = abs(
                    entry_price - (signal_obj.stop_loss or entry_price * 0.99)
                )
                qty = risk_amount / stop_dist if stop_dist > 0 else 0.01

                open_side = (
                    OrderSide.BUY
                    if signal_obj.direction == SignalDirection.LONG
                    else OrderSide.SELL
                )
                open_price = entry_price
                open_qty = qty
                open_sl = signal_obj.stop_loss
                open_tp = signal_obj.take_profit
                open_bar = i + 1

            equity_curve.append(capital)

        # Force-close any open position at the last available price
        if open_side is not None and len(df) > 0:
            last_price = float(df["close"].iloc[-1])
            pnl = self._calculate_pnl(open_side, open_price, last_price, open_qty)
            capital += pnl
            trades.append(
                BacktestTrade(
                    strategy_name=strategy_name,
                    side=open_side,
                    entry_price=open_price,
                    exit_price=last_price,
                    quantity=open_qty,
                    pnl=pnl,
                    entry_bar=open_bar,
                    exit_bar=total_bars - 1,
                    exit_reason="end_of_data",
                )
            )
            equity_curve.append(capital)

        return self._build_result(strategy_name, trades, equity_curve)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _calculate_pnl(
        side: OrderSide, entry: float, exit_: float, qty: float
    ) -> float:
        if side == OrderSide.BUY:
            return (exit_ - entry) * qty
        return (entry - exit_) * qty

    def _build_result(
        self,
        strategy_name: str,
        trades: List[BacktestTrade],
        equity_curve: List[float],
    ) -> BacktestResult:
        initial = self.config.initial_capital
        final = equity_curve[-1] if equity_curve else initial
        total_return = (final - initial) / initial

        eq_series = pd.Series(equity_curve)
        returns = eq_series.pct_change().dropna()

        sharpe = compute_sharpe_ratio(returns, periods_per_year=252)
        max_dd = compute_max_drawdown(eq_series)

        # Annualise return (assume 252 * 24 bars of 1h data ≈ 1 year)
        n_bars = len(equity_curve)
        ann_return = (
            (1 + total_return) ** (252 * 24 / max(n_bars, 1)) - 1 if n_bars > 1 else 0.0
        )

        winning = [t for t in trades if t.pnl > 0]
        losing = [t for t in trades if t.pnl <= 0]
        win_rate = len(winning) / len(trades) if trades else 0.0

        avg_profit = float(np.mean([t.pnl for t in winning])) if winning else 0.0
        avg_loss = float(np.mean([t.pnl for t in losing])) if losing else 0.0

        gross_profit = sum(t.pnl for t in winning)
        gross_loss = abs(sum(t.pnl for t in losing))
        profit_factor = (
            gross_profit / gross_loss if gross_loss > 0 else float("inf")
        )

        return BacktestResult(
            strategy_name=strategy_name,
            initial_capital=initial,
            final_capital=final,
            total_return=total_return,
            annualised_return=ann_return,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            total_trades=len(trades),
            winning_trades=len(winning),
            losing_trades=len(losing),
            win_rate=win_rate,
            avg_profit=avg_profit,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            trades=trades,
            equity_curve=equity_curve,
        )
