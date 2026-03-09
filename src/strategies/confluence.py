"""Strategy 5: Multi-Indicator Confluence (Ensemble / Scoring meta-strategy)."""

from typing import List, Optional

import numpy as np
import pandas as pd

from config.settings import ConfluenceConfig
from src.models.signals import Signal, SignalDirection
from src.strategies.base import BaseStrategy
from src.strategies.breakout import BreakoutStrategy
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.momentum import MomentumStrategy
from src.strategies.vwap import VWAPStrategy


class ConfluenceStrategy(BaseStrategy):
    """
    Multi-Indicator Confluence meta-strategy.

    Collects signals from all four underlying strategies, weights them by
    recent historical performance, and only issues a trade when the total
    weighted vote meets or exceeds ``min_confluence_score``.

    Weighting
    ---------
    Each sub-strategy starts with equal weight (1.0).  After each call to
    ``generate_signal`` the weights are updated using an exponential moving
    average of the sub-strategy vote vs the eventual outcome.  For simplicity
    in the absence of live P&L data the initial equal weights are used unless
    the caller explicitly updates them via ``update_weights``.

    Args:
        config: Confluence strategy configuration.
        symbol: Instrument symbol.
        strategies: Optional list of sub-strategy instances.  Defaults to one
                    instance of each of the four concrete strategies.
    """

    def __init__(
        self,
        config: ConfluenceConfig | None = None,
        symbol: str = "XAUUSD",
        strategies: Optional[List[BaseStrategy]] = None,
    ) -> None:
        super().__init__("Confluence", symbol)
        self.config = config or ConfluenceConfig()
        self._sub_strategies: List[BaseStrategy] = strategies or [
            MomentumStrategy(symbol=symbol),
            MeanReversionStrategy(symbol=symbol),
            BreakoutStrategy(symbol=symbol),
            VWAPStrategy(symbol=symbol),
        ]
        self._weights: List[float] = [1.0] * len(self._sub_strategies)
        self._performance_history: List[List[int]] = [
            [] for _ in self._sub_strategies
        ]

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def generate_signal(self, df: pd.DataFrame) -> Signal:
        """
        Aggregate sub-strategy signals and return a consensus signal.

        Each sub-strategy casts a vote (+1 long, -1 short, 0 neutral).
        Votes are multiplied by the strategy's current weight and summed.
        A trade signal is generated only when the aggregate score is at
        least ``min_confluence_score`` (default 3).

        Args:
            df: OHLCV DataFrame passed to all sub-strategies.

        Returns:
            Aggregated Signal.
        """
        votes: List[int] = []
        sub_signals: List[Signal] = []
        sls: List[float] = []
        tps: List[float] = []

        for strategy in self._sub_strategies:
            sig = strategy.generate_signal(df)
            sub_signals.append(sig)
            votes.append(sig.to_vote())
            if sig.stop_loss is not None:
                sls.append(sig.stop_loss)
            if sig.take_profit is not None:
                tps.append(sig.take_profit)

        # Weighted score
        weighted_score = float(np.dot(votes, self._weights))
        long_votes = sum(1 for v in votes if v == 1)
        short_votes = sum(1 for v in votes if v == -1)

        metadata = {
            "votes": votes,
            "weights": list(self._weights),
            "weighted_score": weighted_score,
            "long_votes": long_votes,
            "short_votes": short_votes,
            "sub_signals": [
                {
                    "strategy": s.strategy_name,
                    "direction": s.direction.value,
                    "strength": s.strength,
                }
                for s in sub_signals
            ],
        }

        self.logger.debug(
            "Confluence votes: %s | weighted_score=%.2f", votes, weighted_score
        )

        # Determine consensus direction
        if weighted_score >= self.config.min_confluence_score:
            # Average SL/TP from all strategies that provided them
            avg_sl = float(np.mean(sls)) if sls else None
            avg_tp = float(np.mean(tps)) if tps else None
            current_price = float(df["close"].iloc[-1])
            return self._build_signal(
                SignalDirection.LONG,
                strength=min(weighted_score / len(self._sub_strategies), 1.0),
                entry_price=current_price,
                stop_loss=avg_sl,
                take_profit=avg_tp,
                metadata=metadata,
            )

        if weighted_score <= -self.config.min_confluence_score:
            avg_sl = float(np.mean(sls)) if sls else None
            avg_tp = float(np.mean(tps)) if tps else None
            current_price = float(df["close"].iloc[-1])
            return self._build_signal(
                SignalDirection.SHORT,
                strength=min(abs(weighted_score) / len(self._sub_strategies), 1.0),
                entry_price=current_price,
                stop_loss=avg_sl,
                take_profit=avg_tp,
                metadata=metadata,
            )

        # Return neutral but always include vote metadata for observability
        signal = self._build_signal(
            SignalDirection.NEUTRAL,
            strength=0.0,
            metadata={**metadata, "reason": f"Insufficient confluence: score={weighted_score:.2f}"},
        )
        return signal

    def update_weights(self, outcome_direction: SignalDirection) -> None:
        """
        Update strategy weights based on the outcome of the last trade.

        Strategies whose last vote agreed with the outcome are rewarded;
        those that disagreed are penalised.  Weights are normalised to
        prevent unbounded growth.

        Args:
            outcome_direction: The actual profitable direction.
        """
        outcome_vote = (
            1 if outcome_direction == SignalDirection.LONG
            else -1 if outcome_direction == SignalDirection.SHORT
            else 0
        )

        for i, strategy in enumerate(self._sub_strategies):
            last_sig = strategy.last_signal
            if last_sig is None:
                continue
            strategy_vote = last_sig.to_vote()
            if strategy_vote == outcome_vote and outcome_vote != 0:
                self._weights[i] = min(self._weights[i] * 1.1, 3.0)
            elif strategy_vote != 0 and outcome_vote != 0:
                self._weights[i] = max(self._weights[i] * 0.9, 0.1)

        # Normalise weights so the mean remains 1.0
        mean_w = float(np.mean(self._weights))
        if mean_w > 0:
            self._weights = [w / mean_w for w in self._weights]

    @property
    def sub_strategies(self) -> List[BaseStrategy]:
        """Read-only access to the underlying sub-strategy list."""
        return list(self._sub_strategies)

    @property
    def weights(self) -> List[float]:
        """Current strategy weights."""
        return list(self._weights)
