"""Risk management module for the XAUUSD Trading Bot."""

from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Optional

from config.settings import RiskConfig
from src.models.signals import Signal, SignalDirection
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RiskManager:
    """
    Professional risk manager that enforces all configurable risk limits.

    Features
    --------
    - Maximum position size as % of account equity
    - Maximum daily loss limit — halts trading when breached
    - Maximum drawdown limit — halts all trading when breached
    - Maximum number of concurrent open positions
    - Cooldown period after a losing trade
    - ATR-based position sizing

    Args:
        config: Risk configuration (loaded from settings/env vars).
        initial_equity: Starting account equity for drawdown tracking.
    """

    def __init__(
        self,
        config: RiskConfig | None = None,
        initial_equity: float = 10_000.0,
    ) -> None:
        self.config = config or RiskConfig()
        self.initial_equity = initial_equity
        self.peak_equity = initial_equity

        # State
        self._current_equity: float = initial_equity
        self._daily_pnl: float = 0.0
        self._daily_pnl_date: date = datetime.now(timezone.utc).date()
        self._open_positions: int = 0
        self._last_loss_time: Optional[datetime] = None
        self._trading_halted: bool = False
        self._halt_reason: str = ""

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def current_equity(self) -> float:
        """Current account equity."""
        return self._current_equity

    @property
    def is_trading_halted(self) -> bool:
        """Whether risk limits have triggered a full trading halt."""
        return self._trading_halted

    @property
    def halt_reason(self) -> str:
        """Description of why trading was halted."""
        return self._halt_reason

    # ------------------------------------------------------------------
    # Core public methods
    # ------------------------------------------------------------------

    def update_equity(self, equity: float) -> None:
        """
        Update the current equity and recalculate daily P&L and drawdown.

        Args:
            equity: Latest account equity from the broker.
        """
        # Reset daily P&L tracking on new day
        today = datetime.now(timezone.utc).date()
        if today != self._daily_pnl_date:
            self._daily_pnl = 0.0
            self._daily_pnl_date = today  

        pnl_delta = equity - self._current_equity
        self._daily_pnl += pnl_delta
        self._current_equity = equity

        if equity > self.peak_equity:
            self.peak_equity = equity

        self._check_limits()

    def record_trade_result(self, pnl: float) -> None:
        """
        Record the result of a completed trade and enforce post-trade rules.

        Args:
            pnl: Realised P&L of the closed trade.
        """
        today = datetime.now(timezone.utc).date()
        if today != self._daily_pnl_date:
            self._daily_pnl = 0.0
            self._daily_pnl_date = today  

        self._daily_pnl += pnl
        self._current_equity += pnl

        if self._current_equity > self.peak_equity:
            self.peak_equity = self._current_equity

        if pnl < 0:
            self._last_loss_time = datetime.now(timezone.utc)
            logger.info(
                "Losing trade recorded. P&L=%.4f. Cooldown starts.", pnl
            )

        self._check_limits()

    def open_position(self) -> None:
        """Increment the open position counter."""
        self._open_positions += 1

    def close_position(self) -> None:
        """Decrement the open position counter (floor at zero)."""
        self._open_positions = max(0, self._open_positions - 1)

    def can_trade(self, signal: Optional[Signal] = None) -> bool:
        """
        Check whether a new trade is permitted given the current risk state.

        Args:
            signal: Optional signal to validate (not used for basic gate,
                    kept for future extensibility).

        Returns:
            True if trading is allowed, False otherwise.
        """
        if self._trading_halted:
            logger.warning("Trading halted: %s", self._halt_reason)
            return False

        # Max concurrent positions
        if self._open_positions >= self.config.max_concurrent_positions:
            logger.info(
                "Max concurrent positions reached (%d).",
                self.config.max_concurrent_positions,
            )
            return False

        # Cooldown after loss
        if self._last_loss_time is not None:
            cooldown = timedelta(minutes=self.config.cooldown_minutes)
            elapsed = datetime.now(timezone.utc) - self._last_loss_time
            if elapsed < cooldown:
                remaining = (cooldown - elapsed).seconds // 60
                logger.info(
                    "Cooldown active. %d minutes remaining.", remaining
                )
                return False

        return True

    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss: float,
        atr: Optional[float] = None,
    ) -> float:
        """
        Calculate the position size based on the configured risk per trade.

        Uses the formula:
            size = (equity × risk_per_trade) / |entry_price - stop_loss|

        If no stop_loss is provided (stop_loss == entry_price) the ATR-based
        fallback is used.

        Args:
            entry_price: Planned trade entry price.
            stop_loss: Planned stop-loss price.
            atr: Current ATR value (used as fallback distance).

        Returns:
            Position size in units (clipped to max_position_size).
        """
        risk_amount = self._current_equity * self.config.risk_per_trade
        distance = abs(entry_price - stop_loss)

        if distance == 0:
            if atr and atr > 0:
                distance = atr
            else:
                logger.warning(
                    "Cannot compute position size: distance=0 and no ATR. Returning 0."
                )
                return 0.0

        raw_size = risk_amount / distance

        # Cap at max_position_size of equity / entry_price
        max_size = (
            self._current_equity * self.config.max_position_size / entry_price
        )
        size = min(raw_size, max_size)

        logger.debug(
            "Position sizing: risk_amount=%.2f distance=%.4f raw_size=%.4f "
            "max_size=%.4f final_size=%.4f",
            risk_amount,
            distance,
            raw_size,
            max_size,
            size,
        )
        return round(size, 4)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_limits(self) -> None:
        """Check all risk limits and halt trading if any are breached."""
        # Daily loss limit
        daily_loss_pct = -self._daily_pnl / self.initial_equity
        if daily_loss_pct >= self.config.max_daily_loss:
            self._halt("Max daily loss exceeded: {:.2%}".format(daily_loss_pct))
            return

        # Max drawdown limit
        if self.peak_equity > 0:
            drawdown = (self.peak_equity - self._current_equity) / self.peak_equity
            if drawdown >= self.config.max_drawdown:
                self._halt("Max drawdown exceeded: {:.2%}".format(drawdown))
                return

    def _halt(self, reason: str) -> None:
        """Trigger a trading halt."""
        if not self._trading_halted:
            self._trading_halted = True
            self._halt_reason = reason
            logger.critical("TRADING HALTED — %s", reason)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def get_risk_summary(self) -> Dict[str, object]:
        """Return a dict summarising the current risk state."""
        drawdown = (
            (self.peak_equity - self._current_equity) / self.peak_equity
            if self.peak_equity > 0
            else 0.0
        )
        return {
            "current_equity": self._current_equity,
            "peak_equity": self.peak_equity,
            "daily_pnl": self._daily_pnl,
            "daily_pnl_pct": self._daily_pnl / self.initial_equity,
            "drawdown": drawdown,
            "open_positions": self._open_positions,
            "trading_halted": self._trading_halted,
            "halt_reason": self._halt_reason,
        }
