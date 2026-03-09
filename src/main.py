"""Main entry point and orchestrator for the XAUUSD Trading Bot."""

import signal
import sys
import time
from typing import List, Optional

import schedule

from config.settings import settings
from src.api.trading212 import Trading212Client, Trading212Error
from src.data.market_data import MarketDataFetcher
from src.models.signals import OrderSide, OrderType, SignalDirection
from src.models.signals import Order
from src.risk.manager import RiskManager
from src.strategies.base import BaseStrategy
from src.strategies.breakout import BreakoutStrategy
from src.strategies.confluence import ConfluenceStrategy
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.momentum import MomentumStrategy
from src.strategies.vwap import VWAPStrategy
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TradingBot:
    """
    XAUUSD Trading Bot orchestrator.

    Initialises all components, runs the main trading loop, and handles
    graceful shutdown on SIGINT/SIGTERM.

    Args:
        dry_run: When True, signals are logged but no orders are submitted.
    """

    def __init__(self, dry_run: bool = True) -> None:
        self.dry_run = dry_run
        self._running = False

        logger.info(
            "Initialising TradingBot | env=%s symbol=%s dry_run=%s",
            settings.api.environment,
            settings.instrument.symbol,
            dry_run,
        )

        # API client
        self.client = Trading212Client(
            api_key=settings.api.api_key,
            base_url=settings.api.base_url,
        )

        # Market data
        self.market_data = MarketDataFetcher(
            client=self.client,
            symbol=settings.instrument.symbol,
            timeframe=settings.instrument.timeframe,
        )

        # Risk manager (equity will be updated from account on start)
        self.risk_manager = RiskManager(
            config=settings.risk,
            initial_equity=10_000.0,
        )

        # Strategies
        self.strategies: List[BaseStrategy] = [
            MomentumStrategy(config=settings.momentum, symbol=settings.instrument.symbol),
            MeanReversionStrategy(
                config=settings.mean_reversion, symbol=settings.instrument.symbol
            ),
            BreakoutStrategy(config=settings.breakout, symbol=settings.instrument.symbol),
            VWAPStrategy(config=settings.vwap, symbol=settings.instrument.symbol),
            ConfluenceStrategy(config=settings.confluence, symbol=settings.instrument.symbol),
        ]

        logger.info("TradingBot initialised with %d strategies.", len(self.strategies))

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the trading bot and run the main loop."""
        self._running = True
        signal.signal(signal.SIGINT, self._shutdown_handler)
        signal.signal(signal.SIGTERM, self._shutdown_handler)

        logger.info("Starting trading bot…")
        self._sync_account()

        # Schedule the main tick
        schedule.every(1).minutes.do(self._tick)

        logger.info("Scheduler started. Press Ctrl+C to stop.")
        while self._running:
            schedule.run_pending()
            time.sleep(1)

    def stop(self) -> None:
        """Gracefully stop the trading bot."""
        self._running = False
        logger.info("Trading bot stopped.")

    # ------------------------------------------------------------------
    # Core trading logic
    # ------------------------------------------------------------------

    def _tick(self) -> None:
        """Execute one iteration of the trading loop."""
        logger.debug("Tick started.")
        try:
            self._sync_account()
            df = self.market_data.get_candles(limit=500)
            if df.empty:
                logger.warning("No market data received — skipping tick.")
                return

            for strategy in self.strategies:
                if not self.risk_manager.can_trade():
                    logger.info(
                        "Risk gate closed — skipping remaining strategies."
                    )
                    break

                signal_obj = strategy.generate_signal(df)
                logger.info(
                    "Strategy=%s direction=%s strength=%.2f",
                    signal_obj.strategy_name,
                    signal_obj.direction.value,
                    signal_obj.strength,
                )

                if signal_obj.direction == SignalDirection.NEUTRAL:
                    continue

                self._execute_signal(signal_obj)

        except Trading212Error as exc:
            logger.error("API error during tick: %s", exc)
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception("Unexpected error during tick: %s", exc)

    def _execute_signal(self, signal_obj) -> None:
        """
        Convert a signal to an order and submit it (or log in dry_run mode).

        Args:
            signal_obj: The Signal to act on.
        """
        if signal_obj.entry_price is None:
            logger.warning(
                "Signal from %s has no entry_price — skipping.",
                signal_obj.strategy_name,
            )
            return

        atr = signal_obj.metadata.get("atr")
        size = self.risk_manager.calculate_position_size(
            entry_price=signal_obj.entry_price,
            stop_loss=signal_obj.stop_loss or (signal_obj.entry_price * 0.99),
            atr=atr,
        )
        if size <= 0:
            logger.warning("Position size is 0 — skipping order.")
            return

        side = (
            OrderSide.BUY
            if signal_obj.direction == SignalDirection.LONG
            else OrderSide.SELL
        )

        order = Order(
            symbol=settings.instrument.symbol,
            side=side,
            order_type=OrderType.MARKET,
            quantity=size,
            stop_loss=signal_obj.stop_loss,
            take_profit=signal_obj.take_profit,
            strategy_name=signal_obj.strategy_name,
        )

        if self.dry_run:
            logger.info(
                "[DRY RUN] Would place order: %s %s qty=%.4f sl=%s tp=%s",
                side.value,
                settings.instrument.symbol,
                size,
                signal_obj.stop_loss,
                signal_obj.take_profit,
            )
            return

        # Live order placement
        signed_qty = size if side == OrderSide.BUY else -size
        result = self.client.place_market_order(
            ticker=settings.instrument.symbol,
            quantity=signed_qty,
        )
        self.risk_manager.open_position()
        logger.info(
            "Order placed: id=%s strategy=%s",
            result.get("id"),
            signal_obj.strategy_name,
        )

    # ------------------------------------------------------------------
    # Account sync
    # ------------------------------------------------------------------

    def _sync_account(self) -> None:
        """Fetch latest account equity and update the risk manager."""
        try:
            cash = self.client.get_account_cash()
            equity = float(cash.get("total", self.risk_manager.current_equity))
            self.risk_manager.update_equity(equity)
            logger.debug("Account synced: equity=%.2f", equity)
        except Trading212Error as exc:
            logger.warning("Could not sync account equity: %s", exc)

    # ------------------------------------------------------------------
    # Shutdown handler
    # ------------------------------------------------------------------

    def _shutdown_handler(self, signum: int, frame) -> None:  # noqa: ANN001
        """Handle SIGINT / SIGTERM gracefully."""
        logger.info("Shutdown signal received (%d). Stopping bot…", signum)
        self.stop()
        sys.exit(0)


def main() -> None:
    """Entry point for the trading bot."""
    import argparse

    parser = argparse.ArgumentParser(description="XAUUSD Trading Bot")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run in live trading mode (default: dry-run / paper trading)",
    )
    args = parser.parse_args()

    bot = TradingBot(dry_run=not args.live)
    bot.start()


if __name__ == "__main__":
    main()
