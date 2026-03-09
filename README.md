# XAUUSD Trading Bot 🥇

A **production-grade, professional algorithmic trading bot** for XAUUSD (Gold vs US Dollar) built in **Python 3.11+**, integrating with the **Trading212 API**. Implements the **top 5 institutional-quality trading strategies** with comprehensive risk management, backtesting, and clean architecture.

> ⚠️ **Disclaimer**: This software is for educational and research purposes only. Algorithmic trading carries significant financial risk. Past performance does not guarantee future results. Always test thoroughly in **demo mode** before considering live trading. The authors assume no responsibility for financial losses.

---

## 📋 Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Top 5 Trading Strategies](#-top-5-trading-strategies)
- [Risk Management](#-risk-management)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Usage](#-usage)
- [Backtesting](#-backtesting)
- [Running Tests](#-running-tests)
- [Project Structure](#-project-structure)
- [Contributing](#-contributing)
- [License](#-license)

---

## ✨ Features

- 🏗️ **Clean Architecture** — Strategy Pattern, dependency injection, separation of concerns
- 📈 **5 Production Strategies** — From simple momentum to multi-indicator ensemble
- 🛡️ **Professional Risk Management** — Max daily loss, drawdown halts, cooldowns, ATR-based sizing
- 🔌 **Full Trading212 API Integration** — Auth, retries, rate limiting, demo/live support
- 📊 **Backtesting Engine** — Event-driven, walk-forward, performance metrics
- 🔍 **Type-Safe** — Full type hints on all functions and classes
- 📝 **Comprehensive Logging** — Rotating file logs, structured console output
- ✅ **Tested** — pytest suite covering strategies, risk manager, and API client
- 🔧 **Config-Driven** — All parameters via `.env` file, no hardcoded values

---

## 🏗️ Architecture

```
+------------------------------------------------------------------+
|                         src/main.py                              |
|                    TradingBot (Orchestrator)                     |
+------+----------------------+--------------+--------------------+
       |                      |              |
       v                      v              v
+----------+        +---------------+  +----------------+
| API      |        | MarketData    |  | RiskManager    |
| Client   |        | Fetcher       |  |                |
+----------+        +---------------+  +----------------+
       |                      |
       |                      v
       |              +----------------+
       |              | Strategies     |
       |              +----------------+
       |              | Momentum       |
       |              | MeanReversion  |
       |              | Breakout       |
       |              | VWAP           |
       |              | Confluence     | <- meta-strategy
       |              +----------------+
       v
Trading212 REST API (demo.trading212.com / live.trading212.com)
```

---

## 📊 Top 5 Trading Strategies

### Strategy 1: Momentum / Trend Following
**File**: `src/strategies/momentum.py`

Uses **EMA crossovers** (9/21 short-term, 50/200 long-term) with **RSI** confirmation and **ADX** trend-strength filtering.

| Component | Value |
|-----------|-------|
| Entry (Long) | Fast EMA crosses above slow EMA + RSI > 50 + ADX > 25 + price above 50/200 EMAs |
| Entry (Short) | Fast EMA crosses below slow EMA + RSI < 50 + ADX > 25 + price below 50/200 EMAs |
| Stop-Loss | 2x ATR below/above entry |
| Take-Profit | 4x ATR above/below entry (2:1 R:R) |

### Strategy 2: Mean Reversion (Bollinger Band Bounce)
**File**: `src/strategies/mean_reversion.py`

Exploits **overextended price** moves using **Bollinger Bands** and **RSI** divergence.

| Component | Value |
|-----------|-------|
| Entry (Long) | Price <= lower Bollinger Band + RSI < 30 |
| Entry (Short) | Price >= upper Bollinger Band + RSI > 70 |
| Stop-Loss | 1.5x ATR beyond entry |
| Take-Profit | Middle Bollinger Band (20-SMA) |

### Strategy 3: Breakout
**File**: `src/strategies/breakout.py`

Identifies **price breakouts** from consolidation using **Donchian Channels** with **volume confirmation**.

| Component | Value |
|-----------|-------|
| Entry (Long) | Close > upper Donchian Channel + volume > 1.5x average |
| Entry (Short) | Close < lower Donchian Channel + volume > 1.5x average |
| Stop-Loss | 1x ATR |
| Take-Profit | 2x ATR |

### Strategy 4: VWAP + Order Flow
**File**: `src/strategies/vwap.py`

Trades **VWAP deviations** with **MACD** and **RSI** confirmation, filtered by **London/NY trading sessions**.

| Component | Value |
|-----------|-------|
| Entry (Long) | Price > 0.5% below VWAP + RSI < 50 + MACD bullish + active session |
| Entry (Short) | Price > 0.5% above VWAP + RSI > 50 + MACD bearish + active session |
| Stop-Loss | 1.5x ATR |
| Take-Profit | VWAP level (mean reversion target) |

### Strategy 5: Multi-Indicator Confluence (Meta-Strategy)
**File**: `src/strategies/confluence.py`

The **ensemble / scoring system** used by professional systematic traders. Aggregates votes from all four strategies above.

| Component | Value |
|-----------|-------|
| Voting | +1 (long), -1 (short), 0 (neutral) per strategy |
| Threshold | >= 3 agreeing votes required |
| Weighting | Dynamic — strategies rewarded/penalised based on past accuracy |
| SL/TP | Averaged from contributing strategies |

---

## 🛡️ Risk Management

**File**: `src/risk/manager.py`

| Feature | Default | Config Key |
|---------|---------|------------|
| Risk per trade | 1% of equity | `RISK_PER_TRADE` |
| Max daily loss | 2% of initial equity | `MAX_DAILY_LOSS` |
| Max drawdown | 5% from peak | `MAX_DRAWDOWN` |
| Max position size | 10% of equity | `MAX_POSITION_SIZE` |
| Max concurrent positions | 3 | `MAX_CONCURRENT_POSITIONS` |
| Cooldown after loss | 30 minutes | `COOLDOWN_MINUTES` |

When either the **daily loss** or **drawdown** limit is exceeded the bot halts all trading immediately and logs a CRITICAL message.

**Position sizing formula:**
```
size = (equity x risk_per_trade) / |entry_price - stop_loss|
size = min(size, equity x max_position_size / entry_price)
```

---

## 📦 Prerequisites

- Python **3.11+**
- A Trading212 account (https://www.trading212.com)
- Trading212 API key (Settings -> API -> Generate Key)

---

## 🚀 Installation

```bash
# 1. Clone the repository
git clone https://github.com/jesse2704/XAUUSD_Trading_Bot.git
cd XAUUSD_Trading_Bot

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate       # macOS / Linux
# .venv\Scripts\activate        # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

---

## ⚙️ Configuration

Copy `.env.example` to `.env` and edit the values:

```bash
cp .env.example .env
```

**Critical settings:**

```
TRADING212_API_KEY=your_api_key_here
TRADING212_ENVIRONMENT=demo
INSTRUMENT=XAUUSD
TIMEFRAME=1h
```

All strategy parameters, risk limits, and logging settings are also configurable via the `.env` file. See `.env.example` for the full list.

---

## 🤖 Usage

### Demo / Paper Trading (default — recommended)

```bash
python -m src.main
```

All signals are logged but **no real orders are placed**.

### Live Trading

Only use live mode after thorough testing in demo mode:

```bash
python -m src.main --live
```

Make sure `TRADING212_ENVIRONMENT=live` is set in your `.env`.

---

## 📈 Backtesting

Prepare a CSV file with columns: `timestamp, open, high, low, close, volume`

```bash
# Backtest all strategies
python scripts/run_backtest.py --data path/to/XAUUSD_hourly.csv

# Backtest a single strategy
python scripts/run_backtest.py --data path/to/data.csv --strategy momentum

# Available: momentum, mean_reversion, breakout, vwap, confluence, all
```

---

## ✅ Running Tests

```bash
# Run all tests
pytest

# With coverage report
pytest --cov=src --cov=config --cov-report=term-missing

# Run specific module
pytest tests/test_strategies.py -v
pytest tests/test_risk_manager.py -v
pytest tests/test_api_client.py -v
```

---

## 📁 Project Structure

```
XAUUSD_Trading_Bot/
├── README.md
├── .env.example
├── .gitignore
├── requirements.txt
├── pyproject.toml
├── config/
│   └── settings.py                    # Typed configuration dataclasses
├── src/
│   ├── main.py                        # Bot entry point & orchestrator
│   ├── api/trading212.py              # Trading212 REST API client
│   ├── strategies/
│   │   ├── base.py                    # Abstract BaseStrategy
│   │   ├── momentum.py                # EMA crossover + RSI + ADX
│   │   ├── mean_reversion.py          # Bollinger Bands + RSI
│   │   ├── breakout.py                # Donchian Channels + volume
│   │   ├── vwap.py                    # VWAP + MACD + session filter
│   │   └── confluence.py              # Ensemble meta-strategy
│   ├── risk/manager.py                # Risk management & position sizing
│   ├── data/market_data.py            # OHLCV fetching & VWAP calculation
│   ├── models/signals.py              # Signal, Order, Position dataclasses
│   └── utils/
│       ├── logger.py                  # Rotating file + console logger
│       └── helpers.py                 # ATR, Sharpe, drawdown utilities
├── tests/
│   ├── test_strategies.py
│   ├── test_risk_manager.py
│   └── test_api_client.py
├── backtest/backtester.py             # Event-driven backtesting engine
└── scripts/run_backtest.py            # CLI backtest runner
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-new-strategy`
3. Make your changes with type hints and docstrings
4. Add tests and ensure they pass: `pytest`
5. Submit a pull request

Please follow **PEP 8** style guidelines.

---

## 📄 License

This project is licensed under the **MIT License**.

---

## ⚠️ Financial Disclaimer

Trading foreign exchange, gold, and other financial instruments involves substantial risk of loss and is not suitable for all investors. This software is provided **for educational purposes only** and should not be construed as financial advice. **Always use demo mode first.** Never risk capital you cannot afford to lose.
