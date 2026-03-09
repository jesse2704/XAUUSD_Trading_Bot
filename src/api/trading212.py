"""Trading212 API client wrapper."""

import time
from typing import Any, Dict, List, Optional

import requests

from src.utils.logger import get_logger

logger = get_logger(__name__)


class Trading212Error(Exception):
    """Base exception for Trading212 API errors."""


class Trading212AuthError(Trading212Error):
    """Raised when authentication fails (401)."""


class Trading212RateLimitError(Trading212Error):
    """Raised when the API rate limit is exceeded (429)."""


class Trading212NotFoundError(Trading212Error):
    """Raised when a requested resource is not found (404)."""


class Trading212ServerError(Trading212Error):
    """Raised on 5xx server-side errors."""


class Trading212Client:
    """
    Wrapper around the Trading212 REST API.

    Handles authentication, request retries, rate-limiting, and provides
    typed methods for every endpoint used by the bot.

    Args:
        api_key: Trading212 API key.
        base_url: Base URL for the target environment (demo or live).
        max_retries: Number of retry attempts for transient errors.
        retry_delay: Initial delay between retries (seconds, doubles each attempt).
        timeout: HTTP request timeout in seconds.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://demo.trading212.com/api/v0",
        max_retries: int = 3,
        retry_delay: float = 1.0,
        timeout: int = 30,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": self.api_key,
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
        )

    # ------------------------------------------------------------------
    # Internal request helpers
    # ------------------------------------------------------------------

    def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        json_body: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Execute an HTTP request with automatic retries and error handling.

        Args:
            method: HTTP method (GET, POST, DELETE, …).
            endpoint: API endpoint path (e.g. '/equity/account/info').
            params: Optional query string parameters.
            json_body: Optional JSON request body.

        Returns:
            Parsed JSON response as a Python object.

        Raises:
            Trading212AuthError: On 401 responses.
            Trading212RateLimitError: On 429 responses.
            Trading212NotFoundError: On 404 responses.
            Trading212ServerError: On 5xx responses.
            Trading212Error: On other unexpected errors.
        """
        url = f"{self.base_url}{endpoint}"
        delay = self.retry_delay

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._session.request(
                    method=method.upper(),
                    url=url,
                    params=params,
                    json=json_body,
                    timeout=self.timeout,
                )
            except requests.RequestException as exc:
                logger.warning(
                    "Request error on attempt %d/%d for %s %s: %s",
                    attempt,
                    self.max_retries,
                    method,
                    url,
                    exc,
                )
                if attempt == self.max_retries:
                    raise Trading212Error(
                        f"Request failed after {self.max_retries} attempts: {exc}"
                    ) from exc
                time.sleep(delay)
                delay *= 2
                continue

            if response.status_code == 200:
                return response.json() if response.content else {}

            if response.status_code == 401:
                raise Trading212AuthError(
                    "Authentication failed. Check your API key."
                )
            if response.status_code == 404:
                raise Trading212NotFoundError(
                    f"Resource not found: {endpoint}"
                )
            if response.status_code == 429:
                retry_after = float(response.headers.get("Retry-After", delay))
                logger.warning(
                    "Rate limit hit. Waiting %.1f seconds before retry.", retry_after
                )
                if attempt == self.max_retries:
                    raise Trading212RateLimitError(
                        "Rate limit exceeded. Try again later."
                    )
                time.sleep(retry_after)
                continue
            if response.status_code >= 500:
                logger.warning(
                    "Server error %d on attempt %d/%d for %s",
                    response.status_code,
                    attempt,
                    self.max_retries,
                    url,
                )
                if attempt == self.max_retries:
                    raise Trading212ServerError(
                        f"Server error {response.status_code}: {response.text}"
                    )
                time.sleep(delay)
                delay *= 2
                continue

            # Other 4xx errors
            raise Trading212Error(
                f"API error {response.status_code}: {response.text}"
            )

        raise Trading212Error("Unexpected exit from retry loop.")  # pragma: no cover

    # ------------------------------------------------------------------
    # Account
    # ------------------------------------------------------------------

    def get_account_info(self) -> Dict[str, Any]:
        """
        Retrieve account metadata.

        Returns:
            Dict with account id, currency code, etc.
        """
        return self._request("GET", "/equity/account/info")

    def get_account_cash(self) -> Dict[str, Any]:
        """
        Retrieve account cash balances.

        Returns:
            Dict with free, invested, result, total, ppl, blocked fields.
        """
        return self._request("GET", "/equity/account/cash")

    # ------------------------------------------------------------------
    # Portfolio / Positions
    # ------------------------------------------------------------------

    def get_positions(self) -> List[Dict[str, Any]]:
        """
        Get all currently open positions.

        Returns:
            List of open position dicts.
        """
        result = self._request("GET", "/equity/portfolio")
        return result if isinstance(result, list) else []

    def get_position(self, ticker: str) -> Dict[str, Any]:
        """
        Get details of a single open position.

        Args:
            ticker: Instrument ticker (e.g. 'XAUUSD').

        Returns:
            Position dict.
        """
        return self._request("GET", f"/equity/portfolio/{ticker}")

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    def get_orders(self) -> List[Dict[str, Any]]:
        """
        List all open / pending orders.

        Returns:
            List of order dicts.
        """
        result = self._request("GET", "/equity/orders")
        return result if isinstance(result, list) else []

    def place_market_order(
        self,
        ticker: str,
        quantity: float,
        time_validity: str = "DAY",
    ) -> Dict[str, Any]:
        """
        Place a market order.

        Args:
            ticker: Instrument ticker.
            quantity: Positive for buy, negative for sell.
            time_validity: 'DAY' or 'GTC'.

        Returns:
            Created order dict.
        """
        body = {
            "ticker": ticker,
            "quantity": quantity,
            "timeValidity": time_validity,
        }
        logger.info(
            "Placing MARKET order: ticker=%s qty=%.4f", ticker, quantity
        )
        return self._request("POST", "/equity/orders/market", json_body=body)

    def place_limit_order(
        self,
        ticker: str,
        quantity: float,
        limit_price: float,
        time_validity: str = "DAY",
    ) -> Dict[str, Any]:
        """
        Place a limit order.

        Args:
            ticker: Instrument ticker.
            quantity: Positive for buy, negative for sell.
            limit_price: Limit price.
            time_validity: 'DAY' or 'GTC'.

        Returns:
            Created order dict.
        """
        body = {
            "ticker": ticker,
            "quantity": quantity,
            "limitPrice": limit_price,
            "timeValidity": time_validity,
        }
        logger.info(
            "Placing LIMIT order: ticker=%s qty=%.4f @ %.5f",
            ticker,
            quantity,
            limit_price,
        )
        return self._request("POST", "/equity/orders/limit", json_body=body)

    def place_stop_order(
        self,
        ticker: str,
        quantity: float,
        stop_price: float,
        time_validity: str = "DAY",
    ) -> Dict[str, Any]:
        """
        Place a stop order.

        Args:
            ticker: Instrument ticker.
            quantity: Positive for buy, negative for sell.
            stop_price: Stop trigger price.
            time_validity: 'DAY' or 'GTC'.

        Returns:
            Created order dict.
        """
        body = {
            "ticker": ticker,
            "quantity": quantity,
            "stopPrice": stop_price,
            "timeValidity": time_validity,
        }
        logger.info(
            "Placing STOP order: ticker=%s qty=%.4f stop=%.5f",
            ticker,
            quantity,
            stop_price,
        )
        return self._request("POST", "/equity/orders/stop", json_body=body)

    def cancel_order(self, order_id: str) -> None:
        """
        Cancel an open order.

        Args:
            order_id: ID of the order to cancel.
        """
        logger.info("Cancelling order id=%s", order_id)
        self._request("DELETE", f"/equity/orders/{order_id}")

    # ------------------------------------------------------------------
    # Instruments
    # ------------------------------------------------------------------

    def get_instruments(self) -> List[Dict[str, Any]]:
        """
        Get the list of tradable instruments.

        Returns:
            List of instrument dicts.
        """
        result = self._request("GET", "/equity/metadata/instruments")
        return result if isinstance(result, list) else []

    # ------------------------------------------------------------------
    # Historical Data
    # ------------------------------------------------------------------

    def get_historical_data(
        self,
        ticker: str,
        period: str = "ONE_HOUR",
        limit: int = 500,
    ) -> List[Dict[str, Any]]:
        """
        Fetch historical OHLCV candles for a ticker.

        Args:
            ticker: Instrument ticker.
            period: Candle period ('ONE_MINUTE', 'FIVE_MINUTES', 'ONE_HOUR',
                    'FOUR_HOURS', 'ONE_DAY').
            limit: Number of candles to retrieve (max 500).

        Returns:
            List of OHLCV dicts with keys: timestamp, open, high, low, close, volume.
        """
        params = {"period": period, "limit": limit}
        result = self._request(
            "GET", f"/equity/history/instrument/{ticker}", params=params
        )
        return result if isinstance(result, list) else []
