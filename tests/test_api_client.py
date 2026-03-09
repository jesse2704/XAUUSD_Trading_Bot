"""Unit tests for the Trading212 API client."""

import pytest
import responses as resp_lib
from requests.exceptions import ConnectionError as RequestsConnectionError

from src.api.trading212 import (
    Trading212AuthError,
    Trading212Client,
    Trading212Error,
    Trading212NotFoundError,
    Trading212RateLimitError,
    Trading212ServerError,
)

BASE_URL = "https://demo.trading212.com/api/v0"


@pytest.fixture
def client() -> Trading212Client:
    """Trading212Client pointed at the demo environment with minimal retries."""
    return Trading212Client(
        api_key="test-api-key",
        base_url=BASE_URL,
        max_retries=2,
        retry_delay=0.01,
        timeout=5,
    )


# ---------------------------------------------------------------------------
# Authentication / headers
# ---------------------------------------------------------------------------


class TestClientInit:
    def test_api_key_in_headers(self, client):
        assert client._session.headers.get("Authorization") == "test-api-key"

    def test_base_url_stored(self, client):
        assert client.base_url == BASE_URL

    def test_base_url_trailing_slash_stripped(self):
        c = Trading212Client(api_key="k", base_url=BASE_URL + "/")
        assert not c.base_url.endswith("/")


# ---------------------------------------------------------------------------
# Account endpoints
# ---------------------------------------------------------------------------


class TestAccountEndpoints:
    @resp_lib.activate
    def test_get_account_info_success(self, client):
        resp_lib.add(
            resp_lib.GET,
            f"{BASE_URL}/equity/account/info",
            json={"id": 123, "currencyCode": "USD"},
            status=200,
        )
        result = client.get_account_info()
        assert result["id"] == 123

    @resp_lib.activate
    def test_get_account_cash_success(self, client):
        resp_lib.add(
            resp_lib.GET,
            f"{BASE_URL}/equity/account/cash",
            json={"free": 5000.0, "total": 10000.0},
            status=200,
        )
        result = client.get_account_cash()
        assert result["total"] == 10000.0


# ---------------------------------------------------------------------------
# Portfolio / Positions
# ---------------------------------------------------------------------------


class TestPositionEndpoints:
    @resp_lib.activate
    def test_get_positions_returns_list(self, client):
        resp_lib.add(
            resp_lib.GET,
            f"{BASE_URL}/equity/portfolio",
            json=[{"ticker": "XAUUSD", "quantity": 1.0}],
            status=200,
        )
        positions = client.get_positions()
        assert isinstance(positions, list)
        assert positions[0]["ticker"] == "XAUUSD"

    @resp_lib.activate
    def test_get_positions_empty_list(self, client):
        resp_lib.add(
            resp_lib.GET,
            f"{BASE_URL}/equity/portfolio",
            json=[],
            status=200,
        )
        positions = client.get_positions()
        assert positions == []


# ---------------------------------------------------------------------------
# Order endpoints
# ---------------------------------------------------------------------------


class TestOrderEndpoints:
    @resp_lib.activate
    def test_place_market_order_success(self, client):
        resp_lib.add(
            resp_lib.POST,
            f"{BASE_URL}/equity/orders/market",
            json={"id": "order-001", "status": "PENDING"},
            status=200,
        )
        result = client.place_market_order("XAUUSD", quantity=1.0)
        assert result["id"] == "order-001"

    @resp_lib.activate
    def test_place_limit_order_success(self, client):
        resp_lib.add(
            resp_lib.POST,
            f"{BASE_URL}/equity/orders/limit",
            json={"id": "order-002"},
            status=200,
        )
        result = client.place_limit_order("XAUUSD", quantity=1.0, limit_price=2000.0)
        assert result["id"] == "order-002"

    @resp_lib.activate
    def test_cancel_order_success(self, client):
        resp_lib.add(
            resp_lib.DELETE,
            f"{BASE_URL}/equity/orders/order-001",
            body=b"",
            status=200,
        )
        # Should not raise
        client.cancel_order("order-001")

    @resp_lib.activate
    def test_place_stop_order_success(self, client):
        resp_lib.add(
            resp_lib.POST,
            f"{BASE_URL}/equity/orders/stop",
            json={"id": "order-003"},
            status=200,
        )
        result = client.place_stop_order("XAUUSD", quantity=-1.0, stop_price=1950.0)
        assert result["id"] == "order-003"


# ---------------------------------------------------------------------------
# Historical data
# ---------------------------------------------------------------------------


class TestHistoricalData:
    @resp_lib.activate
    def test_get_historical_data_returns_list(self, client):
        candles = [
            {
                "timestamp": "2024-01-01T00:00:00Z",
                "open": 2000.0,
                "high": 2010.0,
                "low": 1990.0,
                "close": 2005.0,
                "volume": 1000.0,
            }
        ]
        resp_lib.add(
            resp_lib.GET,
            f"{BASE_URL}/equity/history/instrument/XAUUSD",
            json=candles,
            status=200,
        )
        result = client.get_historical_data("XAUUSD")
        assert isinstance(result, list)
        assert result[0]["close"] == 2005.0

    @resp_lib.activate
    def test_get_historical_data_empty_response(self, client):
        resp_lib.add(
            resp_lib.GET,
            f"{BASE_URL}/equity/history/instrument/XAUUSD",
            json={},  # Non-list response
            status=200,
        )
        result = client.get_historical_data("XAUUSD")
        assert result == []


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    @resp_lib.activate
    def test_raises_auth_error_on_401(self, client):
        resp_lib.add(
            resp_lib.GET,
            f"{BASE_URL}/equity/account/info",
            status=401,
        )
        with pytest.raises(Trading212AuthError):
            client.get_account_info()

    @resp_lib.activate
    def test_raises_not_found_on_404(self, client):
        resp_lib.add(
            resp_lib.GET,
            f"{BASE_URL}/equity/portfolio/FAKEINSTRUMENT",
            status=404,
        )
        with pytest.raises(Trading212NotFoundError):
            client.get_position("FAKEINSTRUMENT")

    @resp_lib.activate
    def test_raises_rate_limit_on_429(self, client):
        resp_lib.add(
            resp_lib.GET,
            f"{BASE_URL}/equity/account/info",
            status=429,
            headers={"Retry-After": "0.01"},
        )
        resp_lib.add(
            resp_lib.GET,
            f"{BASE_URL}/equity/account/info",
            status=429,
            headers={"Retry-After": "0.01"},
        )
        with pytest.raises(Trading212RateLimitError):
            client.get_account_info()

    @resp_lib.activate
    def test_raises_server_error_on_500(self, client):
        for _ in range(client.max_retries):
            resp_lib.add(
                resp_lib.GET,
                f"{BASE_URL}/equity/account/info",
                status=500,
            )
        with pytest.raises(Trading212ServerError):
            client.get_account_info()

    @resp_lib.activate
    def test_raises_trading212_error_on_4xx(self, client):
        resp_lib.add(
            resp_lib.GET,
            f"{BASE_URL}/equity/account/info",
            status=400,
            json={"error": "bad request"},
        )
        with pytest.raises(Trading212Error):
            client.get_account_info()

    @resp_lib.activate
    def test_retries_on_500_then_succeeds(self, client):
        """Client should retry on 500 and succeed on the next attempt."""
        resp_lib.add(
            resp_lib.GET,
            f"{BASE_URL}/equity/account/info",
            status=500,
        )
        resp_lib.add(
            resp_lib.GET,
            f"{BASE_URL}/equity/account/info",
            json={"id": 1},
            status=200,
        )
        result = client.get_account_info()
        assert result["id"] == 1
