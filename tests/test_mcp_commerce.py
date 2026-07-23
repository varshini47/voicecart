"""Unit tests for mcp_commerce's tool functions, mocking mcp_commerce.shopify_client
so nothing here touches the live Shopify store (that's covered by the manual
smoke test against the real dev store, run separately during development).

@mcp.tool() leaves the underlying function directly callable, so these call
the tool functions as plain Python functions rather than going through the
MCP protocol layer.
"""

from __future__ import annotations

import pytest

from mcp_commerce import carts, server
from mcp_commerce.shopify_client import ShopifyError

SESSION = "test-session"

FAKE_CART = {
    "cart_id": "gid://shopify/Cart/1",
    "checkout_url": "https://example.myshopify.com/cart/c/abc",
    "lines": [
        {
            "line_id": "gid://shopify/CartLine/1",
            "variant_id": "gid://shopify/ProductVariant/1",
            "title": "Amul Toned Milk 500ml",
            "brand": "Amul",
            "quantity": 2,
            "line_total": "60.0",
        }
    ],
    "subtotal": "60.0",
    "currency": "USD",
}

EMPTY_CART = {**FAKE_CART, "lines": [], "subtotal": "0.0"}


@pytest.fixture(autouse=True)
def reset_carts():
    carts._carts.clear()
    yield
    carts._carts.clear()


@pytest.fixture(autouse=True)
def fake_create_cart(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(server.shopify, "create_cart", lambda: FAKE_CART["cart_id"])


def test_search_products_returns_products(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        server.shopify,
        "search_products",
        lambda query, limit: [{"variant_id": "v1", "title": "Amul Toned Milk 500ml", "brand": "Amul", "price": "30.0", "currency": "USD"}],
    )
    result = server.search_products("milk", limit=5)
    assert result.error is None
    assert result.products[0].title == "Amul Toned Milk 500ml"


def test_search_products_returns_error_on_shopify_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_error(query, limit):
        raise ShopifyError("boom")

    monkeypatch.setattr(server.shopify, "search_products", raise_error)
    result = server.search_products("milk", limit=5)
    assert result.error == "boom"
    assert result.products == []


def test_add_to_cart_rejects_zero_quantity() -> None:
    result = server.add_to_cart(SESSION, "v1", 0)
    assert result.error == "quantity must be at least 1"


def test_add_to_cart_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server.shopify, "add_line", lambda cart_id, variant_id, quantity: FAKE_CART)
    result = server.add_to_cart(SESSION, "gid://shopify/ProductVariant/1", 2)
    assert result.error is None
    assert result.lines[0].quantity == 2
    assert result.subtotal == "60.0"


def test_get_cart_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server.shopify, "get_cart", lambda cart_id: FAKE_CART)
    result = server.get_cart(SESSION)
    assert result.error is None
    assert len(result.lines) == 1


def test_remove_from_cart_not_in_cart_returns_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server.shopify, "get_cart", lambda cart_id: EMPTY_CART)
    result = server.remove_from_cart(SESSION, "gid://shopify/ProductVariant/999")
    assert result.error == "product gid://shopify/ProductVariant/999 is not in the cart"


def test_remove_from_cart_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server.shopify, "get_cart", lambda cart_id: FAKE_CART)
    monkeypatch.setattr(server.shopify, "remove_line", lambda cart_id, line_id: EMPTY_CART)
    result = server.remove_from_cart(SESSION, "gid://shopify/ProductVariant/1")
    assert result.error is None
    assert result.lines == []


def test_checkout_requires_confirm() -> None:
    result = server.checkout(SESSION)
    assert result.confirmed is False
    assert "confirm=true" in result.error


def test_checkout_empty_cart_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server.shopify, "get_cart", lambda cart_id: EMPTY_CART)
    result = server.checkout(SESSION, confirm=True)
    assert result.confirmed is False
    assert "empty" in result.error


def test_checkout_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server.shopify, "get_cart", lambda cart_id: FAKE_CART)
    result = server.checkout(SESSION, confirm=True)
    assert result.confirmed is True
    assert result.checkout_url == FAKE_CART["checkout_url"]
    assert result.error is None
