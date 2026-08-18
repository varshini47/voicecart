"""MCP server exposing commerce tools backed by the Shopify Storefront API.

The agent never talks to Shopify directly (see CLAUDE.md) — it only ever
sees these five tools. Run standalone for manual testing:

  .venv/Scripts/python.exe -m mcp_commerce.server
"""

from __future__ import annotations

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv()

from mcp_commerce import shopify_client as shopify
from mcp_commerce.carts import get_or_create_cart_id
from mcp_commerce.models import (
    CartLine,
    CartResult,
    CheckoutResult,
    Product,
    SearchProductsResult,
)

mcp = FastMCP("commerce-mcp")


def _to_cart_result(session: str, cart: dict) -> CartResult:
    return CartResult(
        session=session,
        lines=[
            CartLine(
                product_id=line["variant_id"],
                title=line["title"],
                brand=line["brand"],
                quantity=line["quantity"],
                line_total=line["line_total"],
            )
            for line in cart["lines"]
        ],
        subtotal=cart["subtotal"],
        currency=cart["currency"],
    )


@mcp.tool()
def search_products(query: str, limit: int = 5) -> SearchProductsResult:
    """Search the grocery catalog by free-text query (e.g. "milk", "bread")."""
    try:
        raw = shopify.search_products(query, limit)
    except shopify.ShopifyError as exc:
        return SearchProductsResult(error=str(exc))
    return SearchProductsResult(
        products=[
            Product(product_id=p["variant_id"], title=p["title"], brand=p["brand"], price=p["price"], currency=p["currency"])
            for p in raw
        ]
    )


@mcp.tool()
def get_cart(session: str) -> CartResult:
    """Get the current cart contents for a session. Read-only, no confirmation needed."""
    try:
        cart_id = get_or_create_cart_id(session)
        cart = shopify.get_cart(cart_id)
    except shopify.ShopifyError as exc:
        return CartResult(session=session, error=str(exc))
    return _to_cart_result(session, cart)


@mcp.tool()
def add_to_cart(session: str, product_id: str, quantity: int) -> CartResult:
    """Add a quantity of a product (product_id from search_products) to the session's cart."""
    if quantity < 1:
        return CartResult(session=session, error="quantity must be at least 1")
    try:
        cart_id = get_or_create_cart_id(session)
        cart = shopify.add_line(cart_id, product_id, quantity)
    except shopify.ShopifyError as exc:
        return CartResult(session=session, error=str(exc))
    return _to_cart_result(session, cart)


@mcp.tool()
def remove_from_cart(session: str, product_id: str) -> CartResult:
    """Remove a product entirely from the session's cart."""
    try:
        cart_id = get_or_create_cart_id(session)
        cart = shopify.get_cart(cart_id)
        line = next((l for l in cart["lines"] if l["variant_id"] == product_id), None)
        if line is None:
            return CartResult(session=session, error=f"product {product_id} is not in the cart")
        cart = shopify.remove_line(cart_id, line["line_id"])
    except shopify.ShopifyError as exc:
        return CartResult(session=session, error=str(exc))
    return _to_cart_result(session, cart)


@mcp.tool()
def checkout(session: str, confirm: bool = False) -> CheckoutResult:
    """Finalize the session's cart and return a checkout link.

    Requires confirm=true. The agent must get explicit user confirmation
    before ever calling this with confirm=true (see CLAUDE.md). The buyer
    enters their own email and shipping address on Shopify's checkout page
    itself — this tool doesn't collect either in conversation.
    """
    if not confirm:
        return CheckoutResult(
            session=session,
            confirmed=False,
            error="checkout requires confirm=true; ask the user to confirm the order first",
        )
    try:
        cart_id = get_or_create_cart_id(session)
        cart = shopify.get_cart(cart_id)
    except shopify.ShopifyError as exc:
        return CheckoutResult(session=session, confirmed=False, error=str(exc))

    if not cart["lines"]:
        return CheckoutResult(session=session, confirmed=False, error="cart is empty, nothing to check out")

    return CheckoutResult(
        session=session,
        confirmed=True,
        checkout_url=cart["checkout_url"],
        subtotal=cart["subtotal"],
        currency=cart["currency"],
    )


if __name__ == "__main__":
    mcp.run()
