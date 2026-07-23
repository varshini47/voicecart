"""The only module in mcp_commerce that talks to Shopify directly.

Wraps the Storefront GraphQL API (product search, cart create/add/remove/read).
Keeping every Shopify-specific query and response shape in this one module is
what makes the backend swappable later (see CLAUDE.md) — server.py's tool
contracts wouldn't need to change if this were replaced with an ONDC client.
"""

from __future__ import annotations

import os

import requests

API_VERSION = "2024-10"


class ShopifyError(Exception):
    """Raised for GraphQL-level errors, mutation userErrors, or request failures."""


def _endpoint() -> str:
    domain = os.environ["SHOPIFY_STORE_DOMAIN"]
    return f"https://{domain}/api/{API_VERSION}/graphql.json"


def _headers() -> dict[str, str]:
    return {
        "X-Shopify-Storefront-Access-Token": os.environ["SHOPIFY_STOREFRONT_TOKEN"],
        "Content-Type": "application/json",
    }


def _graphql(query: str, variables: dict | None = None) -> dict:
    try:
        response = requests.post(
            _endpoint(),
            headers=_headers(),
            json={"query": query, "variables": variables or {}},
            timeout=15,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise ShopifyError(f"Shopify request failed: {exc}") from exc

    body = response.json()
    if "errors" in body:
        raise ShopifyError("; ".join(e["message"] for e in body["errors"]))
    return body["data"]


def _check_user_errors(user_errors: list[dict]) -> None:
    if user_errors:
        raise ShopifyError("; ".join(e["message"] for e in user_errors))


CART_FIELDS = """
id
checkoutUrl
cost {
  subtotalAmount { amount currencyCode }
}
lines(first: 50) {
  edges {
    node {
      id
      quantity
      cost { totalAmount { amount currencyCode } }
      merchandise {
        ... on ProductVariant {
          id
          product { title vendor }
        }
      }
    }
  }
}
"""


def _parse_cart(cart: dict) -> dict:
    lines = []
    for edge in cart["lines"]["edges"]:
        node = edge["node"]
        lines.append(
            {
                "line_id": node["id"],
                "variant_id": node["merchandise"]["id"],
                "title": node["merchandise"]["product"]["title"],
                "brand": node["merchandise"]["product"]["vendor"],
                "quantity": node["quantity"],
                "line_total": node["cost"]["totalAmount"]["amount"],
            }
        )
    subtotal = cart["cost"]["subtotalAmount"]
    return {
        "cart_id": cart["id"],
        "checkout_url": cart["checkoutUrl"],
        "lines": lines,
        "subtotal": subtotal["amount"],
        "currency": subtotal["currencyCode"],
    }


def search_products(query: str, limit: int) -> list[dict]:
    gql = """
    query($query: String!, $first: Int!) {
      products(first: $first, query: $query) {
        edges {
          node {
            title
            vendor
            variants(first: 1) {
              edges { node { id price { amount currencyCode } } }
            }
          }
        }
      }
    }
    """
    data = _graphql(gql, {"query": query, "first": limit})
    results = []
    for edge in data["products"]["edges"]:
        node = edge["node"]
        variant = node["variants"]["edges"][0]["node"]
        results.append(
            {
                "variant_id": variant["id"],
                "title": node["title"],
                "brand": node["vendor"],
                "price": variant["price"]["amount"],
                "currency": variant["price"]["currencyCode"],
            }
        )
    return results


def create_cart() -> str:
    gql = """
    mutation {
      cartCreate {
        cart { id }
        userErrors { message }
      }
    }
    """
    data = _graphql(gql)
    result = data["cartCreate"]
    _check_user_errors(result["userErrors"])
    return result["cart"]["id"]


def get_cart(cart_id: str) -> dict:
    gql = f"""
    query($cartId: ID!) {{
      cart(id: $cartId) {{ {CART_FIELDS} }}
    }}
    """
    data = _graphql(gql, {"cartId": cart_id})
    cart = data["cart"]
    if cart is None:
        raise ShopifyError(f"cart {cart_id} no longer exists")
    return _parse_cart(cart)


def add_line(cart_id: str, variant_id: str, quantity: int) -> dict:
    gql = f"""
    mutation($cartId: ID!, $lines: [CartLineInput!]!) {{
      cartLinesAdd(cartId: $cartId, lines: $lines) {{
        cart {{ {CART_FIELDS} }}
        userErrors {{ message }}
      }}
    }}
    """
    data = _graphql(
        gql,
        {"cartId": cart_id, "lines": [{"merchandiseId": variant_id, "quantity": quantity}]},
    )
    result = data["cartLinesAdd"]
    _check_user_errors(result["userErrors"])
    return _parse_cart(result["cart"])


def remove_line(cart_id: str, line_id: str) -> dict:
    gql = f"""
    mutation($cartId: ID!, $lineIds: [ID!]!) {{
      cartLinesRemove(cartId: $cartId, lineIds: $lineIds) {{
        cart {{ {CART_FIELDS} }}
        userErrors {{ message }}
      }}
    }}
    """
    data = _graphql(gql, {"cartId": cart_id, "lineIds": [line_id]})
    result = data["cartLinesRemove"]
    _check_user_errors(result["userErrors"])
    return _parse_cart(result["cart"])
