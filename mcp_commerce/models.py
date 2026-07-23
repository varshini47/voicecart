"""Pydantic schemas for mcp_commerce tool inputs/outputs.

Every result includes `error` rather than letting exceptions cross the tool
boundary into the agent loop, per CLAUDE.md.
"""

from __future__ import annotations

from pydantic import BaseModel


class Product(BaseModel):
    product_id: str  # Shopify variant GID; pass this back into add_to_cart
    title: str
    brand: str
    price: str
    currency: str


class SearchProductsResult(BaseModel):
    products: list[Product] = []
    error: str | None = None


class CartLine(BaseModel):
    product_id: str
    title: str
    brand: str
    quantity: int
    line_total: str


class CartResult(BaseModel):
    session: str
    lines: list[CartLine] = []
    subtotal: str | None = None
    currency: str | None = None
    error: str | None = None


class CheckoutResult(BaseModel):
    session: str
    confirmed: bool
    checkout_url: str | None = None
    subtotal: str | None = None
    currency: str | None = None
    error: str | None = None
