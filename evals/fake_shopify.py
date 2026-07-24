"""In-memory stand-in for mcp_commerce.shopify_client, used only by evals.

Implements the same function signatures (search_products, create_cart,
get_cart, add_line, remove_line) and raises the same ShopifyError, so
mcp_commerce/server.py's tool functions run completely unmodified against
this — the only thing evals fake is the Shopify backend, not our own tool
logic (validation, the checkout confirm gate, error shapes all stay real).

The catalog mirrors demo/seed_shopify.py's product list so scenario wording
("add milk", "add a dozen eggs") matches the real store's ambiguity design.
Search is plain case-insensitive substring matching against title — simpler
than Shopify's real fuzzy search, which is a known simplification (see
evals/README or NOTES.md) that's fine for scoring agent *behavior* rather
than search relevance.
"""

from __future__ import annotations

import uuid

from mcp_commerce.shopify_client import ShopifyError

# (title, brand, price) — same products as demo/seed_shopify.py.
CATALOG = [
    ("Amul Toned Milk 500ml", "Amul", "30.00"),
    ("Nandini Toned Milk 500ml", "Nandini", "28.00"),
    ("Mother Dairy Full Cream Milk 500ml", "Mother Dairy", "32.00"),
    ("Amul Butter 100g", "Amul", "52.00"),
    ("Amul Cheese Slices 200g", "Amul", "120.00"),
    ("Nestle a+ Curd 400g", "Nestle", "45.00"),
    ("Britannia Brown Bread 400g", "Britannia", "45.00"),
    ("Modern White Bread 400g", "Modern", "40.00"),
    ("Britannia Whole Wheat Bread 400g", "Britannia", "48.00"),
    ("Tata Salt 1kg", "Tata", "25.00"),
    ("Fortune Sunflower Oil 1L", "Fortune", "150.00"),
    ("Aashirvaad Atta 5kg", "Aashirvaad", "260.00"),
    ("India Gate Basmati Rice 1kg", "India Gate", "110.00"),
    ("Toor Dal 1kg", "Generic", "140.00"),
    ("Moong Dal 1kg", "Generic", "130.00"),
    ("Maggi 2-Minute Noodles 70g", "Maggi", "14.00"),
    ("Lay's Classic Salted 52g", "Lay's", "20.00"),
    ("Parle-G Biscuits 250g", "Parle", "30.00"),
    ("Hide & Seek Chocolate Biscuits 120g", "Britannia", "35.00"),
    ("Bru Instant Coffee 100g", "Bru", "180.00"),
    ("Tata Tea Gold 250g", "Tata", "140.00"),
    ("Coca-Cola 750ml", "Coca-Cola", "40.00"),
    ("Real Orange Juice 1L", "Real", "110.00"),
    ("Onion 1kg", "Generic", "35.00"),
    ("Tomato 1kg", "Generic", "40.00"),
    ("Potato 1kg", "Generic", "30.00"),
    ("Banana 6pc", "Generic", "45.00"),
    ("Apple 1kg", "Generic", "160.00"),
    ("Eggs 6pc Tray", "Generic", "48.00"),
    ("Colgate Toothpaste 100g", "Colgate", "55.00"),
    ("Vim Dishwash Bar 200g", "Vim", "20.00"),
]

_products = [
    {"variant_id": f"gid://fake/Variant/{i}", "title": title, "brand": brand, "price": price, "currency": "USD"}
    for i, (title, brand, price) in enumerate(CATALOG)
]

_carts: dict[str, dict] = {}


def reset() -> None:
    """Wipe all cart state between eval scenarios."""
    _carts.clear()


def search_products(query: str, limit: int) -> list[dict]:
    needle = query.lower()
    matches = [p for p in _products if needle in p["title"].lower() or needle in p["brand"].lower()]
    return matches[:limit]


def create_cart() -> str:
    cart_id = f"fake-cart-{uuid.uuid4().hex}"
    _carts[cart_id] = {}  # variant_id -> quantity
    return cart_id


def _cart_lines(cart_id: str) -> dict:
    if cart_id not in _carts:
        raise ShopifyError(f"cart {cart_id} no longer exists")
    return _carts[cart_id]


def _serialize(cart_id: str) -> dict:
    lines = _cart_lines(cart_id)
    by_id = {p["variant_id"]: p for p in _products}
    line_dicts = []
    subtotal = 0.0
    for variant_id, quantity in lines.items():
        product = by_id[variant_id]
        line_total = float(product["price"]) * quantity
        subtotal += line_total
        line_dicts.append(
            {
                "line_id": f"line-{variant_id}",
                "variant_id": variant_id,
                "title": product["title"],
                "brand": product["brand"],
                "quantity": quantity,
                "line_total": f"{line_total:.2f}",
            }
        )
    return {
        "cart_id": cart_id,
        "checkout_url": f"https://fake.myshopify.com/cart/{cart_id}",
        "lines": line_dicts,
        "subtotal": f"{subtotal:.2f}",
        "currency": "USD",
    }


def get_cart(cart_id: str) -> dict:
    return _serialize(cart_id)


def add_line(cart_id: str, variant_id: str, quantity: int) -> dict:
    lines = _cart_lines(cart_id)
    if variant_id not in {p["variant_id"] for p in _products}:
        raise ShopifyError(f"unknown product {variant_id}")
    lines[variant_id] = lines.get(variant_id, 0) + quantity
    return _serialize(cart_id)


def remove_line(cart_id: str, line_id: str) -> dict:
    lines = _cart_lines(cart_id)
    variant_id = line_id.removeprefix("line-")
    if variant_id not in lines:
        raise ShopifyError(f"line {line_id} not found in cart")
    del lines[variant_id]
    return _serialize(cart_id)
