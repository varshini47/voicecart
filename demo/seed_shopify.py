"""Seed the Shopify dev store with ~30 grocery products via the Admin REST API.

Why it exists: the agent/evals need a realistic small catalog (multiple brands
per item, so the agent has real ambiguity to resolve, e.g. "Amul or Nandini?").

Requires SHOPIFY_STORE_DOMAIN and SHOPIFY_ADMIN_TOKEN in .env (Week 0 setup).

Run: .venv/Scripts/python.exe demo/seed_shopify.py
"""

import os
import sys
import time

import requests
from dotenv import load_dotenv

load_dotenv()

STORE_DOMAIN = os.environ.get("SHOPIFY_STORE_DOMAIN")
ADMIN_TOKEN = os.environ.get("SHOPIFY_ADMIN_TOKEN")
API_VERSION = "2024-10"

if not STORE_DOMAIN or not ADMIN_TOKEN:
    sys.exit("Set SHOPIFY_STORE_DOMAIN and SHOPIFY_ADMIN_TOKEN in .env first.")

BASE_URL = f"https://{STORE_DOMAIN}/admin/api/{API_VERSION}"
HEADERS = {
    "X-Shopify-Access-Token": ADMIN_TOKEN,
    "Content-Type": "application/json",
}

# (title, product_type, price, brand) - multiple brands per category on
# purpose, so "add milk" is genuinely ambiguous for the agent to clarify.
PRODUCTS = [
    ("Amul Toned Milk 500ml", "Dairy", "30.00", "Amul"),
    ("Nandini Toned Milk 500ml", "Dairy", "28.00", "Nandini"),
    ("Mother Dairy Full Cream Milk 500ml", "Dairy", "32.00", "Mother Dairy"),
    ("Amul Butter 100g", "Dairy", "52.00", "Amul"),
    ("Amul Cheese Slices 200g", "Dairy", "120.00", "Amul"),
    ("Nestle a+ Curd 400g", "Dairy", "45.00", "Nestle"),
    ("Britannia Brown Bread 400g", "Bakery", "45.00", "Britannia"),
    ("Modern White Bread 400g", "Bakery", "40.00", "Modern"),
    ("Britannia Whole Wheat Bread 400g", "Bakery", "48.00", "Britannia"),
    ("Tata Salt 1kg", "Staples", "25.00", "Tata"),
    ("Fortune Sunflower Oil 1L", "Staples", "150.00", "Fortune"),
    ("Aashirvaad Atta 5kg", "Staples", "260.00", "Aashirvaad"),
    ("India Gate Basmati Rice 1kg", "Staples", "110.00", "India Gate"),
    ("Toor Dal 1kg", "Staples", "140.00", "Generic"),
    ("Moong Dal 1kg", "Staples", "130.00", "Generic"),
    ("Maggi 2-Minute Noodles 70g", "Snacks", "14.00", "Maggi"),
    ("Lay's Classic Salted 52g", "Snacks", "20.00", "Lay's"),
    ("Parle-G Biscuits 250g", "Snacks", "30.00", "Parle"),
    ("Hide & Seek Chocolate Biscuits 120g", "Snacks", "35.00", "Britannia"),
    ("Bru Instant Coffee 100g", "Beverages", "180.00", "Bru"),
    ("Tata Tea Gold 250g", "Beverages", "140.00", "Tata"),
    ("Coca-Cola 750ml", "Beverages", "40.00", "Coca-Cola"),
    ("Real Orange Juice 1L", "Beverages", "110.00", "Real"),
    ("Onion 1kg", "Produce", "35.00", "Generic"),
    ("Tomato 1kg", "Produce", "40.00", "Generic"),
    ("Potato 1kg", "Produce", "30.00", "Generic"),
    ("Banana 6pc", "Produce", "45.00", "Generic"),
    ("Apple 1kg", "Produce", "160.00", "Generic"),
    ("Eggs 6pc Tray", "Produce", "48.00", "Generic"),
    ("Colgate Toothpaste 100g", "Household", "55.00", "Colgate"),
    ("Vim Dishwash Bar 200g", "Household", "20.00", "Vim"),
]


def create_product(title, product_type, price, brand):
    payload = {
        "product": {
            "title": title,
            "product_type": product_type,
            "vendor": brand,
            "status": "active",
            "variants": [{"price": price}],
        }
    }
    resp = requests.post(f"{BASE_URL}/products.json", json=payload, headers=HEADERS)
    resp.raise_for_status()
    return resp.json()["product"]["id"]


def main():
    created = 0
    for title, product_type, price, brand in PRODUCTS:
        product_id = create_product(title, product_type, price, brand)
        print(f"created {product_id}: {title}")
        created += 1
        time.sleep(0.5)  # stay under the Admin API rate limit
    print(f"\nDone. Created {created} products.")


if __name__ == "__main__":
    main()
