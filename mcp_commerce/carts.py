"""In-memory session_id -> Shopify cart_id mapping.

Same "plain dict, no DB" approach as agent/session.py (see CLAUDE.md).
This is a separate store from the agent's conversation history — it lives
in mcp_commerce because the cart's identity is a Shopify concept the agent
shouldn't need to know about.
"""

from __future__ import annotations

from mcp_commerce import shopify_client as shopify

_carts: dict[str, str] = {}


def get_or_create_cart_id(session: str) -> str:
    if session not in _carts:
        _carts[session] = shopify.create_cart()
    return _carts[session]
