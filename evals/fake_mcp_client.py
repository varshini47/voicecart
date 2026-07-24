"""In-process fake MCP client for evals.

Calls mcp_commerce's real tool functions directly — real pydantic
validation, the real checkout confirm gate, real structured errors — against
the fake Shopify backend (evals/fake_shopify.py). Skips the stdio subprocess
entirely (see agent/mcp_client.py for the real one), which is both faster
and lets us swap the Shopify backend from inside the same process.
"""

from __future__ import annotations

from contextlib import contextmanager

from agent.mcp_client import build_tool_schemas
from mcp_commerce import carts, server
from evals import fake_shopify


@contextmanager
def patched_shopify():
    """Swap the Shopify backend for the in-memory fake, for one scenario.

    Both mcp_commerce.server and mcp_commerce.carts import shopify_client
    under their own local name, so both need patching independently.
    """
    original_server_shopify = server.shopify
    original_carts_shopify = carts.shopify
    server.shopify = fake_shopify
    carts.shopify = fake_shopify
    fake_shopify.reset()
    carts._carts.clear()
    try:
        yield
    finally:
        server.shopify = original_server_shopify
        carts.shopify = original_carts_shopify


class FakeMCPClient:
    def __init__(self) -> None:
        self.tool_schemas: list[dict] = []
        self.tools_requiring_session: set[str] = set()
        self.calls: list[tuple[str, dict]] = []

    async def load_tool_schemas(self) -> None:
        tools = await server.mcp.list_tools()
        self.tool_schemas, self.tools_requiring_session = build_tool_schemas(tools)

    async def call_tool(self, name: str, arguments: dict) -> dict:
        self.calls.append((name, arguments))
        tool_fn = getattr(server, name)
        result = tool_fn(**arguments)
        return result.model_dump()
