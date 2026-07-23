"""Manages the agent's connection to the mcp_commerce MCP server.

The server runs as a subprocess over stdio (spawned once at app startup, not
per-request — see agent/main.py's lifespan), the same transport used for the
standalone verification during Milestone 2.1. Tool schemas are fetched once
from the server itself and converted to the OpenAI function-calling shape,
so the MCP server's schemas stay the single source of truth (per CLAUDE.md:
"tool schemas are the contract").
"""

from __future__ import annotations

import json
import sys
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPClient:
    def __init__(self) -> None:
        self._exit_stack = AsyncExitStack()
        self.session: ClientSession | None = None
        self.tool_schemas: list[dict] = []

    async def connect(self) -> None:
        params = StdioServerParameters(command=sys.executable, args=["-m", "mcp_commerce.server"])
        read, write = await self._exit_stack.enter_async_context(stdio_client(params))
        self.session = await self._exit_stack.enter_async_context(ClientSession(read, write))
        await self.session.initialize()

        tools = await self.session.list_tools()
        self.tool_schemas = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema,
                },
            }
            for tool in tools.tools
        ]

    async def close(self) -> None:
        await self._exit_stack.aclose()

    async def call_tool(self, name: str, arguments: dict) -> dict:
        """Call an mcp_commerce tool, returning its parsed JSON result.

        mcp_commerce tools never raise (they return a structured `error`
        field — see CLAUDE.md), but the MCP layer itself can still report an
        error for things the tool never got to run (unknown tool name, bad
        arguments failing schema validation). Both cases are normalized to
        the same {"error": "..."} shape so the agent loop's retry/explain
        logic doesn't need to special-case them.
        """
        result = await self.session.call_tool(name, arguments)
        if result.isError:
            return {"error": result.content[0].text}
        return json.loads(result.content[0].text)
