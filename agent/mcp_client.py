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
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def build_tool_schemas(tools: list[Any]) -> tuple[list[dict], set[str]]:
    """Convert MCP Tool objects to OpenAI function-calling schemas.

    Strips the "session" parameter from what the LLM sees. session_id is
    agent-loop plumbing the model has no way to know the correct value for
    (it was inventing placeholder strings like "current_session" before this
    fix) — not something to expose as a parameter for it to fill in.
    agent_loop.py injects the real value automatically for any tool name in
    the returned `tools_requiring_session` set.
    """
    schemas = []
    tools_requiring_session = set()
    for tool in tools:
        input_schema = dict(tool.inputSchema)
        properties = dict(input_schema.get("properties", {}))
        if "session" in properties:
            tools_requiring_session.add(tool.name)
            properties = {k: v for k, v in properties.items() if k != "session"}
            input_schema = {
                **input_schema,
                "properties": properties,
                "required": [r for r in input_schema.get("required", []) if r != "session"],
            }
        schemas.append(
            {
                "type": "function",
                "function": {"name": tool.name, "description": tool.description, "parameters": input_schema},
            }
        )
    return schemas, tools_requiring_session


class MCPClient:
    def __init__(self) -> None:
        self._exit_stack = AsyncExitStack()
        self.session: ClientSession | None = None
        self.tool_schemas: list[dict] = []
        self.tools_requiring_session: set[str] = set()

    async def connect(self) -> None:
        params = StdioServerParameters(command=sys.executable, args=["-m", "mcp_commerce.server"])
        read, write = await self._exit_stack.enter_async_context(stdio_client(params))
        self.session = await self._exit_stack.enter_async_context(ClientSession(read, write))
        await self.session.initialize()

        tools = await self.session.list_tools()
        self.tool_schemas, self.tools_requiring_session = build_tool_schemas(tools.tools)

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
