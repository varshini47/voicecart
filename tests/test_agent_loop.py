"""Tests for agent/agent_loop.py's tool-calling orchestration.

Mocks agent.llm.chat_completion (a queue of canned responses) and a fake
MCP client, so these never touch a real LLM or the real mcp_commerce
subprocess.
"""

from __future__ import annotations

import json

import pytest

from agent import agent_loop, session

SESSION = "test-session"


class FakeMCPClient:
    def __init__(self, tool_results: dict[str, dict] | None = None) -> None:
        self.tool_schemas: list[dict] = []
        self.tool_results = tool_results or {}
        self.calls: list[tuple[str, dict]] = []

    async def call_tool(self, name: str, arguments: dict) -> dict:
        self.calls.append((name, arguments))
        return self.tool_results.get(name, {"ok": True})


def _tool_call_message(call_id: str, name: str, arguments: dict) -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }
        ],
    }


def _reply_message(text: str) -> dict:
    return {"role": "assistant", "content": text}


@pytest.fixture(autouse=True)
def reset_sessions():
    session._sessions.clear()
    yield
    session._sessions.clear()


def _queue_responses(monkeypatch: pytest.MonkeyPatch, responses: list[dict]) -> list[list[dict]]:
    calls_seen: list[list[dict]] = []
    queue = list(responses)

    def fake_chat_completion(messages: list[dict], tools=None) -> dict:
        calls_seen.append(list(messages))
        return queue.pop(0)

    monkeypatch.setattr(agent_loop.llm, "chat_completion", fake_chat_completion)
    return calls_seen


@pytest.mark.asyncio
async def test_plain_reply_no_tool_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    _queue_responses(monkeypatch, [_reply_message("Hi there!")])
    mcp_client = FakeMCPClient()

    reply = await agent_loop.run_turn(SESSION, "hello", mcp_client)

    assert reply == "Hi there!"
    assert mcp_client.calls == []
    history = session.get_history(SESSION)
    assert history[0] == {"role": "user", "content": "hello"}
    assert history[1]["content"] == "Hi there!"


@pytest.mark.asyncio
async def test_single_tool_call_round_trip(monkeypatch: pytest.MonkeyPatch) -> None:
    _queue_responses(
        monkeypatch,
        [
            _tool_call_message("call_1", "search_products", {"query": "milk", "limit": 5}),
            _reply_message("Found milk!"),
        ],
    )
    mcp_client = FakeMCPClient(tool_results={"search_products": {"products": [{"title": "Amul Milk"}], "error": None}})

    reply = await agent_loop.run_turn(SESSION, "find milk", mcp_client)

    assert reply == "Found milk!"
    assert mcp_client.calls == [("search_products", {"query": "milk", "limit": 5})]

    history = session.get_history(SESSION)
    tool_messages = [m for m in history if m.get("role") == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0]["tool_call_id"] == "call_1"
    assert json.loads(tool_messages[0]["content"]) == {"products": [{"title": "Amul Milk"}], "error": None}


@pytest.mark.asyncio
async def test_history_carries_across_turns(monkeypatch: pytest.MonkeyPatch) -> None:
    calls_seen = _queue_responses(
        monkeypatch,
        [_reply_message("Got milk noted."), _reply_message("You asked for milk.")],
    )
    mcp_client = FakeMCPClient()

    await agent_loop.run_turn(SESSION, "I want milk", mcp_client)
    await agent_loop.run_turn(SESSION, "what did I ask for?", mcp_client)

    second_request_messages = calls_seen[1]
    contents = [m.get("content") for m in second_request_messages]
    assert "I want milk" in contents
    assert "Got milk noted." in contents
    assert "what did I ask for?" in contents


@pytest.mark.asyncio
async def test_max_tool_rounds_cap_prevents_infinite_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    endless_tool_call = _tool_call_message("call_x", "get_cart", {"session": SESSION})
    _queue_responses(monkeypatch, [endless_tool_call] * agent_loop.MAX_TOOL_ROUNDS)
    mcp_client = FakeMCPClient()

    reply = await agent_loop.run_turn(SESSION, "loop forever", mcp_client)

    assert reply == agent_loop.FALLBACK_REPLY
    assert len(mcp_client.calls) == agent_loop.MAX_TOOL_ROUNDS
