"""In-memory conversation history, keyed by session_id.

Dict-backed per CLAUDE.md — no database until we actually need one. History
holds raw chat/completions message dicts (role, content, and — since
Milestone 2.2's tool-calling loop — tool_calls/tool_call_id where relevant),
so agent/agent_loop.py can splice them straight into the next request and
the LLM sees the full history of what tools were called and what they
returned, not just user-visible text.
"""

from __future__ import annotations

import uuid

_sessions: dict[str, list[dict]] = {}


def new_session_id() -> str:
    return uuid.uuid4().hex


def get_history(session_id: str) -> list[dict]:
    return _sessions.setdefault(session_id, [])


def append(session_id: str, message: dict) -> None:
    get_history(session_id).append(message)
