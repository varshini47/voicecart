"""In-memory conversation history, keyed by session_id.

Dict-backed per CLAUDE.md — no database until we actually need one. History
is stored as plain {"role", "content"} messages so agent/llm.py can splice
them straight into the chat/completions request.
"""

from __future__ import annotations

import uuid

_sessions: dict[str, list[dict[str, str]]] = {}


def new_session_id() -> str:
    return uuid.uuid4().hex


def get_history(session_id: str) -> list[dict[str, str]]:
    return _sessions.setdefault(session_id, [])


def append(session_id: str, role: str, content: str) -> None:
    get_history(session_id).append({"role": role, "content": content})
