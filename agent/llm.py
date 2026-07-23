"""Provider-agnostic LLM client.

All provider config comes from env vars (LLM_API_KEY, LLM_MODEL, LLM_BASE_URL)
per CLAUDE.md — never hardcode a provider. We talk to the OpenAI-compatible
chat/completions REST endpoint directly with `requests` instead of pulling in
a provider SDK, since Groq (our current provider) and most alternatives all
speak this same shape, including tool/function calling.

This module is intentionally domain-agnostic — no system prompt, no
shopping-specific logic. That lives in agent/agent_loop.py, which is the
thing that actually decides what the agent does with the LLM's response.
"""

from __future__ import annotations

import os

import requests


def chat_completion(messages: list[dict], tools: list[dict] | None = None) -> dict:
    """Send `messages` (and optional tool schemas) to the LLM.

    Returns the raw assistant message dict from the API — has a "content"
    string for a plain reply, or a "tool_calls" list if the model wants to
    call one or more tools instead.
    """
    api_key = os.environ["LLM_API_KEY"]
    model = os.environ["LLM_MODEL"]
    base_url = os.environ["LLM_BASE_URL"]

    payload = {"model": model, "messages": messages}
    if tools:
        payload["tools"] = tools

    response = requests.post(
        f"{base_url}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json=payload,
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]
