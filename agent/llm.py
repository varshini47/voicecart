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
import time

import requests

MAX_RETRIES = 5


def chat_completion(messages: list[dict], tools: list[dict] | None = None) -> dict:
    """Send `messages` (and optional tool schemas) to the LLM.

    Returns the raw assistant message dict from the API — has a "content"
    string for a plain reply, or a "tool_calls" list if the model wants to
    call one or more tools instead.

    temperature=0: a shopping assistant should behave consistently rather
    than creatively, and this also makes the eval suite (evals/) far less
    flaky than sampling would.
    """
    api_key = os.environ["LLM_API_KEY"]
    model = os.environ["LLM_MODEL"]
    base_url = os.environ["LLM_BASE_URL"]

    payload = {"model": model, "messages": messages, "temperature": 0}
    if tools:
        payload["tools"] = tools

    for attempt in range(MAX_RETRIES):
        response = requests.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json=payload,
            timeout=30,
        )
        if response.status_code == 429 and attempt < MAX_RETRIES - 1:
            # Free-tier rate limit — the eval suite fires many requests back
            # to back and hits this in practice, so back off and retry.
            # Groq sometimes sends Retry-After; honor it if present, else
            # fall back to exponential backoff.
            retry_after = response.headers.get("retry-after")
            delay = float(retry_after) if retry_after else 2**attempt * 3
            time.sleep(delay)
            continue
        if response.status_code == 400 and attempt < MAX_RETRIES - 1:
            # Groq/Llama occasionally emits a malformed non-JSON function-call
            # token (e.g. "<function=name{...}>") instead of a proper
            # tool_calls response, surfaced as error code "tool_use_failed".
            # This is inference noise, not a deterministic function of the
            # prompt, but it can repeat several times in a row at
            # temperature=0 — confirmed a plain retry at temperature=0 alone
            # isn't reliable, but nudging the temperature up breaks the loop.
            # We keep nudging (rather than one fixed bump) since no single
            # value reliably escapes it either.
            try:
                error_code = response.json().get("error", {}).get("code")
            except ValueError:
                error_code = None
            if error_code == "tool_use_failed":
                payload["temperature"] = 0.2 * (attempt + 1)
                continue
        response.raise_for_status()
        return response.json()["choices"][0]["message"]
