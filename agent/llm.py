"""Provider-agnostic LLM client.

All provider config comes from env vars (LLM_API_KEY, LLM_MODEL, LLM_BASE_URL)
per CLAUDE.md — never hardcode a provider. We talk to the OpenAI-compatible
chat/completions REST endpoint directly with `requests` instead of pulling in
a provider SDK, since Groq (our current provider) and most alternatives all
speak this same shape.
"""

from __future__ import annotations

import os

import requests

SYSTEM_PROMPT = (
    "You are VoiceCart, a friendly voice-based grocery shopping assistant. "
    "Keep replies short and conversational, since they will be spoken aloud. "
    "You cannot place orders yet — just chat helpfully about what the user wants."
)


def reply(user_text: str) -> str:
    """Send `user_text` to the configured LLM with the shopping-assistant persona and return its reply."""
    api_key = os.environ["LLM_API_KEY"]
    model = os.environ["LLM_MODEL"]
    base_url = os.environ["LLM_BASE_URL"]

    response = requests.post(
        f"{base_url}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text},
            ],
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]
