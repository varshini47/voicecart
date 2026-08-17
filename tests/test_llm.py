"""Tests for agent/llm.py's retry logic — added after a live debugging
session where a 429 with a long Retry-After caused chat_completion to block
for an unknown, very long time (see NOTES.md). Retries that would sleep
longer than MAX_RETRY_DELAY_SECONDS should fail fast instead.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

# evals/runner.py monkeypatches agent.llm.chat_completion at import time (a
# module-level `llm.chat_completion = _throttled_chat_completion`), and
# pytest collecting tests/test_evals.py imports evals.runner regardless of
# the `-m eval` marker filter — so by the time these tests run, plain
# `llm.chat_completion` may already be that wrapped version, not the real
# one. evals/runner.py keeps its own reference to the original before
# overwriting it (`_real_chat_completion`); import it here instead of
# reloading agent.llm, which would reset the module for the whole session
# (including undoing the eval suite's own throttle if `pytest -m eval` ever
# collects this file too).
from evals.runner import _real_chat_completion as chat_completion


class FakeResponse:
    def __init__(self, status_code: int, headers: dict | None = None, json_body: dict | None = None) -> None:
        self.status_code = status_code
        self.headers = headers or {}
        self._json_body = json_body or {}

    def json(self) -> dict:
        return self._json_body

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            import requests

            raise requests.exceptions.HTTPError(f"{self.status_code} error", response=self)


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("LLM_BASE_URL", "https://example.invalid/v1")


def test_long_retry_after_fails_fast_instead_of_blocking(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = [FakeResponse(429, headers={"retry-after": "3600"})]

    with patch("agent.llm.requests.post", side_effect=responses):
        with patch("agent.llm.time.sleep") as mock_sleep:
            with pytest.raises(Exception):
                chat_completion([{"role": "user", "content": "hi"}])
            mock_sleep.assert_not_called()


def test_short_retry_after_still_retries_and_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    ok_message = {"role": "assistant", "content": "hello"}
    responses = [
        FakeResponse(429, headers={"retry-after": "1"}),
        FakeResponse(200, json_body={"choices": [{"message": ok_message}]}),
    ]

    with patch("agent.llm.requests.post", side_effect=responses):
        with patch("agent.llm.time.sleep") as mock_sleep:
            result = chat_completion([{"role": "user", "content": "hi"}])
            mock_sleep.assert_called_once_with(1.0)

    assert result == ok_message
