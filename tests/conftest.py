"""Shared fixtures: mock the STT/LLM/TTS boundary so tests never touch real
models (see voice/stt.py and voice/tts.py's lazy-loading docstrings for why
that's safe), and reset session state between tests.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from agent import session
from agent.main import app


class FakePipeline:
    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.reply_calls: list[tuple[list[dict[str, str]], str]] = []
        monkeypatch.setattr("voice.stt.transcribe", self._transcribe)
        monkeypatch.setattr("agent.llm.reply", self._reply)
        monkeypatch.setattr("voice.tts.synthesize", self._synthesize)

    def _transcribe(self, audio_bytes: bytes) -> tuple[str, str]:
        return "hello world", "en"

    def _reply(self, history: list[dict[str, str]], user_text: str) -> str:
        self.reply_calls.append((list(history), user_text))
        return f"reply to: {user_text}"

    def _synthesize(self, text: str) -> bytes:
        return b"FAKE-WAV-BYTES"


@pytest.fixture
def mock_pipeline(monkeypatch: pytest.MonkeyPatch) -> FakePipeline:
    session._sessions.clear()
    return FakePipeline(monkeypatch)


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)
