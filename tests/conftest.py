"""Shared fixtures: mock the STT/agent/TTS boundary so tests never touch real
models, the live LLM, or the real MCP subprocess — see voice/stt.py and
voice/tts.py's lazy-loading docstrings, and agent/agent_loop.py for what
"the agent" means since Milestone 2.2.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from agent import main, session


class FakePipeline:
    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.run_turn_calls: list[tuple[str, str]] = []
        monkeypatch.setattr("voice.stt.transcribe", self._transcribe)
        monkeypatch.setattr("agent.main.agent_loop.run_turn", self._run_turn)
        monkeypatch.setattr("voice.tts.synthesize", self._synthesize)

    def _transcribe(self, audio_bytes: bytes) -> tuple[str, str]:
        return "hello world", "en"

    async def _run_turn(self, session_id: str, user_text: str, mcp_client) -> str:
        self.run_turn_calls.append((session_id, user_text))
        return f"reply to: {user_text}"

    def _synthesize(self, text: str) -> bytes:
        return b"FAKE-WAV-BYTES"


@pytest.fixture
def mock_pipeline(monkeypatch: pytest.MonkeyPatch) -> FakePipeline:
    session._sessions.clear()
    return FakePipeline(monkeypatch)


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    # The real lifespan spawns the mcp_commerce subprocess and needs live
    # Shopify credentials; tests mock agent_loop.run_turn entirely, so the
    # MCP connection is never used and startup/shutdown can be no-ops.
    async def noop(*args, **kwargs) -> None:
        return None

    monkeypatch.setattr(main.mcp_client, "connect", noop)
    monkeypatch.setattr(main.mcp_client, "close", noop)
    with TestClient(main.app) as test_client:
        yield test_client


@pytest.fixture
def text_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    # Same reasoning as `client`, but for the voice-free deploy entrypoint
    # (agent/main_text.py) — its own separate MCPClient instance, same noop
    # lifespan trick.
    from agent import main_text

    async def noop(*args, **kwargs) -> None:
        return None

    monkeypatch.setattr(main_text.mcp_client, "connect", noop)
    monkeypatch.setattr(main_text.mcp_client, "close", noop)
    with TestClient(main_text.app) as test_client:
        yield test_client
