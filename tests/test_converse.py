"""Tests for POST /converse: response shape, session_id handling, and
latency logging. What the agent actually *does* with a turn (tool calls,
clarification, etc.) is agent_loop.py's job — tested in test_agent_loop.py.
Here agent_loop.run_turn is mocked as a black box (see conftest.py).
"""

from __future__ import annotations

import base64
import logging

from fastapi.testclient import TestClient

from tests.conftest import FakePipeline

FAKE_AUDIO = b"fake-audio-bytes"


def _post(client: TestClient, session_id: str | None = None) -> dict:
    data = {"session_id": session_id} if session_id else {}
    response = client.post(
        "/converse",
        files={"audio": ("utterance.webm", FAKE_AUDIO, "audio/webm")},
        data=data,
    )
    assert response.status_code == 200
    return response.json()


def test_converse_returns_expected_shape(client: TestClient, mock_pipeline: FakePipeline) -> None:
    body = _post(client)

    assert body["transcript"] == "hello world"
    assert body["language"] == "en"
    assert body["reply_text"] == "reply to: hello world"
    assert body["session_id"]
    assert base64.b64decode(body["reply_audio_base64"]) == b"FAKE-WAV-BYTES"


def test_new_session_id_minted_when_absent(client: TestClient, mock_pipeline: FakePipeline) -> None:
    first = _post(client)
    second = _post(client)

    assert first["session_id"] != second["session_id"]


def test_run_turn_called_with_session_id_and_transcript(client: TestClient, mock_pipeline: FakePipeline) -> None:
    body = _post(client, session_id="my-session")
    _post(client, session_id="my-session")

    assert mock_pipeline.run_turn_calls == [
        ("my-session", "hello world"),
        ("my-session", "hello world"),
    ]
    assert body["session_id"] == "my-session"


def test_latency_logged_without_leaking_audio(
    client: TestClient, mock_pipeline: FakePipeline, caplog
) -> None:
    with caplog.at_level(logging.INFO, logger="voicecart.agent"):
        _post(client)

    [record] = [r for r in caplog.records if r.name == "voicecart.agent"]
    message = record.getMessage()

    for field in ("stt_ms=", "agent_ms=", "tts_ms=", "total_ms=", "hello world"):
        assert field in message
    assert FAKE_AUDIO.decode() not in message
    assert "FAKE-WAV-BYTES" not in message
