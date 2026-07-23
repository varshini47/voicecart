"""Tests for POST /converse: response shape, session continuity/isolation,
and latency logging — mocking STT/LLM/TTS throughout (see conftest.py).
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


def test_session_history_threaded_into_llm(client: TestClient, mock_pipeline: FakePipeline) -> None:
    first = _post(client)
    session_id = first["session_id"]
    _post(client, session_id=session_id)

    assert len(mock_pipeline.reply_calls) == 2
    first_history, _ = mock_pipeline.reply_calls[0]
    second_history, _ = mock_pipeline.reply_calls[1]

    assert first_history == []
    assert second_history == [
        {"role": "user", "content": "hello world"},
        {"role": "assistant", "content": "reply to: hello world"},
    ]


def test_separate_sessions_dont_share_history(client: TestClient, mock_pipeline: FakePipeline) -> None:
    _post(client, session_id="session-a")
    _post(client, session_id="session-b")

    assert len(mock_pipeline.reply_calls) == 2
    for history, _ in mock_pipeline.reply_calls:
        assert history == []


def test_latency_logged_without_leaking_audio(
    client: TestClient, mock_pipeline: FakePipeline, caplog
) -> None:
    with caplog.at_level(logging.INFO, logger="voicecart.agent"):
        _post(client)

    [record] = [r for r in caplog.records if r.name == "voicecart.agent"]
    message = record.getMessage()

    for field in ("stt_ms=", "llm_ms=", "tts_ms=", "total_ms=", "hello world"):
        assert field in message
    assert FAKE_AUDIO.decode() not in message
    assert "FAKE-WAV-BYTES" not in message
