"""Tests for the DEMO_PASSCODE gate (agent/access.py) on both voice
endpoints, plus /health, which Render's health check uses and must stay
open. With no passcode configured (local dev), nothing is gated — that
path is what the rest of the test suite already exercises.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from agent import access
from tests.conftest import FakePipeline
from tests.test_ws_stream import FRAME, _use_fake_endpointer

PASSCODE = "open-sesame"


@pytest.fixture
def gated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEMO_PASSCODE", PASSCODE)


def test_passcode_ok_when_none_configured(monkeypatch) -> None:
    monkeypatch.delenv("DEMO_PASSCODE", raising=False)

    assert access.passcode_ok(None)
    assert access.passcode_ok("anything")


def test_passcode_ok_checks_value_when_configured(monkeypatch) -> None:
    monkeypatch.setenv("DEMO_PASSCODE", PASSCODE)

    assert access.passcode_ok(PASSCODE)
    assert not access.passcode_ok("wrong")
    assert not access.passcode_ok("")
    assert not access.passcode_ok(None)


def _post(client: TestClient, passcode: str | None) -> int:
    data = {"passcode": passcode} if passcode is not None else {}
    response = client.post(
        "/converse", files={"audio": ("utterance.webm", b"fake-audio", "audio/webm")}, data=data
    )
    return response.status_code


def test_converse_rejects_missing_or_wrong_passcode(
    client: TestClient, mock_pipeline: FakePipeline, gated
) -> None:
    assert _post(client, None) == 401
    assert _post(client, "wrong") == 401
    assert mock_pipeline.run_turn_calls == []


def test_converse_accepts_correct_passcode(client: TestClient, mock_pipeline: FakePipeline, gated) -> None:
    assert _post(client, PASSCODE) == 200
    assert len(mock_pipeline.run_turn_calls) == 1


def test_stream_rejects_wrong_passcode_before_any_audio(
    client: TestClient, mock_pipeline: FakePipeline, gated, monkeypatch
) -> None:
    _use_fake_endpointer(monkeypatch, frames_per_utterance=1)

    with client.websocket_connect("/converse/stream") as ws:
        ws.send_text('{"type": "auth", "passcode": "wrong"}')
        error = ws.receive_json()
        assert error == {"type": "error", "detail": "Wrong or missing passcode."}
        with pytest.raises(WebSocketDisconnect) as closed:
            ws.receive_json()
        assert closed.value.code == 1008

    assert mock_pipeline.run_turn_calls == []


def test_stream_rejects_audio_sent_instead_of_auth(
    client: TestClient, mock_pipeline: FakePipeline, gated, monkeypatch
) -> None:
    # A client that skips the auth message and streams audio straight away
    # must not get a turn processed.
    _use_fake_endpointer(monkeypatch, frames_per_utterance=1)

    with client.websocket_connect("/converse/stream") as ws:
        ws.send_bytes(FRAME)
        assert ws.receive_json()["type"] == "error"

    assert mock_pipeline.run_turn_calls == []


def test_stream_accepts_correct_passcode(
    client: TestClient, mock_pipeline: FakePipeline, gated, monkeypatch
) -> None:
    _use_fake_endpointer(monkeypatch, frames_per_utterance=1)

    with client.websocket_connect("/converse/stream") as ws:
        ws.send_text(f'{{"type": "auth", "passcode": "{PASSCODE}"}}')
        assert ws.receive_json()["type"] == "ready"

        ws.send_bytes(FRAME)
        assert ws.receive_json()["type"] == "turn"

    assert len(mock_pipeline.run_turn_calls) == 1


def test_auth_message_is_harmless_when_no_passcode_configured(
    client: TestClient, mock_pipeline: FakePipeline, monkeypatch
) -> None:
    # The page always sends the auth message; locally, with no passcode
    # set, the stream must just ignore it and work as before.
    _use_fake_endpointer(monkeypatch, frames_per_utterance=1)

    with client.websocket_connect("/converse/stream") as ws:
        assert ws.receive_json()["type"] == "ready"
        ws.send_text('{"type": "auth", "passcode": ""}')
        ws.send_bytes(FRAME)
        assert ws.receive_json()["type"] == "turn"


def test_health_is_open_even_when_gated(client: TestClient, gated) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
