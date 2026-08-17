"""Tests for POST /converse/text and /health — the deployed, voice-free
text-mode API (Week 4, Milestone 4.2). Mirrors test_converse.py's
structure; agent_loop.run_turn is mocked the same way (see conftest.py).
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from tests.conftest import FakePipeline


def test_converse_text_returns_expected_shape(text_client: TestClient, mock_pipeline: FakePipeline) -> None:
    response = text_client.post("/converse/text", json={"text": "add milk to my cart"})

    assert response.status_code == 200
    body = response.json()
    assert body["reply_text"] == "reply to: add milk to my cart"
    assert body["session_id"]


def test_new_session_id_minted_when_absent(text_client: TestClient, mock_pipeline: FakePipeline) -> None:
    first = text_client.post("/converse/text", json={"text": "hi"}).json()
    second = text_client.post("/converse/text", json={"text": "hi"}).json()

    assert first["session_id"] != second["session_id"]


def test_run_turn_called_with_session_id_and_text(text_client: TestClient, mock_pipeline: FakePipeline) -> None:
    text_client.post("/converse/text", json={"text": "add milk", "session_id": "my-session"})
    text_client.post("/converse/text", json={"text": "add milk", "session_id": "my-session"})

    assert mock_pipeline.run_turn_calls == [
        ("my-session", "add milk"),
        ("my-session", "add milk"),
    ]


def test_health_endpoint(text_client: TestClient) -> None:
    response = text_client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
