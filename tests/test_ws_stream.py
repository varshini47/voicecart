"""Tests for the /converse/stream WebSocket endpoint (Milestone 3.1). The
real Endpointer (voice/vad.py) is replaced with a fake that finalizes an
"utterance" after a fixed number of frames, regardless of content — VAD
logic itself is covered by tests/test_vad.py. This just verifies the
WebSocket wiring: frame-size validation, one JSON "turn" message per
finalized utterance, multi-turn over one connection, and clean "stop".
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from tests.conftest import FakePipeline
from voice.vad import FRAME_BYTES

FRAME = b"\x00" * FRAME_BYTES


class FakeEndpointer:
    """Finalizes an utterance every `frames_per_utterance` frames, no VAD involved."""

    def __init__(self, frames_per_utterance: int = 2) -> None:
        self.frames_per_utterance = frames_per_utterance
        self._buffer: list[bytes] = []

    def accept_frame(self, frame: bytes) -> bytes | None:
        self._buffer.append(frame)
        if len(self._buffer) < self.frames_per_utterance:
            return None
        audio = b"".join(self._buffer)
        self._buffer = []
        return audio

    def force_finalize(self) -> bytes | None:
        if not self._buffer:
            return None
        audio = b"".join(self._buffer)
        self._buffer = []
        return audio


def _use_fake_endpointer(monkeypatch, frames_per_utterance: int = 2) -> None:
    monkeypatch.setattr(
        "agent.ws_stream.Endpointer",
        lambda: FakeEndpointer(frames_per_utterance=frames_per_utterance),
    )


def test_ready_message_on_connect(client: TestClient, mock_pipeline: FakePipeline, monkeypatch) -> None:
    _use_fake_endpointer(monkeypatch)

    with client.websocket_connect("/converse/stream") as ws:
        ready = ws.receive_json()
        assert ready["type"] == "ready"
        assert ready["session_id"]


def test_turn_sent_after_endpointer_finalizes(client: TestClient, mock_pipeline: FakePipeline, monkeypatch) -> None:
    _use_fake_endpointer(monkeypatch, frames_per_utterance=2)

    with client.websocket_connect("/converse/stream") as ws:
        ws.receive_json()  # ready

        ws.send_bytes(FRAME)  # not enough frames yet
        ws.send_bytes(FRAME)  # finalizes the utterance

        turn = ws.receive_json()
        assert turn["type"] == "turn"
        assert turn["transcript"] == "hello world"
        assert turn["reply_text"] == "reply to: hello world"
        assert turn["reply_audio_base64"]

    assert mock_pipeline.run_turn_calls == [(turn["session_id"], "hello world")]


def test_multiple_utterances_on_one_connection(client: TestClient, mock_pipeline: FakePipeline, monkeypatch) -> None:
    _use_fake_endpointer(monkeypatch, frames_per_utterance=1)

    with client.websocket_connect("/converse/stream") as ws:
        ws.receive_json()  # ready

        ws.send_bytes(FRAME)
        first = ws.receive_json()
        ws.send_bytes(FRAME)
        second = ws.receive_json()

    assert first["type"] == "turn"
    assert second["type"] == "turn"
    assert len(mock_pipeline.run_turn_calls) == 2
    assert mock_pipeline.run_turn_calls[0][0] == mock_pipeline.run_turn_calls[1][0]


def test_wrong_frame_size_returns_error(client: TestClient, mock_pipeline: FakePipeline, monkeypatch) -> None:
    _use_fake_endpointer(monkeypatch)

    with client.websocket_connect("/converse/stream") as ws:
        ws.receive_json()  # ready

        ws.send_bytes(b"\x00" * (FRAME_BYTES - 1))

        error = ws.receive_json()
        assert error["type"] == "error"

    assert mock_pipeline.run_turn_calls == []


def test_stop_message_closes_cleanly(client: TestClient, mock_pipeline: FakePipeline, monkeypatch) -> None:
    _use_fake_endpointer(monkeypatch)

    with client.websocket_connect("/converse/stream") as ws:
        ws.receive_json()  # ready
        ws.send_text('{"type": "stop"}')
        # Connection should close without the server raising.


def test_failed_turn_sends_error_and_keeps_connection_open(
    client: TestClient, mock_pipeline: FakePipeline, monkeypatch
) -> None:
    _use_fake_endpointer(monkeypatch, frames_per_utterance=1)

    call_count = 0

    async def flaky_run_turn(session_id: str, user_text: str, mcp_client) -> str:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("simulated LLM provider failure")
        return f"reply to: {user_text}"

    monkeypatch.setattr("agent.main.agent_loop.run_turn", flaky_run_turn)

    with client.websocket_connect("/converse/stream") as ws:
        ws.receive_json()  # ready

        ws.send_bytes(FRAME)  # triggers the simulated failure
        error = ws.receive_json()
        assert error["type"] == "error"

        ws.send_bytes(FRAME)  # connection should still be alive for the next turn
        turn = ws.receive_json()
        assert turn["type"] == "turn"
        assert turn["reply_text"] == "reply to: hello world"


def test_finalize_message_forces_turn_before_endpointer_would(
    client: TestClient, mock_pipeline: FakePipeline, monkeypatch
) -> None:
    # frames_per_utterance=5 so accept_frame alone would never finalize from
    # just 2 frames — only the manual "finalize" signal should trigger a turn.
    _use_fake_endpointer(monkeypatch, frames_per_utterance=5)

    with client.websocket_connect("/converse/stream") as ws:
        ws.receive_json()  # ready

        ws.send_bytes(FRAME)
        ws.send_bytes(FRAME)
        ws.send_text('{"type": "finalize"}')

        turn = ws.receive_json()
        assert turn["type"] == "turn"
        assert turn["transcript"] == "hello world"

    assert mock_pipeline.run_turn_calls == [(turn["session_id"], "hello world")]


def test_finalize_with_nothing_buffered_is_a_no_op(
    client: TestClient, mock_pipeline: FakePipeline, monkeypatch
) -> None:
    _use_fake_endpointer(monkeypatch)

    with client.websocket_connect("/converse/stream") as ws:
        ws.receive_json()  # ready

        ws.send_text('{"type": "finalize"}')
        ws.send_text('{"type": "stop"}')
        # No turn or error should be sent — nothing was buffered.

    assert mock_pipeline.run_turn_calls == []


def test_session_id_from_query_param_is_reused(client: TestClient, mock_pipeline: FakePipeline, monkeypatch) -> None:
    _use_fake_endpointer(monkeypatch, frames_per_utterance=1)

    with client.websocket_connect("/converse/stream?session_id=my-session") as ws:
        ready = ws.receive_json()
        assert ready["session_id"] == "my-session"

        ws.send_bytes(FRAME)
        turn = ws.receive_json()
        assert turn["session_id"] == "my-session"

    assert mock_pipeline.run_turn_calls == [("my-session", "hello world")]
