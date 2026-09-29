"""Tests for voice.tts's deploy knobs: the TTS_NUM_THREADS onnxruntime
thread cap and the TTS_PRELOAD startup hook in agent/main.py. Piper and
onnxruntime are faked, so no model file is needed (CI has none).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from agent import main
from voice import tts


class FakeVoice:
    session = "piper-default-session"


class FakeInferenceSession:
    def __init__(self, path: str, sess_options, providers: list[str]) -> None:
        self.path = path
        self.options = sess_options
        self.providers = providers


@pytest.fixture
def fake_piper(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    loads: list[int] = []

    def fake_load(model, config_path=None):
        loads.append(1)
        return FakeVoice()

    monkeypatch.setattr(tts, "_voice", None)
    monkeypatch.setattr(tts.PiperVoice, "load", staticmethod(fake_load))
    monkeypatch.setattr(tts.onnxruntime, "InferenceSession", FakeInferenceSession)
    monkeypatch.setattr(tts, "MODEL", tts.Path(__file__))  # any existing file
    return loads


def test_default_keeps_piper_session(monkeypatch, fake_piper) -> None:
    monkeypatch.delenv("TTS_NUM_THREADS", raising=False)

    voice = tts._get_voice()

    assert voice.session == "piper-default-session"


def test_thread_cap_rebuilds_session_with_limit(monkeypatch, fake_piper) -> None:
    monkeypatch.setenv("TTS_NUM_THREADS", "1")

    session = tts._get_voice().session

    assert isinstance(session, FakeInferenceSession)
    assert session.options.intra_op_num_threads == 1
    assert session.options.inter_op_num_threads == 1
    assert session.providers == ["CPUExecutionProvider"]


def test_voice_loaded_once_and_reused(monkeypatch, fake_piper) -> None:
    monkeypatch.delenv("TTS_NUM_THREADS", raising=False)

    tts.preload()
    tts._get_voice()

    assert fake_piper == [1]


def _start_app(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    preloads: list[int] = []

    async def noop(*args, **kwargs) -> None:
        return None

    monkeypatch.setattr(main.mcp_client, "connect", noop)
    monkeypatch.setattr(main.mcp_client, "close", noop)
    monkeypatch.setattr(main.tts, "preload", lambda: preloads.append(1))
    with TestClient(main.app):
        pass
    return preloads


def test_startup_preloads_voice_when_enabled(monkeypatch) -> None:
    monkeypatch.setenv("TTS_PRELOAD", "1")

    assert _start_app(monkeypatch) == [1]


def test_startup_skips_preload_by_default(monkeypatch) -> None:
    monkeypatch.delenv("TTS_PRELOAD", raising=False)

    assert _start_app(monkeypatch) == []
