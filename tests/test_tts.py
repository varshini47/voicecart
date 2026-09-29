"""Tests for voice.tts's deploy tuning: onnxruntime session options (thread
cap, memory arena off), reply chunking (bounds Piper's peak memory), the
chunk-joining WAV writer, and the TTS_PRELOAD startup hook in
agent/main.py. Piper and onnxruntime are faked, so no model file is needed
(CI has none).
"""

from __future__ import annotations

import io
import wave
from dataclasses import dataclass

import pytest
from fastapi.testclient import TestClient

from agent import main
from voice import tts


class FakeInferenceSession:
    def __init__(self, path: str, sess_options, providers: list[str]) -> None:
        self.options = sess_options
        self.providers = providers


@pytest.fixture
def fake_onnx(monkeypatch: pytest.MonkeyPatch, tmp_path) -> list[FakeInferenceSession]:
    sessions: list[FakeInferenceSession] = []

    def make_session(*args, **kwargs) -> FakeInferenceSession:
        session = FakeInferenceSession(*args, **kwargs)
        sessions.append(session)
        return session

    config = tmp_path / "voice.onnx.json"
    config.write_text("{}")
    monkeypatch.setattr(tts, "_voice", None)
    monkeypatch.setattr(tts, "MODEL", config)  # any existing file
    monkeypatch.setattr(tts, "CONFIG", config)
    monkeypatch.setattr(tts.PiperConfig, "from_dict", staticmethod(lambda d: "fake-config"))
    monkeypatch.setattr(tts.onnxruntime, "InferenceSession", make_session)
    return sessions


def test_memory_arena_off_and_default_threads(monkeypatch, fake_onnx) -> None:
    monkeypatch.delenv("TTS_NUM_THREADS", raising=False)

    options = tts._get_voice().session.options

    assert options.enable_cpu_mem_arena is False
    assert options.enable_mem_pattern is False
    assert options.intra_op_num_threads == 0  # 0 = onnxruntime picks


def test_thread_cap_applied_to_session(monkeypatch, fake_onnx) -> None:
    monkeypatch.setenv("TTS_NUM_THREADS", "1")

    session = tts._get_voice().session

    assert session.options.intra_op_num_threads == 1
    assert session.options.inter_op_num_threads == 1
    assert session.providers == ["CPUExecutionProvider"]


def test_model_loaded_once_and_reused(monkeypatch, fake_onnx) -> None:
    monkeypatch.delenv("TTS_NUM_THREADS", raising=False)

    tts.preload()
    tts._get_voice()

    assert len(fake_onnx) == 1


def test_short_reply_is_one_chunk() -> None:
    assert tts._chunks("Amul, Nandini, or Mother Dairy milk?") == ["Amul, Nandini, or Mother Dairy milk?"]


def test_sentences_become_separate_chunks() -> None:
    assert tts._chunks("Added two packets. Anything else?") == ["Added two packets.", "Anything else?"]


def test_long_sentence_split_at_clause_breaks(monkeypatch) -> None:
    text = "one two three, four five six, seven eight nine"

    assert tts._chunks(text, max_words=6) == ["one two three, four five six,", "seven eight nine"]


def test_overlong_clause_split_by_word_count() -> None:
    words = [f"w{i}" for i in range(45)]

    chunks = tts._chunks(" ".join(words), max_words=20)

    assert [len(c.split()) for c in chunks] == [20, 20, 5]
    assert " ".join(chunks).split() == words


def test_no_chunk_exceeds_limit_and_no_words_lost() -> None:
    text = (
        "Your cart has two packets of Amul Toned Milk at thirty dollars each, one bag of India Gate "
        "Basmati Rice at one hundred and ten dollars, a dozen eggs at eighty dollars — and two packets "
        "of chips at twenty dollars each for a total of two hundred and ninety dollars. Check out now?"
    )

    chunks = tts._chunks(text)

    assert all(len(c.split()) <= tts.MAX_CHUNK_WORDS for c in chunks)
    assert " ".join(chunks).split() == text.split()


@dataclass
class FakeConfig:
    sample_rate: int = 22050


class FakeVoice:
    """Writes two frames per word, so the output length shows which chunks
    were spoken."""

    config = FakeConfig()

    def __init__(self) -> None:
        self.spoken: list[str] = []

    def synthesize_wav(self, text: str, wav_file, set_wav_format: bool = True) -> None:
        assert set_wav_format is False  # format is set once, up front
        self.spoken.append(text)
        wav_file.writeframes(b"\x00\x00" * 2 * len(text.split()))


def test_synthesize_joins_chunks_into_one_wav(monkeypatch) -> None:
    voice = FakeVoice()
    monkeypatch.setattr(tts, "_voice", voice)

    audio = tts.synthesize("Added two **packets**. Anything else?")

    assert voice.spoken == ["Added two packets.", "Anything else?"]
    with wave.open(io.BytesIO(audio)) as wav:
        assert wav.getframerate() == 22050
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getnframes() == 2 * 5


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
