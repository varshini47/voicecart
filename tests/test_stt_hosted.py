"""Tests for voice.stt_hosted (the hosted Whisper API backend used by the
Render deploy) and voice.stt's STT_BACKEND switch. requests.post is faked,
so no audio ever leaves the machine and no quota is spent.
"""

from __future__ import annotations

import pytest
import requests

from voice import stt, stt_hosted

WAV_AUDIO = b"RIFF\x00\x00\x00\x00WAVEfmt "
WEBM_AUDIO = b"\x1a\x45\xdf\xa3fake-webm"


class FakeResponse:
    def __init__(self, body: dict, status_code: int = 200) -> None:
        self._body = body
        self.status_code = status_code

    def json(self) -> dict:
        return self._body

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error")


class FakePost:
    """Records each call's arguments and returns a canned response."""

    def __init__(self, response: FakeResponse) -> None:
        self.response = response
        self.calls: list[dict] = []

    def __call__(self, url: str, **kwargs) -> FakeResponse:
        self.calls.append({"url": url, **kwargs})
        return self.response


@pytest.fixture
def groq_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_API_KEY", "llm-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://api.example.com/v1")
    for name in ("STT_API_KEY", "STT_BASE_URL", "STT_MODEL"):
        monkeypatch.delenv(name, raising=False)


def _fake_post(monkeypatch: pytest.MonkeyPatch, body: dict, status_code: int = 200) -> FakePost:
    fake = FakePost(FakeResponse(body, status_code))
    monkeypatch.setattr(stt_hosted.requests, "post", fake)
    return fake


def test_sends_audio_to_transcription_endpoint_with_llm_credentials(monkeypatch, groq_env) -> None:
    fake = _fake_post(monkeypatch, {"text": "add milk", "language": "english"})

    stt_hosted.transcribe(WAV_AUDIO)

    [call] = fake.calls
    assert call["url"] == "https://api.example.com/v1/audio/transcriptions"
    assert call["headers"] == {"Authorization": "Bearer llm-key"}
    assert call["data"]["model"] == stt_hosted.DEFAULT_MODEL
    assert call["data"]["response_format"] == "verbose_json"
    assert call["files"]["file"] == ("audio.wav", WAV_AUDIO)


def test_stt_specific_env_vars_override_llm_ones(monkeypatch, groq_env) -> None:
    monkeypatch.setenv("STT_API_KEY", "stt-key")
    monkeypatch.setenv("STT_BASE_URL", "https://stt.example.com/v1")
    monkeypatch.setenv("STT_MODEL", "whisper-large-v3-turbo")
    fake = _fake_post(monkeypatch, {"text": "add milk", "language": "english"})

    stt_hosted.transcribe(WAV_AUDIO)

    [call] = fake.calls
    assert call["url"] == "https://stt.example.com/v1/audio/transcriptions"
    assert call["headers"] == {"Authorization": "Bearer stt-key"}
    assert call["data"]["model"] == "whisper-large-v3-turbo"


def test_webm_audio_uploaded_with_webm_filename(monkeypatch, groq_env) -> None:
    # The API infers the format from the extension; the record-button page
    # sends WebM, the streaming page sends WAV.
    fake = _fake_post(monkeypatch, {"text": "add milk", "language": "english"})

    stt_hosted.transcribe(WEBM_AUDIO)

    assert fake.calls[0]["files"]["file"][0] == "audio.webm"


def test_full_language_name_mapped_to_iso_code(monkeypatch, groq_env) -> None:
    _fake_post(monkeypatch, {"text": "do packet doodh add karo", "language": "hindi"})

    text, language = stt_hosted.transcribe(WAV_AUDIO)

    assert text == "do packet doodh add karo"
    assert language == "hi"


def test_drops_high_no_speech_prob_segments(monkeypatch, groq_env) -> None:
    _fake_post(
        monkeypatch,
        {
            "text": "add milk thank you",
            "language": "english",
            "segments": [
                {"text": " add milk", "no_speech_prob": 0.1},
                {"text": " thank you", "no_speech_prob": 0.9},
            ],
        },
    )

    text, language = stt_hosted.transcribe(WAV_AUDIO)

    assert text == "add milk"
    assert language == "en"


def test_http_error_is_raised_for_the_caller_to_handle(monkeypatch, groq_env) -> None:
    # agent/ws_stream.py turns any failed turn into a plain-language error
    # message, so this module just needs to raise, not swallow it.
    _fake_post(monkeypatch, {"error": "rate limited"}, status_code=429)

    with pytest.raises(requests.HTTPError):
        stt_hosted.transcribe(WAV_AUDIO)


def test_stt_backend_hosted_routes_to_hosted_module(monkeypatch) -> None:
    monkeypatch.setenv("STT_BACKEND", "hosted")
    monkeypatch.setattr(stt_hosted, "transcribe", lambda audio: ("from hosted", "en"))

    def fail_if_called():
        raise AssertionError("local model must not load when STT_BACKEND=hosted")

    monkeypatch.setattr(stt, "_get_model", fail_if_called)

    assert stt.transcribe(b"audio") == ("from hosted", "en")


def test_unknown_stt_backend_is_rejected(monkeypatch) -> None:
    monkeypatch.setenv("STT_BACKEND", "typo")

    with pytest.raises(ValueError, match="STT_BACKEND"):
        stt.transcribe(b"audio")
