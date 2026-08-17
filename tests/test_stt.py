"""Tests for voice.stt.transcribe's no_speech_prob filtering. A fake model
stands in for WhisperModel so this exercises just the filtering logic, not
the real model (see conftest.py's docstring on why tests never load it).
"""

from __future__ import annotations

from dataclasses import dataclass

import voice.stt as stt


@dataclass
class FakeSegment:
    text: str
    no_speech_prob: float


class FakeInfo:
    language = "en"


class FakeModel:
    def __init__(self, segments: list[FakeSegment]) -> None:
        self._segments = segments

    def transcribe(self, audio):
        return iter(self._segments), FakeInfo()


def _use_fake_model(monkeypatch, segments: list[FakeSegment]) -> None:
    monkeypatch.setattr(stt, "_get_model", lambda: FakeModel(segments))


def test_transcribe_joins_low_no_speech_prob_segments(monkeypatch) -> None:
    _use_fake_model(
        monkeypatch,
        [FakeSegment("hello", 0.1), FakeSegment("world", 0.2)],
    )

    text, language = stt.transcribe(b"fake-audio")

    assert text == "hello world"
    assert language == "en"


def test_transcribe_drops_high_no_speech_prob_segments(monkeypatch) -> None:
    # A noise burst Whisper hallucinates plausible-looking (or gibberish)
    # text on, but flags itself as likely not speech.
    _use_fake_model(monkeypatch, [FakeSegment("ʕ ʕ ʕ ʔ", 0.9)])

    text, _ = stt.transcribe(b"fake-audio")

    assert text == ""


def test_transcribe_keeps_only_confident_segments_in_a_mixed_utterance(monkeypatch) -> None:
    _use_fake_model(
        monkeypatch,
        [
            FakeSegment("add milk", 0.1),
            FakeSegment("thank you", 0.8),
        ],
    )

    text, _ = stt.transcribe(b"fake-audio")

    assert text == "add milk"
