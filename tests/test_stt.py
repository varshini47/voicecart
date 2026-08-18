"""Tests for voice.stt.transcribe's no_speech_prob filtering and low-
confidence language retry. A fake model stands in for WhisperModel so this
exercises just that logic, not the real model (see conftest.py's docstring
on why tests never load it).
"""

from __future__ import annotations

from dataclasses import dataclass

from voice import stt


@dataclass
class FakeSegment:
    text: str
    no_speech_prob: float


@dataclass
class FakeInfo:
    language: str = "en"
    language_probability: float = 0.95


class FakeModel:
    """Returns `calls` in order, one per transcribe() invocation, so tests
    can drive the language-confidence retry path (a second call happens
    only when the first call's info warrants it)."""

    def __init__(self, calls: list[tuple[list[FakeSegment], FakeInfo]]) -> None:
        self._calls = list(calls)
        self.languages_requested: list[str | None] = []

    def transcribe(self, audio, language=None):
        self.languages_requested.append(language)
        segments, info = self._calls.pop(0)
        return iter(segments), info


def _use_fake_model(monkeypatch, calls: list[tuple[list[FakeSegment], FakeInfo]]) -> FakeModel:
    model = FakeModel(calls)
    monkeypatch.setattr(stt, "_get_model", lambda: model)
    return model


def test_transcribe_joins_low_no_speech_prob_segments(monkeypatch) -> None:
    _use_fake_model(
        monkeypatch,
        [([FakeSegment("hello", 0.1), FakeSegment("world", 0.2)], FakeInfo())],
    )

    text, language = stt.transcribe(b"fake-audio")

    assert text == "hello world"
    assert language == "en"


def test_transcribe_drops_high_no_speech_prob_segments(monkeypatch) -> None:
    # A noise burst Whisper hallucinates plausible-looking (or gibberish)
    # text on, but flags itself as likely not speech.
    _use_fake_model(monkeypatch, [([FakeSegment("ʕ ʕ ʕ ʔ", 0.9)], FakeInfo())])

    text, _ = stt.transcribe(b"fake-audio")

    assert text == ""


def test_transcribe_keeps_only_confident_segments_in_a_mixed_utterance(monkeypatch) -> None:
    _use_fake_model(
        monkeypatch,
        [([FakeSegment("add milk", 0.1), FakeSegment("thank you", 0.8)], FakeInfo())],
    )

    text, _ = stt.transcribe(b"fake-audio")

    assert text == "add milk"


def test_low_confidence_non_english_detection_retries_with_english_forced(monkeypatch) -> None:
    # Seen live: a short English utterance misdetected as Hindi at low
    # confidence, transcribed entirely in Devanagari script.
    model = _use_fake_model(
        monkeypatch,
        [
            ([FakeSegment("गलत भाषा", 0.1)], FakeInfo(language="hi", language_probability=0.4)),
            ([FakeSegment("add milk", 0.1)], FakeInfo(language="en", language_probability=0.95)),
        ],
    )

    text, language = stt.transcribe(b"fake-audio")

    assert text == "add milk"
    assert language == "en"
    assert model.languages_requested == [None, "en"]


def test_confident_non_english_detection_is_not_retried(monkeypatch) -> None:
    # Genuine non-English speech, confidently detected, must not be
    # second-guessed just because it isn't English.
    model = _use_fake_model(
        monkeypatch,
        [([FakeSegment("दूध जोड़ें", 0.1)], FakeInfo(language="hi", language_probability=0.9))],
    )

    text, language = stt.transcribe(b"fake-audio")

    assert text == "दूध जोड़ें"
    assert language == "hi"
    assert model.languages_requested == [None]


def test_low_confidence_english_detection_is_not_retried(monkeypatch) -> None:
    # Already English — nothing to gain from forcing English again.
    model = _use_fake_model(
        monkeypatch,
        [([FakeSegment("add milk", 0.1)], FakeInfo(language="en", language_probability=0.3))],
    )

    text, language = stt.transcribe(b"fake-audio")

    assert text == "add milk"
    assert language == "en"
    assert model.languages_requested == [None]
