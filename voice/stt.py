"""Speech-to-text wrapper around faster-whisper.

Loads the `small` model lazily on first use (CPU, int8), not at import time
— importing this module (e.g. via agent/main.py) shouldn't pay a multi-second
model-load cost, especially in tests where `transcribe` is mocked outright
and the model should never load at all. Language auto-detect stays on so
Hinglish (mixed Hindi/English) input works without a hint, per CLAUDE.md.
"""

from __future__ import annotations

import io

from faster_whisper import WhisperModel

_model: WhisperModel | None = None

NO_SPEECH_PROB_THRESHOLD = 0.6
# faster-whisper's per-segment estimate of "this segment isn't speech at
# all". voice/vad.py's MIN_SPEECH_FRAMES already filters short noise blips
# by duration, but sustained non-speech noise (a cough, a mic bump,
# background chatter) can pass that frame-level check while still being
# acoustically not-speech — Whisper then hallucinates plausible-looking
# text on it (a stray word, or oddly, phonetic-looking gibberish). Dropping
# high-no_speech_prob segments catches that case using Whisper's own
# confidence signal instead of guessing from duration alone.

LOW_LANGUAGE_CONFIDENCE_THRESHOLD = 0.6
# info.language_probability below this means language auto-detect itself
# wasn't confident. Auto-detect stays on (needed for Hinglish, per
# CLAUDE.md), but it's least reliable on exactly the short/unclear
# utterances where it matters most — seen live: a short English utterance
# misdetected as Hindi, transcribed entirely in Devanagari script, which
# then can't match anything in the (English) product catalog. Retried once
# with English forced only when detection was already unsure — a confident
# non-English detection is left alone, so genuine Hindi/other-language
# speech isn't second-guessed.


def _get_model() -> WhisperModel:
    global _model
    if _model is None:
        _model = WhisperModel("small", device="cpu", compute_type="int8")
    return _model


def _transcribe_once(audio_bytes: bytes, language: str | None = None):
    segments, info = _get_model().transcribe(io.BytesIO(audio_bytes), language=language)
    text = " ".join(
        segment.text.strip() for segment in segments if segment.no_speech_prob < NO_SPEECH_PROB_THRESHOLD
    )
    return text, info


def transcribe(audio_bytes: bytes) -> tuple[str, str]:
    """Transcribe audio bytes (any ffmpeg/PyAV-decodable format) to text.

    Returns (text, detected_language). A segment Whisper itself flags as
    likely non-speech is dropped rather than included in the transcript.
    """
    text, info = _transcribe_once(audio_bytes)

    if info.language != "en" and info.language_probability < LOW_LANGUAGE_CONFIDENCE_THRESHOLD:
        text, info = _transcribe_once(audio_bytes, language="en")
        return text, "en"

    return text, info.language
