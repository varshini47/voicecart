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


def _get_model() -> WhisperModel:
    global _model
    if _model is None:
        _model = WhisperModel("small", device="cpu", compute_type="int8")
    return _model


def transcribe(audio_bytes: bytes) -> tuple[str, str]:
    """Transcribe audio bytes (any ffmpeg/PyAV-decodable format) to text.

    Returns (text, detected_language). A segment Whisper itself flags as
    likely non-speech is dropped rather than included in the transcript.
    """
    segments, info = _get_model().transcribe(io.BytesIO(audio_bytes))
    text = " ".join(
        segment.text.strip() for segment in segments if segment.no_speech_prob < NO_SPEECH_PROB_THRESHOLD
    )
    return text, info.language
