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


def _get_model() -> WhisperModel:
    global _model
    if _model is None:
        _model = WhisperModel("small", device="cpu", compute_type="int8")
    return _model


def transcribe(audio_bytes: bytes) -> tuple[str, str]:
    """Transcribe audio bytes (any ffmpeg/PyAV-decodable format) to text.

    Returns (text, detected_language).
    """
    segments, info = _get_model().transcribe(io.BytesIO(audio_bytes))
    text = " ".join(segment.text.strip() for segment in segments)
    return text, info.language
