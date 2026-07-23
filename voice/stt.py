"""Speech-to-text wrapper around faster-whisper.

Loads the `small` model once at import time (CPU, int8) and exposes a single
`transcribe` function. Language auto-detect stays on so Hinglish (mixed
Hindi/English) input works without a hint, per CLAUDE.md.
"""

from __future__ import annotations

import io

from faster_whisper import WhisperModel

_model = WhisperModel("small", device="cpu", compute_type="int8")


def transcribe(audio_bytes: bytes) -> tuple[str, str]:
    """Transcribe audio bytes (any ffmpeg/PyAV-decodable format) to text.

    Returns (text, detected_language).
    """
    segments, info = _model.transcribe(io.BytesIO(audio_bytes))
    text = " ".join(segment.text.strip() for segment in segments)
    return text, info.language
