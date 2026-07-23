"""Text-to-speech wrapper around Piper (local, free — see CLAUDE.md).

Loads the voice once at import time via Piper's Python API. The original
Week 0 check script (voice/tts_check.py) shelled out to `python -m piper`
per call, which respawns a process and reloads the ONNX model every time —
that was the dominant chunk of /converse latency, so we load it once here
instead.
"""

from __future__ import annotations

import io
import wave
from pathlib import Path

from piper.voice import PiperVoice

MODEL_DIR = Path(__file__).parent / "models"
MODEL = MODEL_DIR / "en_US-lessac-medium.onnx"
CONFIG = MODEL_DIR / "en_US-lessac-medium.onnx.json"

if not MODEL.exists():
    raise FileNotFoundError(
        f"Voice model not found at {MODEL}. Run:\n"
        f"  python -m piper.download_voices --download-dir voice/models en_US-lessac-medium"
    )

_voice = PiperVoice.load(MODEL, config_path=CONFIG)


def synthesize(text: str) -> bytes:
    """Synthesize `text` to WAV bytes using the local Piper voice."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        _voice.synthesize_wav(text, wav_file)
    return buffer.getvalue()
