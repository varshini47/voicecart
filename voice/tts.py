"""Text-to-speech wrapper around Piper (local, free — see CLAUDE.md).

Loads the voice lazily on first use via Piper's Python API, for the same
reason as voice/stt.py: importing this module shouldn't require the ONNX
voice file to exist or pay a load cost, since tests mock `synthesize`
outright. The original Week 0 check script (voice/tts_check.py) shelled out
to `python -m piper` per call, which respawns a process and reloads the
model every time — that was the dominant chunk of /converse latency, so we
load it once (in-process) here instead.

TTS_NUM_THREADS caps onnxruntime's thread count. By default it starts one
thread per host core, which is fine on a laptop but disastrous on Render's
free plan: the container sees 16 host cores but gets 0.1 CPU of quota, so
16 threads fight over a tenth of a core. Measured there: ~26s per reply
with the default, ~4.5s with 1 thread. Unset locally = onnxruntime default.
"""

from __future__ import annotations

import io
import os
import re
import wave
from pathlib import Path

import onnxruntime
from piper.voice import PiperVoice

MODEL_DIR = Path(__file__).parent / "models"
MODEL = MODEL_DIR / "en_US-lessac-medium.onnx"
CONFIG = MODEL_DIR / "en_US-lessac-medium.onnx.json"

_voice: PiperVoice | None = None

_MARKDOWN_EMPHASIS = re.compile(r"(\*{1,2}|_{1,2})(.+?)\1")
_MARKDOWN_HEADER = re.compile(r"^#{1,6}\s+", re.MULTILINE)
_MARKDOWN_BULLET = re.compile(r"^\s*(?:[-*+]|\d+\.)\s+", re.MULTILINE)


def _strip_markdown(text: str) -> str:
    """Drop markdown syntax before synthesis.

    Piper has no notion of markdown, so a literal "**bold**" from the LLM
    gets read aloud as "asterisk asterisk bold asterisk asterisk". The
    system prompt tells the model not to use markdown at all; this is a
    defensive second layer for whatever slips through anyway.
    """
    text = _MARKDOWN_EMPHASIS.sub(r"\2", text)
    text = _MARKDOWN_HEADER.sub("", text)
    text = _MARKDOWN_BULLET.sub("", text)
    return text


def _get_voice() -> PiperVoice:
    global _voice
    if _voice is None:
        if not MODEL.exists():
            raise FileNotFoundError(
                f"Voice model not found at {MODEL}. Run:\n"
                f"  python -m piper.download_voices --download-dir voice/models en_US-lessac-medium"
            )
        _voice = PiperVoice.load(MODEL, config_path=CONFIG)
        threads = os.environ.get("TTS_NUM_THREADS")
        if threads:
            # PiperVoice.load() doesn't accept onnxruntime session options,
            # so swap in a session built with the thread cap.
            options = onnxruntime.SessionOptions()
            options.intra_op_num_threads = int(threads)
            options.inter_op_num_threads = 1
            _voice.session = onnxruntime.InferenceSession(
                str(MODEL), sess_options=options, providers=["CPUExecutionProvider"]
            )
    return _voice


def preload() -> None:
    """Load the voice now instead of on the first reply. Used by the Render
    deploy (TTS_PRELOAD, see agent/main.py), where loading on 0.1 CPU takes
    long enough that the first visitor would otherwise wait for it."""
    _get_voice()


def synthesize(text: str) -> bytes:
    """Synthesize `text` to WAV bytes using the local Piper voice."""
    text = _strip_markdown(text)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        _get_voice().synthesize_wav(text, wav_file)
    return buffer.getvalue()
