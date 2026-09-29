"""Text-to-speech wrapper around Piper (local, free — see CLAUDE.md).

Loads the voice lazily on first use via Piper's Python API, for the same
reason as voice/stt.py: importing this module shouldn't require the ONNX
voice file to exist or pay a load cost, since tests mock `synthesize`
outright. The original Week 0 check script (voice/tts_check.py) shelled out
to `python -m piper` per call, which respawns a process and reloads the
model every time — that was the dominant chunk of /converse latency, so we
load it once (in-process) here instead.

Memory and CPU tuning, both found by testing under Render's free-plan
limits (0.1 CPU / 512 MB, see NOTES.md Milestone 4.5):

- Piper's peak memory grows with the length of the sentence it's speaking
  (~5-6 MB per word; one 80-word sentence alone went past 512 MB), so
  replies are spoken in chunks of at most MAX_CHUNK_WORDS words.
- onnxruntime's memory arena keeps its peak allocation forever, so it's
  turned off: memory is handed back after each chunk.
- TTS_NUM_THREADS caps onnxruntime's thread count. By default it starts
  one thread per host core; a Render container sees 16 cores but gets 0.1
  CPU of quota, so 16 threads fought over a tenth of a core (~26s per
  reply vs ~4.5s with 1 thread). Unset locally = onnxruntime default.
"""

from __future__ import annotations

import io
import json
import os
import re
import wave
from pathlib import Path

import onnxruntime
from piper.config import PiperConfig
from piper.voice import PiperVoice

MODEL_DIR = Path(__file__).parent / "models"
MODEL = MODEL_DIR / "en_US-lessac-medium.onnx"
CONFIG = MODEL_DIR / "en_US-lessac-medium.onnx.json"

MAX_CHUNK_WORDS = 20  # ~230 MB peak for Piper itself, measured

_voice: PiperVoice | None = None

_MARKDOWN_EMPHASIS = re.compile(r"(\*{1,2}|_{1,2})(.+?)\1")
_MARKDOWN_HEADER = re.compile(r"^#{1,6}\s+", re.MULTILINE)
_MARKDOWN_BULLET = re.compile(r"^\s*(?:[-*+]|\d+\.)\s+", re.MULTILINE)

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_CLAUSE_END = re.compile(r"(?<=[,;:—])\s*")


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


def _chunks(text: str, max_words: int = MAX_CHUNK_WORDS) -> list[str]:
    """Split text into pieces of at most `max_words` words, breaking at
    sentence ends first, then at commas/semicolons/colons/dashes (natural
    pauses), and only mid-clause when a single clause is itself too long."""
    chunks: list[str] = []
    for sentence in _SENTENCE_END.split(text.strip()):
        current: list[str] = []
        for clause in _CLAUSE_END.split(sentence):
            words = clause.split()
            if current and len(current) + len(words) > max_words:
                chunks.append(" ".join(current))
                current = []
            while len(words) > max_words:
                chunks.append(" ".join(words[:max_words]))
                words = words[max_words:]
            current.extend(words)
        if current:
            chunks.append(" ".join(current))
    return chunks


def _session_options() -> onnxruntime.SessionOptions:
    options = onnxruntime.SessionOptions()
    options.enable_cpu_mem_arena = False
    options.enable_mem_pattern = False
    threads = os.environ.get("TTS_NUM_THREADS")
    if threads:
        options.intra_op_num_threads = int(threads)
        options.inter_op_num_threads = 1
    return options


def _get_voice() -> PiperVoice:
    global _voice
    if _voice is None:
        if not MODEL.exists():
            raise FileNotFoundError(
                f"Voice model not found at {MODEL}. Run:\n"
                f"  python -m piper.download_voices --download-dir voice/models en_US-lessac-medium"
            )
        # Built directly rather than via PiperVoice.load(), which doesn't
        # accept onnxruntime session options (and loading it that way and
        # then swapping the session meant loading the model twice).
        with open(CONFIG, encoding="utf-8") as config_file:
            config = PiperConfig.from_dict(json.load(config_file))
        session = onnxruntime.InferenceSession(
            str(MODEL), sess_options=_session_options(), providers=["CPUExecutionProvider"]
        )
        _voice = PiperVoice(session=session, config=config)
    return _voice


def preload() -> None:
    """Load the voice now instead of on the first reply. Used by the Render
    deploy (TTS_PRELOAD, see agent/main.py), where loading on 0.1 CPU takes
    long enough that the first visitor would otherwise wait for it."""
    _get_voice()


def synthesize(text: str) -> bytes:
    """Synthesize `text` to WAV bytes using the local Piper voice."""
    voice = _get_voice()
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        # Format set once up front (Piper outputs 16-bit mono), so each
        # chunk's audio is appended to the same WAV.
        wav_file.setframerate(voice.config.sample_rate)
        wav_file.setsampwidth(2)
        wav_file.setnchannels(1)
        for chunk in _chunks(_strip_markdown(text)):
            voice.synthesize_wav(chunk, wav_file, set_wav_format=False)
    return buffer.getvalue()
