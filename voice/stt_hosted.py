"""Hosted speech-to-text over an OpenAI-compatible transcription API.

Exists for the free Render deploy (see render.yaml): that box has 0.1 CPU
and 512 MB RAM, which can't run faster-whisper locally at any usable speed.
Instead the audio is sent to a hosted Whisper model (Groq's
`whisper-large-v3` by default) and the text comes back — the same pattern
agent/llm.py already uses for the LLM. Local development keeps using
on-device faster-whisper; voice/stt.py picks between the two via the
STT_BACKEND env var.

Provider config comes from env vars, like agent/llm.py. STT_API_KEY and
STT_BASE_URL fall back to the LLM_* values, so a Groq setup needs no new
variables at all.
"""

from __future__ import annotations

import os

import requests

DEFAULT_MODEL = "whisper-large-v3"
# The full large-v3, not the faster "-turbo" variant: this deploy is the one
# people try Hinglish on, and turbo trades some multilingual accuracy for
# speed that Groq's hardware already provides.

NO_SPEECH_PROB_THRESHOLD = 0.6  # same meaning as in voice/stt.py

# verbose_json reports the detected language as a full name ("english"),
# while the local backend and the rest of the app use ISO codes ("en").
_LANGUAGE_CODES = {"english": "en", "hindi": "hi"}


def _filename_for(audio_bytes: bytes) -> str:
    """The API infers the audio format from the upload's file extension.
    Callers pass WAV (the streaming path, see agent/ws_stream.py) or WebM
    (the record-button page), which are easy to tell apart by header."""
    if audio_bytes[:4] == b"RIFF":
        return "audio.wav"
    return "audio.webm"


def transcribe(audio_bytes: bytes) -> tuple[str, str]:
    """Same contract as voice.stt.transcribe: returns (text, language)."""
    api_key = os.environ.get("STT_API_KEY") or os.environ["LLM_API_KEY"]
    base_url = os.environ.get("STT_BASE_URL") or os.environ["LLM_BASE_URL"]
    model = os.environ.get("STT_MODEL") or DEFAULT_MODEL

    response = requests.post(
        f"{base_url}/audio/transcriptions",
        headers={"Authorization": f"Bearer {api_key}"},
        files={"file": (_filename_for(audio_bytes), audio_bytes)},
        data={"model": model, "response_format": "verbose_json", "temperature": "0"},
        timeout=30,
    )
    response.raise_for_status()
    body = response.json()

    segments = body.get("segments")
    if segments is None:
        text = body.get("text", "").strip()
    else:
        # Same noise filter as the local backend: drop segments Whisper
        # itself flags as probably not speech.
        text = " ".join(
            segment["text"].strip()
            for segment in segments
            if segment.get("no_speech_prob", 0.0) < NO_SPEECH_PROB_THRESHOLD
        )

    language = str(body.get("language", "")).lower()
    return text, _LANGUAGE_CODES.get(language, language)
