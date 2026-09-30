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

# Hinglish handling. Left to auto-detect, Groq's Whisper wrote Hinglish in
# Devanagari ("नो प्लीज चकाओके" for "no please, checkout karo"), and the
# agent then acted on a misreading. The local backend's fix (retry with
# English when detection is unsure) needs a language probability, which
# this API doesn't return. Measured on Hindi-voiced test audio instead:
#   - auto-detect: Devanagari every time
#   - prompt only: romanized 4/5
#   - language="en" only: *translates*, sometimes wrongly
#     ("bread cart se hata do" -> "Remove the bread cut")
#   - prompt + language="en": romanized and faithful 5/5, plain English
#     unaffected
# Whisper imitates the style and spelling of its prompt, so this is written
# as romanized Hinglish and names the catalog's brands (demo/seed_shopify.py).
LANGUAGE = "en"
PROMPT = (
    "Do packet Amul doodh add karo. Nandini wala daal do. Bread cart se hata do. "
    "Ek kilo basmati rice chahiye, checkout karo. Mother Dairy, Britannia, Modern, "
    "Tata Salt, Fortune, Aashirvaad atta, India Gate, toor dal, moong dal, Maggi, "
    "Lay's, Parle-G, Bru, Tata Tea, Coca-Cola, Colgate, Vim."
)

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
        data={
            "model": model,
            "response_format": "verbose_json",
            "temperature": "0",
            "language": LANGUAGE,
            "prompt": PROMPT,
        },
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
