"""FastAPI agent loop: the /converse endpoint wires STT -> LLM -> TTS.

Week 1: single-turn STT/LLM/TTS plus per-session conversation history
(Milestone 1.2), no commerce tools yet. The LLM just chats in the
shopping-assistant persona, but now remembers earlier turns in the session.
"""

from __future__ import annotations

import base64
import logging
import time
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Form, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

load_dotenv()

from agent import llm, session
from voice import stt, tts

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("voicecart.agent")

app = FastAPI(title="VoiceCart Agent")

STATIC_DIR = Path(__file__).parent / "static"


class ConverseResponse(BaseModel):
    session_id: str
    transcript: str
    language: str
    reply_text: str
    reply_audio_base64: str


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/converse", response_model=ConverseResponse)
async def converse(audio: UploadFile, session_id: str | None = Form(None)) -> ConverseResponse:
    if session_id is None:
        session_id = session.new_session_id()

    audio_bytes = await audio.read()

    t0 = time.perf_counter()
    transcript, language = stt.transcribe(audio_bytes)
    t1 = time.perf_counter()

    history = session.get_history(session_id)
    reply_text = llm.reply(history, transcript)
    t2 = time.perf_counter()

    session.append(session_id, "user", transcript)
    session.append(session_id, "assistant", reply_text)

    reply_audio = tts.synthesize(reply_text)
    t3 = time.perf_counter()

    # Log transcripts and tool calls, never raw audio or API keys (CLAUDE.md).
    logger.info(
        "session=%s stt_ms=%.0f llm_ms=%.0f tts_ms=%.0f total_ms=%.0f transcript=%r reply_text=%r",
        session_id,
        (t1 - t0) * 1000,
        (t2 - t1) * 1000,
        (t3 - t2) * 1000,
        (t3 - t0) * 1000,
        transcript,
        reply_text,
    )

    return ConverseResponse(
        session_id=session_id,
        transcript=transcript,
        language=language,
        reply_text=reply_text,
        reply_audio_base64=base64.b64encode(reply_audio).decode("ascii"),
    )
