"""FastAPI agent loop: the /converse endpoint wires STT -> LLM -> TTS.

Week 1: single-turn STT/LLM/TTS plus per-session conversation history
(Milestone 1.2), no commerce tools yet. The LLM just chats in the
shopping-assistant persona, but now remembers earlier turns in the session.
"""

from __future__ import annotations

import base64
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Form, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

load_dotenv()

from agent import llm, session
from voice import stt, tts

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
    transcript, language = stt.transcribe(audio_bytes)

    history = session.get_history(session_id)
    reply_text = llm.reply(history, transcript)
    session.append(session_id, "user", transcript)
    session.append(session_id, "assistant", reply_text)

    reply_audio = tts.synthesize(reply_text)

    return ConverseResponse(
        session_id=session_id,
        transcript=transcript,
        language=language,
        reply_text=reply_text,
        reply_audio_base64=base64.b64encode(reply_audio).decode("ascii"),
    )
