"""FastAPI agent loop: the /converse endpoint wires STT -> LLM -> TTS.

Week 1 skeleton: single-turn only, no tools, no session memory yet (that's
Milestone 1.2). The LLM just chats in the shopping-assistant persona.
"""

from __future__ import annotations

import base64
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

load_dotenv()

from agent import llm
from voice import stt, tts

app = FastAPI(title="VoiceCart Agent")

STATIC_DIR = Path(__file__).parent / "static"


class ConverseResponse(BaseModel):
    transcript: str
    language: str
    reply_text: str
    reply_audio_base64: str


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/converse", response_model=ConverseResponse)
async def converse(audio: UploadFile) -> ConverseResponse:
    audio_bytes = await audio.read()

    transcript, language = stt.transcribe(audio_bytes)
    reply_text = llm.reply(transcript)
    reply_audio = tts.synthesize(reply_text)

    return ConverseResponse(
        transcript=transcript,
        language=language,
        reply_text=reply_text,
        reply_audio_base64=base64.b64encode(reply_audio).decode("ascii"),
    )
