"""Minimal FastAPI app for the deployed, text-mode API — no STT/TTS.

Week 4, Milestone 4.2: the free-tier EC2 deploy target. This is a separate
entrypoint from agent/main.py (not a flag on the same one) specifically so
this module never imports voice.stt or voice.tts — faster-whisper and
piper-tts (and their model weights) stay entirely out of this process and
out of the deployed image. Voice runs locally in demos; this proves the
deployment/CI/CD story without the RAM cost of loading Whisper on a
free-tier instance.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

load_dotenv()

from agent import agent_loop, session
from agent.mcp_client import MCPClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("voicecart.agent")

mcp_client = MCPClient()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await mcp_client.connect()
    yield
    await mcp_client.close()


app = FastAPI(title="VoiceCart Agent (text mode)", lifespan=lifespan)


class ConverseTextRequest(BaseModel):
    text: str
    session_id: str | None = None


class ConverseTextResponse(BaseModel):
    session_id: str
    reply_text: str


@app.post("/converse/text", response_model=ConverseTextResponse)
async def converse_text(body: ConverseTextRequest) -> ConverseTextResponse:
    session_id = body.session_id or session.new_session_id()
    reply_text = await agent_loop.run_turn(session_id, body.text, mcp_client)
    logger.info("session=%s reply_text=%r", session_id, reply_text)
    return ConverseTextResponse(session_id=session_id, reply_text=reply_text)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
