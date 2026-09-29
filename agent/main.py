"""FastAPI agent loop: the /converse endpoint wires STT -> agent -> TTS.

Week 2, Milestone 2.2: the agent is now tool-calling (agent/agent_loop.py),
backed by the mcp_commerce MCP server (Milestone 2.1) instead of the Week 1
echo persona. Session history (Milestone 1.2) now holds full tool-calling
message dicts, not just plain text turns.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Form, HTTPException, UploadFile, WebSocket
from fastapi.responses import FileResponse
from pydantic import BaseModel

load_dotenv()

from agent import access, agent_loop, session
from agent.mcp_client import MCPClient
from agent.ws_stream import handle_stream
from voice import stt, tts

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("voicecart.agent")

mcp_client = MCPClient()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await mcp_client.connect()
    if os.environ.get("TTS_PRELOAD"):
        # Off by default (local dev and tests load lazily, and CI has no
        # voice model on disk); the Render image turns it on — see
        # voice/tts.py's preload().
        await asyncio.to_thread(tts.preload)
    yield
    await mcp_client.close()


app = FastAPI(title="VoiceCart Agent", lifespan=lifespan)

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


@app.get("/stream")
def stream_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "stream.html")


AUTH_TIMEOUT_SECONDS = 10


@app.websocket("/converse/stream")
async def converse_stream(websocket: WebSocket, session_id: str | None = None) -> None:
    await websocket.accept()
    if access.passcode_required():
        # The passcode arrives as the first message ({"type": "auth", ...}),
        # not as a URL query param: uvicorn's access log records full URLs,
        # and secrets must never be logged (CLAUDE.md). The page always
        # sends this message; when no passcode is configured, handle_stream
        # just ignores it like any other unknown message type.
        try:
            message = await asyncio.wait_for(websocket.receive(), timeout=AUTH_TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            message = {"type": "websocket.receive"}
        if message["type"] == "websocket.disconnect":
            return
        try:
            passcode = json.loads(message.get("text") or "{}").get("passcode")
        except (ValueError, AttributeError):
            passcode = None
        if not access.passcode_ok(passcode):
            await websocket.send_json({"type": "error", "detail": "Wrong or missing passcode."})
            await websocket.close(code=1008)  # 1008 = policy violation
            return
    await handle_stream(websocket, mcp_client, session_id)


@app.post("/converse", response_model=ConverseResponse)
async def converse(
    audio: UploadFile, session_id: str | None = Form(None), passcode: str | None = Form(None)
) -> ConverseResponse:
    if not access.passcode_ok(passcode):
        raise HTTPException(status_code=401, detail="Wrong or missing passcode.")
    if session_id is None:
        session_id = session.new_session_id()

    audio_bytes = await audio.read()

    t0 = time.perf_counter()
    transcript, language = stt.transcribe(audio_bytes)
    t1 = time.perf_counter()

    reply_text = await agent_loop.run_turn(session_id, transcript, mcp_client)
    t2 = time.perf_counter()

    reply_audio = tts.synthesize(reply_text)
    t3 = time.perf_counter()

    # Log transcripts and tool calls, never raw audio or API keys (CLAUDE.md).
    logger.info(
        "session=%s stt_ms=%.0f agent_ms=%.0f tts_ms=%.0f total_ms=%.0f transcript=%r reply_text=%r",
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


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
