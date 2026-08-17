"""WebSocket streaming pipeline — Milestone 3.1.

Replaces manual click-to-stop recording with continuous mic capture: the
client streams fixed-size raw PCM frames over the WebSocket, voice.vad's
Endpointer runs frame-level VAD to detect utterance boundaries, and each
finalized utterance runs through the same STT -> agent_loop -> TTS pipeline
as POST /converse. One connection can carry many turns (the endpointer
resets after each finalized utterance and keeps listening).

The old POST /converse endpoint is left as-is — it's simple, already has
test coverage, and is still the easier path for one-shot/non-browser
callers (e.g. the eval runner's style of testing). This is an additional
transport, not a replacement.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from dataclasses import dataclass

from fastapi import WebSocket, WebSocketDisconnect

from agent import agent_loop, session
from agent.mcp_client import MCPClient
from voice import stt, tts
from voice.pcm import pcm16_to_wav
from voice.vad import FRAME_BYTES, Endpointer

logger = logging.getLogger("voicecart.agent")


@dataclass
class _TurnState:
    """Holds the currently-running turn's task so a "barge_in" message (see
    handle_stream) can cancel it from a different coroutine. Plain mutable
    state, not a queue — there's only ever at most one turn running."""

    current_task: asyncio.Task | None = None


async def handle_stream(websocket: WebSocket, mcp_client: MCPClient, session_id: str | None) -> None:
    await websocket.accept()
    if session_id is None:
        session_id = session.new_session_id()
    await websocket.send_json({"type": "ready", "session_id": session_id})

    endpointer = Endpointer()
    was_speech_confirmed = False

    # A turn (STT -> LLM -> TTS) can take several seconds. Processing it
    # inline here would block this loop's `websocket.receive()`, so any
    # audio the browser sends *while a turn is in flight* (it never stops
    # streaming just because we're thinking) piles up in the socket buffer
    # instead of reaching the endpointer in real time — it then arrives all
    # at once the moment we free up, well after it was actually spoken,
    # which is what made replies feel badly delayed and out of sync with
    # what was said in between. A background consumer decouples the two:
    # this loop keeps reading/end-pointing frames continuously, and finalized
    # utterances are handed off to run one at a time, in order, without
    # blocking frame ingestion.
    utterance_queue: asyncio.Queue[bytes] = asyncio.Queue()
    turn_state = _TurnState()
    consumer_task = asyncio.create_task(
        _consume_utterances(websocket, mcp_client, session_id, utterance_queue, turn_state)
    )

    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break

            frame = message.get("bytes")
            if frame is not None:
                if len(frame) != FRAME_BYTES:
                    await websocket.send_json(
                        {"type": "error", "detail": f"expected {FRAME_BYTES}-byte frames, got {len(frame)}"}
                    )
                    continue
                utterance_pcm = endpointer.accept_frame(frame)
                if endpointer.speech_confirmed and not was_speech_confirmed:
                    # Milestone 3.2 (barge-in): tell the client speech has
                    # started well before the utterance finishes, so it can
                    # stop any reply audio it's playing right away instead
                    # of waiting for a full "turn" round trip. Gated on
                    # speech_confirmed rather than the raw in_speech flag —
                    # in_speech alone flips true after ~90ms, which is too
                    # trigger-happy for this: a false positive here means
                    # wrongly cutting off audio that's actively playing
                    # (e.g. from acoustic echo of that same audio hitting
                    # the mic), not just discarding an empty buffer later.
                    await websocket.send_json({"type": "speech_started"})
                was_speech_confirmed = endpointer.speech_confirmed
                if utterance_pcm is not None:
                    utterance_queue.put_nowait(utterance_pcm)
                continue

            text = message.get("text")
            if text is not None:
                msg_type = json.loads(text).get("type")
                if msg_type == "stop":
                    break
                if msg_type == "finalize":
                    # Manual "I'm done talking" signal — bypasses waiting on
                    # VAD silence detection entirely. Added after VAD tuning
                    # for automatic end-pointing proved too environment-
                    # dependent (mic/room noise) to rely on alone; this is
                    # now the reliable primary control, with VAD's own
                    # silence detection left running underneath as a backup
                    # in case the user forgets to click it.
                    utterance_pcm = endpointer.force_finalize()
                    if utterance_pcm is not None:
                        utterance_queue.put_nowait(utterance_pcm)
                elif msg_type == "barge_in":
                    # The client sends this the moment it hears speech_started
                    # while a reply is playing (or still being generated).
                    # Drop anything still queued — it's stale now that the
                    # user is talking again — and cancel the turn actually
                    # running, if any. If nothing's running (the previous
                    # turn already finished and only its audio was playing),
                    # this is a no-op; stopping that audio is client-side.
                    while not utterance_queue.empty():
                        utterance_queue.get_nowait()
                    if turn_state.current_task is not None and not turn_state.current_task.done():
                        turn_state.current_task.cancel()
    except WebSocketDisconnect:
        pass
    finally:
        consumer_task.cancel()
        try:
            await consumer_task
        except (asyncio.CancelledError, WebSocketDisconnect):
            pass


async def _consume_utterances(
    websocket: WebSocket, mcp_client: MCPClient, session_id: str, queue: asyncio.Queue[bytes], turn_state: _TurnState
) -> None:
    """Process finalized utterances one at a time, in order, off the main
    receive loop (see handle_stream's comment for why this is separate).
    Each turn runs as its own task, rather than a bare await, so a
    "barge_in" message can cancel just that task without killing this loop —
    cancelling doesn't stop a blocking call already handed to a thread (see
    _run_turn's asyncio.to_thread calls), only our own wait on its result.
    """
    while True:
        utterance_pcm = await queue.get()
        turn_state.current_task = asyncio.create_task(
            _process_utterance(websocket, mcp_client, session_id, utterance_pcm)
        )
        try:
            await turn_state.current_task
        except asyncio.CancelledError:
            pass
        finally:
            turn_state.current_task = None


async def _process_utterance(
    websocket: WebSocket, mcp_client: MCPClient, session_id: str, utterance_pcm: bytes
) -> None:
    try:
        await _run_turn(websocket, mcp_client, session_id, utterance_pcm)
    except WebSocketDisconnect:
        raise
    except Exception:
        # A failed turn (e.g. the LLM provider erroring out after exhausting
        # its own retries) shouldn't kill the whole streaming session — log
        # it, tell the user in plain language, and keep listening.
        logger.exception("session=%s turn failed", session_id)
        await websocket.send_json(
            {"type": "error", "detail": "Something went wrong processing that — please try again."}
        )


async def _run_turn(websocket: WebSocket, mcp_client: MCPClient, session_id: str, utterance_pcm: bytes) -> None:
    wav_bytes = pcm16_to_wav(utterance_pcm)

    # stt.transcribe and tts.synthesize are blocking, CPU-bound calls (several
    # seconds each) — offload them to a thread so the event loop stays free
    # to service this WebSocket's keepalive pings while they run. Without
    # this, a long enough turn gets the connection closed by ping-timeout
    # before the reply can be sent (see agent_loop.py's matching fix for the
    # LLM call, which is the same underlying issue).
    t0 = time.perf_counter()
    transcript, language = await asyncio.to_thread(stt.transcribe, wav_bytes)
    t1 = time.perf_counter()

    if not transcript.strip():
        # A VAD false-positive (e.g. a noise burst) with nothing to transcribe.
        return

    reply_text = await agent_loop.run_turn(session_id, transcript, mcp_client)
    t2 = time.perf_counter()

    reply_audio = await asyncio.to_thread(tts.synthesize, reply_text)
    t3 = time.perf_counter()

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

    await websocket.send_json(
        {
            "type": "turn",
            "session_id": session_id,
            "transcript": transcript,
            "language": language,
            "reply_text": reply_text,
            "reply_audio_base64": base64.b64encode(reply_audio).decode("ascii"),
        }
    )
