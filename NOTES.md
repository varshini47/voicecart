# NOTES

Milestone log: what was built, key decisions, and what the owner should be able to explain in an interview.

---

## Week 0 — Setup (2026-07-22)

**Built:** Repo scaffold, `.env`/`.gitignore`, `demo/seed_shopify.py` (Shopify Admin API seeding script), `voice/stt_check.py` and `voice/tts_check.py` verification scripts.

**Key decisions:**
- LLM provider: Groq free tier, OpenAI-compatible API, `llama-3.3-70b-versatile`. Chosen for speed and zero cost; swappable later since nothing is hardcoded to it.
- Shopify dev store seeded with 31 grocery products across 7 categories, deliberately including multiple brands per item (e.g. multiple milk brands) — needed later for ambiguity-clarification evals ("Amul or Nandini?").

**Review:** confirm the Shopify dev store (`voicecart-karukcuy.myshopify.com`) looks right in admin; confirm `.env` is populated locally and never committed.

---

## Week 1, Milestone 1.1 — Voice loop skeleton (2026-07-23)

**Built:**
- `voice/stt.py` — thin wrapper around faster-whisper (`small`, CPU, int8, language auto-detect). `transcribe(audio_bytes) -> (text, language)`.
- `voice/tts.py` — thin wrapper around Piper. Shells out to the Piper CLI against a temp WAV file (same mechanism as the Week 0 check script) and returns the bytes. `synthesize(text) -> wav_bytes`.
- `agent/llm.py` — provider-agnostic client: a single `requests.post` to `{LLM_BASE_URL}/chat/completions` with a fixed shopping-assistant system prompt. No SDK, no tool-calling yet (that's Week 2).
- `agent/main.py` — FastAPI app, one endpoint `POST /converse`: audio upload → STT → LLM → TTS → JSON response (`transcript`, `language`, `reply_text`, `reply_audio_base64`). `GET /` serves `agent/static/index.html`.
- `agent/static/index.html` — minimal record-button page (MediaRecorder, webm/opus), plays back the base64 WAV reply.

**Key decisions:**
- Audio in is webm/opus (browser's native `MediaRecorder` format); faster-whisper decodes it directly via PyAV, so no client-side transcoding needed. Audio out is WAV, returned base64-encoded inside the JSON body rather than as a separate binary response — keeps the endpoint single-shot and easy to test with curl.
- No session/conversation memory yet — each `/converse` call is a single independent turn. That's explicitly Milestone 1.2.
- No pytest suite yet — this session was scoped to the skeleton only (Milestone 1.3 covers mocked tests + latency logging). Verified instead with a manual end-to-end smoke test (curl + a real browser mic recording): posted/spoke, confirmed transcript → Groq reply → synthesized audio all round-tripped.
- Record button is click-to-start/click-to-stop, not press-and-hold — changed after owner testing found holding a button while framing a sentence awkward.

**Perf fix (owner-reported latency):** first end-to-end test came in at ~10.3s per turn, which felt too slow. Profiled each stage in isolation and found `voice/tts.py` was the dominant cost — it shelled out to `python -m piper` per request, respawning a process and reloading the ONNX model every single call. Switched to Piper's Python API (`PiperVoice.load` once at import time, `synthesize_wav` reused per call) — TTS dropped from ~9s to ~0.6s, total to ~7s. Remaining ~5s is faster-whisper `small` on CPU transcribing a ~5s clip; tried `beam_size=1` (greedy decoding) and raising `cpu_threads`, neither moved the needle meaningfully. Didn't drop to a smaller Whisper model to chase more speed — CLAUDE.md explicitly locks `small`/CPU as a decision made for accuracy (matters for the Hinglish stretch goal), so that tradeoff needs a conscious discussion, not a silent downgrade. If this becomes a real blocker, the actual fix is Week 3's streaming STT (hides latency instead of removing it) or explicitly renegotiating the model-size decision.

**Review:**
- Run `uvicorn agent.main:app --reload`, open `http://127.0.0.1:8000/`, click to record, talk, click to stop — this was tested live in-browser already, but re-confirm on your machine.
- Be able to explain: why turn-based before streaming (simpler to get end-to-end correct first, no VAD/barge-in complexity yet), why `requests` instead of an LLM SDK (keeps the provider swap trivial and the HTTP call fully visible), why audio comes back as base64-in-JSON instead of a raw audio response, and the TTS latency root-cause (process-per-request vs. loading the model once).
