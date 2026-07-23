# VoiceCart — 4-Week Build Plan

How to use this: each week has milestones sized for one Claude Code session each. Start every session by pointing Claude Code at the milestone text. After each milestone: review the diff, ask Claude Code to explain any design decision you couldn't defend in an interview, run the tests, commit.

---

## Week 0 (one evening) — Setup

- [x] Create GitHub repo `voicecart`, add this file and CLAUDE.md
- [x] Create a Shopify Partner account → free development store → seed it with ~30 grocery products (Claude Code can write a seeding script using the Admin API)
- [x] Get Storefront API token; put it in `.env`, add `.env` to `.gitignore` immediately
- [x] Pick your LLM provider (free tier is fine to start) and set `LLM_API_KEY` / `LLM_MODEL` env vars
- [x] `pip install faster-whisper` and verify the small model transcribes a test recording of your voice on your laptop
- [x] Install Piper TTS and generate one test WAV

Definition of done: you can transcribe your own voice and synthesize a reply locally, and your Shopify store shows products in its admin.

Done 2026-07-22. LLM provider: Groq (free tier, OpenAI-compatible API, `llama-3.3-70b-versatile`). Shopify dev store `voicecart-karukcuy.myshopify.com` seeded with 31 grocery products across 7 categories, multiple brands per item on purpose (for ambiguity-clarification evals later). STT/TTS verified end-to-end (Piper → Whisper round-trip, plus real voice recording transcribed correctly).

---

## Week 1 — Turn-based voice pipeline

**Milestone 1.1 — Voice loop skeleton.** [x]
FastAPI app with one endpoint: POST /converse accepts an audio file, runs STT, sends transcript to the LLM with a simple system prompt (no tools yet — just a helpful shopping assistant persona), synthesizes the reply with TTS, returns audio + transcript JSON. Plus a minimal HTML page with a record button that hits this endpoint and plays the response.

Done 2026-07-23. `agent/main.py` (FastAPI, single POST /converse + static index page), `voice/stt.py` / `voice/tts.py` (thin wrappers around the Week 0 check scripts), `agent/llm.py` (plain `requests` call to the OpenAI-compatible chat/completions endpoint — no provider SDK). Smoke-tested end-to-end locally with a real WAV and live in-browser via the record button: transcript → Groq reply → synthesized audio all round-tripped correctly. No session memory yet (Milestone 1.2) and no pytest suite yet (Milestone 1.3) — this milestone was scoped to the skeleton only.

Perf fix during owner testing: initial round-trip was ~10.3s, almost entirely `voice/tts.py` shelling out to `python -m piper` per request (process spawn + ONNX model reload every call). Switched to Piper's Python API (`PiperVoice.load` once at import, `synthesize_wav` in-process) — brought total to ~7s. Remaining time is faster-whisper `small` on CPU (~5s for a ~5s clip); tried `beam_size=1` and higher `cpu_threads`, neither helped meaningfully, and swapping to a smaller Whisper model wasn't done since CLAUDE.md locks `small`/CPU as a deliberate decision (accuracy matters for Hinglish later) — that's a tradeoff to revisit together if latency becomes a real blocker, not Week 3's planned streaming-STT fix. Also swapped the record button from press-and-hold to click-to-start/click-to-stop per owner feedback (holding was awkward).

**Milestone 1.2 — Session state.**
Add session_id handling and conversation history so multi-turn works ("what did I just ask?"). In-memory store.

**Milestone 1.3 — Tests + latency logging.**
pytest coverage for the endpoint (mock STT/LLM/TTS). Log per-stage latency (STT ms, LLM ms, TTS ms) — you'll cite these numbers later.

Interview checkpoint — you should be able to explain: why turn-based first, where the latency goes, what faster-whisper's model sizes trade off.

---

## Week 2 — Agent + MCP commerce tools + evals

**Milestone 2.1 — MCP server.**
`mcp_commerce` exposing: `search_products(query, limit)`, `add_to_cart(session, product_id, qty)`, `remove_from_cart`, `get_cart`, `checkout(confirm=true)`. Each wraps the Shopify Storefront API. Strict pydantic schemas, structured errors.

**Milestone 2.2 — Tool-calling agent loop.**
Replace the echo agent: LLM now decides between replying, calling a tool, or asking a clarifying question. Implement confirmation-before-checkout and ambiguity clarification ("Amul or Nandini?"). Retry-once on recoverable tool errors.

**Milestone 2.3 — Evals v1.**
YAML scenarios + runner as specified in CLAUDE.md. 20 scenarios: happy paths, ambiguous items, quantity edge cases ("half a dozen eggs"), a checkout-without-confirmation trap the agent must not fall into. Wire into pytest.

Interview checkpoint: why MCP instead of direct function calls (swappable backend, standard contract), how you score an agent, what your pass rate was before/after prompt fixes.

---

## Week 3 — Streaming, interruptions, Hinglish

**Milestone 3.1 — Streaming STT over WebSocket.**
Switch /converse to a WebSocket session: client streams mic audio chunks, server runs VAD (silence detection) + incremental transcription.

**Milestone 3.2 — Barge-in.**
If the user speaks while TTS audio is playing, stop playback, cancel the in-flight agent turn, process the new utterance. This is the hardest milestone in the project — budget a full session, possibly two.

**Milestone 3.3 — Hinglish.**
Add 10 code-switched eval scenarios ("do packet doodh add karo"). Whisper's auto-detect handles a lot; fix the gaps with prompt guidance to the agent ("users may mix Hindi and English; item names may be in either"). Measure eval pass rate on the Hinglish set separately.

Interview checkpoint: how VAD works, why cancellation is hard (race conditions between audio pipeline and agent loop), Hinglish pass rate numbers.

---

## Week 4 — Ship it

**Milestone 4.1 — Docker.**
Dockerfile for the agent+MCP services, docker-compose for local run. Keep STT/TTS out of the deployed image (see 4.2).

**Milestone 4.2 — AWS deploy (text mode).**
Deploy the agent + MCP to a free-tier EC2 instance behind text-mode endpoints (voice runs locally in demos; the cloud instance proves deployment skills without the RAM cost of Whisper). Set the ₹100 billing alarm FIRST.

**Milestone 4.3 — CI/CD.**
GitHub Actions: on push → lint, pytest, run evals, report pass rate; on main → build image and deploy. This workflow file is itself interview material.

**Milestone 4.4 — Demo + README.**
2-minute screen recording: voice order end-to-end, order appearing in the Shopify admin, a barge-in moment, one Hinglish command. README with architecture diagram, latency numbers, eval pass rates, and a "design decisions" section.

---

## Resume bullet targets (write these only when they're true)

- Built a real-time voice commerce agent (STT → LLM tool-calling → TTS) with barge-in interruption handling and Hindi-English code-switched input
- Designed an MCP server exposing swappable commerce tools over the Shopify API, with confirmation-gated mutations and structured error recovery
- Authored a 30+ scenario eval suite scoring tool selection and cart-state correctness, integrated into GitHub Actions CI with automated AWS deployment

## Scope discipline

If you're behind schedule, cut in this order: Hinglish → barge-in (keep streaming) → AWS (demo locally). Never cut: evals, tests, README. A smaller project with evals beats a bigger one without.
