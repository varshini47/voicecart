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

**Milestone 1.2 — Session state.** [x]
Add session_id handling and conversation history so multi-turn works ("what did I just ask?"). In-memory store.

Done 2026-07-23. `agent/session.py`: plain dict keyed by `session_id`, storing each turn's {"role", "content"}. `/converse` now accepts an optional `session_id` form field (mints one via `uuid4().hex` if absent, always returns it) and threads the session's history into `agent/llm.py`'s `reply()` before the current turn. Demo page tracks `session_id` in a JS variable and appends every turn to a running transcript instead of overwriting. Verified via curl: told the agent "my name is Varshini and I want to buy some milk," then in the same session asked "what did I just say I wanted to buy?" — got back "milk, Varshini" correctly; repeating the second question with no session_id (fresh session) correctly came back with no memory of it.

**Milestone 1.3 — Tests + latency logging.** [x]
pytest coverage for the endpoint (mock STT/LLM/TTS). Log per-stage latency (STT ms, LLM ms, TTS ms) — you'll cite these numbers later.

Done 2026-07-23. `tests/conftest.py` + `tests/test_converse.py`, 5 tests: response shape, session_id minted when absent, history correctly threaded into the LLM for repeat calls with the same session_id, isolation between different session_ids, and latency fields present in the log line without leaking raw audio bytes. All pass in ~0.05s. That speed only works because `voice/stt.py` and `voice/tts.py` were changed to load their models lazily (on first real call) instead of at import time — otherwise every test run would pay the ~6s Whisper+Piper load cost, and CI wouldn't even have the (gitignored) Piper model file available. Tradeoff: the very first real `/converse` call after server startup now eats that load cost instead of it happening at startup (measured: first call tts_ms≈1900 vs ≈200 on later calls) — acceptable for a local demo, worth knowing if this becomes a prod service. `agent/main.py` now times each stage with `time.perf_counter()` and logs one line per turn: `session=... stt_ms=... llm_ms=... tts_ms=... total_ms=... transcript=... reply_text=...` — no audio bytes or API keys logged, per CLAUDE.md. Latest real numbers: stt_ms≈5000 (dominates, expected for `small` on CPU), llm_ms≈550, tts_ms≈200-1900 depending on cold/warm.

Interview checkpoint — you should be able to explain: why turn-based first, where the latency goes, what faster-whisper's model sizes trade off.

---

## Week 2 — Agent + MCP commerce tools + evals

**Milestone 2.1 — MCP server.** [x]
`mcp_commerce` exposing: `search_products(query, limit)`, `add_to_cart(session, product_id, qty)`, `remove_from_cart`, `get_cart`, `checkout(confirm=true)`. Each wraps the Shopify Storefront API. Strict pydantic schemas, structured errors.

Done 2026-07-23. `mcp_commerce/shopify_client.py` (only module that talks to Shopify — GraphQL Storefront API: product search, cart create/read/add/remove), `mcp_commerce/carts.py` (in-memory `session_id -> Shopify cart_id` map), `mcp_commerce/models.py` (pydantic result schemas, every one with an `error` field), `mcp_commerce/server.py` (FastMCP server, 5 `@mcp.tool()`-decorated functions). `product_id` in every tool is the Shopify variant GID — the agent never needs to know Shopify's product/variant distinction. `checkout` defaults `confirm=False` and returns a structured error until called with `confirm=True`; the agent (Milestone 2.2) is responsible for only passing `True` after explicit user confirmation.

Blocker hit and fixed before this could start: the Storefront API returned zero results for all 31 seeded products — turned out the custom app's Admin API token was missing `read_publications`/`write_publications` scope, so there was no way to confirm/fix product-to-Online-Store-channel publication. Owner added the scopes and regenerated the token; after that, all 31 products (44 total including a few of Shopify's own sample products) were confirmed published, and Storefront search worked correctly, including returning all 3 milk brands for a "milk" query — exactly the ambiguity the Week 0 seed data was designed for.

Verified two ways: (1) `tests/test_mcp_commerce.py`, 10 tests mocking `shopify_client` — covers structured errors, the checkout confirm-gate, empty-cart rejection, add/remove/get cart shapes. (2) A one-off manual script driving a real MCP `ClientSession` over stdio against the live server (not committed — dev-only smoke test) that exercised the whole flow against the actual Shopify dev store: searched "milk" (got all 3 brands), added 2x Amul milk, fetched the cart, checkout without confirm (correctly errored), checkout with confirm=true (got back a real `checkoutUrl`), removed the item, then confirmed removing again correctly errors since it's no longer in the cart.

**Milestone 2.2 — Tool-calling agent loop.** [x]
Replace the echo agent: LLM now decides between replying, calling a tool, or asking a clarifying question. Implement confirmation-before-checkout and ambiguity clarification ("Amul or Nandini?"). Retry-once on recoverable tool errors.

Done 2026-07-23. `agent/mcp_client.py` (spawns `mcp_commerce.server` over stdio once at app startup via a FastAPI lifespan, fetches tool schemas once and converts them to OpenAI function-calling format — the MCP server's own schemas stay the single source of truth). `agent/agent_loop.py` — `run_turn(session_id, user_text, mcp_client)`: loops calling `agent/llm.py`'s new `chat_completion(messages, tools)` (replaces the old `reply()`), executing any `tool_calls` via the MCP client, feeding results back, until a plain-text reply or a `MAX_TOOL_ROUNDS=6` cap. `agent/session.py`'s history now stores full message dicts (not just role/content), since tool-calling messages carry `tool_calls`/`tool_call_id`. All agent behavior (ambiguity clarification, confirm-before-checkout, quantity/unit conversion, retry-once-then-explain on tool errors) is driven by `agent_loop.py`'s system prompt, not hardcoded Python logic — the tool layer's own `confirm=False` default (Milestone 2.1) is the structural backstop in case the prompt ever fails.

Verified two ways: (1) `tests/test_agent_loop.py` (4 tests mocking `llm.chat_completion` and a fake MCP client — plain reply, single tool-call round-trip, history carrying across turns, the `MAX_TOOL_ROUNDS` cap actually stopping an endless-tool-call loop) plus `tests/test_converse.py` updated to mock `agent_loop.run_turn` as a black box. (2) Manual end-to-end runs against the live Groq + Shopify dev store: asking for "milk" correctly triggered a brand-clarification question (Amul/Nandini/Mother Dairy) instead of guessing; naming a brand after that correctly resolved and added to cart; a multi-turn conversation built up a 2-item cart and `get_cart` reported the right running total; "check out my order" correctly read back the cart and total and asked for confirmation *without* calling `checkout(confirm=true)`; only after "yes, I confirm" did it call checkout and return a real Shopify checkout URL.

Notable non-bug found during testing: faster-whisper consistently mis-transcribed "Amul" as "Emil" (an Indian brand name outside Whisper's English-biased vocabulary) — the agent handled this gracefully (recognized "Emil" wasn't a real brand and re-asked) rather than adding the wrong thing, but it's a real STT accuracy limitation worth remembering for the Hinglish work in Week 3.

Not exhaustively tested by hand: the specific "retry once with corrected arguments" path (hard to force deterministically in a one-off conversation) and quantity/unit edge cases like "half a dozen eggs." Both are exactly what Milestone 2.3's eval suite is for — systematic scenario coverage instead of ad hoc manual conversations.

**Milestone 2.3 — Evals v1.** [x]
YAML scenarios + runner as specified in CLAUDE.md. 20 scenarios: happy paths, ambiguous items, quantity edge cases ("half a dozen eggs"), a checkout-without-confirmation trap the agent must not fall into. Wire into pytest.

Done 2026-07-25. `evals/fake_shopify.py` (in-memory catalog + cart backend, same function signatures as `mcp_commerce.shopify_client`), `evals/fake_mcp_client.py` (in-process fake MCP client — calls the real `mcp_commerce.server` tool functions directly via `list_tools()`, no subprocess needed), `evals/scenarios/*.yaml` (20 scenarios), `evals/runner.py` (loads scenarios, runs each through the real `agent_loop.run_turn`, scores tool selection/clarification/checkout-safety/final-cart-state, proactively throttles LLM calls 3s apart). `tests/test_evals.py` parametrizes pytest over all 20 scenarios, marked `@pytest.mark.eval` and excluded from the default `pytest` run (see `pytest.ini`) since they hit the live LLM and are slow/quota-consuming — run explicitly with `pytest -m eval` or `python -m evals.runner`.

Final result: **20/20 scenarios passing (100%)** against live Groq (`llama-3.3-70b-versatile`), Shopify mocked.

Real bugs found and fixed along the way (not just eval-writing — genuine agent/infra fixes):
1. Session-id leakage: the LLM was inventing placeholder `session` values (e.g. `"current_session"`) because `session` was exposed as a normal fillable tool parameter. Fixed by stripping `session` from the tool schemas shown to the LLM (`agent/mcp_client.py`'s `build_tool_schemas()`) and having `agent_loop.run_turn` auto-inject the real `session_id`.
2. Half-a-dozen-eggs quantity bug: agent added quantity=6 instead of 1 for "Eggs 6pc Tray." Fixed via a tightened system-prompt example (`agent/agent_loop.py`).
3. Bread brand ambiguity: agent silently narrowed its own search instead of asking, then treated the user's brand answer as an *additional* item instead of resolving the same request — ended up with two breads in the cart. Fixed via explicit "don't guess by re-searching, and a follow-up answer resolves the same request" prompt guidance.
4. Checkout-on-empty-cart: agent called `checkout(confirm=true)` on the very first message ("Check out my order.") without any prior confirmation exchange — worked correctly when the cart already had an item, only failed on an empty first-turn cart. Fixed via explicit "even on the first message, even if empty" prompt guidance.
5. Reliability bug in `agent/llm.py`: Groq/Llama-3.3 occasionally emits a malformed non-JSON function-call token (`<function=name{...}>`) instead of proper `tool_calls`, surfaced as a 400 `tool_use_failed` error — confirmed via direct API probing that this is inference noise (not deterministic on the prompt: identical requests sometimes fail every time, sometimes succeed at a later moment) and that a plain temperature=0 retry isn't reliable, but nudging the temperature up breaks the loop. Added a bounded retry with increasing temperature specifically for this error code, alongside the existing 429 retry.

One eval design correction: `remove_item_not_in_cart` originally asserted `remove_from_cart` must be called even for an item that was never added. The agent instead called `get_cart`, saw the item wasn't there, and answered correctly without calling `remove_from_cart` (there was no `product_id` to pass — nothing to remove). Rewrote the scenario to `forbid_tools: [remove_from_cart]` and tightened the prompt to make this flow (search → check cart → remove-or-explain) explicit and consistent, rather than forcing a pointless tool call.

Groq free-tier daily quota (100k TPD) was hit and fully recovered twice during this milestone (once mid-session on 2026-07-23, once again during verification) — a real constraint of the free tier, worked around by conserving quota (targeted scenario batches instead of full 20-scenario re-runs) rather than by creating additional accounts, which would violate Groq's terms.

Interview checkpoint: why MCP instead of direct function calls (swappable backend, standard contract), how you score an agent, what your pass rate was before/after prompt fixes, the session-id leakage bug and why tool schemas shouldn't expose infra-only parameters to the LLM, and the malformed-tool-call reliability story (inference noise vs. deterministic bug, and why the fix is a retry with varied sampling rather than a fixed backoff).

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
