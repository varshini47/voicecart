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
- `voice/tts.py` — thin wrapper around Piper's Python API (`PiperVoice`, loaded once and reused across calls — see Milestone 1.3 for why loading became lazy instead of at-import). `synthesize(text) -> wav_bytes`. (Originally shelled out to the Piper CLI per call like the Week 0 check script — see the perf fix below for why that changed.)
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

---

## Week 1, Milestone 1.2 — Session state (2026-07-23)

**Built:**
- `agent/session.py` — in-memory store: a plain `dict[session_id, list[{"role", "content"}]]`. `new_session_id()`, `get_history(session_id)`, `append(session_id, role, content)`.
- `agent/llm.py`'s `reply()` now takes `(history, user_text)` and splices `history` between the system prompt and the current turn.
- `agent/main.py`'s `/converse` accepts an optional `session_id` form field, mints one (`uuid4().hex`) if absent, always returns it, and appends both sides of each turn to that session's history.
- `agent/static/index.html` tracks `session_id` in a JS variable across turns and now appends every turn to a running transcript in the page instead of overwriting the last one.

**Key decisions:**
- Plain dict, no TTL/eviction — matches CLAUDE.md's "no database until needed." Fine for a single-process local demo; will need addressing before any real deployment (sessions currently live and die with the process, memory grows unbounded for long-running processes).
- Session ends when the browser tab reloads (the JS variable resets), not via any explicit "new conversation" button — kept minimal since that's enough to demonstrate both continuity and isolation.

**Verified:** via curl, told the agent "my name is Varshini and I want to buy some milk," then in the same session asked "what did I just say I wanted to buy?" — reply correctly included "milk" and "Varshini." Repeating the second question with no `session_id` (a fresh session) correctly produced a reply with no memory of the milk mention. Also confirmed live in-browser by the owner.

**Review:**
- Be able to explain: why a dict instead of Postgres at this stage, why session_id is client-supplied rather than cookie-based (keeps curl/testing trivial, no cookie/CORS complexity yet), and the memory-growth caveat above (relevant when we get to Week 4 deployment).

---

## Week 1, Milestone 1.3 — Tests + latency logging (2026-07-23)

**Built:**
- `tests/conftest.py` — `FakePipeline` fixture monkeypatches `voice.stt.transcribe`, `agent.llm.reply`, `voice.tts.synthesize` module-level, and clears `agent.session._sessions` before each test.
- `tests/test_converse.py` — 5 tests: response shape, a fresh `session_id` minted when absent, history correctly threaded into the LLM across repeat calls with the same `session_id`, isolation between different `session_id`s, and the latency log line containing the expected fields without leaking raw audio bytes.
- `pytest.ini` — `pythonpath = .` so `tests/` can `from agent.main import app` and `from tests.conftest import FakePipeline` without an installed package or `__init__.py` files.
- `agent/main.py` now times STT/LLM/TTS with `time.perf_counter()` and logs one line per turn via the stdlib `logging` module: `session=... stt_ms=... llm_ms=... tts_ms=... total_ms=... transcript=... reply_text=...`.

**Key decision — the one worth defending in an interview:** `voice/stt.py` and `voice/tts.py` originally loaded their models (Whisper, Piper) at *import* time. That's fine for a running server (load once, serve fast) but wrong for tests: importing `agent.main` transitively imports both wrapper modules, so every test run would eat the real ~6s model-load cost even though the tests mock `transcribe`/`synthesize` and never touch the real models — and CI wouldn't even have the gitignored Piper `.onnx` file to load. Fixed by making both loads *lazy* (load on first real call, memoized after). Tests now run in ~0.05s. Tradeoff: the first real `/converse` call after server startup now pays that load cost instead of it happening at startup — measured `tts_ms≈1900` on the first call after a fresh start vs `≈200` on every call after. Acceptable for a local demo; worth another look if this ever becomes a warm, always-on prod service (e.g. a startup event that forces the load proactively).

**Latency numbers (for the resume/interview):** stt_ms≈5000 (faster-whisper `small`, CPU — the dominant cost, expected), llm_ms≈550 (Groq), tts_ms≈200 warm / ≈1900 cold-start. Total per turn ≈6-7s warm.

**Review:**
- Run `pytest` — should be 5 passed in well under a second, no network/model access.
- Be able to explain: why STT dominates and what the fix path is (Week 3 streaming hides it, or a smaller model trades accuracy for speed), why models had to become lazy-loaded for tests to be fast/possible at all, and what's *not* logged and why (no raw audio, no API keys, per CLAUDE.md's observability rule).

---

## Week 2, Milestone 2.1 — MCP commerce server (2026-07-23)

**Built:**
- `mcp_commerce/shopify_client.py` — the only module that speaks to Shopify. Wraps the Storefront GraphQL API: `search_products`, `create_cart`, `get_cart`, `add_line`, `remove_line`. Raises `ShopifyError` on GraphQL errors, mutation `userErrors`, or request failures — the one exception type the rest of mcp_commerce needs to catch.
- `mcp_commerce/carts.py` — plain `dict[session_id, shopify_cart_id]`, same "no DB" pattern as `agent/session.py`.
- `mcp_commerce/models.py` — pydantic result models (`SearchProductsResult`, `CartResult`, `CheckoutResult`), every one carrying an `error: str | None` field.
- `mcp_commerce/server.py` — a `FastMCP` server (official `mcp` SDK) exposing 5 tools: `search_products`, `get_cart`, `add_to_cart`, `remove_from_cart`, `checkout`. Runs standalone via `python -m mcp_commerce.server` (stdio transport).
- `tests/test_mcp_commerce.py` — 10 tests, mocking `shopify_client` entirely, covering structured errors, the checkout confirm-gate, empty-cart rejection, and add/remove/get cart shapes.

**Blocker found and fixed (external system, not code):** the Storefront API returned zero products for every query, even though the Week 0 seed script had created 31 of them. Root cause: the custom Shopify app's Admin API token didn't have `read_publications`/`write_publications` scope, so there was no way via the API to check or fix which sales channel the products were published to. This wasn't something I could resolve by writing code — the owner had to go into the Shopify Partner Dashboard, add the scopes to the custom app's configuration, and regenerate the Admin API token. After that, all 31 grocery products (44 total, including a handful of Shopify's own default sample products) were confirmed published to "Online Store," and Storefront search started working immediately — no further fix needed, so it really was a token-scope gap, not a Storefront-visibility bug on our side.

**Key decisions:**
- `product_id` everywhere in the tool contracts is actually a Shopify **variant** GID, not a product GID. Since every seeded product has exactly one variant, there's no meaningful product/variant distinction to expose to the agent — keeping the tool contract to a single `product_id` is simpler and the agent never needs to know Shopify's data model has two levels here.
- `checkout(session, confirm=False)` — confirm defaults to `False` and returns a structured error ("ask the user to confirm first") rather than proceeding. This is a safety gate at the tool layer itself, independent of whatever the agent decides to do — Milestone 2.2 (tool-calling agent) is responsible for only ever passing `confirm=True` after the user has explicitly said yes, but even if the agent got that wrong, the tool won't finalize checkout without it.
- "Checkout" here means: validate the cart isn't empty, then return Shopify's real hosted `checkoutUrl` and a subtotal — it does **not** complete payment. That's what CLAUDE.md's "test-mode checkout only" means in practice: we prove the whole pipeline can reach a real, valid Shopify checkout link without ever touching payment.
- Transport is stdio (`mcp.run()` default), matching how most local MCP servers run (e.g. Claude Desktop's config). The agent doesn't talk to this server yet — that wiring is explicitly Milestone 2.2, so 2.1 was scoped to "the server works correctly on its own," verified via a standalone MCP client script, not via the agent.

**Verified:** `pytest` (10 mocked unit tests) plus a one-off manual script (not committed — a dev-only smoke test, similar in spirit to the Week 1 curl checks) that opened a real MCP `ClientSession` over stdio against the running server and exercised the full flow against the live Shopify dev store: searched "milk" and got all 3 brands back (Amul/Nandini/Mother Dairy — confirming the deliberate Week 0 ambiguity design actually works end-to-end), added 2x Amul milk, fetched the cart, called checkout without confirm (correctly got a structured error), called it again with `confirm=True` (got back a real Shopify `checkoutUrl`), removed the item, and confirmed removing it again correctly errors since it's no longer in the cart.

**Review:**
- Be able to explain: why Shopify-specific code lives only in `shopify_client.py` (swappable backend — replacing Shopify with ONDC later would mean rewriting that one file, not `server.py`'s tool contracts), why `product_id` is secretly a variant ID, why `checkout` needs a confirm flag at the tool layer *in addition to* agent-level confirmation logic, and the publications-scope story above (it's a good example of "the bug was in account configuration, not code").
