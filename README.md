# VoiceCart — Voice Commerce Agent

A voice-driven shopping agent for quick commerce. You speak ("add two packets of
milk and a loaf of bread"), it transcribes you, an LLM agent decides what to do
and calls commerce tools over MCP, and it talks back. Supports barge-in
(interrupt it mid-reply) and Hindi/English code-switched (Hinglish) input.

Built by [Varshini Pandiri](https://github.com/varshini47) as a portfolio
project — see [CLAUDE.md](CLAUDE.md) for the full spec this was built against,
and [PLAN.md](PLAN.md) / [NOTES.md](NOTES.md) for the week-by-week build log,
every real bug found, and every design decision made along the way.

## Architecture

```
Mic (browser)
   │  raw PCM over WebSocket
   ▼
VAD end-pointing (webrtcvad)  ──  detects utterance boundaries, drives barge-in
   │  finalized utterance
   ▼
STT (faster-whisper, small, local, language auto-detect)
   │  transcript
   ▼
Agent loop (FastAPI)  ──  LLM decides: reply / call a tool / ask a clarifying question
   │  tool calls (JSON-schema contract)
   ▼
MCP server ("commerce-mcp")  ──  search_products / get_cart / add_to_cart /
   │                             remove_from_cart / checkout
   ▼
Shopify dev store (Storefront API)
   │
   ▼
TTS (Piper, local) → audio out, streamed back over the same WebSocket
```

**Key design principle:** the agent never talks to Shopify directly. Every
commerce operation goes through the MCP tool layer, with strict pydantic
schemas and structured `error` fields instead of exceptions — so the backend
(Shopify today) is swappable without touching the agent loop or the prompt.

## What's built

| Week | Milestone | Status |
|---|---|---|
| 1 | Turn-based voice pipeline (record → STT → LLM → TTS) | ✅ |
| 2 | MCP commerce server, tool-calling agent, evals v1 (20 scenarios) | ✅ |
| 3 | Streaming STT over WebSocket, barge-in, Hinglish evals (+10 scenarios) | ✅ |
| 4.1 | Docker (text-mode deploy image) | ✅ |
| 4.2 | AWS EC2 deploy (text-mode, SSM-only access) | ✅ |
| 4.3 | GitHub Actions CI/CD (lint, tests, evals, image build+push) | ✅ |
| 4.4 | Demo video + this README | ✅ |

## Latency (turn-based, CPU, warm)

| Stage | Time | Notes |
|---|---|---|
| STT (faster-whisper `small`, CPU) | ~5000ms | Dominant cost — see [design decisions](#design-decisions) |
| LLM (Groq, tool-calling) | ~550ms–2s | Varies with tool-call rounds |
| TTS (Piper) | ~200ms warm / ~1900ms cold-start | Cold cost paid once per process, not per turn |
| **Total, single turn** | **~6–7s warm** | Streaming mode (Week 3) hides this behind continuous capture instead of removing it |

## Eval results

30 scenarios in `evals/scenarios/` (20 English, 10 Hinglish), scored against
the real agent loop with Shopify mocked — tool selection, argument
correctness, clarification behavior, and checkout-confirmation safety.
Verified live in CI (GitHub Actions, `master`-only, live Groq calls):

**29/30 passing (97%).** The sole failure, `hinglish_remove_without_respecifying_brand`,
is a documented non-deterministic flake in this model's tool-calling (passes
3 of 4 attempts across sessions) — not a prompt gap. See NOTES.md's Milestone
3.3 and 4.3 entries for the full diagnosis.

Run it yourself: `python -m evals.runner` (needs `LLM_API_KEY` in `.env`; Shopify
is mocked, no real store needed).

## Design decisions

A few of the ones most worth defending in an interview — full reasoning and
the real bugs behind each one are in NOTES.md.

- **MCP as the only path to Shopify.** `mcp_commerce/shopify_client.py` is the
  one module that speaks to Shopify; every tool contract is Shopify-agnostic
  (`product_id` hides that it's actually a variant GID). Swapping Shopify for
  ONDC later means rewriting one file, not the agent or its prompt.
- **Confirmation and clarification are prompt-driven, with a tool-layer
  backstop.** There's no hand-rolled dialogue state machine — the system
  prompt instructs the model to confirm before checkout and ask before
  guessing an ambiguous brand, and `checkout(confirm=False)` defaulting to a
  structured error is the code-level safety net if the prompt ever fails.
  Verified with the 30-scenario eval suite, not vibes.
- **In-memory session store, no database.** A `dict[session_id, ...]` for
  both conversation history and cart mapping. Correct for a single-process
  local demo; the growth-is-unbounded and process-restart-loses-everything
  tradeoffs are real and would need addressing before any multi-instance
  deployment.
- **"Streaming STT" means VAD end-pointing, not live partial captions.**
  faster-whisper has no incremental decode mode; re-transcribing a growing
  buffer for live captions would add real latency-management complexity for
  an effect nothing downstream needs. A manual "Done talking" button is the
  primary end-of-utterance signal (automatic VAD is a backup) — added after
  automatic VAD proved too trial-and-error to tune blindly across mics/rooms.
- **Barge-in cancels the in-flight `asyncio.Task`, not the underlying LLM
  call.** Cancelling a task only stops our own code from waiting on it — a
  blocking call already handed to a thread (the LLM HTTP request) keeps
  running to completion regardless, since Python can't kill a thread. Not
  dangerous (nothing irreversible; `checkout` has its own confirm gate), but
  a genuine, explainable limitation of using threads for blocking I/O.
- **Acoustic echo (the TTS reply triggering its own barge-in) is mitigated,
  not solved.** Unlike a noise blip, an echo of your own voice through
  speakers is real, continuous, natural speech — no duration threshold can
  distinguish it from an actual interruption. Browser-level echo cancellation
  plus a headphones recommendation is the honest fix; full AEC for arbitrary
  local audio playback needs dedicated hardware this project doesn't have.
- **CI evals run on `master`-only pushes, not every commit, and don't block
  the build.** Groq's free-tier quota was hit repeatedly during real
  development (documented 429s in Milestones 2.3 and 3.3); running 30
  live-LLM scenarios on every push would make that worse for no benefit. The
  job still reports its pass rate to the run summary either way.
- **CD stops at pushing a Docker image to GHCR, not auto-deploying to EC2.**
  The EC2 box is a family member's personal AWS account with no CI
  credentials configured — wiring that up is a deliberate access-control
  decision to make separately, not something to default into silently.

## Running it locally

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in LLM_API_KEY, SHOPIFY_* — see .env.example

# Turn-based voice demo (record button):
uvicorn agent.main:app --reload
# open http://127.0.0.1:8000/

# Streaming voice demo (continuous mic, barge-in, Hinglish) — needs the
# longer WebSocket keepalive, see NOTES.md Milestone 3.1 for why:
uvicorn agent.main:app --reload --ws-ping-interval 20 --ws-ping-timeout 90
# open http://127.0.0.1:8000/stream
```

Tests: `pytest` (56 tests, mocked, no network — runs in well under a second).
Evals: `pytest -m eval` or `python -m evals.runner` (live LLM, Shopify mocked).

## Deployment (text-mode)

The deployed target drops the voice stack entirely (`agent/main_text.py`,
`POST /converse/text` + `GET /health`) — Whisper's RAM footprint doesn't fit
a free-tier instance, and voice is meant to be demoed locally.

```bash
docker build -f infra/Dockerfile -t voicecart .
docker compose -f infra/docker-compose.yml up --build   # local
```

Deployed to a free-tier EC2 instance (Amazon Linux 2023), administered
entirely through AWS Systems Manager Session Manager — zero inbound security
group rules, no SSH keys. See NOTES.md Milestone 4.2 for the full cost-safety
and access-model reasoning.

GitHub Actions (`.github/workflows/ci.yml`) runs lint + tests on every push,
runs the live eval suite and builds/pushes the deploy image to GHCR on pushes
to `master`. See NOTES.md Milestone 4.3 for the CI scope decisions.

## Repository layout

```
agent/            FastAPI app: agent loop, session state, LLM client, WS streaming
mcp_commerce/     MCP server wrapping the Shopify Storefront API
voice/            STT (faster-whisper) + TTS (Piper) + VAD wrappers
evals/            YAML eval scenarios + runner (fake Shopify, real agent loop)
tests/            pytest unit + integration tests
infra/            Dockerfile, docker-compose.yml
demo/             Shopify seed script, demo recording script
.github/workflows/ CI/CD
```

## Known limitations

- Acoustic echo can still occasionally self-trigger barge-in without
  headphones (see [design decisions](#design-decisions)).
- In-memory session/cart state — restarting the server loses all sessions.
- One Hinglish eval scenario is flaky due to underlying LLM non-determinism,
  not a fixed bug.
- `checkout` returns a real Shopify hosted checkout URL but never completes
  payment — this is deliberately test-mode only, per CLAUDE.md.
