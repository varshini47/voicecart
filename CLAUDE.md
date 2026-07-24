# CLAUDE.md — VoiceCart (Voice Commerce Agent)

## What this project is

A voice-driven shopping agent for quick commerce. A user speaks ("add two packets of milk and a loaf of bread"), the system transcribes it, an LLM agent interprets intent and calls commerce tools (search, cart, checkout) exposed via an MCP server that wraps the Shopify Storefront/Admin APIs, and responds by voice. Stretch goal: Hindi/English code-switched (Hinglish) input.

Built by Varshini Pandiri as a portfolio project. Primary goals, in order:
1. **Understandable** — the owner must be able to explain every design decision in an interview. Prefer simple, explicit code over clever abstractions.
2. **Demoable** — a working end-to-end voice demo running locally, plus a deployed text-mode API on AWS.
3. **Production-flavored** — tests, Docker, CI/CD, basic observability, an eval suite for the agent.

## Architecture (target state)

```
Mic/audio in
   │
   ▼
STT (faster-whisper, local)          ── Week 1: turn-based; Week 3: streaming
   │  transcript
   ▼
Agent loop (FastAPI service)
   │  LLM decides: reply / call tool / ask clarification
   ▼
MCP server ("commerce-mcp")          ── clean tool contracts, swappable backend
   │  search_products / get_cart / add_to_cart / remove_from_cart / checkout
   ▼
Shopify dev store (Storefront API)
   │
   ▼
TTS (Piper, local) → audio out
```

Key design principle: **the agent never talks to Shopify directly.** All commerce operations go through the MCP tool layer, so the backend is swappable (Shopify today; ONDC or a custom store later). Tool schemas are the contract.

## Repository layout

```
voicecart/
  agent/            # FastAPI app: agent loop, session state, LLM client
  mcp_commerce/     # MCP server wrapping Shopify APIs
  voice/            # STT + TTS wrappers (faster-whisper, piper)
  evals/            # eval scenarios + runner + reports
  tests/            # pytest unit + integration tests
  infra/            # Dockerfile(s), docker-compose.yml, GitHub Actions
  demo/             # demo scripts, sample audio
```

## Tech decisions (do not change without discussing)

- **Python 3.11+, FastAPI, uv or pip-tools for deps.** Type hints everywhere; pydantic models for all tool inputs/outputs.
- **STT:** faster-whisper, `small` model, CPU is fine for turn-based. Language auto-detect on (needed for Hinglish).
- **TTS:** Piper (local, free). Premium TTS only for the final demo video, if at all.
- **LLM:** provider-agnostic client. Default to a cheap/fast model via env var `LLM_MODEL`; all provider config through env vars, never hardcoded. Must support tool calling.
- **MCP:** official Python MCP SDK. Tools defined with strict JSON schemas; every tool returns structured results including an `error` field rather than raising into the agent loop.
- **Shopify:** free development store. Credentials in `.env` (never committed). Storefront API for search/cart, test-mode checkout only.
- **State:** in-memory session store first (dict keyed by session_id); Postgres only if/when needed. Do not add a database prematurely.
- **No frontend framework** until Week 4 polish. A minimal HTML page with a record button is enough for the demo.

## Conventions

- Small commits, imperative messages ("add cart tools", not "added stuff").
- Every tool and agent behavior change needs a test. `pytest` must pass before commit.
- Every module gets a short docstring saying what it does and why it exists.
- When you (Claude Code) finish a milestone, write a brief `NOTES.md` entry: what was built, key decisions, what the owner should review and be able to explain.
- Never log full audio or API keys. Log transcripts and tool calls (that's the observability story).
- If a task is ambiguous, ask before building. Do not invent requirements.

## Agent behavior requirements

- **Confirmation before mutation:** the agent must confirm before checkout, and confirm ambiguous items ("Amul or Nandini?"). Read-only tools (search, get_cart) need no confirmation.
- **Error recovery:** on a failed tool call, the agent retries once with corrected input if the error is recoverable, otherwise tells the user what went wrong in plain language.
- **Quantity/unit handling:** "two milk" → quantity 2 of the resolved milk product. Ambiguity → clarifying question, not a guess.
- **Session memory:** the agent remembers the cart context within a session ("remove the bread" must work without re-specifying which bread).

## Evals (this is a headline feature, not an afterthought)

- `evals/scenarios/` holds YAML scenarios: user utterance(s) → expected tool calls + expected final cart state.
- Runner executes scenarios against the live agent (mocked Shopify), scores: correct tool selection, correct arguments, correct clarification behavior, no unauthorized mutations.
- Target: 20+ scenarios by Week 2, including Hinglish ones by Week 3. CI runs evals and reports the pass rate in the PR.

## Current status

- [x] Week 1: turn-based voice pipeline (record → STT → LLM echo agent → TTS)
- [x] Week 2: MCP commerce server + tool-calling agent + evals v1
- [ ] Week 3: streaming STT, barge-in/interruptions, Hinglish scenarios
- [ ] Week 4: Docker, AWS deploy (text-mode API), GitHub Actions CI/CD, demo video, README with architecture diagram

Update these checkboxes as milestones complete.
