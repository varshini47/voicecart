"""Eval runner: executes YAML scenarios (evals/scenarios/) against the real
agent_loop against a fake Shopify backend, and scores tool selection,
clarification behavior, and mutation safety per CLAUDE.md's Evals section.

Run standalone:  .venv/Scripts/python.exe -m evals.runner
Also wired into pytest via tests/test_evals.py, one test per scenario.
"""

from __future__ import annotations

import asyncio
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

from agent import agent_loop, llm, session
from evals.fake_mcp_client import FakeMCPClient, patched_shopify

SCENARIOS_DIR = Path(__file__).parent / "scenarios"

# Groq's free tier rate-limits by requests/tokens per minute. A tool-calling
# eval run fires many LLM calls back to back (unlike normal interactive use),
# so we throttle proactively here rather than relying on agent/llm.py's
# reactive 429 retry, which is far slower once the limit is actually hit.
MIN_SECONDS_BETWEEN_LLM_CALLS = 3.0
_last_llm_call_time = 0.0
_real_chat_completion = llm.chat_completion


def _throttled_chat_completion(messages: list[dict], tools: list[dict] | None = None) -> dict:
    global _last_llm_call_time
    wait = MIN_SECONDS_BETWEEN_LLM_CALLS - (time.monotonic() - _last_llm_call_time)
    if wait > 0:
        time.sleep(wait)
    _last_llm_call_time = time.monotonic()
    return _real_chat_completion(messages, tools=tools)


llm.chat_completion = _throttled_chat_completion


@dataclass
class TurnResult:
    user: str
    reply: str
    tool_calls: list[tuple[str, dict]]
    failures: list[str] = field(default_factory=list)


@dataclass
class ScenarioResult:
    name: str
    passed: bool
    turn_results: list[TurnResult]
    failures: list[str]


def load_scenarios() -> list[dict]:
    scenarios = []
    for path in sorted(SCENARIOS_DIR.glob("*.yaml")):
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        data["_path"] = path
        scenarios.append(data)
    return scenarios


async def run_scenario(scenario: dict) -> ScenarioResult:
    session_id = f"eval-{uuid.uuid4().hex}"
    failures: list[str] = []
    turn_results: list[TurnResult] = []

    with patched_shopify():
        mcp_client = FakeMCPClient()
        await mcp_client.load_tool_schemas()

        for i, turn in enumerate(scenario["turns"]):
            calls_before = len(mcp_client.calls)
            reply = await agent_loop.run_turn(session_id, turn["user"], mcp_client)
            this_turn_calls = mcp_client.calls[calls_before:]
            tool_names = [name for name, _ in this_turn_calls]

            turn_failures: list[str] = []

            for expected in turn.get("expect_tools", []):
                if expected not in tool_names:
                    turn_failures.append(f"turn {i + 1}: expected tool '{expected}' to be called, calls were {tool_names}")

            for forbidden in turn.get("forbid_tools", []):
                if forbidden in tool_names:
                    turn_failures.append(f"turn {i + 1}: tool '{forbidden}' must not be called, calls were {tool_names}")

            if turn.get("expect_clarification") and "?" not in reply:
                turn_failures.append(f"turn {i + 1}: expected a clarifying question, got reply: {reply!r}")

            if turn.get("forbid_confirmed_checkout"):
                if any(name == "checkout" and args.get("confirm") is True for name, args in this_turn_calls):
                    turn_failures.append(f"turn {i + 1}: checkout was called with confirm=true without expecting it")

            if turn.get("expect_confirmed_checkout"):
                confirmed = any(name == "checkout" and args.get("confirm") is True for name, args in this_turn_calls)
                if not confirmed:
                    turn_failures.append(f"turn {i + 1}: expected checkout(confirm=true), calls were {tool_names}")

            turn_results.append(
                TurnResult(user=turn["user"], reply=reply, tool_calls=this_turn_calls, failures=turn_failures)
            )
            failures.extend(turn_failures)

        if "expected_final_cart" in scenario:
            cart = await mcp_client.call_tool("get_cart", {"session": session_id})
            actual = sorted((line["title"], line["quantity"]) for line in cart["lines"])
            expected = sorted((item["title"], item["quantity"]) for item in scenario["expected_final_cart"])
            if actual != expected:
                failures.append(f"final cart mismatch: expected {expected}, got {actual}")

    session._sessions.pop(session_id, None)

    return ScenarioResult(name=scenario["name"], passed=not failures, turn_results=turn_results, failures=failures)


async def run_all(name_filter: str | None = None) -> list[ScenarioResult]:
    results = []
    scenarios = load_scenarios()
    if name_filter:
        scenarios = [s for s in scenarios if name_filter in s["name"]]
    for scenario in scenarios:
        print(f"running {scenario['name']}...", flush=True)
        result = await run_scenario(scenario)
        results.append(result)
        status = "PASS" if result.passed else "FAIL"
        print(f"[{status}] {result.name}", flush=True)
    return results


def print_report(results: list[ScenarioResult]) -> None:
    passed = sum(1 for r in results if r.passed)
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"[{status}] {r.name}")
        for failure in r.failures:
            print(f"    - {failure}")
    rate = passed / len(results) * 100 if results else 0
    print(f"\n{passed}/{len(results)} scenarios passed ({rate:.0f}%)")


if __name__ == "__main__":
    name_filter = sys.argv[1] if len(sys.argv) > 1 else None
    results = asyncio.run(run_all(name_filter))
    print_report(results)
