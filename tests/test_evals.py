"""pytest wrapper around evals/: one test per YAML scenario in evals/scenarios/.

These call the live LLM (only Shopify is mocked, via evals/fake_shopify.py —
see CLAUDE.md's Evals section), so they're marked `eval` and excluded from
the default `pytest` run (see pytest.ini's addopts). Run them explicitly
with `pytest -m eval`, or use `python -m evals.runner` for the human-readable
pass-rate report.
"""

from __future__ import annotations

import pytest

from evals import runner

SCENARIOS = runner.load_scenarios()


@pytest.mark.eval
@pytest.mark.parametrize("scenario", SCENARIOS, ids=[s["name"] for s in SCENARIOS])
async def test_scenario(scenario: dict) -> None:
    result = await runner.run_scenario(scenario)
    assert result.passed, "\n".join(result.failures)
