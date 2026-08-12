from dataclasses import replace
from pathlib import Path
from threading import Thread
import time

from tokenos.budget import BudgetManager, estimate_usage_cost
from tokenos.models import ModelUsage, RunConfig


def config(**changes) -> RunConfig:
    base = RunConfig(
        run_id="run",
        strategies=("no-memory",),
        task_ids=("HumanEval/0",),
        dataset_path=str(Path("data.gz")),
        input_usd_per_mtok=1.0,
        cached_input_usd_per_mtok=0.1,
        output_usd_per_mtok=10.0,
        budget_usd=0.1,
        max_output_tokens=1000,
    )
    return replace(base, **changes)


def test_usage_cost_separates_cached_tokens() -> None:
    usage = ModelUsage(input_tokens=1000, cached_input_tokens=400, output_tokens=200)
    assert estimate_usage_cost(config(), usage) == 0.00264


def test_reservation_is_conservative_and_reconciled() -> None:
    budget = BudgetManager(config())
    reservation = budget.acquire("hello")
    assert reservation is not None
    assert budget.reserved_usd > 0
    actual = budget.reconcile(
        reservation, ModelUsage(input_tokens=10, output_tokens=10)
    )
    assert budget.reserved_usd == 0
    assert budget.committed_usd == actual


def test_hard_cap_rejects_call_that_cannot_fit() -> None:
    budget = BudgetManager(config(budget_usd=0.0001))
    assert budget.acquire("hello") is None


def test_waiting_reservation_rechecks_after_release() -> None:
    cfg = config(budget_usd=0.015, max_output_tokens=1000)
    budget = BudgetManager(cfg)
    first = budget.acquire("first")
    assert first is not None
    acquired = []

    def wait_for_budget() -> None:
        acquired.append(budget.acquire("second"))

    thread = Thread(target=wait_for_budget)
    thread.start()
    time.sleep(0.02)
    assert thread.is_alive()
    budget.release(first)
    thread.join(timeout=1)
    assert acquired[0] is not None

