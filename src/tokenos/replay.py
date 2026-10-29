"""Offline analysis of recorded runs: context growth, budget sweeps, stopping.

All read a completed run's append-only records and spend no tokens. Growth
uses API-reported input tokens, so it is measured rather than estimated. A
sweep recompiles every recorded bounded-memory attempt under candidate budgets
to show what each budget would have cut before paying to find out. A stop
simulation truncates trajectories where an early-stopping rule would have
ended them; because stopping only removes later attempts, it is exact.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence

from tokenos.compiler import ContextBudgetError, compile_context
from tokenos.models import AttemptRecord, Problem
from tokenos.patchsearch.retriever import retrieve_with_store
from tokenos.strategies import (
    PATCHSEARCH_STRATEGIES,
    stuck_streak,
    system_instructions,
)


def context_growth(records: Iterable[AttemptRecord]) -> dict[str, list[dict]]:
    """Mean and peak input tokens at each attempt number, per strategy."""
    grouped: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        grouped[record.strategy][record.attempt].append(record.usage.input_tokens)
    return {
        strategy: [
            {
                "attempt": attempt,
                "requests": len(values),
                "mean_input_tokens": round(sum(values) / len(values)),
                "max_input_tokens": max(values),
            }
            for attempt, values in sorted(by_attempt.items())
        ]
        for strategy, by_attempt in sorted(grouped.items())
    }


def _trajectories(
    records: Iterable[AttemptRecord],
) -> dict[tuple[str, str], list[AttemptRecord]]:
    grouped: dict[tuple[str, str], list[AttemptRecord]] = defaultdict(list)
    for record in records:
        # Cross-task runs depend on a store that is not part of the run.
        if record.strategy in PATCHSEARCH_STRATEGIES and (
            record.strategy != "patchsearch-crosstask"
        ):
            grouped[(record.strategy, record.task_id)].append(record)
    for items in grouped.values():
        items.sort(key=lambda item: item.attempt)
    return grouped


def budget_sweep(
    records: Iterable[AttemptRecord],
    problems: dict[str, Problem],
    budgets: Sequence[int | None],
) -> dict[str, list[dict]]:
    """Recompile each recorded attempt under each budget.

    The unbounded row doubles as a fidelity check: `reproduced` counts attempts
    whose recompiled prompt is byte-identical to the one actually sent, and
    `estimate_covers` those whose estimate is at least the API-reported count.
    """
    totals: dict[str, dict[int | None, dict[str, int]]] = defaultdict(
        lambda: {budget: defaultdict(int) for budget in budgets}
    )
    for (strategy, task_id), items in sorted(_trajectories(records).items()):
        problem = problems.get(task_id)
        if problem is None:
            continue
        instructions = system_instructions(problem.benchmark)
        for index, record in enumerate(items):
            history = items[:index]
            memories, transferred = retrieve_with_store(problem, history, None)
            for budget in budgets:
                row = totals[strategy][budget]
                try:
                    context = compile_context(
                        strategy,
                        problem,
                        record.attempt,
                        history,
                        memories=memories,
                        transferred=transferred,
                        budget_tokens=budget,
                        instructions=instructions,
                    )
                except ContextBudgetError:
                    row["unfit"] += 1
                    continue
                row["attempts"] += 1
                row["estimated_tokens"] += context.estimated_tokens
                row["trimmed"] += context.trimmed
                row["dropped_code"] += context.dropped_code
                row["dropped_memories"] += len(context.dropped_memories)
                if budget is None:
                    row["reproduced"] += context.prompt == record.prompt
                    row["estimate_covers"] += (
                        context.estimated_tokens >= record.usage.input_tokens
                    )
    fields = ("attempts", "unfit", "trimmed", "dropped_code", "dropped_memories",
              "estimated_tokens")
    return {
        strategy: [
            {
                "budget_tokens": budget,
                **{key: row[key] for key in fields},
                **(
                    {key: row[key] for key in ("reproduced", "estimate_covers")}
                    if budget is None
                    else {}
                ),
            }
            for budget, row in by_budget.items()
        ]
        for strategy, by_budget in totals.items()
    }


def stop_simulation(
    records: Iterable[AttemptRecord], limits: Sequence[int]
) -> dict[str, list[dict]]:
    """What stopping after K identical failures in a row would have changed.

    Each recorded trajectory is cut at the first failed attempt whose stuck
    streak reaches K -- the same rule `--stop-after-identical K` enforces live.
    Attempts before the cut happened exactly as recorded, so solves kept and
    tokens spent are measured, not estimated.
    """
    result: dict[str, list[dict]] = {}
    for strategy, task_items in _by_strategy(_trajectories(records)).items():
        rows = [_stop_row(task_items, None)]
        rows.extend(_stop_row(task_items, limit) for limit in limits)
        result[strategy] = rows
    return result


def _by_strategy(
    trajectories: dict[tuple[str, str], list[AttemptRecord]],
) -> dict[str, list[list[AttemptRecord]]]:
    grouped: dict[str, list[list[AttemptRecord]]] = defaultdict(list)
    for (strategy, _), items in sorted(trajectories.items()):
        grouped[strategy].append(items)
    return grouped


def _stop_row(trajectories: list[list[AttemptRecord]], limit: int | None) -> dict:
    row = {"stop_after_identical": limit, "trajectories": len(trajectories),
           "solved": 0, "abandoned": 0, "attempts": 0, "output_tokens": 0,
           "input_tokens": 0, "estimated_cost_usd": 0.0}
    for items in trajectories:
        for index, record in enumerate(items):
            row["attempts"] += 1
            row["output_tokens"] += record.usage.output_tokens
            row["input_tokens"] += record.usage.input_tokens
            row["estimated_cost_usd"] += record.estimated_cost_usd
            if record.evaluation.passed:
                row["solved"] += 1
                break
            if limit is not None and stuck_streak(items[: index + 1]) >= limit:
                row["abandoned"] += index + 1 < len(items)
                break
    row["estimated_cost_usd"] = round(row["estimated_cost_usd"], 6)
    return row


def export_workload(records: Iterable[AttemptRecord], instructions: str) -> list[dict]:
    """Every recorded request as a replayable serving workload.

    One row per request, in the order a trajectory sent them, carrying the
    exact prompt and system instructions plus what the API reported, so a
    self-hosted server can replay the real traffic shape of each strategy.
    """
    rows = []
    for record in sorted(
        records, key=lambda item: (item.strategy, item.task_id, item.attempt)
    ):
        rows.append(
            {
                "run_id": record.run_id,
                "strategy": record.strategy,
                "task_id": record.task_id,
                "attempt": record.attempt,
                "instructions": instructions,
                "prompt": record.prompt,
                "recorded_input_tokens": record.usage.input_tokens,
                "recorded_output_tokens": record.usage.output_tokens,
                "passed": record.evaluation.passed,
            }
        )
    return rows
