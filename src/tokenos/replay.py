"""Offline analysis of recorded runs: context growth and budget sweeps.

Both read a completed run's append-only records and spend no tokens. Growth
uses API-reported input tokens, so it is measured rather than estimated. A
sweep recompiles every recorded bounded-memory attempt under candidate budgets
to show what each budget would have cut before paying to find out.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence

from tokenos.compiler import ContextBudgetError, compile_context
from tokenos.models import AttemptRecord, Problem
from tokenos.patchsearch.retriever import retrieve_with_store
from tokenos.strategies import PATCHSEARCH_STRATEGIES, system_instructions


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
