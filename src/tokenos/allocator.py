"""Heuristic token allocation.

With a cross-task store the candidate pool exceeds the retrieval budget, so
selection becomes a real decision rather than "include everything". This
implements the heuristic from the design:

    utility = (relevance x importance x confidence) / token_cost

Deliberately not learned. The point is to make allocation observable and
measurable first; a learned ranker can replace the scoring function later
without changing the selection or accounting around it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from tokenos.models import FailureMemory
from tokenos.store import StoredMemory

# A failure kind carries different weight as transferable evidence. A summary
# of a wrong answer describes this task's contract; a runtime error more often
# describes the environment, which is what actually generalises.
_IMPORTANCE_BY_KIND = {
    "runtime_error": 1.0,
    "timeout": 0.9,
    "wrong_answer": 0.7,
    "syntax_error": 0.5,
    "missing_entry_point": 0.5,
    "invalid_output": 0.2,
}
DEFAULT_IMPORTANCE = 0.5
MIN_RELEVANCE = 0.05


@dataclass(frozen=True, slots=True)
class Allocation:
    """One selected memory and the score that earned it its tokens."""

    memory: FailureMemory
    relevance: float
    importance: float
    utility: float
    cost_chars: int


def relevance(task_keywords: frozenset[str], entry: StoredMemory) -> float:
    """Share of a memory's vocabulary that the current task also uses."""
    if not entry.source_keywords:
        return 0.0
    shared = task_keywords & entry.source_keywords
    return len(shared) / len(entry.source_keywords)


def importance(memory: FailureMemory) -> float:
    return _IMPORTANCE_BY_KIND.get(memory.failure_kind, DEFAULT_IMPORTANCE)


def confidence(memory: FailureMemory) -> float:
    # Only validated memories are stored, so confidence currently reduces to
    # whether the evidence was complete rather than truncated.
    return 1.0 if memory.validated else 0.0


def cost_chars(memory: FailureMemory) -> int:
    return len(memory.content) + len(memory.explanation or "")


def score(task_keywords: frozenset[str], entry: StoredMemory) -> Allocation | None:
    """Score one candidate, or None when it is not worth any tokens."""
    match = relevance(task_keywords, entry)
    if match < MIN_RELEVANCE:
        return None
    weight = importance(entry.memory)
    certainty = confidence(entry.memory)
    cost = cost_chars(entry.memory)
    if certainty <= 0 or cost <= 0:
        return None
    return Allocation(
        memory=entry.memory,
        relevance=match,
        importance=weight,
        utility=(match * weight * certainty) / cost,
        cost_chars=cost,
    )


def allocate(
    task_keywords: frozenset[str],
    candidates: Sequence[StoredMemory],
    *,
    budget_chars: int,
    limit: int | None = None,
) -> list[Allocation]:
    """Select the highest-utility memories that fit the budget.

    Greedy by utility, which is the standard approximation for this shape of
    knapsack and is what the design calls for. Ties break on task id and
    fingerprint so a run is reproducible.
    """
    if budget_chars <= 0:
        return []
    scored = [
        allocation
        for allocation in (score(task_keywords, entry) for entry in candidates)
        if allocation is not None
    ]
    scored.sort(
        key=lambda item: (-item.utility, item.memory.task_id, item.memory.fingerprint)
    )
    selected: list[Allocation] = []
    used = 0
    for allocation in scored:
        if limit is not None and len(selected) >= limit:
            break
        if used + allocation.cost_chars > budget_chars:
            continue
        selected.append(allocation)
        used += allocation.cost_chars
    return selected
