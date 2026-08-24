from __future__ import annotations

from collections.abc import Sequence
from hashlib import sha256

from tokenos.models import AttemptRecord, FailureMemory, Patch

MAX_MEMORY_ITEM_CHARS = 600
MAX_MEMORY_SECTION_CHARS = 3000


def retrieve_memories(
    history: Sequence[AttemptRecord],
    *,
    max_chars: int = MAX_MEMORY_SECTION_CHARS,
) -> list[FailureMemory]:
    """Select distinct task-local memory within a bounded prompt budget."""
    distinct: dict[str, FailureMemory] = {}
    for attempt in sorted(history, key=lambda item: item.attempt):
        for memory in attempt.memories:
            if memory.validated and memory.task_id == attempt.task_id:
                distinct.setdefault(memory.fingerprint, memory)
        for patch in attempt.patches:
            if patch.validated and patch.task_id == attempt.task_id:
                fingerprint = sha256(patch.assertion.encode()).hexdigest()
                distinct.setdefault(
                    fingerprint,
                    FailureMemory(
                        task_id=patch.task_id,
                        kind="executable_constraint",
                        content=patch.assertion,
                        source_attempt=patch.source_attempt,
                        failure_kind=patch.failure_kind,
                        fingerprint=fingerprint,
                        validated=True,
                        provenance="legacy-v1",
                        explanation=patch.explanation,
                    ),
                )
    executable = sorted(
        (item for item in distinct.values() if item.kind == "executable_constraint"),
        key=lambda item: item.source_attempt,
    )
    summaries = sorted(
        (item for item in distinct.values() if item.kind == "failure_summary"),
        key=lambda item: item.source_attempt,
        reverse=True,
    )
    selected: list[FailureMemory] = []
    used = 0
    for memory in executable + summaries:
        if len(memory.content) > MAX_MEMORY_ITEM_CHARS:
            continue
        size = len(memory.content) + len(memory.explanation or "") + 32
        if used + size > max_chars:
            continue
        selected.append(memory)
        used += size
    return sorted(selected, key=lambda item: item.source_attempt)


def retrieve_patches(history: Sequence[AttemptRecord]) -> list[Patch]:
    """Retrieve validated task-local patches, removing duplicate assertions."""
    selected: dict[str, Patch] = {}
    for attempt in sorted(history, key=lambda item: item.attempt):
        for patch in attempt.patches:
            if patch.validated and patch.task_id == attempt.task_id:
                selected.setdefault(patch.assertion, patch)
    return list(selected.values())
