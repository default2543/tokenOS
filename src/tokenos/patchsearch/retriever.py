from __future__ import annotations

from collections.abc import Sequence

from tokenos.models import AttemptRecord, Patch


def retrieve_patches(history: Sequence[AttemptRecord]) -> list[Patch]:
    """Retrieve validated task-local patches, removing duplicate assertions."""
    selected: dict[str, Patch] = {}
    for attempt in sorted(history, key=lambda item: item.attempt):
        for patch in attempt.patches:
            if patch.validated and patch.task_id == attempt.task_id:
                selected.setdefault(patch.assertion, patch)
    return list(selected.values())
