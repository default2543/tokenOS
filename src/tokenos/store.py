"""Cross-task memory store.

Task-local retrieval can only ever surface the handful of memories one task's
own attempts produced, so the retrieval budget never binds and an allocator has
nothing to allocate. This store accumulates memories across tasks and offers
them as candidates for another task's attempts.

Two constraints shape the design.

Determinism: a benchmark comparison cannot depend on thread scheduling, so the
store is *frozen*. It is built from a completed run's artifacts and does not
change while a run reads it. Online accumulation would make a task's context
depend on which other tasks happened to finish first.

Safety: an executable constraint names a specific function. Every
BigCodeBench task defines `task_func`, so a constraint learned on one task
would assert against an unrelated function of the same name and be silently
false. Only summaries cross task boundaries.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import re

from tokenos.models import AttemptRecord, FailureMemory, Problem

_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]{3,}")

# Boilerplate that appears in nearly every task prompt or failure summary and
# would otherwise make every memory look relevant to every task. `task_func` is
# the entry point of every BigCodeBench task.
_STOPWORDS = frozenset(
    """
    task_func self none true false return returns param params parameter
    parameters example examples requirements note notes args kwargs type types
    value values object objects test tests case cases previous candidate raised
    failed failure assertion error exception result results output input given
    should must will with without from that this than then they them their there
    which what when where while have base plus wrong answer runtime invalid
    """.split()
)


def keywords(*texts: str) -> frozenset[str]:
    """Extract a deterministic, dependency-free relevance signal."""
    found: set[str] = set()
    for text in texts:
        found.update(word.lower() for word in _WORD.findall(text))
    return frozenset(found) - _STOPWORDS


@dataclass(frozen=True, slots=True)
class StoredMemory:
    """A memory plus the vocabulary of the task that produced it."""

    memory: FailureMemory
    source_keywords: frozenset[str]
    strategy: str

    @property
    def task_id(self) -> str:
        return self.memory.task_id


class MemoryStore:
    """A frozen, benchmark-neutral pool of memories drawn from other tasks."""

    def __init__(self, entries: Iterable[StoredMemory] = ()) -> None:
        # Deduplicate by fingerprint so a memory repeated across attempts does
        # not win the budget by appearing many times.
        unique: dict[str, StoredMemory] = {}
        for entry in entries:
            unique.setdefault(entry.memory.fingerprint, entry)
        self._entries = sorted(
            unique.values(), key=lambda item: (item.task_id, item.memory.fingerprint)
        )

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def entries(self) -> tuple[StoredMemory, ...]:
        return tuple(self._entries)

    def candidates(self, task_id: str) -> list[StoredMemory]:
        """Memories safe to offer to `task_id`.

        Excludes the task's own memories, which the task-local retriever already
        supplies, and executable constraints, which do not survive the move to a
        different function of the same name.
        """
        return [
            entry
            for entry in self._entries
            if entry.task_id != task_id
            and entry.memory.kind != "executable_constraint"
            and entry.memory.validated
        ]

    @classmethod
    def from_records(
        cls,
        records: Sequence[AttemptRecord],
        problems: dict[str, Problem],
        *,
        strategies: Sequence[str] | None = None,
    ) -> "MemoryStore":
        allowed = None if strategies is None else set(strategies)
        entries: list[StoredMemory] = []
        for record in records:
            if allowed is not None and record.strategy not in allowed:
                continue
            problem = problems.get(record.task_id)
            source = problem.prompt if problem is not None else ""
            for memory in record.memories:
                entries.append(
                    StoredMemory(
                        memory=memory,
                        source_keywords=keywords(source, memory.content),
                        strategy=record.strategy,
                    )
                )
        return cls(entries)

    @classmethod
    def from_run(
        cls,
        events_path: Path,
        problems: dict[str, Problem],
        *,
        strategies: Sequence[str] | None = None,
    ) -> "MemoryStore":
        """Build a store from a completed run's append-only event log."""
        if not events_path.is_file():
            raise FileNotFoundError(f"run events not found: {events_path}")
        records: list[AttemptRecord] = []
        with events_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                event = json.loads(line)
                if event.get("type") != "attempt_completed":
                    continue
                records.append(AttemptRecord.from_dict(event["record"]))
        return cls.from_records(records, problems, strategies=strategies)
