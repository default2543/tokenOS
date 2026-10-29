import json

import pytest

from tokenos.models import (
    AttemptRecord,
    EvaluationResult,
    FailureMemory,
    ModelUsage,
    Problem,
)
from tokenos.store import MemoryStore, keywords

PROBLEMS = {
    "B/1": Problem("B/1", "def task_func(server):\n  connect to an ftp server\n",
                   "task_func", "bigcodebench-hard"),
    "B/2": Problem("B/2", "def task_func(path):\n  archive image files\n",
                   "task_func", "bigcodebench-hard"),
}


def _memory(task_id, kind="failure_summary", content="x", validated=True,
            failure_kind="runtime_error", fingerprint=None):
    return FailureMemory(
        task_id=task_id,
        kind=kind,
        content=content,
        source_attempt=1,
        failure_kind=failure_kind,
        fingerprint=fingerprint or f"{task_id}:{content}",
        validated=validated,
    )


def _record(task_id, memories, strategy="patchsearch"):
    return AttemptRecord(
        run_id="run", task_id=task_id, strategy=strategy, attempt=1,
        prompt="p", completion="c", solution="s", response_id=None, request_id=None,
        usage=ModelUsage(), evaluation=EvaluationResult("fail", "fail", False),
        model_latency_seconds=0, evaluation_latency_seconds=0, total_latency_seconds=0,
        estimated_cost_usd=0, created_at="now", memories=tuple(memories),
    )


def test_keywords_drop_boilerplate_shared_by_every_task() -> None:
    # Every BigCodeBench task defines task_func, so it carries no signal.
    assert "task_func" not in keywords("def task_func(server): return None")
    assert "server" in keywords("def task_func(server): return None")


def test_store_excludes_the_asking_task_own_memories() -> None:
    store = MemoryStore.from_records(
        [_record("B/1", [_memory("B/1")]), _record("B/2", [_memory("B/2")])], PROBLEMS
    )
    assert [c.task_id for c in store.candidates("B/1")] == ["B/2"]


def test_executable_constraints_never_cross_task_boundaries() -> None:
    # Every task names its entry point task_func, so a constraint moved to
    # another task would assert against an unrelated function of that name.
    store = MemoryStore.from_records(
        [_record("B/2", [_memory("B/2", kind="executable_constraint",
                                 content="assert task_func(*(1,)) == 2")])],
        PROBLEMS,
    )
    assert store.candidates("B/1") == []


def test_unvalidated_memories_are_not_offered() -> None:
    store = MemoryStore.from_records(
        [_record("B/2", [_memory("B/2", validated=False)])], PROBLEMS
    )
    assert store.candidates("B/1") == []


def test_repeated_memories_do_not_win_the_budget_by_repetition() -> None:
    same = _memory("B/2", fingerprint="fp")
    store = MemoryStore.from_records(
        [_record("B/2", [same]), _record("B/2", [same]), _record("B/2", [same])],
        PROBLEMS,
    )
    assert len(store) == 1


def test_store_can_be_restricted_to_one_strategy() -> None:
    records = [
        _record("B/2", [_memory("B/2", content="a")], strategy="patchsearch"),
        _record("B/2", [_memory("B/2", content="b")], strategy="patchsearch-explained"),
    ]
    both = MemoryStore.from_records(records, PROBLEMS)
    one = MemoryStore.from_records(records, PROBLEMS, strategies=["patchsearch"])
    assert len(both) == 2
    assert len(one) == 1


def test_store_ordering_is_deterministic() -> None:
    records = [_record(f"B/{i}", [_memory(f"B/{i}", content=f"c{i}")]) for i in (2, 1)]
    first = MemoryStore.from_records(records, PROBLEMS).entries
    second = MemoryStore.from_records(list(reversed(records)), PROBLEMS).entries
    assert [e.memory.fingerprint for e in first] == [
        e.memory.fingerprint for e in second
    ]


def test_store_loads_from_a_completed_run(tmp_path) -> None:
    events = tmp_path / "events.jsonl"
    with events.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"type": "run_started"}) + "\n")
        handle.write(json.dumps({
            "type": "attempt_completed",
            "record": _record("B/2", [_memory("B/2", content="learned")]).to_dict(),
        }) + "\n")
    store = MemoryStore.from_run(events, PROBLEMS)
    assert [c.memory.content for c in store.candidates("B/1")] == ["learned"]


def test_missing_run_is_reported_clearly(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="run events not found"):
        MemoryStore.from_run(tmp_path / "absent.jsonl", PROBLEMS)


def test_source_keywords_come_from_the_producing_task() -> None:
    store = MemoryStore.from_records(
        [_record("B/2", [_memory("B/2", content="permission denied")])], PROBLEMS
    )
    entry = store.candidates("B/1")[0]
    assert "archive" in entry.source_keywords   # from B/2's prompt
    assert "permission" in entry.source_keywords  # from the memory itself
