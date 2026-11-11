from tokenos.allocator import MIN_RELEVANCE, allocate, importance, relevance, score
from tokenos.models import FailureMemory
from tokenos.store import StoredMemory


def _entry(task_id, words, content="x" * 100, failure_kind="runtime_error"):
    return StoredMemory(
        memory=FailureMemory(
            task_id=task_id,
            kind="failure_summary",
            content=content,
            source_attempt=1,
            failure_kind=failure_kind,
            fingerprint=f"{task_id}:{content[:8]}",
        ),
        source_keywords=frozenset(words),
        strategy="patchsearch",
    )


def test_relevance_is_the_shared_share_of_a_memory_vocabulary() -> None:
    entry = _entry("B/2", {"ftp", "socket", "timeout", "retry"})
    assert relevance(frozenset({"ftp", "socket"}), entry) == 0.5
    assert relevance(frozenset(), entry) == 0.0


def test_a_memory_with_no_vocabulary_overlap_earns_no_tokens() -> None:
    entry = _entry("B/2", {"matplotlib", "axes"})
    assert score(frozenset({"ftp", "socket"}), entry) is None


def test_environment_failures_outrank_contract_failures() -> None:
    # A runtime error more often describes the environment, which is what
    # actually transfers; a wrong answer describes one task's contract.
    assert importance(_entry("B/1", set(), failure_kind="runtime_error").memory) > (
        importance(_entry("B/1", set(), failure_kind="wrong_answer").memory)
    )
    assert importance(_entry("B/1", set(), failure_kind="invalid_output").memory) < (
        importance(_entry("B/1", set(), failure_kind="wrong_answer").memory)
    )


def test_allocation_never_exceeds_the_budget() -> None:
    task = frozenset({"ftp"})
    entries = [_entry(f"B/{i}", {"ftp"}, content="x" * 100) for i in range(20)]
    selected = allocate(task, entries, budget_chars=450)
    assert sum(a.cost_chars for a in selected) <= 450
    assert len(selected) == 4


def test_a_zero_budget_selects_nothing() -> None:
    assert allocate(frozenset({"ftp"}), [_entry("B/1", {"ftp"})], budget_chars=0) == []


def test_cheaper_memory_wins_when_relevance_ties() -> None:
    # utility divides by token cost, so the same evidence for fewer tokens wins.
    task = frozenset({"ftp"})
    cheap = _entry("B/1", {"ftp"}, content="x" * 50)
    dear = _entry("B/2", {"ftp"}, content="y" * 400)
    selected = allocate(task, [dear, cheap], budget_chars=100)
    assert [a.memory.task_id for a in selected] == ["B/1"]


def test_more_relevant_memory_wins_when_cost_ties() -> None:
    task = frozenset({"ftp", "socket"})
    weak = _entry("B/1", {"ftp", "a", "b", "c"}, content="x" * 100)
    strong = _entry("B/2", {"ftp", "socket"}, content="y" * 100)
    selected = allocate(task, [weak, strong], budget_chars=100)
    assert [a.memory.task_id for a in selected] == ["B/2"]


def test_selection_is_order_independent() -> None:
    task = frozenset({"ftp", "socket"})
    entries = [
        _entry("B/1", {"ftp"}, content="x" * 80),
        _entry("B/2", {"socket", "ftp"}, content="y" * 120),
        _entry("B/3", {"ftp", "a", "b"}, content="z" * 60),
    ]
    forward = allocate(task, entries, budget_chars=200)
    backward = allocate(task, list(reversed(entries)), budget_chars=200)
    assert [a.memory.fingerprint for a in forward] == [
        a.memory.fingerprint for a in backward
    ]


def test_limit_caps_the_number_of_memories() -> None:
    entries = [_entry(f"B/{i}", {"ftp"}, content="x" * 10) for i in range(10)]
    assert len(allocate(frozenset({"ftp"}), entries, budget_chars=9999, limit=3)) == 3


def test_relevance_floor_is_enforced() -> None:
    entry = _entry("B/2", {f"w{i}" for i in range(100)} | {"ftp"})
    assert relevance(frozenset({"ftp"}), entry) < MIN_RELEVANCE
    assert allocate(frozenset({"ftp"}), [entry], budget_chars=9999) == []
