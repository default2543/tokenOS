from tokenos.models import (
    AttemptRecord,
    EvidenceObservation,
    EvaluationResult,
    FailureFeedback,
    FailureMemory,
    ModelUsage,
    Problem,
)
from tokenos.patchsearch.extractor import (
    MAX_EXPLANATION_CHARS,
    explain_failure,
    extract_patch,
    extract_memory,
)
from tokenos.patchsearch.retriever import retrieve_memories, retrieve_patches
from tokenos.strategies import build_prompt


PROBLEM = Problem("HumanEval/0", "def answer(value):\n", "answer")
FEEDBACK = FailureFeedback(
    kind="wrong_answer",
    message="mismatch",
    input_repr="(0,)",
    expected_repr="42",
    actual_repr="0",
)


def _attempt(number: int, patches=()) -> AttemptRecord:
    return AttemptRecord(
        run_id="run",
        task_id=PROBLEM.task_id,
        strategy="patchsearch",
        attempt=number,
        prompt="prompt",
        completion="    return 0\n",
        solution="def answer(value):\n    return 0\n",
        response_id="resp",
        request_id="req",
        usage=ModelUsage(),
        evaluation=EvaluationResult("fail", "fail", False, FEEDBACK),
        model_latency_seconds=0,
        evaluation_latency_seconds=0,
        total_latency_seconds=0,
        estimated_cost_usd=0,
        created_at="now",
        patches=tuple(patches),
    )


def test_extracts_validated_executable_patch() -> None:
    patch = extract_patch(PROBLEM, FEEDBACK, 1)
    assert patch is not None
    assert patch.assertion == "assert answer(*(0,)) == 42"
    compile(patch.assertion, "patch", "exec")
    assert patch.explanation is None


def test_explained_patch_contains_deterministic_wrong_answer_context() -> None:
    patch = extract_patch(PROBLEM, FEEDBACK, 1, include_explanation=True)
    assert patch is not None
    assert patch.explanation == "Previous candidate returned 0 but expected 42."


def test_extracts_runtime_and_timeout_oracle_patches() -> None:
    for kind in ("runtime_error", "timeout"):
        feedback = FailureFeedback(
            kind=kind,
            message="candidate failed",
            input_repr="(0,)",
            expected_repr="42",
        )
        patch = extract_patch(PROBLEM, feedback, 2)
        assert patch is not None
        assert patch.failure_kind == kind
        assert patch.assertion == "assert answer(*(0,)) == 42"


def test_explanations_cover_failure_kinds_and_missing_details() -> None:
    runtime = FailureFeedback(
        kind="runtime_error",
        message="bad\n  value",
        exception_type="ValueError",
    )
    timeout = FailureFeedback(kind="timeout", message="slow")
    missing_actual = FailureFeedback(kind="wrong_answer", message="mismatch")
    assert explain_failure(runtime) == (
        "Previous candidate raised ValueError: bad value."
    )
    assert explain_failure(timeout) == (
        "Previous candidate did not terminate on this input."
    )
    assert explain_failure(missing_actual) == (
        "Previous candidate produced an incorrect result."
    )


def test_explanation_is_single_line_and_bounded() -> None:
    feedback = FailureFeedback(
        kind="wrong_answer",
        message="mismatch",
        actual_repr="line one\n" + "x" * 300,
    )
    explanation = explain_failure(feedback)
    assert "\n" not in explanation
    assert len(explanation) == MAX_EXPLANATION_CHARS
    assert explanation.endswith("...")


def test_rejects_non_oracle_and_non_literal_feedback() -> None:
    runtime = FailureFeedback(
        kind="runtime_error", message="boom", input_repr="(0,)"
    )
    unsafe = FailureFeedback(
        kind="wrong_answer",
        message="bad",
        input_repr="(__import__('os'),)",
        expected_repr="42",
    )
    assert extract_patch(PROBLEM, runtime, 1) is None
    assert extract_patch(PROBLEM, unsafe, 1) is None


def test_rejects_ineligible_truncated_and_malformed_feedback() -> None:
    cases = [
        FailureFeedback(
            kind="syntax_error",
            message="bad syntax",
            input_repr="(0,)",
            expected_repr="42",
        ),
        FailureFeedback(
            kind="timeout", message="slow", input_repr="(0,)"
        ),
        FailureFeedback(
            kind="runtime_error",
            message="boom",
            input_repr="(0,)",
            expected_repr="'value...<truncated>'",
        ),
        FailureFeedback(
            kind="timeout",
            message="slow",
            input_repr="[0]",
            expected_repr="42",
        ),
    ]
    assert all(extract_patch(PROBLEM, feedback, 1) is None for feedback in cases)


def test_retrieval_deduplicates_and_prompt_excludes_full_history() -> None:
    patch = extract_patch(PROBLEM, FEEDBACK, 1)
    assert patch is not None
    history = [_attempt(1, [patch]), _attempt(2, [patch])]
    assert retrieve_patches(history) == [patch]
    prompt = build_prompt("patchsearch", PROBLEM, 3, history)
    assert prompt.count(patch.assertion) == 1
    assert "return 0" not in prompt
    assert "Actual: 0" not in prompt


def test_explained_prompt_includes_only_bounded_memory() -> None:
    patch = extract_patch(PROBLEM, FEEDBACK, 1, include_explanation=True)
    assert patch is not None
    history = [_attempt(1, [patch])]
    prompt = build_prompt("patchsearch-explained", PROBLEM, 2, history)
    assert f"Explanation: {patch.explanation}" in prompt
    assert f"Constraint: {patch.assertion}" in prompt
    assert "return 0" not in prompt
    assert "Actual: 0" not in prompt


def test_old_patch_artifact_without_explanation_is_readable() -> None:
    patch = extract_patch(PROBLEM, FEEDBACK, 1)
    assert patch is not None
    payload = _attempt(1, [patch]).to_dict()
    payload["patches"][0].pop("explanation")
    restored = AttemptRecord.from_dict(payload)
    assert restored.patches[0].explanation is None
    assert restored.memories[0].kind == "executable_constraint"


def test_non_oracle_failure_always_becomes_bounded_summary() -> None:
    problem = Problem("BigCodeBench/1", "def answer():\n", "answer", benchmark="bigcodebench-hard")
    evidence = FailureFeedback(
        kind="runtime_error",
        message="bad value " * 200,
        benchmark="bigcodebench-hard",
        test_id="test_bad_value",
        observations=(EvidenceObservation("exception", "ValueError"),),
    )
    memory = extract_memory(problem, evidence, 1)
    assert memory is not None
    assert memory.kind == "failure_summary"
    assert len(memory.content) <= 600
    assert "test_bad_value" in memory.content


def test_untrusted_or_malicious_call_evidence_cannot_become_executable() -> None:
    evidence = FailureFeedback(
        kind="wrong_answer",
        message="mismatch",
        observations=(
            EvidenceObservation("function_call", "(__import__('os').system('id'),)", True),
            EvidenceObservation("expected_value", "0", True),
        ),
    )
    memory = extract_memory(PROBLEM, evidence, 1)
    assert memory is not None
    assert memory.kind == "failure_summary"
    assert "__import__" not in memory.content


def test_retrieval_prefers_constraints_and_obeys_total_budget() -> None:
    records = []
    for number in range(1, 9):
        evidence = FailureFeedback(kind="runtime_error", message="x" * 500)
        memory = extract_memory(PROBLEM, evidence, number)
        assert memory is not None
        record = _attempt(number)
        records.append(record.__class__(**{**record.to_dict(), "usage": record.usage,
            "evaluation": record.evaluation, "patches": (), "memories": (memory,),
            "retrieved_memories": ()}))
    selected = retrieve_memories(records, max_chars=1000)
    assert sum(len(item.content) for item in selected) <= 1000
    assert len({item.fingerprint for item in selected}) == len(selected)


def test_assertion_diff_is_not_reported_as_a_returned_value() -> None:
    problem = Problem(
        "BigCodeBench/13", "def task_func(server):\n", "task_func", "bigcodebench-hard"
    )
    diff = "'a to FTP server' != 'a to FTP server invalid_server: a to FTP server'"
    evidence = FailureFeedback(
        kind="wrong_answer",
        message="unittest failure",
        benchmark="bigcodebench-hard",
        test_id="test_case_2",
        observations=(EvidenceObservation("assertion_diff", diff),),
    )
    memory = extract_memory(problem, evidence, 1)
    assert memory is not None
    assert memory.kind == "failure_summary"
    assert "failed an assertion" in memory.content
    assert "returned" not in memory.content
    assert diff in memory.content


def test_assertion_diff_never_becomes_an_executable_constraint() -> None:
    evidence = FailureFeedback(
        kind="wrong_answer",
        message="unittest failure",
        observations=(
            EvidenceObservation("assertion_diff", "0 != 42"),
            EvidenceObservation("test_name", "test_case_1"),
        ),
    )
    memory = extract_memory(PROBLEM, evidence, 1)
    assert memory is not None
    assert memory.kind == "failure_summary"
    assert extract_patch(PROBLEM, evidence, 1) is None


def test_summary_fallback_may_use_the_full_memory_budget() -> None:
    # The one-line explanation bound is for captions; a summary fallback is the
    # only evidence a summary-only benchmark provides.
    evidence = FailureFeedback(
        kind="wrong_answer",
        message="unittest failure",
        observations=(EvidenceObservation("assertion_diff", "x" * 900),),
    )
    memory = extract_memory(PROBLEM, evidence, 1)
    assert memory is not None
    assert MAX_EXPLANATION_CHARS < len(memory.content) <= 600


def test_actual_value_still_wins_over_an_assertion_diff() -> None:
    evidence = FailureFeedback(
        kind="wrong_answer",
        message="mismatch",
        observations=(
            EvidenceObservation("actual_value", "0"),
            EvidenceObservation("assertion_diff", "0 != 42"),
        ),
    )
    assert explain_failure(evidence) == "Previous candidate returned 0."


def _crosstask_store():
    from tokenos.store import MemoryStore
    other = Problem("HumanEval/9", "def other(path):\n  archive files\n", "other")
    record = AttemptRecord(
        run_id="run", task_id="HumanEval/9", strategy="patchsearch", attempt=1,
        prompt="p", completion="c", solution="s", response_id=None, request_id=None,
        usage=ModelUsage(), evaluation=EvaluationResult("fail", "fail", False),
        model_latency_seconds=0, evaluation_latency_seconds=0,
        total_latency_seconds=0, estimated_cost_usd=0, created_at="now",
        memories=(FailureMemory(
            task_id="HumanEval/9", kind="failure_summary",
            content="RUNTIME ERROR: archive path did not exist.", source_attempt=1,
            failure_kind="runtime_error", fingerprint="fp-other"),),
    )
    return MemoryStore.from_records([record], {"HumanEval/9": other})


def test_task_local_memory_is_kept_separate_from_transferred_memory() -> None:
    from tokenos.patchsearch.retriever import retrieve_with_store
    problem = Problem("HumanEval/0", "def answer(archive):\n  archive files\n", "answer")
    memory = extract_memory(problem, FEEDBACK, 1)
    record = _attempt(1)
    record = AttemptRecord(**{**record.to_dict(), "usage": record.usage,
        "evaluation": record.evaluation, "patches": (), "memories": (memory,),
        "retrieved_memories": ()})
    local, transferred = retrieve_with_store(problem, [record], _crosstask_store())
    assert [m.task_id for m in local] == ["HumanEval/0"]
    assert [m.task_id for m in transferred] == ["HumanEval/9"]


def test_transferred_memory_is_labelled_by_its_source_task() -> None:
    problem = Problem("HumanEval/0", "def answer(archive):\n  archive files\n", "answer")
    from tokenos.patchsearch.retriever import retrieve_with_store
    _, transferred = retrieve_with_store(problem, [], _crosstask_store())
    prompt = build_prompt("patchsearch-crosstask", problem, 2, [],
                          memories=[], transferred=transferred)
    assert "RELATED FAILURES FROM OTHER TASKS" in prompt
    assert "HumanEval/9:" in prompt
    # The model must not read another task's failure as a requirement here.
    assert "never as requirements for this task" in prompt


def test_no_store_means_no_transferred_memory() -> None:
    from tokenos.patchsearch.retriever import retrieve_with_store
    local, transferred = retrieve_with_store(PROBLEM, [], None)
    assert transferred == []
