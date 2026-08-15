from tokenos.models import (
    AttemptRecord,
    EvaluationResult,
    FailureFeedback,
    ModelUsage,
    Problem,
)
from tokenos.patchsearch.extractor import extract_patch
from tokenos.patchsearch.retriever import retrieve_patches
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
