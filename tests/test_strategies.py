from tokenos.models import (
    AttemptRecord,
    EvaluationResult,
    FailureFeedback,
    ModelUsage,
    Problem,
)
from tokenos.strategies import build_prompt, parse_completion


PROBLEM = Problem("HumanEval/0", "def answer():\n    \"\"\"Return 42.\"\"\"\n", "answer")


def prior_attempt() -> AttemptRecord:
    return AttemptRecord(
        run_id="run",
        task_id=PROBLEM.task_id,
        strategy="full-history",
        attempt=1,
        prompt="old prompt",
        completion="    return 0\n",
        solution=PROBLEM.prompt + "    return 0\n",
        response_id="resp",
        request_id="req",
        usage=ModelUsage(input_tokens=10, output_tokens=5),
        evaluation=EvaluationResult(
            base_status="fail",
            plus_status="fail",
            passed=False,
            feedback=FailureFeedback(
                kind="wrong_answer",
                message="mismatch",
                input_repr="()",
                expected_repr="42",
                actual_repr="0",
            ),
        ),
        model_latency_seconds=1,
        evaluation_latency_seconds=1,
        total_latency_seconds=2,
        estimated_cost_usd=0.01,
        created_at="now",
    )


def test_no_memory_excludes_history() -> None:
    prompt = build_prompt("no-memory", PROBLEM, 2, [prior_attempt()])
    assert "return 0" not in prompt
    assert "Expected: 42" not in prompt
    assert "Previous attempts are unavailable" in prompt


def test_full_history_includes_completion_and_one_feedback() -> None:
    prompt = build_prompt("full-history", PROBLEM, 2, [prior_attempt()])
    assert "return 0" in prompt
    assert "Expected: 42" in prompt
    assert "Actual: 0" in prompt


def test_parse_structured_completion() -> None:
    assert parse_completion('{"completion":"    return 42"}') == "    return 42\n"


def test_parse_completion_rejects_extra_fields() -> None:
    try:
        parse_completion('{"completion":"x", "explanation":"no"}')
    except ValueError as exc:
        assert "exactly" in str(exc)
    else:
        raise AssertionError("invalid structured output was accepted")

