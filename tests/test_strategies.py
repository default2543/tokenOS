from tokenos.models import (
    AttemptRecord,
    EvaluationResult,
    EvidenceObservation,
    FailureFeedback,
    ModelUsage,
    Problem,
)
from tokenos.strategies import (
    SYSTEM_INSTRUCTIONS,
    _feedback_text,
    build_prompt,
    parse_completion,
    system_instructions,
)


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



def test_bigcodebench_instructions_permit_what_its_tasks_require() -> None:
    # The HumanEval instructions forbid file, network, and I/O work, which is
    # what most BigCodeBench-Hard reference solutions do.
    humaneval = system_instructions("humaneval-plus")
    library = system_instructions("bigcodebench-hard")
    assert "access the network" in humaneval
    assert library != humaneval
    assert "permitted" in library


def test_unknown_benchmark_falls_back_to_the_strict_instructions() -> None:
    assert system_instructions("mbpp-plus") == SYSTEM_INSTRUCTIONS


def test_assertion_diff_is_labelled_in_full_history_feedback() -> None:
    feedback = FailureFeedback(
        kind="wrong_answer",
        message="unittest failure",
        observations=(EvidenceObservation("assertion_diff", "0 != 42"),),
    )
    text = _feedback_text(feedback)
    assert "Assertion detail: 0 != 42" in text


def test_with_code_variant_carries_only_the_most_recent_completion() -> None:
    # Isolates one variable against full-history: the prior code, not the
    # richer feedback and not every earlier attempt.
    first = prior_attempt()
    second = AttemptRecord(**{**first.to_dict(), "usage": first.usage,
        "evaluation": first.evaluation, "attempt": 2,
        "completion": "    return 99\n"})
    prompt = build_prompt("patchsearch-with-code", PROBLEM, 3, [first, second])
    assert "MOST RECENT ATTEMPT (attempt 2)" in prompt
    assert "return 99" in prompt
    assert "return 0" not in prompt


def test_plain_patchsearch_still_carries_no_code() -> None:
    prompt = build_prompt("patchsearch", PROBLEM, 2, [prior_attempt()])
    assert "MOST RECENT ATTEMPT" not in prompt
    assert "return 0" not in prompt


def test_with_code_variant_has_no_code_on_the_first_attempt() -> None:
    prompt = build_prompt("patchsearch-with-code", PROBLEM, 1, [])
    assert "MOST RECENT ATTEMPT" not in prompt
