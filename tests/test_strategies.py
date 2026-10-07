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


def _attempt(number: int, completion: str, kind: str, failed: int | None = None):
    from dataclasses import replace

    from tokenos.models import FailureMemory

    base = prior_attempt()
    memory = FailureMemory(
        task_id=PROBLEM.task_id,
        kind="failure_summary",
        content=f"{kind.upper()} at attempt {number}",
        source_attempt=number,
        failure_kind=kind,
        fingerprint=f"{kind}-{number}",
    )
    return replace(
        base,
        strategy="patchsearch-best-code",
        attempt=number,
        completion=completion,
        evaluation=EvaluationResult(
            base_status="fail",
            plus_status="fail",
            passed=False,
            feedback=FailureFeedback(
                kind=kind, message="failed", failed_test_count=failed
            ),
        ),
        memories=(memory,),
    )


def test_best_code_prefers_the_attempt_with_fewest_failing_tests() -> None:
    history = [
        _attempt(1, "    return 1\n", "wrong_answer", failed=1),
        _attempt(2, "    return 2\n", "wrong_answer", failed=3),
    ]
    prompt = build_prompt("patchsearch-best-code", PROBLEM, 3, history)
    assert "BEST PREVIOUS ATTEMPT (attempt 1, 1 failing test(s))" in prompt
    assert "return 1" in prompt and "return 2" not in prompt


def test_best_code_skips_unparsed_responses_and_their_memories() -> None:
    history = [
        _attempt(1, "    return 1\n", "runtime_error"),
        _attempt(2, '{"completion": "    ret', "invalid_output"),
    ]
    prompt = build_prompt("patchsearch-best-code", PROBLEM, 3, history)
    assert "BEST PREVIOUS ATTEMPT (attempt 1)" in prompt
    assert '{"completion"' not in prompt
    assert "INVALID_OUTPUT" not in prompt
    assert "RUNTIME_ERROR at attempt 1" in prompt


def test_best_code_breaks_ties_toward_the_latest_attempt() -> None:
    history = [
        _attempt(1, "    return 1\n", "wrong_answer", failed=2),
        _attempt(2, "    return 2\n", "wrong_answer", failed=2),
    ]
    prompt = build_prompt("patchsearch-best-code", PROBLEM, 3, history)
    assert "attempt 2, 2 failing" in prompt


def test_best_code_shows_no_code_when_every_response_was_unparsed() -> None:
    history = [_attempt(1, '{"compl', "invalid_output")]
    prompt = build_prompt("patchsearch-best-code", PROBLEM, 2, history)
    assert "PREVIOUS ATTEMPT" not in prompt
    assert '{"compl' not in prompt


def test_with_code_still_shows_the_latest_attempt_even_if_unparsed() -> None:
    # The measured strategy is unchanged, so recorded prompts still replay.
    history = [
        _attempt(1, "    return 1\n", "wrong_answer", failed=1),
        _attempt(2, '{"compl', "invalid_output"),
    ]
    prompt = build_prompt("patchsearch-with-code", PROBLEM, 3, history)
    assert "MOST RECENT ATTEMPT (attempt 2)" in prompt


def test_sliding_window_keeps_the_newest_attempts_that_fit() -> None:
    from tokenos.compiler import compile_context

    history = [
        _attempt(n, f"    return {n}  # {'x' * 300}\n", "wrong_answer", failed=1)
        for n in (1, 2, 3)
    ]
    unbounded = compile_context("sliding-window", PROBLEM, 4, history)
    assert unbounded.dropped_attempts == 0
    assert all(f"Attempt {n} completion" in unbounded.prompt for n in (1, 2, 3))

    one_less = unbounded.estimated_tokens - 1
    bounded = compile_context(
        "sliding-window", PROBLEM, 4, history, budget_tokens=one_less
    )
    assert bounded.dropped_attempts == 1
    assert "Attempt 1 completion" not in bounded.prompt
    assert "Attempt 3 completion" in bounded.prompt
    assert bounded.estimated_tokens <= one_less


def test_sliding_window_without_a_budget_matches_full_history() -> None:
    from tokenos.compiler import compile_context

    history = [_attempt(1, "    return 1\n", "wrong_answer", failed=1)]
    window = compile_context("sliding-window", PROBLEM, 2, history).prompt
    assert window == build_prompt("full-history", PROBLEM, 2, history)
