from __future__ import annotations

import json
from typing import Sequence

from tokenos.models import (
    AttemptRecord,
    FailureFeedback,
    FailureMemory,
    Patch,
    Problem,
    StrategyName,
)
from tokenos.patchsearch.retriever import retrieve_memories

PATCHSEARCH_STRATEGIES = frozenset(
    {
        "patchsearch",
        "patchsearch-explained",
        "patchsearch-crosstask",
        "patchsearch-with-code",
        "patchsearch-best-code",
        "patchsearch-adaptive",
    }
)
# Strategies that show the model one earlier implementation alongside memory.
CODE_STRATEGIES = frozenset(
    {"patchsearch-with-code", "patchsearch-best-code", "patchsearch-adaptive"}
)
# Raw test output carried by patchsearch-adaptive. Summaries are capped at 600
# characters and lose the traceback detail full history keeps.
MAX_RAW_FEEDBACK_CHARS = 1500


def shown_attempt(
    strategy: StrategyName, history: Sequence[AttemptRecord]
) -> AttemptRecord | None:
    """The earlier implementation a code-carrying strategy shows, if any.

    `patchsearch-with-code` shows the latest attempt, whatever it was.
    `patchsearch-best-code` skips responses that never parsed -- their
    "completion" is a truncated JSON fragment, not code -- and prefers the
    attempt that failed the fewest tests, so a retry that fixes one test and
    breaks another does not discard the better version. Ties go to the latest.
    """
    if strategy not in CODE_STRATEGIES or not history:
        return None
    if strategy == "patchsearch-with-code":
        return max(history, key=lambda item: item.attempt)
    valid = [item for item in history if _failure_kind(item) != "invalid_output"]
    if not valid:
        return None
    if strategy == "patchsearch-adaptive":
        return max(valid, key=lambda item: item.attempt)

    def rank(item: AttemptRecord) -> tuple[int, int, int]:
        count = _failed_test_count(item)
        if count is None:
            return (1, 0, -item.attempt)
        return (0, count, -item.attempt)

    return min(valid, key=rank)


def strategy_memories(
    strategy: StrategyName, memories: Sequence[FailureMemory]
) -> list[FailureMemory]:
    """Memories a strategy shows. "The response was not valid JSON" says
    nothing about the task, so best-code drops it rather than spend a slot."""
    if strategy in ("patchsearch-best-code", "patchsearch-adaptive"):
        return [item for item in memories if item.failure_kind != "invalid_output"]
    return list(memories)


def repeat_counts(history: Sequence[AttemptRecord]) -> dict[str, int]:
    """How many attempts produced each memory, by fingerprint."""
    counts: dict[str, int] = {}
    for item in history:
        for memory in item.memories:
            counts[memory.fingerprint] = counts.get(memory.fingerprint, 0) + 1
    return counts


def stuck_streak(history: Sequence[AttemptRecord]) -> int:
    """Consecutive latest attempts that failed exactly as the latest did.

    Retrieval deduplicates identical failures, which is what keeps memory
    bounded -- but it also hides that the model is repeating itself. Full
    history shows the repetition; this restores it in one number.
    """
    ordered = sorted(history, key=lambda item: item.attempt)
    if not ordered or not ordered[-1].memories:
        return 0
    latest = ordered[-1].memories[0].fingerprint
    streak = 0
    for item in reversed(ordered):
        if not item.memories or item.memories[0].fingerprint != latest:
            break
        streak += 1
    return streak


def _bounded_feedback(item: AttemptRecord) -> str:
    text = _feedback_text(item.evaluation.feedback)
    if len(text) <= MAX_RAW_FEEDBACK_CHARS:
        return text
    return text[: MAX_RAW_FEEDBACK_CHARS - 3].rstrip() + "..."


def _failure_kind(item: AttemptRecord) -> str | None:
    feedback = item.evaluation.feedback
    return None if feedback is None else feedback.kind


def _failed_test_count(item: AttemptRecord) -> int | None:
    feedback = item.evaluation.feedback
    if feedback is None or feedback.kind != "wrong_answer":
        return None
    return getattr(feedback, "failed_test_count", None)

SYSTEM_INSTRUCTIONS = """You solve standalone Python function-completion tasks.
Return a JSON object with exactly one field named completion. Its value must be the
Python source that follows the supplied prompt. Do not repeat the prompt, use Markdown,
read files, access the network, or perform input/output. Preserve the required function
signature and implement the complete function body."""

# BigCodeBench-Hard tasks are library-composition problems whose reference
# solutions read files, spawn subprocesses, and call network clients. The
# HumanEval instructions forbid exactly that, so they cannot be shared.
LIBRARY_TASK_INSTRUCTIONS = """You solve Python function-completion tasks that compose
standard-library and third-party APIs. Return a JSON object with exactly one field named
completion. Its value must be the Python source that follows the supplied prompt. Do not
repeat the prompt or use Markdown. Preserve the required function signature and implement
the complete function body. File, subprocess, and network operations are permitted when
the task requires them; write any imports the body needs. Match the documented return
value, raised exception types, and side effects exactly."""

_BENCHMARK_INSTRUCTIONS = {
    "humaneval-plus": SYSTEM_INSTRUCTIONS,
    "bigcodebench-hard": LIBRARY_TASK_INSTRUCTIONS,
}


def system_instructions(benchmark: str) -> str:
    """Return the system prompt matching a benchmark's task shape."""
    return _BENCHMARK_INSTRUCTIONS.get(benchmark, SYSTEM_INSTRUCTIONS)


def _feedback_text(feedback: FailureFeedback | None) -> str:
    if feedback is None:
        return "No diagnostic feedback was available."
    lines = [f"Failure kind: {feedback.kind}", f"Message: {feedback.message}"]
    if feedback.input_repr is not None:
        lines.append(f"Failing input: {feedback.input_repr}")
    if feedback.expected_repr is not None:
        lines.append(f"Expected: {feedback.expected_repr}")
    if feedback.actual_repr is not None:
        lines.append(f"Actual: {feedback.actual_repr}")
    if feedback.exception_type is not None:
        lines.append(f"Exception: {feedback.exception_type}")
    if feedback.test_id is not None:
        lines.append(f"Test: {feedback.test_id}")
    labels = {
        "assertion_diff": "Assertion detail",
        "stdout_mismatch": "Output mismatch",
        "stderr": "Standard error",
        "property": "Failed property",
        "traceback": "Traceback",
        "timeout": "Timeout",
    }
    for observation in feedback.observations:
        if observation.kind in labels:
            lines.append(f"{labels[observation.kind]}: {observation.value}")
    return "\n".join(lines)


def build_prompt(
    strategy: StrategyName,
    problem: Problem,
    attempt: int,
    history: Sequence[AttemptRecord],
    max_attempts: int = 5,
    patches: Sequence[Patch] | None = None,
    memories: Sequence[FailureMemory] | None = None,
    transferred: Sequence[FailureMemory] | None = None,
) -> str:
    sections = [
        f"TASK {problem.task_id}",
        problem.prompt.rstrip(),
        f"ATTEMPT {attempt} OF {max_attempts}",
    ]
    if strategy == "no-memory":
        sections.append(
            "Previous attempts are unavailable. Produce a fresh, independent completion."
        )
    elif strategy in ("full-history", "sliding-window"):
        if history:
            sections.append("PREVIOUS ATTEMPTS")
            for item in history:
                sections.extend(
                    [
                        f"Attempt {item.attempt} completion:",
                        item.completion,
                        f"Attempt {item.attempt} test feedback:",
                        _feedback_text(item.evaluation.feedback),
                    ]
                )
            sections.append(
                "Use the history to produce a corrected completion, not a patch or explanation."
            )
        else:
            sections.append("There are no previous attempts.")
    elif strategy in PATCHSEARCH_STRATEGIES:
        if memories is not None:
            selected = list(memories)
        elif patches is not None:
            selected = [
                FailureMemory(
                    task_id=patch.task_id,
                    kind="executable_constraint",
                    content=patch.assertion,
                    source_attempt=patch.source_attempt,
                    failure_kind=patch.failure_kind,
                    fingerprint=patch.assertion,
                    validated=patch.validated,
                    provenance="legacy-v1",
                    explanation=patch.explanation,
                )
                for patch in patches
            ]
        else:
            selected = retrieve_memories(history)
        selected = strategy_memories(strategy, selected)
        shown = shown_attempt(strategy, history)
        if shown is not None and strategy == "patchsearch-with-code":
            sections.extend(
                [f"MOST RECENT ATTEMPT (attempt {shown.attempt})", shown.completion]
            )
        elif shown is not None and strategy == "patchsearch-adaptive":
            sections.extend(
                [
                    f"MOST RECENT VALID ATTEMPT (attempt {shown.attempt})",
                    shown.completion,
                    f"Attempt {shown.attempt} test feedback:",
                    _bounded_feedback(shown),
                ]
            )
        elif shown is not None:
            count = _failed_test_count(shown)
            detail = f", {count} failing test(s)" if count is not None else ""
            sections.extend(
                [f"BEST PREVIOUS ATTEMPT (attempt {shown.attempt}{detail})",
                 shown.completion]
            )
        counts = (
            repeat_counts(history) if strategy == "patchsearch-adaptive" else {}
        )
        if selected:
            sections.append("FAILURE MEMORY")
            for memory in selected:
                seen = counts.get(memory.fingerprint, 1)
                times = f" (failed this way {seen} times)" if seen > 1 else ""
                if memory.kind == "executable_constraint":
                    if strategy == "patchsearch-explained" and memory.explanation:
                        sections.append(f"Explanation: {memory.explanation}")
                    sections.append(f"Constraint{times}: {memory.content}")
                else:
                    sections.append(f"Summary{times}: {memory.content}")
            sections.append(
                "Use this bounded failure memory to produce a complete corrected "
                "completion, not a patch or explanation. Satisfy every executable "
                "constraint."
            )
        else:
            sections.append("There is no validated failure memory from previous attempts.")
        streak = stuck_streak(history) if strategy == "patchsearch-adaptive" else 0
        if streak >= 2:
            sections.append(
                f"STUCK: the last {streak} attempts failed in exactly the same way. "
                "Small edits to that approach are not working. Re-read the task "
                "and the failing test, then write the function with a different "
                "approach."
            )
        if transferred:
            # Labelled separately and by source task: these describe a
            # different problem and are hints, not constraints on this one.
            sections.append("RELATED FAILURES FROM OTHER TASKS")
            for memory in transferred:
                sections.append(f"{memory.task_id}: {memory.content}")
            sections.append(
                "These describe different tasks. Treat them as hints about "
                "recurring mistakes, never as requirements for this task."
            )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    return "\n\n".join(sections).rstrip() + "\n"


def parse_completion(output_text: str) -> str:
    try:
        payload = json.loads(output_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"response was not valid JSON: {exc.msg}") from exc
    if not isinstance(payload, dict) or set(payload) != {"completion"}:
        raise ValueError("response must contain exactly the completion field")
    completion = payload["completion"]
    if not isinstance(completion, str) or not completion.strip():
        raise ValueError("completion must be a non-empty string")
    return completion.rstrip() + "\n"
