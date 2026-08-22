from __future__ import annotations

from tokenos.models import FailureFeedback, Patch, Problem
from tokenos.patchsearch.validator import validate_patch

PATCHABLE_FAILURE_KINDS = {"wrong_answer", "runtime_error", "timeout"}
MAX_EXPLANATION_CHARS = 200


def _one_line(value: str) -> str:
    return " ".join(value.split())


def _bounded(value: str) -> str:
    normalized = _one_line(value)
    if len(normalized) <= MAX_EXPLANATION_CHARS:
        return normalized
    return normalized[: MAX_EXPLANATION_CHARS - 3].rstrip() + "..."


def explain_failure(feedback: FailureFeedback) -> str:
    """Render a deterministic, bounded explanation without another model call."""
    if feedback.kind == "wrong_answer":
        if feedback.actual_repr is None:
            return "Previous candidate produced an incorrect result."
        return _bounded(f"Previous candidate returned {feedback.actual_repr}.")
    if feedback.kind == "runtime_error":
        exception = _one_line(feedback.exception_type or "an exception")
        message = _one_line(feedback.message)
        if message:
            return _bounded(f"Previous candidate raised {exception}: {message}.")
        return _bounded(f"Previous candidate raised {exception}.")
    if feedback.kind == "timeout":
        return "Previous candidate did not terminate on this input."
    raise ValueError(f"Cannot explain ineligible failure kind: {feedback.kind}")


def extract_patch(
    problem: Problem,
    feedback: FailureFeedback | None,
    source_attempt: int,
    *,
    include_explanation: bool = False,
) -> Patch | None:
    """Convert an oracle counterexample into a small executable assertion."""
    if (
        feedback is None
        or feedback.kind not in PATCHABLE_FAILURE_KINDS
        or feedback.input_repr is None
        or feedback.expected_repr is None
    ):
        return None
    if "<truncated>" in feedback.input_repr or "<truncated>" in feedback.expected_repr:
        return None
    assertion = (
        f"assert {problem.entry_point}(*{feedback.input_repr}) == "
        f"{feedback.expected_repr}"
    )
    patch = Patch(
        task_id=problem.task_id,
        entry_point=problem.entry_point,
        assertion=assertion,
        source_attempt=source_attempt,
        failure_kind=feedback.kind,
        explanation=explain_failure(feedback) if include_explanation else None,
    )
    return patch if validate_patch(patch, feedback) else None
