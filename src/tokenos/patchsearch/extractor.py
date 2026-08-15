from __future__ import annotations

from tokenos.models import FailureFeedback, Patch, Problem
from tokenos.patchsearch.validator import validate_patch

PATCHABLE_FAILURE_KINDS = {"wrong_answer", "runtime_error", "timeout"}


def extract_patch(
    problem: Problem,
    feedback: FailureFeedback | None,
    source_attempt: int,
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
    )
    return patch if validate_patch(patch, feedback) else None
