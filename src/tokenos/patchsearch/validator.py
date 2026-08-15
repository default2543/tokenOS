from __future__ import annotations

import ast

from tokenos.models import FailureFeedback, Patch

PATCHABLE_FAILURE_KINDS = {"wrong_answer", "runtime_error", "timeout"}


def validate_patch(patch: Patch, feedback: FailureFeedback) -> bool:
    """Reject malformed or lossy assertions before they enter memory."""
    if (
        feedback.kind not in PATCHABLE_FAILURE_KINDS
        or patch.failure_kind != feedback.kind
        or not patch.assertion.startswith("assert ")
    ):
        return False
    if feedback.input_repr is None or feedback.expected_repr is None:
        return False
    expected_assertion = (
        f"assert {patch.entry_point}(*{feedback.input_repr}) == "
        f"{feedback.expected_repr}"
    )
    if patch.assertion != expected_assertion or len(patch.assertion) > 4096:
        return False
    try:
        arguments = ast.literal_eval(feedback.input_repr)
        ast.literal_eval(feedback.expected_repr)
        tree = ast.parse(patch.assertion, mode="exec")
    except (SyntaxError, ValueError, TypeError, MemoryError, RecursionError):
        return False
    if not isinstance(arguments, tuple) or len(tree.body) != 1:
        return False
    statement = tree.body[0]
    if not isinstance(statement, ast.Assert) or not isinstance(statement.test, ast.Compare):
        return False
    return patch.entry_point.isidentifier()
