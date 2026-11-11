from __future__ import annotations

import ast

from tokenos.models import FailureEvidence, FailureMemory, Patch

PATCHABLE_FAILURE_KINDS = {"wrong_answer", "runtime_error", "timeout"}


def _typed_value(evidence: FailureEvidence, kind: str, legacy: str | None) -> str | None:
    for item in evidence.observations:
        if item.kind == kind:
            if not item.trusted or item.truncated:
                return None
            return item.value
    return legacy


def validate_memory(memory: FailureMemory, evidence: FailureEvidence, entry_point: str) -> bool:
    """Validate executable memory structurally without executing benchmark code."""
    if memory.kind == "failure_summary":
        return bool(memory.content) and len(memory.content) <= 600
    if (
        evidence.kind not in PATCHABLE_FAILURE_KINDS
        or memory.failure_kind != evidence.kind
        or not memory.content.startswith("assert ")
        or evidence.truncated
    ):
        return False
    input_repr = _typed_value(evidence, "function_call", evidence.input_repr)
    expected_repr = _typed_value(evidence, "expected_value", evidence.expected_repr)
    if input_repr is None or expected_repr is None:
        return False
    expected_assertion = f"assert {entry_point}(*{input_repr}) == {expected_repr}"
    if memory.content != expected_assertion or len(memory.content) > 600:
        return False
    try:
        arguments = ast.literal_eval(input_repr)
        ast.literal_eval(expected_repr)
        tree = ast.parse(memory.content, mode="exec")
    except (SyntaxError, ValueError, TypeError, MemoryError, RecursionError):
        return False
    if not isinstance(arguments, tuple) or len(tree.body) != 1:
        return False
    statement = tree.body[0]
    return (
        isinstance(statement, ast.Assert)
        and isinstance(statement.test, ast.Compare)
        and entry_point.isidentifier()
    )


def validate_patch(patch: Patch, feedback: FailureEvidence) -> bool:
    """Legacy validation wrapper."""
    memory = FailureMemory(
        task_id=patch.task_id,
        kind="executable_constraint",
        content=patch.assertion,
        source_attempt=patch.source_attempt,
        failure_kind=patch.failure_kind,
        fingerprint="legacy",
        validated=patch.validated,
        explanation=patch.explanation,
    )
    return validate_memory(memory, feedback, patch.entry_point)
