from __future__ import annotations

from hashlib import sha256

from tokenos.models import FailureEvidence, FailureMemory, Patch, Problem
from tokenos.patchsearch.validator import validate_memory

PATCHABLE_FAILURE_KINDS = {"wrong_answer", "runtime_error", "timeout"}
MAX_EXPLANATION_CHARS = 200
MAX_MEMORY_CHARS = 600


def _one_line(value: str) -> str:
    return " ".join(value.split())


def _bounded(value: str, limit: int = MAX_EXPLANATION_CHARS) -> str:
    normalized = _one_line(value)
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."


def _observation(evidence: FailureEvidence, kind: str) -> tuple[str, bool, bool] | None:
    for item in evidence.observations:
        if item.kind == kind:
            return item.value, item.trusted, item.truncated
    legacy = {
        "function_call": evidence.input_repr,
        "expected_value": evidence.expected_repr,
        "actual_value": evidence.actual_repr,
        "exception": evidence.exception_type,
    }.get(kind)
    if legacy is not None:
        return legacy, kind in {"function_call", "expected_value"}, "<truncated>" in legacy
    return None


def explain_failure(
    evidence: FailureEvidence, *, limit: int = MAX_EXPLANATION_CHARS
) -> str:
    """Render a deterministic, bounded explanation without another model call."""
    actual = _observation(evidence, "actual_value")
    diff = _observation(evidence, "assertion_diff")
    exception = _observation(evidence, "exception")
    if evidence.kind == "wrong_answer":
        if actual is not None:
            return _bounded(f"Previous candidate returned {actual[0]}.", limit)
        # An assertion diff compares two values; it is not a returned value.
        if diff is not None:
            return _bounded(
                f"Previous candidate failed an assertion: {diff[0]}.", limit
            )
        return "Previous candidate produced an incorrect result."
    if evidence.kind == "runtime_error":
        exception_name = _one_line(exception[0] if exception else "an exception")
        message = _one_line(evidence.message)
        if message:
            return _bounded(
                f"Previous candidate raised {exception_name}: {message}.", limit
            )
        return _bounded(f"Previous candidate raised {exception_name}.", limit)
    if evidence.kind == "timeout":
        return "Previous candidate did not terminate on this input."
    if evidence.kind == "syntax_error":
        return _bounded(
            f"Previous candidate had invalid Python syntax: {evidence.message}.", limit
        )
    if evidence.kind == "missing_entry_point":
        return _bounded(
            f"Previous candidate omitted the required callable: {evidence.message}.",
            limit,
        )
    if evidence.kind == "invalid_output":
        return _bounded(f"Previous model response was invalid: {evidence.message}.", limit)
    return _bounded(f"Previous candidate failed: {evidence.message}.", limit)


def _fingerprint(kind: str, content: str) -> str:
    return sha256(f"{kind}\0{content}".encode()).hexdigest()


def extract_memory(
    problem: Problem,
    evidence: FailureEvidence | None,
    source_attempt: int,
    *,
    include_explanation: bool = False,
) -> FailureMemory | None:
    """Create a safe executable constraint or a bounded typed summary."""
    if evidence is None or evidence.kind == "infrastructure_error":
        return None
    call = _observation(evidence, "function_call")
    expected = _observation(evidence, "expected_value")
    if (
        evidence.kind in PATCHABLE_FAILURE_KINDS
        and call is not None
        and expected is not None
        and call[1] and expected[1]
        and not call[2] and not expected[2]
        and not evidence.truncated
    ):
        assertion = f"assert {problem.entry_point}(*{call[0]}) == {expected[0]}"
        memory = FailureMemory(
            task_id=problem.task_id,
            kind="executable_constraint",
            content=assertion,
            source_attempt=source_attempt,
            failure_kind=evidence.kind,
            fingerprint=_fingerprint("executable_constraint", assertion),
            provenance=evidence.provenance,
            explanation=explain_failure(evidence) if include_explanation else None,
        )
        if validate_memory(memory, evidence, problem.entry_point):
            return memory
    label = evidence.kind.replace("_", " ").upper()
    test = f" Test: {evidence.test_id}." if evidence.test_id else ""
    # The one-line explanation budget is for `patchsearch-explained` captions.
    # A summary fallback is the only evidence the model receives, so it may use
    # the whole memory budget.
    details = explain_failure(
        evidence, limit=MAX_MEMORY_CHARS - len(label) - len(test) - 2
    )
    content = _bounded(f"{label}: {details}{test}", MAX_MEMORY_CHARS)
    return FailureMemory(
        task_id=problem.task_id,
        kind="failure_summary",
        content=content,
        source_attempt=source_attempt,
        failure_kind=evidence.kind,
        fingerprint=_fingerprint("failure_summary", content),
        provenance=evidence.provenance,
    )


def extract_patch(
    problem: Problem,
    feedback: FailureEvidence | None,
    source_attempt: int,
    *,
    include_explanation: bool = False,
) -> Patch | None:
    """Legacy v1 wrapper returning only executable memories as Patch objects."""
    memory = extract_memory(problem, feedback, source_attempt, include_explanation=include_explanation)
    if memory is None or memory.kind != "executable_constraint":
        return None
    return Patch(
        task_id=memory.task_id,
        entry_point=problem.entry_point,
        assertion=memory.content,
        source_attempt=memory.source_attempt,
        failure_kind=memory.failure_kind,
        validated=memory.validated,
        explanation=memory.explanation,
    )
