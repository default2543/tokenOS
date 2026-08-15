from __future__ import annotations

import json
from typing import Sequence

from tokenos.models import AttemptRecord, FailureFeedback, Patch, Problem, StrategyName
from tokenos.patchsearch.retriever import retrieve_patches

SYSTEM_INSTRUCTIONS = """You solve standalone Python function-completion tasks.
Return a JSON object with exactly one field named completion. Its value must be the
Python source that follows the supplied prompt. Do not repeat the prompt, use Markdown,
read files, access the network, or perform input/output. Preserve the required function
signature and implement the complete function body."""


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
    return "\n".join(lines)


def build_prompt(
    strategy: StrategyName,
    problem: Problem,
    attempt: int,
    history: Sequence[AttemptRecord],
    max_attempts: int = 5,
    patches: Sequence[Patch] | None = None,
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
    elif strategy == "full-history":
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
    elif strategy == "patchsearch":
        selected = list(patches) if patches is not None else retrieve_patches(history)
        if selected:
            sections.append("EXECUTABLE MEMORY")
            sections.extend(patch.assertion for patch in selected)
            sections.append(
                "Satisfy every executable constraint and produce a complete corrected "
                "completion, not a patch or explanation."
            )
        else:
            sections.append("There are no validated patches from previous attempts.")
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
