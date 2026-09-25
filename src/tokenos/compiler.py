"""Budgeted context compilation.

Retrieval bounds memory in characters, but nothing bounded the prompt a model
actually reads. The compiler does: given a token budget, it decides which
pieces of context survive and renders them through the same `build_prompt` the
benchmark measured, so a budget that binds nowhere reproduces the benchmarked
prompt byte for byte.

Priority is taken from the measurements, not intuition. On BigCodeBench-Hard,
adding bounded failure memory lifted 10/32 to 18/32 and adding the latest
implementation on top lifted it to 21/32, so under pressure task-local memory
outranks the code, and the code outranks memory transferred from other tasks,
which never beat its own absence.

Token counts are estimated, not tokenized, to keep the package free of a
tokenizer dependency. Across 707 recorded requests the densest text reached
3.34 UTF-8 bytes per API-reported input token; dividing by 3.0 therefore
overestimates every one of them, so a compiled context never exceeds its budget
on text shaped like the benchmark's.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math

from tokenos.models import AttemptRecord, FailureMemory, Problem, StrategyName
from tokenos.provider import structured_output_overhead_bytes
from tokenos.strategies import build_prompt

BYTES_PER_TOKEN = 3.0


class ContextBudgetError(ValueError):
    """The task itself does not fit, so no context can be compiled."""


def estimate_tokens(text: str) -> int:
    """A conservative token estimate for prompt text."""
    return tokens_for_bytes(len(text.encode("utf-8")))


def tokens_for_bytes(size: int) -> int:
    return math.ceil(size / BYTES_PER_TOKEN)


@dataclass(frozen=True, slots=True)
class CompiledContext:
    """A rendered prompt and an account of what the budget kept and dropped."""

    prompt: str
    estimated_tokens: int
    budget_tokens: int | None
    required_tokens: int
    memories: tuple[FailureMemory, ...]
    transferred: tuple[FailureMemory, ...]
    included_code: bool
    dropped_memories: tuple[FailureMemory, ...] = ()
    dropped_transferred: tuple[FailureMemory, ...] = ()
    dropped_code: bool = False

    @property
    def trimmed(self) -> bool:
        return bool(
            self.dropped_memories or self.dropped_transferred or self.dropped_code
        )

    def to_dict(self) -> dict:
        return {
            "estimated_tokens": self.estimated_tokens,
            "budget_tokens": self.budget_tokens,
            "required_tokens": self.required_tokens,
            "memories": len(self.memories),
            "transferred": len(self.transferred),
            "included_code": self.included_code,
            "dropped_memories": len(self.dropped_memories),
            "dropped_transferred": len(self.dropped_transferred),
            "dropped_code": self.dropped_code,
        }


def compile_context(
    strategy: StrategyName,
    problem: Problem,
    attempt: int,
    history: Sequence[AttemptRecord],
    *,
    memories: Sequence[FailureMemory] = (),
    transferred: Sequence[FailureMemory] = (),
    budget_tokens: int | None = None,
    instructions: str | None = None,
    max_attempts: int = 5,
) -> CompiledContext:
    """Fit retrieved context into `budget_tokens`, highest priority first.

    `budget_tokens` covers everything the model reads: the system
    instructions, the output schema, and the prompt. Each candidate is kept
    only if the fully rendered prompt still fits, so section headers and
    instructions are paid for exactly rather than approximated. Code is kept or
    dropped whole; a truncated implementation would mislead more than none.
    """
    overhead = structured_output_overhead_bytes(instructions)
    wants_code = strategy == "patchsearch-with-code" and bool(history)
    local = list(memories)
    remote = list(transferred)

    def render(keep_local: list, keep_code: bool, keep_remote: list) -> tuple[str, int]:
        prompt = build_prompt(
            strategy,
            problem,
            attempt,
            history if keep_code else (),
            max_attempts=max_attempts,
            memories=keep_local,
            transferred=keep_remote,
        )
        return prompt, tokens_for_bytes(len(prompt.encode("utf-8")) + overhead)

    if budget_tokens is None:
        prompt, tokens = render(local, wants_code, remote)
        _, required = render([], False, [])
        return CompiledContext(
            prompt=prompt,
            estimated_tokens=tokens,
            budget_tokens=None,
            required_tokens=required,
            memories=tuple(local),
            transferred=tuple(remote),
            included_code=wants_code,
        )

    _, required = render([], False, [])
    if required > budget_tokens:
        raise ContextBudgetError(
            f"{problem.task_id} needs {required} tokens before any memory, "
            f"over the {budget_tokens}-token budget"
        )

    # Executable constraints first, then summaries newest first: the order the
    # retriever already uses when its own character budget binds.
    ranked_local = sorted(
        local,
        key=lambda item: (
            item.kind != "executable_constraint",
            item.source_attempt
            if item.kind == "executable_constraint"
            else -item.source_attempt,
        ),
    )
    kept_local: list[FailureMemory] = []
    for memory in ranked_local:
        trial = [item for item in local if item in kept_local or item is memory]
        if render(trial, False, [])[1] <= budget_tokens:
            kept_local = trial
    keep_code = wants_code and render(kept_local, True, [])[1] <= budget_tokens
    kept_remote: list[FailureMemory] = []
    for memory in remote:
        trial = kept_remote + [memory]
        if render(kept_local, keep_code, trial)[1] <= budget_tokens:
            kept_remote = trial

    prompt, tokens = render(kept_local, keep_code, kept_remote)
    return CompiledContext(
        prompt=prompt,
        estimated_tokens=tokens,
        budget_tokens=budget_tokens,
        required_tokens=required,
        memories=tuple(kept_local),
        transferred=tuple(kept_remote),
        included_code=keep_code,
        dropped_memories=tuple(item for item in local if item not in kept_local),
        dropped_transferred=tuple(item for item in remote if item not in kept_remote),
        dropped_code=wants_code and not keep_code,
    )
