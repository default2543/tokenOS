from __future__ import annotations

from dataclasses import dataclass
from threading import Condition
from typing import Final

from tokenos.models import ModelUsage, RunConfig
from tokenos.provider import structured_output_overhead_bytes

MTOK: Final = 1_000_000


@dataclass(frozen=True, slots=True)
class Reservation:
    identifier: int
    amount_usd: float


class BudgetManager:
    def __init__(self, config: RunConfig, *, committed_usd: float = 0.0) -> None:
        self.config = config
        self._committed = committed_usd
        self._reserved = 0.0
        self._next_id = 1
        self._condition = Condition()

    @property
    def committed_usd(self) -> float:
        with self._condition:
            return self._committed

    @property
    def reserved_usd(self) -> float:
        with self._condition:
            return self._reserved

    def worst_case_cost(self, prompt: str) -> float:
        # UTF-8 bytes are a conservative upper bound for text token count.
        input_tokens = len(prompt.encode("utf-8")) + structured_output_overhead_bytes()
        return (
            input_tokens * self.config.input_usd_per_mtok
            + self.config.max_output_tokens * self.config.output_usd_per_mtok
        ) / MTOK

    def acquire(self, prompt: str) -> Reservation | None:
        amount = self.worst_case_cost(prompt)
        with self._condition:
            while self._committed + self._reserved + amount > self.config.budget_usd:
                if self._reserved == 0:
                    return None
                self._condition.wait()
            reservation = Reservation(self._next_id, amount)
            self._next_id += 1
            self._reserved += amount
            return reservation

    def reconcile(self, reservation: Reservation, usage: ModelUsage) -> float:
        actual = estimate_usage_cost(self.config, usage)
        with self._condition:
            self._reserved -= reservation.amount_usd
            self._committed += actual
            self._condition.notify_all()
        return actual

    def release(self, reservation: Reservation) -> None:
        with self._condition:
            self._reserved -= reservation.amount_usd
            self._condition.notify_all()


def estimate_usage_cost(config: RunConfig, usage: ModelUsage) -> float:
    cached = min(usage.cached_input_tokens, usage.input_tokens)
    uncached = usage.input_tokens - cached
    return (
        uncached * config.input_usd_per_mtok
        + cached * config.cached_input_usd_per_mtok
        + usage.output_tokens * config.output_usd_per_mtok
    ) / MTOK

