from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
from threading import Lock
from typing import Any, Iterable

from tokenos.config import utc_now
from tokenos.models import AttemptRecord, RunConfig, to_jsonable


class ArtifactError(RuntimeError):
    pass


class EventStore:
    def __init__(self, root: Path, run_id: str, *, create: bool = False) -> None:
        self.run_dir = root / run_id
        self.manifest_path = self.run_dir / "manifest.json"
        self.events_path = self.run_dir / "events.jsonl"
        self.summary_path = self.run_dir / "summary.json"
        self.samples_path = self.run_dir / "samples.jsonl"
        self._lock = Lock()
        if create:
            self.run_dir.mkdir(parents=True, exist_ok=False)
        elif not self.run_dir.is_dir():
            raise ArtifactError(f"run does not exist: {run_id}")

    def write_manifest(self, config: RunConfig) -> None:
        payload = {
            "schema_version": 1,
            "config_fingerprint": config.fingerprint(),
            "config": config.public_dict(),
        }
        self._atomic_json(self.manifest_path, payload)

    def read_manifest(self) -> dict[str, Any]:
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def validate_config(self, config: RunConfig) -> None:
        expected = self.read_manifest()["config_fingerprint"]
        if config.fingerprint() != expected:
            raise ArtifactError("resume configuration does not match the run manifest")

    def append(self, event_type: str, **payload: Any) -> None:
        event = {"type": event_type, "timestamp": utc_now(), **payload}
        line = json.dumps(event, default=to_jsonable, sort_keys=True) + "\n"
        with self._lock:
            with self.events_path.open("a", encoding="utf-8") as handle:
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())

    def events(self) -> list[dict[str, Any]]:
        if not self.events_path.exists():
            return []
        result = []
        for number, line in enumerate(
            self.events_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            try:
                result.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ArtifactError(f"invalid event JSON on line {number}") from exc
        return result

    def completed_attempts(self) -> list[AttemptRecord]:
        records = []
        for event in self.events():
            if event.get("type") == "attempt_completed":
                records.append(AttemptRecord.from_dict(event["record"]))
        records.sort(key=lambda item: (item.strategy, item.task_id, item.attempt))
        return records

    def committed_cost(self) -> float:
        total = 0.0
        for event in self.events():
            if event.get("type") == "attempt_completed":
                total += float(event["record"]["estimated_cost_usd"])
            elif event.get("type") == "evaluation_failed":
                total += float(event.get("estimated_cost_usd", 0.0))
        return total

    def write_summary(self, summary: dict[str, Any]) -> None:
        self._atomic_json(self.summary_path, summary)

    def write_samples(self, records: Iterable[AttemptRecord]) -> None:
        final: dict[tuple[str, str], AttemptRecord] = {}
        for record in records:
            final[(record.strategy, record.task_id)] = record
        lines = [
            json.dumps(
                {
                    "task_id": record.task_id,
                    "solution": record.solution,
                    "strategy": record.strategy,
                    "attempt": record.attempt,
                },
                sort_keys=True,
            )
            for record in sorted(final.values(), key=lambda r: (r.strategy, r.task_id))
        ]
        content = "\n".join(lines) + ("\n" if lines else "")
        self._atomic_text(self.samples_path, content)

    @staticmethod
    def _atomic_json(path: Path, value: Any) -> None:
        EventStore._atomic_text(
            path, json.dumps(value, default=to_jsonable, indent=2, sort_keys=True) + "\n"
        )

    @staticmethod
    def _atomic_text(path: Path, content: str) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)
