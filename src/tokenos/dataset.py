from __future__ import annotations

import gzip
from hashlib import sha256
import json
from pathlib import Path
from typing import Iterable

from tokenos.models import Problem

HUMANEVAL_PLUS_VERSION = "v0.1.10"
HUMANEVAL_PLUS_URL = (
    "https://github.com/evalplus/humanevalplus_release/releases/download/"
    "v0.1.10/HumanEvalPlus.jsonl.gz"
)
HUMANEVAL_PLUS_SHA256 = (
    "272720b90ac375502c8ed23cd791c2a93dfb22a911641a494da74a426c09f101"
)
DEFAULT_DATASET_PATH = Path(
    ".tokenos/cache/HumanEvalPlus-v0.1.10.jsonl.gz"
)


class DatasetError(RuntimeError):
    pass


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_humaneval_plus(
    path: Path = DEFAULT_DATASET_PATH,
    *,
    url: str = HUMANEVAL_PLUS_URL,
    expected_sha256: str | None = HUMANEVAL_PLUS_SHA256,
) -> tuple[Path, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        import httpx

        with httpx.stream("GET", url, follow_redirects=True, timeout=60) as response:
            response.raise_for_status()
            with temporary.open("wb") as target:
                for chunk in response.iter_bytes():
                    target.write(chunk)
        _validate_dataset(temporary)
        digest = file_sha256(temporary)
        if expected_sha256 is not None and digest != expected_sha256:
            raise DatasetError(
                "HumanEval+ SHA-256 mismatch: "
                f"expected {expected_sha256}, received {digest}"
            )
        temporary.replace(path)
    except Exception as exc:
        temporary.unlink(missing_ok=True)
        if isinstance(exc, DatasetError):
            raise
        raise DatasetError(f"failed to download HumanEval+ from {url}: {exc}") from exc
    return path, digest


def _validate_dataset(path: Path) -> None:
    problems = load_problems(path)
    if len(problems) != 164:
        raise DatasetError(f"expected 164 HumanEval+ tasks, found {len(problems)}")


def load_problems(path: Path = DEFAULT_DATASET_PATH) -> dict[str, Problem]:
    if not path.is_file():
        raise DatasetError(f"dataset not found: {path}; run 'tokenos dataset fetch'")
    problems: dict[str, Problem] = {}
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                raw = json.loads(line)
                problem = Problem(
                    task_id=raw["task_id"],
                    prompt=raw["prompt"],
                    entry_point=raw["entry_point"],
                )
                problems[problem.task_id] = problem
    except (OSError, json.JSONDecodeError, KeyError) as exc:
        raise DatasetError(f"invalid HumanEval+ dataset: {exc}") from exc
    return problems


def select_problems(
    problems: dict[str, Problem], task_ids: Iterable[str]
) -> list[Problem]:
    selected = []
    for task_id in task_ids:
        try:
            selected.append(problems[task_id])
        except KeyError as exc:
            raise DatasetError(f"unknown task id: {task_id}") from exc
    return selected


def sorted_task_ids(problems: dict[str, Problem]) -> tuple[str, ...]:
    def key(task_id: str) -> tuple[str, int]:
        prefix, _, suffix = task_id.partition("/")
        return prefix, int(suffix)

    return tuple(sorted(problems, key=key))
