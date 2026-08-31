from __future__ import annotations

import ast
import gzip
from hashlib import sha256
import io
import json
from pathlib import Path
import sys
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
BIGCODEBENCH_HARD_VERSION = "v0.1.4"
BIGCODEBENCH_HARD_DATASET = "bigcode/bigcodebench-hard"
BIGCODEBENCH_HARD_ROWS_URL = "https://datasets-server.huggingface.co/rows"
DEFAULT_BIGCODEBENCH_HARD_PATH = Path(
    ".tokenos/cache/BigCodeBench-Hard-v0.1.4.jsonl.gz"
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


def fetch_bigcodebench_hard(
    path: Path = DEFAULT_BIGCODEBENCH_HARD_PATH,
    *,
    rows_url: str = BIGCODEBENCH_HARD_ROWS_URL,
) -> tuple[Path, str]:
    """Fetch the pinned 148-task Hard split without adding a parquet dependency."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    rows: list[dict] = []
    try:
        import httpx

        offset = 0
        while True:
            response = httpx.get(
                rows_url,
                params={
                    "dataset": BIGCODEBENCH_HARD_DATASET,
                    "config": "default",
                    "split": BIGCODEBENCH_HARD_VERSION,
                    "offset": offset,
                    "length": 100,
                },
                timeout=60,
                follow_redirects=True,
            )
            response.raise_for_status()
            payload = response.json()
            page = payload.get("rows", [])
            rows.extend(item["row"] for item in page)
            offset += len(page)
            if offset >= payload["num_rows_total"]:
                break
            if not page:
                raise DatasetError("BigCodeBench-Hard rows API returned an empty page")
        if len(rows) != 148:
            raise DatasetError(f"expected 148 BigCodeBench-Hard tasks, found {len(rows)}")
        # mtime=0 and canonical JSON make repeated fetches byte-identical.
        with temporary.open("wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
                with io.TextIOWrapper(compressed, encoding="utf-8") as text:
                    for row in rows:
                        text.write(json.dumps(row, sort_keys=True, separators=(",", ":")))
                        text.write("\n")
        _validate_bigcodebench_hard(temporary)
        digest = file_sha256(temporary)
        temporary.replace(path)
    except Exception as exc:
        temporary.unlink(missing_ok=True)
        if isinstance(exc, DatasetError):
            raise
        raise DatasetError(f"failed to download BigCodeBench-Hard: {exc}") from exc
    return path, digest


def _validate_dataset(path: Path) -> None:
    problems = load_problems(path, dataset_name="humaneval-plus")
    if len(problems) != 164:
        raise DatasetError(f"expected 164 HumanEval+ tasks, found {len(problems)}")


def _validate_bigcodebench_hard(path: Path) -> None:
    problems = load_problems(path, dataset_name="bigcodebench-hard")
    if len(problems) != 148:
        raise DatasetError(
            f"expected 148 BigCodeBench-Hard tasks, found {len(problems)}"
        )


def load_problems(
    path: Path = DEFAULT_DATASET_PATH,
    *,
    dataset_name: str = "humaneval-plus",
) -> dict[str, Problem]:
    if not path.is_file():
        raise DatasetError(f"dataset not found: {path}; run 'tokenos dataset fetch'")
    problems: dict[str, Problem] = {}
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                raw = json.loads(line)
                if dataset_name == "humaneval-plus":
                    problem = Problem(
                        task_id=raw["task_id"],
                        prompt=raw["prompt"],
                        entry_point=raw["entry_point"],
                    )
                elif dataset_name == "bigcodebench-hard":
                    problem = Problem(
                        task_id=raw["task_id"],
                        prompt=raw["complete_prompt"],
                        entry_point=raw["entry_point"],
                        benchmark=dataset_name,
                        test_code=raw["test"],
                        libraries=tuple(ast.literal_eval(raw["libs"])),
                    )
                else:
                    raise DatasetError(f"unknown dataset: {dataset_name}")
                problems[problem.task_id] = problem
    except (OSError, json.JSONDecodeError, KeyError, ValueError, SyntaxError) as exc:
        raise DatasetError(f"invalid {dataset_name} dataset: {exc}") from exc
    return problems


def load_canonical_solutions(
    path: Path = DEFAULT_DATASET_PATH,
    *,
    dataset_name: str = "humaneval-plus",
) -> dict[str, str]:
    """Load reference solution bodies for evaluator preflight.

    Deliberately separate from `load_problems`: a reference solution must never
    be reachable from a `Problem`, because `Problem` is what builds prompts.
    """
    if not path.is_file():
        raise DatasetError(f"dataset not found: {path}; run 'tokenos dataset fetch'")
    if dataset_name not in {"humaneval-plus", "bigcodebench-hard"}:
        raise DatasetError(f"unknown dataset: {dataset_name}")
    solutions: dict[str, str] = {}
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                raw = json.loads(line)
                solutions[raw["task_id"]] = raw["canonical_solution"]
    except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
        raise DatasetError(f"invalid {dataset_name} dataset: {exc}") from exc
    return solutions


def is_stdlib_problem(problem: Problem) -> bool:
    modules = {library.partition(".")[0] for library in problem.libraries}
    try:
        tree = ast.parse(problem.prompt + "\n" + (problem.test_code or ""))
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name.partition(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module.partition(".")[0])
    return modules <= sys.stdlib_module_names


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
