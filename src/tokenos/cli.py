from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Sequence

from tokenos.artifacts import ArtifactError, EventStore
from tokenos.config import (
    ConfigurationError,
    config_from_environment,
    config_from_manifest,
)
from tokenos.dataset import (
    DEFAULT_DATASET_PATH,
    DatasetError,
    fetch_humaneval_plus,
    file_sha256,
    load_problems,
    select_problems,
    sorted_task_ids,
)
from tokenos.evaluator import DockerEvaluator, docker_status
from tokenos.models import StrategyName
from tokenos.provider import AzureOpenAIProvider
from tokenos.runner import BenchmarkRunner
from tokenos.strategies import parse_completion

DEFAULT_RUNS_ROOT = Path("runs")
VALID_STRATEGIES = {"no-memory", "full-history"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tokenos")
    subparsers = parser.add_subparsers(dest="command", required=True)

    dataset = subparsers.add_parser("dataset", help="manage benchmark datasets")
    dataset_subparsers = dataset.add_subparsers(dest="dataset_command", required=True)
    fetch = dataset_subparsers.add_parser("fetch", help="download HumanEval+ v0.1.10")
    fetch.add_argument("--path", type=Path, default=DEFAULT_DATASET_PATH)

    doctor = subparsers.add_parser("doctor", help="validate local and Azure setup")
    doctor.add_argument("--live", action="store_true", help="make a small Azure request")
    doctor.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    doctor.add_argument("--evaluator-image", default="tokenos-evalplus:0.3.1")

    benchmark = subparsers.add_parser("benchmark", help="start a benchmark run")
    benchmark.add_argument(
        "--strategies", default="no-memory,full-history", help="comma-separated"
    )
    selection = benchmark.add_mutually_exclusive_group(required=True)
    selection.add_argument("--all", action="store_true", help="run all 164 tasks")
    selection.add_argument("--task-ids", help="comma-separated HumanEval task IDs")
    benchmark.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    benchmark.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    benchmark.add_argument("--max-attempts", type=int, default=5)
    benchmark.add_argument("--concurrency", type=int, default=4)
    benchmark.add_argument("--budget-usd", type=float, default=10.0)
    benchmark.add_argument("--evaluator-image", default="tokenos-evalplus:0.3.1")

    resume = subparsers.add_parser("resume", help="resume an interrupted run")
    resume.add_argument("run_id")
    resume.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    resume.add_argument(
        "--budget-usd",
        type=float,
        help="explicitly raise the run's total estimated-spend cap",
    )

    report = subparsers.add_parser("report", help="print a run summary")
    report.add_argument("run_id")
    report.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    report.add_argument("--json", action="store_true")
    return parser


def _strategies(raw: str) -> tuple[StrategyName, ...]:
    values = tuple(item.strip() for item in raw.split(",") if item.strip())
    invalid = set(values) - VALID_STRATEGIES
    if not values or invalid:
        raise ConfigurationError(
            f"strategies must be drawn from {sorted(VALID_STRATEGIES)}"
        )
    if len(set(values)) != len(values):
        raise ConfigurationError("strategies cannot contain duplicates")
    return values  # type: ignore[return-value]


def _task_ids(raw: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in raw.split(",") if item.strip())
    if not values:
        raise ConfigurationError("--task-ids cannot be empty")
    return values


def command_dataset_fetch(args: argparse.Namespace) -> int:
    path, digest = fetch_humaneval_plus(args.path)
    print(f"Downloaded HumanEval+ v0.1.10 to {path}")
    print(f"SHA-256: {digest}")
    return 0


def command_doctor(args: argparse.Namespace) -> int:
    checks: list[tuple[str, bool, str]] = []
    version_ok = sys.version_info[:2] == (3, 12)
    checks.append(
        (
            "Python 3.12",
            version_ok,
            f"running {sys.version.split()[0]}" + (" (warning)" if not version_ok else ""),
        )
    )
    try:
        problems = load_problems(args.dataset_path)
        dataset_ok = len(problems) == 164
        dataset_note = f"{len(problems)} tasks"
    except DatasetError as exc:
        problems = {}
        dataset_ok = False
        dataset_note = str(exc)
    checks.append(("HumanEval+ v0.1.10", dataset_ok, dataset_note))

    docker = docker_status(args.evaluator_image)
    checks.extend(
        [
            ("Docker CLI", docker["cli"], "installed" if docker["cli"] else "missing"),
            (
                "Docker daemon",
                docker["daemon"],
                "available" if docker["daemon"] else "unavailable",
            ),
            (
                "Evaluator image",
                docker["image"],
                args.evaluator_image if docker["image"] else "not built",
            ),
        ]
    )

    env_ok = True
    notes = []
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "").strip()
    if not endpoint:
        env_ok = False
        notes.append("AZURE_OPENAI_ENDPOINT missing")
    auth = os.environ.get("AZURE_OPENAI_AUTH", "api_key").strip().lower()
    if auth not in {"api_key", "entra"}:
        env_ok = False
        notes.append("AZURE_OPENAI_AUTH invalid")
    if auth == "api_key" and not os.environ.get("AZURE_OPENAI_API_KEY", "").strip():
        env_ok = False
        notes.append("AZURE_OPENAI_API_KEY missing")
    for name in (
        "AZURE_OPENAI_INPUT_USD_PER_MTOK",
        "AZURE_OPENAI_CACHED_INPUT_USD_PER_MTOK",
        "AZURE_OPENAI_OUTPUT_USD_PER_MTOK",
    ):
        try:
            if float(os.environ.get(name, "")) < 0:
                raise ValueError
        except ValueError:
            env_ok = False
            notes.append(f"{name} missing or invalid")
    checks.append(("Azure configuration", env_ok, "; ".join(notes) or auth))

    # Python 3.12 is the reproducible target but a newer compatible host is not fatal.
    hard_failures = [
        check for check in checks[1:] if not check[1]
    ]
    for name, ok, note in checks:
        marker = "OK" if ok else ("WARN" if name == "Python 3.12" else "FAIL")
        print(f"[{marker:4}] {name}: {note}")

    if args.live:
        if hard_failures:
            print("[SKIP] Azure live check: fix failed prerequisites first")
        else:
            assert problems
            config = config_from_environment(
                strategies=("no-memory",),
                task_ids=("HumanEval/0",),
                dataset_path=args.dataset_path,
                dataset_sha256=file_sha256(args.dataset_path),
                max_attempts=1,
                concurrency=1,
                budget_usd=10.0,
            )
            config = replace(config, evaluator_image=args.evaluator_image)
            response = AzureOpenAIProvider(config).generate(
                "TASK doctor\n\ndef tokenos_doctor():\n    \"\"\"Return True.\"\"\"\n"
            )
            parse_completion(response.output_text)
            print(
                f"[OK  ] Azure live check: response {response.response_id}, "
                f"model {response.resolved_model or 'not reported'}"
            )
    return 1 if hard_failures else 0


def command_benchmark(args: argparse.Namespace) -> int:
    problems = load_problems(args.dataset_path)
    task_ids = sorted_task_ids(problems) if args.all else _task_ids(args.task_ids)
    selected = select_problems(problems, task_ids)
    config = config_from_environment(
        strategies=_strategies(args.strategies),
        task_ids=task_ids,
        dataset_path=args.dataset_path,
        dataset_sha256=file_sha256(args.dataset_path),
        max_attempts=args.max_attempts,
        concurrency=args.concurrency,
        budget_usd=args.budget_usd,
    )
    config = replace(config, evaluator_image=args.evaluator_image)
    store = EventStore(args.runs_root, config.run_id, create=True)
    store.write_manifest(config)
    summary = BenchmarkRunner(
        config,
        AzureOpenAIProvider(config),
        DockerEvaluator(config),
        store,
    ).run(selected)
    _print_summary(summary)
    print(f"Artifacts: {store.run_dir}")
    return 0 if summary["complete"] else 2


def command_resume(args: argparse.Namespace) -> int:
    store = EventStore(args.runs_root, args.run_id)
    manifest = store.read_manifest()
    config = config_from_manifest(manifest)
    store.validate_config(config)
    if args.budget_usd is not None:
        if args.budget_usd < config.budget_usd:
            raise ConfigurationError("resume budget cannot be lower than the manifest cap")
        if args.budget_usd < store.committed_cost():
            raise ConfigurationError("resume budget cannot be below already committed spend")
        if args.budget_usd != config.budget_usd:
            store.append(
                "budget_cap_raised",
                previous_budget_usd=config.budget_usd,
                budget_usd=args.budget_usd,
            )
            config = replace(config, budget_usd=args.budget_usd)
    problems = load_problems(Path(config.dataset_path))
    if file_sha256(Path(config.dataset_path)) != config.dataset_sha256:
        raise DatasetError("dataset hash no longer matches the run manifest")
    selected = select_problems(problems, config.task_ids)
    summary = BenchmarkRunner(
        config,
        AzureOpenAIProvider(config),
        DockerEvaluator(config),
        store,
    ).run(selected)
    _print_summary(summary)
    return 0 if summary["complete"] else 2


def command_report(args: argparse.Namespace) -> int:
    store = EventStore(args.runs_root, args.run_id)
    if not store.summary_path.exists():
        raise ArtifactError(f"summary not found for run {args.run_id}")
    summary = json.loads(store.summary_path.read_text(encoding="utf-8"))
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        _print_summary(summary)
    return 0


def _print_summary(summary: dict) -> None:
    state = "complete" if summary.get("complete") else "partial"
    print(f"Run {summary['run_id']}: {state}")
    for strategy, metrics in summary.get("strategies", {}).items():
        print(
            f"  {strategy}: solved {metrics['solved_tasks']}/{summary['planned_tasks']}, "
            f"solve@1={metrics['solve@1']:.3f}, solve@5={metrics['solve@5']:.3f}, "
            f"tokens={metrics['usage']['input_tokens'] + metrics['usage']['output_tokens']}"
        )
    print(f"  estimated cost: ${summary.get('estimated_cost_usd', 0):.4f}")
    if summary.get("stop_reason"):
        print(f"  stop reason: {summary['stop_reason']}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "dataset":
            return command_dataset_fetch(args)
        if args.command == "doctor":
            return command_doctor(args)
        if args.command == "benchmark":
            return command_benchmark(args)
        if args.command == "resume":
            return command_resume(args)
        if args.command == "report":
            return command_report(args)
    except (ConfigurationError, DatasetError, ArtifactError) as exc:
        parser.error(str(exc))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
