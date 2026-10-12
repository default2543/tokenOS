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
from tokenos.compiler import ContextBudgetError
from tokenos.config import (
    ConfigurationError,
    config_from_environment,
    config_from_manifest,
)
from tokenos.dataset import (
    BIGCODEBENCH_HARD_VERSION,
    DEFAULT_DATASET_PATH,
    DEFAULT_BIGCODEBENCH_HARD_PATH,
    DatasetError,
    fetch_bigcodebench_hard,
    fetch_humaneval_plus,
    file_sha256,
    load_canonical_solutions,
    load_problems,
    is_stdlib_problem,
    select_problems,
    sorted_task_ids,
)
from tokenos.evaluator import DockerEvaluator, docker_status
from tokenos.models import StrategyName
from tokenos.preflight import (
    PreflightError,
    preflight_path,
    run_preflight,
    viable_task_ids,
    write_report,
)
from tokenos.provider import AzureOpenAIProvider
from tokenos.replay import budget_sweep, context_growth, stop_simulation
from tokenos.runner import BenchmarkRunner
from tokenos.runtime import DEFAULT_CONTEXT_BUDGET, DEFAULT_STRATEGY, Runtime
from tokenos.store import MemoryStore
from tokenos.strategies import PATCHSEARCH_STRATEGIES, parse_completion

DEFAULT_RUNS_ROOT = Path("runs")
DATASETS = ("humaneval-plus", "bigcodebench-hard")
DATASET_DEFAULTS = {
    "humaneval-plus": (DEFAULT_DATASET_PATH, "v0.1.10", "tokenos-evalplus:0.3.1"),
    "bigcodebench-hard": (
        DEFAULT_BIGCODEBENCH_HARD_PATH,
        BIGCODEBENCH_HARD_VERSION,
        "tokenos-bigcodebench-stdlib:0.3.0",
    ),
}
VALID_STRATEGIES = {
    "no-memory",
    "full-history",
    "sliding-window",
    "patchsearch",
    "patchsearch-explained",
    "patchsearch-crosstask",
    "patchsearch-with-code",
    "patchsearch-best-code",
    "patchsearch-adaptive",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tokenos")
    subparsers = parser.add_subparsers(dest="command", required=True)

    dataset = subparsers.add_parser("dataset", help="manage benchmark datasets")
    dataset_subparsers = dataset.add_subparsers(dest="dataset_command", required=True)
    fetch = dataset_subparsers.add_parser("fetch", help="download a pinned dataset")
    fetch.add_argument("--name", choices=DATASETS, default="humaneval-plus")
    fetch.add_argument("--path", type=Path)

    doctor = subparsers.add_parser("doctor", help="validate local and Azure setup")
    doctor.add_argument("--live", action="store_true", help="make a small Azure request")
    doctor.add_argument("--dataset", choices=DATASETS, default="humaneval-plus")
    doctor.add_argument("--dataset-path", type=Path)
    doctor.add_argument("--evaluator-image")

    preflight = subparsers.add_parser(
        "preflight",
        help="check which tasks the evaluator image can actually run (no tokens)",
    )
    preflight.add_argument("--dataset", choices=DATASETS, default="bigcodebench-hard")
    preflight_selection = preflight.add_mutually_exclusive_group()
    preflight_selection.add_argument(
        "--all", action="store_true", help="check every dataset task"
    )
    preflight_selection.add_argument("--task-ids", help="comma-separated benchmark task IDs")
    preflight.add_argument("--dataset-path", type=Path)
    preflight.add_argument("--evaluator-image")
    preflight.add_argument("--concurrency", type=int, default=4)
    preflight.add_argument("--report-path", type=Path)

    benchmark = subparsers.add_parser("benchmark", help="start a benchmark run")
    benchmark.add_argument("--dataset", choices=DATASETS, default="humaneval-plus")
    benchmark.add_argument(
        "--strategies", default="full-history,patchsearch", help="comma-separated"
    )
    selection = benchmark.add_mutually_exclusive_group(required=True)
    selection.add_argument("--all", action="store_true", help="run every dataset task")
    selection.add_argument("--task-ids", help="comma-separated benchmark task IDs")
    benchmark.add_argument("--dataset-path", type=Path)
    benchmark.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    benchmark.add_argument("--max-attempts", type=int, default=5)
    benchmark.add_argument("--concurrency", type=int, default=4)
    benchmark.add_argument("--budget-usd", type=float, default=10.0)
    benchmark.add_argument("--evaluator-image")
    benchmark.add_argument(
        "--max-output-tokens",
        type=int,
        default=16384,
        help="per-request output cap, reasoning included; responses that hit it "
        "are truncated JSON and score as invalid_output",
    )
    benchmark.add_argument(
        "--stop-after-identical",
        type=int,
        help="abandon a bounded-memory trajectory after K identical failures in a row",
    )
    benchmark.add_argument(
        "--context-budget",
        type=int,
        help="token budget the context compiler enforces on bounded-memory "
        "strategies (default: unbounded)",
    )
    benchmark.add_argument(
        "--skip-unviable-tasks",
        action="store_true",
        help="drop tasks whose reference solution fails preflight in this image",
    )
    benchmark.add_argument("--preflight-path", type=Path)
    benchmark.add_argument(
        "--memory-store",
        help="run id whose memories seed cross-task retrieval",
    )

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

    solve = subparsers.add_parser(
        "solve", help="solve one task with the TokenOS runtime under a context budget"
    )
    solve.add_argument("task_id")
    solve.add_argument("--dataset", choices=DATASETS, default="humaneval-plus")
    solve.add_argument("--dataset-path", type=Path)
    solve.add_argument("--context-budget", type=int, default=DEFAULT_CONTEXT_BUDGET)
    solve.add_argument(
        "--strategy",
        choices=sorted(PATCHSEARCH_STRATEGIES),
        default=DEFAULT_STRATEGY,
    )
    solve.add_argument("--max-attempts", type=int, default=5)
    solve.add_argument(
        "--stop-after-identical",
        type=int,
        help="give up after K identical failures in a row",
    )
    solve.add_argument("--budget-usd", type=float, default=1.0)
    solve.add_argument("--evaluator-image")
    solve.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    solve.add_argument(
        "--memory-store", help="run id whose memories seed cross-task retrieval"
    )
    solve.add_argument("--json", action="store_true")

    growth = subparsers.add_parser(
        "growth", help="show measured prompt size by attempt for a run (no tokens)"
    )
    growth.add_argument("run_id")
    growth.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    growth.add_argument("--json", action="store_true")

    sweep = subparsers.add_parser(
        "sweep",
        help="recompile a run's bounded-memory prompts under context budgets (no tokens)",
    )
    sweep.add_argument("run_id")
    sweep.add_argument(
        "--budgets", default="4000,1500,1000", help="comma-separated token budgets"
    )
    sweep.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    sweep.add_argument("--json", action="store_true")

    stopsim = subparsers.add_parser(
        "stopsim",
        help="replay runs as if trajectories stopped after K identical failures (no tokens)",
    )
    stopsim.add_argument("run_ids", nargs="+", help="one or more runs, pooled")
    stopsim.add_argument(
        "--k", default="2,3,4,5", help="comma-separated identical-failure limits"
    )
    stopsim.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    stopsim.add_argument("--json", action="store_true")
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
    default_path, version, _ = DATASET_DEFAULTS[args.name]
    target = args.path or default_path
    if args.name == "humaneval-plus":
        path, digest = fetch_humaneval_plus(target)
        label = "HumanEval+"
    else:
        path, digest = fetch_bigcodebench_hard(target)
        label = "BigCodeBench-Hard"
    print(f"Downloaded {label} {version} to {path}")
    print(f"SHA-256: {digest}")
    return 0


def command_doctor(args: argparse.Namespace) -> int:
    default_path, version, default_image = DATASET_DEFAULTS[args.dataset]
    dataset_path = args.dataset_path or default_path
    evaluator_image = args.evaluator_image or default_image
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
        problems = load_problems(dataset_path, dataset_name=args.dataset)
        expected_tasks = 164 if args.dataset == "humaneval-plus" else 148
        dataset_ok = len(problems) == expected_tasks
        dataset_note = f"{len(problems)} tasks"
    except DatasetError as exc:
        problems = {}
        dataset_ok = False
        dataset_note = str(exc)
    dataset_label = (
        "HumanEval+" if args.dataset == "humaneval-plus" else "BigCodeBench-Hard"
    )
    checks.append((f"{dataset_label} {version}", dataset_ok, dataset_note))

    docker = docker_status(evaluator_image)
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
                evaluator_image if docker["image"] else "not built",
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
                task_ids=(next(iter(problems)),),
                dataset_path=dataset_path,
                dataset_sha256=file_sha256(dataset_path),
                dataset_name=args.dataset,
                dataset_version=version,
                max_attempts=1,
                concurrency=1,
                budget_usd=10.0,
            )
            config = replace(config, evaluator_image=evaluator_image)
            response = AzureOpenAIProvider(config).generate(
                "TASK doctor\n\ndef tokenos_doctor():\n    \"\"\"Return True.\"\"\"\n"
            )
            parse_completion(response.output_text)
            print(
                f"[OK  ] Azure live check: response {response.response_id}, "
                f"model {response.resolved_model or 'not reported'}"
            )
    return 1 if hard_failures else 0


def command_preflight(args: argparse.Namespace) -> int:
    default_path, version, default_image = DATASET_DEFAULTS[args.dataset]
    dataset_path = args.dataset_path or default_path
    evaluator_image = args.evaluator_image or default_image
    problems = load_problems(dataset_path, dataset_name=args.dataset)
    task_ids = sorted_task_ids(problems) if not args.task_ids else _task_ids(args.task_ids)
    selected = select_problems(problems, task_ids)
    solutions = load_canonical_solutions(dataset_path, dataset_name=args.dataset)
    config = _preflight_config(dataset_path, args.dataset, version, evaluator_image)
    report = run_preflight(
        selected,
        solutions,
        DockerEvaluator(config),
        dataset_name=args.dataset,
        dataset_version=version,
        dataset_sha256=config.dataset_sha256,
        evaluator_image=evaluator_image,
        concurrency=args.concurrency,
    )
    path = args.report_path or preflight_path(args.dataset, version)
    write_report(report, path)
    checked = report["checked_tasks"]
    viable = report["viable_tasks"]
    print(f"Preflight {args.dataset} {version} in {evaluator_image}")
    print(f"  reference solution passes: {viable}/{checked}")
    excluded = [item for item in report["results"] if not item["viable"]]
    for item in excluded[:10]:
        print(f"  [DROP] {item['task_id']}: {item['status']} {item['detail']}".rstrip())
    if len(excluded) > 10:
        print(f"  ... and {len(excluded) - 10} more")
    print(f"Report: {path}")
    return 0 if viable else 1


def _preflight_config(
    dataset_path: Path, dataset_name: str, version: str, evaluator_image: str
):
    """Build a evaluator-only config; preflight makes no model calls."""
    from tokenos.config import utc_now
    from tokenos.models import RunConfig

    return RunConfig(
        run_id="preflight",
        strategies=("no-memory",),
        task_ids=(),
        dataset_path=str(dataset_path),
        dataset_name=dataset_name,
        dataset_version=version,
        dataset_sha256=file_sha256(dataset_path),
        evaluator_image=evaluator_image,
        created_at=utc_now(),
    )


def command_benchmark(args: argparse.Namespace) -> int:
    if args.max_output_tokens <= 0:
        raise ConfigurationError("--max-output-tokens must be positive")
    if args.stop_after_identical is not None and args.stop_after_identical < 1:
        raise ConfigurationError("--stop-after-identical must be at least 1")
    if args.context_budget is not None:
        if args.context_budget <= 0:
            raise ConfigurationError("--context-budget must be positive")
        if not (PATCHSEARCH_STRATEGIES | {"sliding-window"}).intersection(
            _strategies(args.strategies)
        ):
            raise ConfigurationError(
                "--context-budget only affects bounded-memory strategies and "
                "sliding-window; add one to --strategies or drop the flag"
            )
    elif "sliding-window" in _strategies(args.strategies):
        raise ConfigurationError(
            "sliding-window needs --context-budget; without one it is full-history"
        )
    default_path, version, default_image = DATASET_DEFAULTS[args.dataset]
    dataset_path = args.dataset_path or default_path
    evaluator_image = args.evaluator_image or default_image
    problems = load_problems(dataset_path, dataset_name=args.dataset)
    task_ids = sorted_task_ids(problems) if args.all else _task_ids(args.task_ids)
    if args.skip_unviable_tasks:
        path = args.preflight_path or preflight_path(args.dataset, version)
        viable = viable_task_ids(
            path,
            dataset_sha256=file_sha256(dataset_path),
            evaluator_image=evaluator_image,
        )
        dropped = [task_id for task_id in task_ids if task_id not in viable]
        task_ids = tuple(task_id for task_id in task_ids if task_id in viable)
        if not task_ids:
            raise PreflightError(
                f"no selected task passes preflight in {evaluator_image}"
            )
        if dropped:
            preview = ", ".join(dropped[:5])
            suffix = "..." if len(dropped) > 5 else ""
            print(
                f"Preflight excluded {len(dropped)} unviable task(s): {preview}{suffix}"
            )
    selected = select_problems(problems, task_ids)
    if args.dataset == "bigcodebench-hard" and evaluator_image == default_image:
        unsupported = [problem.task_id for problem in selected if not is_stdlib_problem(problem)]
        if unsupported:
            preview = ", ".join(unsupported[:5])
            suffix = "..." if len(unsupported) > 5 else ""
            raise DatasetError(
                "the default BigCodeBench smoke evaluator supports stdlib-only tasks; "
                f"unsupported selection: {preview}{suffix}; use an official full "
                "evaluator image explicitly for the complete suite"
            )
    # Validate the strategy/store pairing before touching Azure configuration:
    # it is a local mistake and deserves a local error.
    strategies = _strategies(args.strategies)
    memory_store = None
    if args.memory_store:
        if "patchsearch-crosstask" not in strategies:
            raise ConfigurationError(
                "--memory-store only affects patchsearch-crosstask; add it to "
                "--strategies or drop the flag"
            )
        source = EventStore(args.runs_root, args.memory_store)
        memory_store = MemoryStore.from_run(source.events_path, problems)
        print(
            f"Seeded cross-task memory from {args.memory_store}: "
            f"{len(memory_store)} memories"
        )
    elif "patchsearch-crosstask" in strategies:
        raise ConfigurationError(
            "patchsearch-crosstask requires --memory-store RUN_ID"
        )
    config = config_from_environment(
        strategies=strategies,
        task_ids=task_ids,
        dataset_path=dataset_path,
        dataset_sha256=file_sha256(dataset_path),
        dataset_name=args.dataset,
        dataset_version=version,
        max_attempts=args.max_attempts,
        concurrency=args.concurrency,
        budget_usd=args.budget_usd,
    )
    config = replace(
        config,
        evaluator_image=evaluator_image,
        context_budget=args.context_budget,
        max_output_tokens=args.max_output_tokens,
        stop_after_identical=args.stop_after_identical,
    )
    store = EventStore(args.runs_root, config.run_id, create=True)
    store.write_manifest(config)
    summary = BenchmarkRunner(
        config,
        AzureOpenAIProvider(config),
        DockerEvaluator(config),
        store,
        memory_store=memory_store,
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
    problems = load_problems(
        Path(config.dataset_path), dataset_name=config.dataset_name
    )
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


def command_solve(args: argparse.Namespace) -> int:
    default_path, _, default_image = DATASET_DEFAULTS[args.dataset]
    dataset_path = args.dataset_path or default_path
    problems = load_problems(dataset_path, dataset_name=args.dataset)
    (problem,) = select_problems(problems, (args.task_id,))
    memory_store = None
    if args.memory_store:
        source = EventStore(args.runs_root, args.memory_store)
        memory_store = MemoryStore.from_run(source.events_path, problems)
    runtime = Runtime(
        args.context_budget,
        strategy=args.strategy,
        max_attempts=args.max_attempts,
        stop_after_identical=args.stop_after_identical,
        memory_store=memory_store,
        budget_usd=args.budget_usd,
        evaluator_image=args.evaluator_image or default_image,
    )
    result = runtime.run(problem)
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        return 0 if result.passed else 2
    print(
        f"{problem.task_id} with {args.strategy}, "
        f"context budget {args.context_budget} tokens"
    )
    for item in result.attempts:
        context = item.context
        kept = f"{len(context.memories)} memories"
        if context.included_code:
            kept += " + code"
        dropped = ""
        if context.trimmed:
            dropped = (
                f", dropped {len(context.dropped_memories)} memories"
                + (" and code" if context.dropped_code else "")
            )
        feedback = item.record.evaluation.feedback
        status = "pass" if item.passed else feedback.kind if feedback else "fail"
        print(
            f"  attempt {item.record.attempt}: {status:<14} "
            f"~{context.estimated_tokens} tokens compiled, "
            f"{item.record.usage.input_tokens} billed ({kept}{dropped})"
        )
    verdict = "solved" if result.passed else f"unsolved ({result.stop_reason})"
    print(
        f"  {verdict}: {result.total_tokens} tokens, "
        f"${result.estimated_cost_usd:.4f} estimated"
    )
    return 0 if result.passed else 2


def _run_records(args: argparse.Namespace) -> tuple[EventStore, list]:
    store = EventStore(args.runs_root, args.run_id)
    records = store.completed_attempts()
    if not records:
        raise ArtifactError(f"run {args.run_id} has no completed attempts")
    return store, records


def command_growth(args: argparse.Namespace) -> int:
    _, records = _run_records(args)
    growth = context_growth(records)
    if args.json:
        print(json.dumps(growth, indent=2, sort_keys=True))
        return 0
    print(f"Mean input tokens by attempt, run {args.run_id}")
    width = max(len(strategy) for strategy in growth)
    attempts = sorted({row["attempt"] for rows in growth.values() for row in rows})
    print(" " * (width + 2) + "".join(f"{f'#{n}':>8}" for n in attempts))
    for strategy, rows in growth.items():
        by_attempt = {row["attempt"]: row["mean_input_tokens"] for row in rows}
        cells = "".join(f"{by_attempt.get(n, ''):>8}" for n in attempts)
        print(f"  {strategy:<{width}}{cells}")
    return 0


def command_sweep(args: argparse.Namespace) -> int:
    store, records = _run_records(args)
    budgets: list[int | None] = [None]
    for raw in args.budgets.split(","):
        if raw.strip():
            value = int(raw)
            if value <= 0:
                raise ConfigurationError("budgets must be positive")
            budgets.append(value)
    config = store.read_manifest()["config"]
    problems = load_problems(
        Path(config["dataset_path"]),
        dataset_name=config.get("dataset_name", "humaneval-plus"),
    )
    sweep = budget_sweep(records, problems, budgets)
    if args.json:
        print(json.dumps(sweep, indent=2, sort_keys=True))
        return 0
    if not sweep:
        print(f"Run {args.run_id} has no bounded-memory trajectories to recompile")
        return 0
    print(f"Context budget sweep, run {args.run_id} (estimated tokens)")
    for strategy, rows in sweep.items():
        print(f"  {strategy}")
        for row in rows:
            if row["budget_tokens"] is None:
                print(
                    f"    unbounded: {row['attempts']} attempts, "
                    f"{row['estimated_tokens']} tokens; "
                    f"{row['reproduced']} prompts reproduced byte-for-byte, "
                    f"estimate covers billed tokens on {row['estimate_covers']}"
                )
                continue
            print(
                f"    {row['budget_tokens']:>6}: {row['estimated_tokens']} tokens, "
                f"{row['trimmed']} trimmed (code dropped {row['dropped_code']}, "
                f"memories dropped {row['dropped_memories']}), "
                f"{row['unfit']} could not fit the task"
            )
    return 0


def command_stopsim(args: argparse.Namespace) -> int:
    limits = []
    for raw in args.k.split(","):
        if raw.strip():
            value = int(raw)
            if value < 1:
                raise ConfigurationError("--k values must be at least 1")
            limits.append(value)
    records = []
    for run_id in args.run_ids:
        attempts = EventStore(args.runs_root, run_id).completed_attempts()
        # Pooled runs repeat task IDs; keep each run's trajectories separate.
        if len(args.run_ids) > 1:
            attempts = [
                replace(item, task_id=f"{run_id}/{item.task_id}") for item in attempts
            ]
        records.extend(attempts)
    simulation = stop_simulation(records, limits)
    if args.json:
        print(json.dumps(simulation, indent=2, sort_keys=True))
        return 0
    if not simulation:
        print("No bounded-memory trajectories to replay")
        return 0
    print(f"Stop-when-stuck simulation, {len(args.run_ids)} run(s) (recorded usage)")
    for strategy, rows in simulation.items():
        base = rows[0]
        print(f"  {strategy}: {base['trajectories']} trajectories")
        for row in rows:
            label = (
                "never stop" if row["stop_after_identical"] is None
                else f"K={row['stop_after_identical']}"
            )
            print(
                f"    {label:>10}: solved {row['solved']} "
                f"({row['solved'] - base['solved']:+d}), "
                f"abandoned {row['abandoned']}, "
                f"output {_change(row['output_tokens'], base['output_tokens'])}, "
                f"cost ${row['estimated_cost_usd']:.2f} "
                f"({_change(row['estimated_cost_usd'], base['estimated_cost_usd'])})"
            )
    return 0


def _change(value: float, base: float) -> str:
    return f"{value / base - 1:+.1%}" if base else "n/a"


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
        if args.command == "preflight":
            return command_preflight(args)
        if args.command == "benchmark":
            return command_benchmark(args)
        if args.command == "resume":
            return command_resume(args)
        if args.command == "report":
            return command_report(args)
        if args.command == "solve":
            return command_solve(args)
        if args.command == "growth":
            return command_growth(args)
        if args.command == "sweep":
            return command_sweep(args)
        if args.command == "stopsim":
            return command_stopsim(args)
    except (
        ConfigurationError,
        ContextBudgetError,
        DatasetError,
        ArtifactError,
        PreflightError,
    ) as exc:
        parser.error(str(exc))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
