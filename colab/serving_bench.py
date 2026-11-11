"""Replay real TokenOS traffic against a self-hosted vLLM server.

Runs in Colab on a T4. Each configuration starts its own OpenAI-compatible
vLLM server, replays the same sampled trajectories (each trajectory's requests
in order, trajectories concurrent), and records time to first token, decode
rate, throughput, prefix-cache hit rate, and an estimated $ per million output
tokens. Generation length and temperature are fixed so configurations differ
only in how they serve, not in what they generate.

The workload comes from `tokenos export-prompts`; only its prompts are used.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
import gzip
import json
import os
from pathlib import Path
import random
import re
import secrets
import statistics
import subprocess
import sys
import time
from typing import Iterator
import urllib.request

# Pinned so results are reproducible; this release runs the V0 engine, which
# supports Turing (T4), AWQ, prefix caching and n-gram speculative decoding.
VLLM_VERSION = "0.6.6.post1"
TARGET_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ"
SMALL_FP16 = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
SMALL_AWQ = "Qwen/Qwen2.5-Coder-1.5B-Instruct-AWQ"
# Reference on-demand price for a T4; only used to convert throughput to $.
T4_USD_PER_HOUR = 0.35
PORT = 8000


@dataclass(frozen=True)
class ServeConfig:
    name: str
    model: str
    prefix_caching: bool = False
    ngram_speculation: bool = False
    quantization: str | None = None
    extra: tuple[str, ...] = ()

    def args(self, max_model_len: int) -> list[str]:
        args = [
            "--model", self.model,
            "--dtype", "half",  # T4 has no bf16
            "--max-model-len", str(max_model_len),
            "--gpu-memory-utilization", "0.90",
            "--disable-log-requests",
        ]
        if self.quantization:
            args += ["--quantization", self.quantization]
        if self.prefix_caching:
            args.append("--enable-prefix-caching")
        if self.ngram_speculation:
            # Retries rewrite code that is already in the prompt; prompt lookup
            # can propose those tokens without a draft model.
            args += [
                "--speculative-model", "[ngram]",
                "--num-speculative-tokens", "5",
                "--ngram-prompt-lookup-max", "4",
            ]
        return args + list(self.extra)


CONFIGS = {
    "baseline": ServeConfig("baseline", TARGET_MODEL, quantization="awq"),
    "prefix-cache": ServeConfig(
        "prefix-cache", TARGET_MODEL, quantization="awq", prefix_caching=True
    ),
    "prefix-cache+ngram": ServeConfig(
        "prefix-cache+ngram", TARGET_MODEL, quantization="awq",
        prefix_caching=True, ngram_speculation=True,
    ),
    "small-fp16": ServeConfig("small-fp16", SMALL_FP16),
    "small-awq": ServeConfig("small-awq", SMALL_AWQ, quantization="awq"),
}


# ---------------------------------------------------------------- workload


def load_workload(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def context_overflow(rows: list[dict], max_prompt_tokens: int) -> dict[str, dict]:
    """Per strategy: trajectories with any request too long for the server.

    Uses API-reported input tokens. This is the failure full history is
    headed for: its prompt grows until a fixed context window cannot hold it.
    """
    longest: dict[tuple[str, str], int] = defaultdict(int)
    for row in rows:
        key = (row["strategy"], row["task_id"])
        longest[key] = max(longest[key], row["recorded_input_tokens"])
    report: dict[str, dict] = {}
    for (strategy, _), tokens in longest.items():
        entry = report.setdefault(strategy, {"trajectories": 0, "overflow": 0,
                                             "longest_prompt_tokens": 0})
        entry["trajectories"] += 1
        entry["overflow"] += tokens > max_prompt_tokens
        entry["longest_prompt_tokens"] = max(entry["longest_prompt_tokens"], tokens)
    return report


def sample_trajectories(
    rows: list[dict], *, tasks: int, min_attempts: int, seed: int = 0,
    max_prompt_tokens: int | None = None,
) -> dict[str, list[list[dict]]]:
    """Whole trajectories for the same tasks under every strategy.

    Prefix caching can only hit across a trajectory's own attempts, so requests
    are kept together and in order. Tasks are those every strategy retried at
    least `min_attempts` times, so long-context retries are represented, and
    whose every request fits `max_prompt_tokens` under every strategy.
    """
    by_key: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        by_key[(row["strategy"], row["task_id"])].append(row)
    strategies = sorted({strategy for strategy, _ in by_key})
    def fits(task: str) -> bool:
        return max_prompt_tokens is None or all(
            row["recorded_input_tokens"] <= max_prompt_tokens
            for s in strategies for row in by_key.get((s, task), [])
        )

    eligible = sorted(
        task
        for task in {task for _, task in by_key}
        if all(len(by_key.get((s, task), [])) >= min_attempts for s in strategies)
        and fits(task)
    )
    chosen = random.Random(seed).sample(eligible, min(tasks, len(eligible)))
    return {
        strategy: [
            sorted(by_key[(strategy, task)], key=lambda row: row["attempt"])
            for task in sorted(chosen)
        ]
        for strategy in strategies
    }


# ------------------------------------------------------------------ server


def _get(url: str, token: str, timeout: float = 5.0) -> str:
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode()


@contextmanager
def vllm_server(
    config: ServeConfig, *, max_model_len: int, log_dir: Path
) -> Iterator[tuple[str, str]]:
    token = secrets.token_hex(16)
    log_dir.mkdir(parents=True, exist_ok=True)
    log = (log_dir / f"{config.name}.log").open("w")
    command = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--host", "127.0.0.1", "--port", str(PORT), "--api-key", token,
        *config.args(max_model_len),
    ]
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
    base = f"http://127.0.0.1:{PORT}"
    try:
        deadline = time.monotonic() + 900
        while True:
            if process.poll() is not None:
                raise RuntimeError(
                    f"vLLM exited during startup; see {log_dir / (config.name + '.log')}"
                )
            try:
                _get(f"{base}/v1/models", token)
                break
            except OSError:
                if time.monotonic() > deadline:
                    raise TimeoutError("vLLM did not become ready in 15 minutes")
                time.sleep(5)
        yield base, token
    finally:
        process.terminate()
        try:
            process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            process.kill()
        log.close()


def prefix_cache_counters(metrics_text: str) -> dict[str, float]:
    """Prefix-cache counters from Prometheus text, across vLLM metric names."""
    found: dict[str, float] = {}
    for line in metrics_text.splitlines():
        if line.startswith("#"):
            continue
        match = re.match(r"^(vllm:[a-z_]*prefix_cache[a-z_]*)(\{[^}]*\})?\s+(\S+)$", line)
        if match:
            found[match.group(1)] = found.get(match.group(1), 0.0) + float(match.group(3))
    return found


def gpu_memory_mib() -> int | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        ).stdout
        return int(out.split()[0])
    except (OSError, subprocess.CalledProcessError, ValueError, IndexError):
        return None


# ------------------------------------------------------------------ replay


@dataclass
class RequestResult:
    strategy: str
    task_id: str
    attempt: int
    prompt_tokens: int
    completion_tokens: int
    ttft_s: float
    total_s: float

    @property
    def decode_tokens_per_s(self) -> float | None:
        decode_time = self.total_s - self.ttft_s
        if self.completion_tokens < 2 or decode_time <= 0:
            return None
        return (self.completion_tokens - 1) / decode_time


async def _one_request(client, model: str, row: dict, max_tokens: int) -> RequestResult:
    started = time.perf_counter()
    first: float | None = None
    usage = None
    stream = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": row["instructions"]},
            {"role": "user", "content": row["prompt"]},
        ],
        max_tokens=max_tokens,
        temperature=0.0,
        stream=True,
        stream_options={"include_usage": True},
        # Fixed length: every configuration decodes the same number of tokens.
        extra_body={"ignore_eos": True, "min_tokens": max_tokens},
    )
    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content and first is None:
            first = time.perf_counter()
        if chunk.usage is not None:
            usage = chunk.usage
    ended = time.perf_counter()
    return RequestResult(
        strategy=row["strategy"],
        task_id=row["task_id"],
        attempt=row["attempt"],
        prompt_tokens=usage.prompt_tokens if usage else 0,
        completion_tokens=usage.completion_tokens if usage else 0,
        ttft_s=(first or ended) - started,
        total_s=ended - started,
    )


async def replay(
    base: str, token: str, model: str, trajectories: list[list[dict]], *,
    concurrency: int, max_tokens: int,
) -> tuple[list[RequestResult], float]:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(base_url=f"{base}/v1", api_key=token, max_retries=0,
                         timeout=600)
    gate = asyncio.Semaphore(concurrency)
    results: list[RequestResult] = []

    async def run_trajectory(rows: list[dict]) -> None:
        async with gate:
            for row in rows:  # attempts in order, as the agent sent them
                results.append(await _one_request(client, model, row, max_tokens))

    started = time.perf_counter()
    await asyncio.gather(*(run_trajectory(rows) for rows in trajectories))
    return results, time.perf_counter() - started


# ----------------------------------------------------------------- summary


def summarize(
    results: list[RequestResult], wall_s: float, *, usd_per_hour: float
) -> dict:
    def stats(values: list[float]) -> dict:
        values = [v for v in values if v is not None]
        if not values:
            return {}
        return {"median": round(statistics.median(values), 4),
                "mean": round(statistics.fmean(values), 4)}

    output_tokens = sum(r.completion_tokens for r in results)
    throughput = output_tokens / wall_s if wall_s > 0 else 0.0
    by_strategy = defaultdict(list)
    for r in results:
        by_strategy[r.strategy].append(r)
    return {
        "requests": len(results),
        "wall_s": round(wall_s, 2),
        "output_tokens": output_tokens,
        "prompt_tokens": sum(r.prompt_tokens for r in results),
        "output_tokens_per_s": round(throughput, 2),
        "usd_per_million_output_tokens": (
            round(usd_per_hour / 3600 / throughput * 1e6, 4) if throughput else None
        ),
        "ttft_s": stats([r.ttft_s for r in results]),
        "decode_tokens_per_s": stats([r.decode_tokens_per_s for r in results]),
        "by_strategy": {
            strategy: {
                "requests": len(items),
                "mean_prompt_tokens": round(
                    statistics.fmean(r.prompt_tokens for r in items)
                ),
                "ttft_s": stats([r.ttft_s for r in items]),
                "decode_tokens_per_s": stats([r.decode_tokens_per_s for r in items]),
            }
            for strategy, items in sorted(by_strategy.items())
        },
    }


@dataclass
class RunPlan:
    workload: Path
    configs: list[str]
    tasks: int = 12
    min_attempts: int = 4
    concurrency: int = 4
    max_tokens: int = 256
    max_model_len: int = 16384
    usd_per_hour: float = T4_USD_PER_HOUR
    out: Path = Path("results/serving.json")
    seed: int = 0
    concurrency_sweep: list[int] = field(default_factory=list)


def run(plan: RunPlan) -> dict:
    rows = load_workload(plan.workload)
    # Leave room for the generated tokens and for tokenizer differences
    # between the recorded model and the served one.
    max_prompt = int(plan.max_model_len * 0.85) - plan.max_tokens
    sampled = sample_trajectories(
        rows, tasks=plan.tasks, min_attempts=plan.min_attempts, seed=plan.seed,
        max_prompt_tokens=max_prompt,
    )
    report: dict = {"vllm_version": VLLM_VERSION, "plan": {
        k: str(v) if isinstance(v, Path) else v for k, v in asdict(plan).items()
    }, "tasks": sorted({t[0]["task_id"] for ts in sampled.values() for t in ts}),
        "context_overflow": context_overflow(rows, max_prompt), "configs": {}}
    print("Trajectories that cannot fit this server's context:",
          report["context_overflow"])
    log_dir = plan.out.parent / "logs"
    for name in plan.configs:
        config = CONFIGS[name]
        entry: dict = {"config": asdict(config)}
        concurrencies = plan.concurrency_sweep or [plan.concurrency]
        with vllm_server(config, max_model_len=plan.max_model_len,
                         log_dir=log_dir) as (base, token):
            entry["gpu_memory_mib"] = gpu_memory_mib()
            for concurrency in concurrencies:
                for strategy, trajectories in sampled.items():
                    before = prefix_cache_counters(_get(f"{base}/metrics", token))
                    results, wall = asyncio.run(replay(
                        base, token, config.model, trajectories,
                        concurrency=concurrency, max_tokens=plan.max_tokens,
                    ))
                    after = prefix_cache_counters(_get(f"{base}/metrics", token))
                    summary = summarize(results, wall, usd_per_hour=plan.usd_per_hour)
                    summary["prefix_cache_metrics"] = {
                        key: after.get(key, 0.0) - before.get(key, 0.0)
                        if key.endswith("_total") else after.get(key)
                        for key in after
                    }
                    entry.setdefault(f"concurrency_{concurrency}", {})[strategy] = summary
                    print(f"{name} c={concurrency} {strategy}: "
                          f"{summary['output_tokens_per_s']} tok/s, "
                          f"TTFT {summary['ttft_s'].get('median')}s, "
                          f"${summary['usd_per_million_output_tokens']}/M out")
        report["configs"][name] = entry
        plan.out.parent.mkdir(parents=True, exist_ok=True)
        plan.out.write_text(json.dumps(report, indent=2))  # checkpoint per config
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--configs", default="baseline,prefix-cache,prefix-cache+ngram")
    parser.add_argument("--tasks", type=int, default=12)
    parser.add_argument("--min-attempts", type=int, default=4)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--concurrency-sweep", default="")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--usd-per-hour", type=float, default=T4_USD_PER_HOUR)
    parser.add_argument("--out", type=Path, default=Path("results/serving.json"))
    args = parser.parse_args(argv)
    unknown = set(args.configs.split(",")) - set(CONFIGS)
    if unknown:
        parser.error(f"unknown configs {sorted(unknown)}; choose from {sorted(CONFIGS)}")
    run(RunPlan(
        workload=args.workload, configs=args.configs.split(","), tasks=args.tasks,
        min_attempts=args.min_attempts, concurrency=args.concurrency,
        max_tokens=args.max_tokens, max_model_len=args.max_model_len,
        usd_per_hour=args.usd_per_hour, out=args.out,
        concurrency_sweep=[int(c) for c in args.concurrency_sweep.split(",") if c],
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
