from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "colab"))
import serving_bench as bench  # noqa: E402


def _row(strategy, task, attempt):
    return {"strategy": strategy, "task_id": task, "attempt": attempt,
            "instructions": "i", "prompt": f"{strategy}-{task}-{attempt}"}


def test_sampling_keeps_whole_ordered_trajectories_for_shared_tasks() -> None:
    rows = [_row(s, t, a) for s in ("full-history", "patchsearch-adaptive")
            for t, n in (("A", 5), ("B", 2), ("C", 4)) for a in range(n, 0, -1)]
    sampled = bench.sample_trajectories(rows, tasks=5, min_attempts=4)
    for strategy, trajectories in sampled.items():
        assert [t[0]["task_id"] for t in trajectories] == ["A", "C"]  # B too short
        assert [r["attempt"] for r in trajectories[0]] == [1, 2, 3, 4, 5]


def test_prefix_cache_counters_sum_labelled_series() -> None:
    text = "\n".join([
        "# HELP vllm:gpu_prefix_cache_hit_rate x",
        'vllm:gpu_prefix_cache_hit_rate{model_name="m"} 0.25',
        'vllm:prefix_cache_queries_total{model_name="m"} 100.0',
        'vllm:prefix_cache_hits_total{model_name="m"} 40.0',
        'vllm:num_requests_running{model_name="m"} 1.0',
    ])
    assert bench.prefix_cache_counters(text) == {
        "vllm:gpu_prefix_cache_hit_rate": 0.25,
        "vllm:prefix_cache_queries_total": 100.0,
        "vllm:prefix_cache_hits_total": 40.0,
    }


def test_summary_converts_throughput_to_dollars() -> None:
    results = [bench.RequestResult("s", "t", 1, 1000, 101, 0.5, 1.5)]
    summary = bench.summarize(results, wall_s=1.0, usd_per_hour=3.6)
    assert summary["output_tokens_per_s"] == 101.0
    # $3.60/hour = $0.001/s; at 101 tok/s that is $0.001 / 101 per token.
    assert summary["usd_per_million_output_tokens"] == round(0.001 / 101 * 1e6, 4)
    assert summary["decode_tokens_per_s"]["median"] == 100.0  # 100 tokens in 1s


def test_server_flags_for_each_optimization() -> None:
    args = bench.CONFIGS["prefix-cache+ngram"].args(8192)
    assert "--enable-prefix-caching" in args
    assert args[args.index("--speculative-model") + 1] == "[ngram]"
    assert args[args.index("--quantization") + 1] == "awq"
    assert args[args.index("--dtype") + 1] == "half"
    assert "--enable-prefix-caching" not in bench.CONFIGS["baseline"].args(8192)


def test_overflowing_trajectories_are_counted_and_not_sampled() -> None:
    rows = [dict(_row(s, t, a), recorded_input_tokens=tokens)
            for s, t, tokens in (("full-history", "A", 9000),
                                 ("patchsearch-adaptive", "A", 1200),
                                 ("full-history", "B", 3000),
                                 ("patchsearch-adaptive", "B", 1100))
            for a in (1, 2, 3, 4)]
    overflow = bench.context_overflow(rows, max_prompt_tokens=8000)
    assert overflow["full-history"] == {"trajectories": 2, "overflow": 1,
                                        "longest_prompt_tokens": 9000}
    assert overflow["patchsearch-adaptive"]["overflow"] == 0
    sampled = bench.sample_trajectories(rows, tasks=5, min_attempts=4,
                                        max_prompt_tokens=8000)
    assert [t[0]["task_id"] for t in sampled["full-history"]] == ["B"]
