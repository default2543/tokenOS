# Output cost on a self-hosted GPU

TokenOS cuts the tokens an agent sends; this folder measures what it costs to
*generate* tokens on hardware you control, using TokenOS's real traffic. It
runs in Google Colab on a free T4 (16 GB, Turing: fp16 only, no
FlashAttention-2). Nothing here spends API money.

| Notebook | What it measures | Time |
|---|---|---|
| [`serving_bench.ipynb`](serving_bench.ipynb) | vLLM serving of Qwen2.5-Coder on recorded TokenOS prompts: prefix caching, n-gram speculative decoding, quantization, concurrency | ~1.5 h |
| [`kernels.ipynb`](kernels.ipynb) | Hand-written Triton kernels: fused softmax, decode attention (single, split-K and GQA-grouped) vs PyTorch, as achieved memory bandwidth | ~3 min |

The serving notebook installs vLLM 0.6.6 into a separate Python 3.12
environment (`/content/vllm-env`), because Colab's Python is 3.13 and that
vLLM supports up to 3.12; Colab's own PyTorch and Triton are left alone.

## The workload

`workloads/bcb-hard-8039bc.jsonl.gz` is every request from the full
BigCodeBench-Hard run `20261011T184316Z-8039bc` (458 full-history, 516
adaptive), made with:

```bash
tokenos export-prompts 20261011T184316Z-8039bc --out workload.jsonl
```

`serving_bench.py` samples whole trajectories for the same tasks under both
strategies and replays each trajectory's attempts in order, so a prefix cache
can hit across attempts exactly as it would for a live agent. Generation is
fixed at 256 tokens at temperature 0, so configurations differ only in how
they serve.

Before sampling, it reports how many trajectories cannot fit the server's
context at all. Using recorded prompt sizes:

| Context window | Full history overflows | Adaptive overflows |
|---|---|---|
| 8K | 14 of 148 | 0 (longest prompt 3,095) |
| 16K | 5 of 148 | 0 |
| 32K | 0 of 148 | 0 |

## What each serving configuration tests

1. **baseline**: Qwen2.5-Coder-7B-Instruct, AWQ 4-bit (fits the T4).
2. **+ prefix caching**: reuses the cached prompt prefix across requests. Full
   history's prompts mostly extend the previous attempt's; adaptive rebuilds
   them. This answers whether bounded memory gives up cache reuse.
3. **+ n-gram speculative decoding**: a retry mostly rewrites the previous
   attempt's code, which is in the prompt. Prompt lookup proposes those tokens
   without a draft model; acceptance should be high for exactly the requests
   TokenOS generates.
4. **quantization**: Qwen2.5-Coder-1.5B, fp16 vs AWQ.
5. **concurrency**: 1, 4 and 16 trajectories at once.

`$ per million output tokens` converts throughput at a reference T4 price of
$0.35/hour (`--usd-per-hour`); it compares configurations, it is not a quote.

## Status

The serving script's sampling, metrics parsing and summaries are unit-tested
locally (`tests/test_colab_serving.py`). The Triton kernels pass their
correctness tests under Triton's CPU interpreter:

```bash
docker run --rm -v "$PWD":/repo:ro -w /repo -e TRITON_INTERPRET=1 python:3.12-slim \
  sh -c "pip install -q torch --index-url https://download.pytorch.org/whl/cpu && \
         pip install -q numpy triton pytest && python -m pytest -q tests/test_colab_kernels.py"
```

Raw results for every run are in [`results/`](results/) (`kernels-t4-run1.json` to
`kernels-t4-run4-vllm.json`).

### First kernel run (T4, before the grouped kernel)

Decode attention, Qwen2.5-7B shapes, fp16, median of `do_bench`:

| Batch | Context | Single-pass | Split-K (8) | Split-K speedup |
|--:|--:|--:|--:|--:|
| 1 | 1,536 | 132 µs (24 GB/s) | 78 µs (40 GB/s) | 1.7x |
| 1 | 8,192 | 650 µs (26 GB/s) | 247 µs (68 GB/s) | 2.6x |
| 1 | 16,384 | 1,283 µs (26 GB/s) | 473 µs (71 GB/s) | 2.7x |
| 8 | 8,192 | 1,980 µs (68 GB/s) | 1,790 µs (75 GB/s) | 1.1x |

Split-K helps exactly when batch × heads underfills the T4's 40 SMs. At the
split-K kernel's speeds, attention for one output token across all 28 layers
costs 2.2 ms at TokenOS's ~1.5K context and 6.9 ms at full history's ~8K.

Two problems this run exposed, both addressed in the current code:

* **Both kernels peak near 22% of bandwidth.** They run one program per
  *query* head, so each KV block is read once per query head in its group
  (7x for Qwen2.5-7B) while the bandwidth figure counts it once.
  `decode_attention_grouped` runs one program per KV head and scores the
  whole group with `tl.dot`.
* **`torch-sdpa` measured ~4 GB/s**, an unfused fallback with `enable_gqa` on
  Turing, not a meaningful baseline. `torch-grouped` (one batched matmul per KV
  head, no KV copy) is now the eager PyTorch baseline; SDPA is still reported.

### Second kernel run (T4, grouped kernel and fair baseline)

Decode attention, µs (GB/s). Triton 3.6, PyTorch 2.11:

| Batch | Context | Split-K | Grouped | `torch-grouped` | `torch-sdpa` |
|--:|--:|--:|--:|--:|--:|
| 1 | 1,536 | 78 (41) | 101 (31) | 82 (38) | 674 (5) |
| 1 | 8,192 | 255 (66) | 354 (47) | 323 (52) | 4,183 (4) |
| 1 | 16,384 | 505 (66) | 664 (51) | 645 (52) | 8,292 (4) |
| 8 | 1,536 | 375 (67) | 506 (50) | 280 (90) | 4,959 (5) |
| 8 | 8,192 | 1,937 (69) | 2,430 (55) | 1,626 (83) | 32,831 (4) |

* **The grouped kernel is slower than split-K**, so redundant KV reads were
  not the main limit: L2 absorbs most of them. At batch 1 it launches only
  4 KV heads x 8 splits = 32 programs for 40 SMs.
* **Eager PyTorch with grouped matmuls beats every Triton kernel at batch 8**
  (by about 20%) and loses to split-K at batch 1 from 4K context (1.2-1.3x).
* **Everything sits at 20-28% of peak,** while the fused softmax reaches
  230 GB/s (72%), so about 230 GB/s is attainable on this T4.
* `torch-sdpa` stays at ~4 GB/s: the fallback path, as suspected.

`bench.py` now sweeps splits, block size and warps for both kernels and
checks whether `tl.dot` compiles to tensor-core `mma.sync` on Turing.

### Third kernel run (T4, launch-parameter sweep)

45 configurations per kernel and shape (splits 4-64, block 32-128 tokens,
2-8 warps), each checked against the reference before timing. Best split-K
configuration vs the defaults used above and vs `torch-grouped`, in µs:

| Batch | Context | Default split-K | Tuned split-K (config) | GB/s, % of peak | `torch-grouped` | Tuned vs PyTorch |
|--:|--:|--:|--:|--:|--:|--:|
| 1 | 1,536 | 78 | 66 (4 splits, 128, 2 warps) | 48, 15% | 81 | 1.23x |
| 1 | 8,192 | 246 | 147 (32 splits, 128, 2 warps) | 114, 36% | 317 | 2.15x |
| 8 | 1,536 | 356 | 198 (4 splits, 128, 2 warps) | 127, 40% | 266 | 1.34x |
| 8 | 8,192 | 1,805 | 981 (4-8 splits, 128, 2 warps) | 137, 43% | 1,540 | 1.57x |

* **Tuning reverses the PyTorch comparison.** The defaults (8 splits,
  64-token blocks, 4 warps) lost to eager PyTorch at batch 8; tuned split-K
  wins at every shape. Sweep timings use a shorter repetition window than
  the main table, so the next run confirms them with the tuned defaults.
* **Large blocks with few warps win everywhere.** Each thread then has more
  independent loads in flight, which a memory-bound kernel on Turing needs:
  the T4 has no asynchronous copies to pipeline loads in hardware.
* **The grouped kernel is a dead end on this GPU.** `tl.dot` does not compile
  to tensor-core `mma.sync` on Turing with Triton 3.6, every 128-token block
  configuration ran out of shared memory, and its best time trails split-K
  at every shape.

`decode_attention_split` now defaults to 128-token blocks, 2 warps and about
256 cached tokens per split (`default_splits`). The sweep now extends past the
old grid's edge (256-token blocks, 1 warp) for split-K only.

### Fourth kernel run (T4, vLLM 0.6.6 environment: Triton 3.1, PyTorch 2.5.1)

Run in the serving notebook's Python 3.12 environment, next to vLLM's own
decode kernel. Every implementation passed the correctness check first.
Eager timings, µs; bandwidth is the grouped kernel's:

| Batch | Context | Triton grouped | vLLM PagedAttention | vs vLLM | vs `torch-grouped` | Bandwidth |
|--:|--:|--:|--:|--:|--:|--:|
| 1 | 1,024 | 18.3 | 30.5 | 1.67x | 3.4x | 115 GB/s, 36% |
| 1 | 1,536 | 23.8 | 34.8 | 1.46x | 3.3x | 132 GB/s, 41% |
| 1 | 4,096 | 49.4 | 71.4 | 1.44x | 3.5x | 170 GB/s, 53% |
| 1 | 8,192 | 81.0 | 115.0 | 1.42x | 3.9x | 207 GB/s, 65% |
| 1 | 16,384 | 141.3 | 218.1 | 1.54x | 4.5x | 237 GB/s, 74% |
| 8 | 1,024 | 89.4 | 115.4 | 1.29x | 2.1x | 188 GB/s, 59% |
| 8 | 1,536 | 122.0 | 172.0 | 1.41x | 2.3x | 206 GB/s, 64% |
| 8 | 4,096 | 282.7 | 442.0 | 1.56x | 2.7x | 237 GB/s, 74% |
| 8 | 8,192 | 541.3 | 937.3 | 1.73x | 2.9x | 248 GB/s, 77% |
| 8 | 16,384 | 1,062.9 | 2,026.4 | 1.91x | 3.1x | 253 GB/s, 79% |

* **The Triton version decides the design.** Under Triton 3.1 the grouped
  kernel's `tl.dot` compiles to tensor-core `mma.sync` on Turing; under
  Colab's Triton 3.6 it did not, and the same kernel was the slowest Triton
  variant. With tensor cores it is 2.1-4.5x faster than eager PyTorch and the
  fastest kernel measured at every shape.
* **vs vLLM, with the caveat:** the Triton kernels read a contiguous KV cache;
  vLLM's reads a paged one (16-token blocks through a block table), which is
  what lets it serve many requests without fragmentation. Part of the margin
  is that indirection, not kernel quality alone.
* Split-K is now slower than vLLM at every shape (e.g. 190 vs 115 µs at
  batch 1, 8K), and its tuned defaults came from Triton 3.6; the sweep under
  3.1 preferred 64-token blocks and 1 warp.
* **CUDA-graph timing failed to capture** for every implementation in this
  environment, so only eager timings exist; the cause is not yet known.

The serving notebook has not run on a GPU yet. vLLM is pinned to 0.6.6.post1, the V0 engine, which
supports Turing; if a flag fails on that version, the smoke-test cell shows it
before the long runs.

## Optional: run TokenOS end to end against Colab

The Docker evaluator needs your machine, so serve in Colab and evaluate
locally. Start vLLM with an API key, expose it with a `cloudflared` quick
tunnel, and point TokenOS at it:

```bash
# Colab
python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-Coder-7B-Instruct-AWQ \
  --quantization awq --dtype half --enable-prefix-caching --api-key "$KEY" &
cloudflared tunnel --url http://localhost:8000

# Local
TOKENOS_PROVIDER_API=chat AZURE_OPENAI_ENDPOINT=https://<tunnel>/v1 \
AZURE_OPENAI_API_KEY="$KEY" AZURE_OPENAI_DEPLOYMENT=Qwen/Qwen2.5-Coder-7B-Instruct-AWQ \
TOKENOS_MODEL_NAME=qwen2.5-coder-7b-awq \
AZURE_OPENAI_INPUT_USD_PER_MTOK=0 AZURE_OPENAI_CACHED_INPUT_USD_PER_MTOK=0 \
AZURE_OPENAI_OUTPUT_USD_PER_MTOK=0 \
tokenos benchmark --dataset bigcodebench-hard --all --skip-unviable-tasks \
  --strategies full-history,patchsearch-adaptive --max-attempts 10 --budget-usd 1
```

A quick tunnel is public: the API key is the only thing protecting the
server, so always set one and stop the tunnel when finished.
