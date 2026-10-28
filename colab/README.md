# Output cost on a self-hosted GPU

TokenOS cuts the tokens an agent sends; this folder measures what it costs to
*generate* tokens on hardware you control, using TokenOS's real traffic. It
runs in Google Colab on a free T4 (16 GB, Turing: fp16 only, no
FlashAttention-2). Nothing here spends API money.

| Notebook | What it measures | Time |
|---|---|---|
| [`serving_bench.ipynb`](serving_bench.ipynb) | vLLM serving of Qwen2.5-Coder on recorded TokenOS prompts: prefix caching, n-gram speculative decoding, quantization, concurrency | ~1.5 h |
| [`kernels.ipynb`](kernels.ipynb) | Hand-written Triton kernels: fused softmax, decode attention (single, split-K and GQA-grouped) vs PyTorch, as achieved memory bandwidth | ~3 min |

Run each in its **own runtime**: installing vLLM replaces Colab's PyTorch.

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
