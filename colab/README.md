# Output cost on a self-hosted GPU

TokenOS cuts the tokens an agent sends; this folder measures what it costs to
*generate* tokens on hardware you control, using TokenOS's real traffic. It
runs in Google Colab on a free T4 (16 GB, Turing: fp16 only, no
FlashAttention-2). Nothing here spends API money.

| Notebook | What it measures | Time |
|---|---|---|
| [`serving_bench.ipynb`](serving_bench.ipynb) | vLLM serving of Qwen2.5-Coder on recorded TokenOS prompts: prefix caching, n-gram speculative decoding, quantization, concurrency | ~1.5 h |
| [`kernels.ipynb`](kernels.ipynb) | Hand-written Triton kernels: fused softmax, decode attention (single and split-K) vs PyTorch, as achieved memory bandwidth | ~10 min |

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
         pip install -q triton pytest && python -m pytest -q tests/test_colab_kernels.py"
```

Neither has run on a real GPU yet: no performance number exists until the
notebooks are run. vLLM is pinned to 0.6.6.post1, the V0 engine, which
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
