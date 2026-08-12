# TokenOS

TokenOS is an experimental context-management runtime for coding agents. Phase 1
provides a reproducible benchmark harness for comparing two retry strategies:

- **No Memory:** each attempt receives only the original problem.
- **Full History:** each attempt receives every prior completion and one test
  counterexample per failure.

The harness uses Azure OpenAI for generation, HumanEval+ for problems and oracle
tests, and a locked-down Docker container for generated-code execution.

## Prerequisites

- Python 3.12
- Docker Desktop with at least 2 GB available memory
- An Azure subscription with access to Microsoft Foundry Models
- Azure OpenAI token prices for the selected region and deployment type
- For Entra ID authentication, Azure CLI or another credential supported by
  `DefaultAzureCredential`

## Azure Deployment

Provision the model manually in the Microsoft Foundry portal:

1. Create or select an Azure OpenAI resource in a region that supports the
   Responses API and `gpt-5.4-mini`.
2. Create a **Global Standard** deployment.
3. Select model `gpt-5.4-mini`, version `2026-03-17`.
4. Name the deployment `tokenos-gpt-54-mini`.
5. Confirm that the deployment quota supports four concurrent trajectories.
6. Copy the Azure OpenAI endpoint. For API-key authentication, also copy a key.

TokenOS passes the deployment name in the API `model` field. It uses the v1
endpoint at `<endpoint>/openai/v1/`, reasoning effort `medium`, a 4,096-token
output limit, structured JSON output, and `store=false`.

## Installation

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
docker build -t tokenos-evalplus:0.3.1 -f docker/Dockerfile .
tokenos dataset fetch
```

The evaluator image pins EvalPlus `0.3.1` and preloads HumanEval+ `v0.1.10`.
The downloaded host dataset is checked as a valid 164-task gzip JSONL file and
against SHA-256 `272720b90ac375502c8ed23cd791c2a93dfb22a911641a494da74a426c09f101`.
That digest is also written into every run manifest.

## Configuration

Set the environment variables shown in `.env.example`. Token prices must match
the Azure region and deployment type because they drive the hard spend guard.

API-key authentication:

```bash
export AZURE_OPENAI_AUTH=api_key
export AZURE_OPENAI_ENDPOINT=https://YOUR-RESOURCE.openai.azure.com
export AZURE_OPENAI_DEPLOYMENT=tokenos-gpt-54-mini
export AZURE_OPENAI_API_KEY=...
```

Entra ID authentication:

```bash
az login
export AZURE_OPENAI_AUTH=entra
export AZURE_OPENAI_ENDPOINT=https://YOUR-RESOURCE.openai.azure.com
export AZURE_OPENAI_DEPLOYMENT=tokenos-gpt-54-mini
```

For both authentication modes:

```bash
export AZURE_OPENAI_INPUT_USD_PER_MTOK=...
export AZURE_OPENAI_CACHED_INPUT_USD_PER_MTOK=...
export AZURE_OPENAI_OUTPUT_USD_PER_MTOK=...
```

Validate local setup without spending tokens, then make an explicit live check:

```bash
tokenos doctor
tokenos doctor --live
```

## Running Benchmarks

Run the fixed five-task smoke suite:

```bash
tokenos benchmark \
  --strategies no-memory,full-history \
  --task-ids HumanEval/0,HumanEval/1,HumanEval/2,HumanEval/3,HumanEval/4 \
  --budget-usd 10 \
  --concurrency 4
```

Run all 164 HumanEval+ tasks once per strategy:

```bash
tokenos benchmark \
  --strategies no-memory,full-history \
  --all \
  --budget-usd 10 \
  --concurrency 4
```

If the hard spend cap or an infrastructure error stops a run, its artifacts stay
resumable:

```bash
tokenos resume RUN_ID
tokenos resume RUN_ID --budget-usd 20
tokenos report RUN_ID
tokenos report RUN_ID --json
```

The `$10` guard is an estimate based on configured rates and API-reported usage,
not an Azure billing guarantee. Before each call, TokenOS reserves a conservative
worst-case amount for the prompt and maximum output. It stops scheduling calls
that cannot fit within the remaining budget. Raising the total cap requires an
explicit resume argument and is recorded as an append-only run event.

## Results

Each run writes an ignored `runs/<run-id>/` directory:

- `manifest.json` contains immutable configuration and a reproducibility hash.
- `events.jsonl` records append-only request and attempt lifecycle events.
- `summary.json` contains completeness, correctness, token, cost, and latency metrics.
- `samples.jsonl` contains the final EvalPlus-compatible sample per task and strategy.

Adaptive retries are reported as `solve@1`, `solve@3`, and `solve@5`. These are
cumulative solve rates, not the classical independent-sampling `pass@k` estimator.
A task succeeds only when both HumanEval base and HumanEval+ tests pass.

## Execution Boundary

Every generated program runs in a fresh container with no network, no workspace
mount, no Azure credentials, a read-only root filesystem, dropped Linux
capabilities, `no-new-privileges`, process and memory caps, one CPU, temporary
storage, and a host timeout. The official EvalPlus checker determines correctness.
A separate diagnostic process returns at most one sanitized counterexample.

Docker substantially reduces risk but is not a perfect hostile-code sandbox. Do
not weaken these controls or pass host secrets into the evaluator container.

## Tests

```bash
python -m pytest
```

Live checks are opt-in:

```bash
TOKENOS_RUN_AZURE_TEST=1 python -m pytest -m azure
TOKENOS_RUN_DOCKER_SECURITY_TESTS=1 python -m pytest -m docker
```

PatchSearch, generalized memory objects, token allocation, and the interactive
demo begin in later phases and are intentionally absent from this implementation.
