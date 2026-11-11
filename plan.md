# TokenOS — Adaptive Context Management for LLMs

## 1. Project Overview

**TokenOS** is a context-management runtime designed to reduce the number of tokens required by LLM applications while preserving—or improving—model performance.

Modern LLM applications frequently accumulate large amounts of context: conversation history, previous model outputs, failed attempts, tool results, error messages, retrieved documents, and intermediate reasoning. Applications often solve this by retaining the entire history, truncating older information, or summarizing it.

TokenOS takes a different approach.

Before every model call, TokenOS determines **which information is actually valuable for the current task, how that information should be represented, and whether it deserves space in the model's limited context window.**

The long-term goal is to create something analogous to a **memory manager for LLM context**.

Instead of:

```text
Application
    ↓
Entire History
    ↓
LLM
```

TokenOS introduces an optimization layer:

```text
Application
    ↓
Raw Context / History
    ↓
TokenOS
    ├── Analyze
    ├── Store
    ├── Compress
    ├── Retrieve
    ├── Prioritize
    └── Allocate Token Budget
    ↓
Optimized Context
    ↓
LLM
```

The primary users would be developers building **LLM agents, coding agents, and long-running AI applications** where context continuously grows.

---

# 2. Problem

LLM context windows are finite and expensive.

Consider a coding agent attempting the same problem several times.

After five attempts, its context may contain:

* original task
* five generated solutions
* compiler/runtime errors
* test failures
* tool outputs
* debugging explanations
* model critiques
* previous corrections

Much of this information is redundant.

However, simply deleting old information creates another problem: the model may repeat mistakes it already encountered.

TokenOS therefore asks:

> **What is the minimum amount of context necessary for an LLM to retain the useful information it has learned?**

The goal is not merely to shorten prompts.

The goal is to maximize **useful information per token**.

---

# 3. Core Hypothesis

An LLM does not need its entire interaction history to benefit from previous experience.

Important information can often be converted into smaller representations while retaining most of its usefulness.

For example, suppose an agent generates:

```python
def solve(nums):
    return max(nums)
```

The implementation fails because `nums` can be empty.

A traditional agent may retain:

```text
Previous attempt:
[entire implementation]

Test execution:
[entire test output]

Traceback:
[stack trace]

Analysis:
The previous implementation failed because...
```

TokenOS could instead retain:

```python
assert solve([]) == expected_output
```

Hundreds or thousands of tokens of debugging history have now been converted into a tiny executable memory.

This leads to the central hypothesis:

> **Adaptive memory representations can achieve comparable or better LLM task performance using substantially fewer context tokens than full-history prompting.**

---

# 4. PatchSearch

The first specialized memory mechanism inside TokenOS will be **PatchSearch**.

PatchSearch converts useful information from failed attempts into compact, reusable constraints.

Pipeline:

```text
LLM generates solution
        ↓
Run tests
        ↓
Failure detected
        ↓
Analyze failure
        ↓
Extract minimal constraint
        ↓
Validate constraint
        ↓
Store patch
        ↓
Retrieve for future attempt
```

For example:

```text
Failure:
Implementation crashes on an empty array.

Full history:
~800 tokens

Patch:
assert solve([]) == []
```

Future generations receive the patch rather than the complete failure history.

PatchSearch therefore acts as a form of **executable memory**.

---

# 5. TokenOS Memory Model

TokenOS will support several types of memory.

### RAW

Original information preserved verbatim.

Examples:

* current user request
* important source code
* recent tool result

### FACT

Structured information extracted from previous context.

```text
language = Python
complexity_requirement = O(n log n)
database = PostgreSQL
```

### SUMMARY

Natural-language compression of older context.

```text
Previous attempts failed primarily because edge cases involving
negative numbers were not handled.
```

### PATCH

Executable constraints derived from failures.

```python
assert solve([-5, -2]) == -2
```

### DECISION

Previously established decisions that should not be repeatedly reconsidered.

```text
Selected algorithm: merge sort
Reason: stable O(n log n) requirement
```

### EPHEMERAL

Information useful temporarily but safe to remove later.

Examples:

* intermediate tool output
* temporary compiler messages
* old retrieval results

---

# 6. Memory Object

Every piece of context can be represented internally as a memory object.

Conceptually:

```python
Memory(
    content="assert solve([]) == []",
    type="PATCH",
    token_cost=8,
    relevance=0.91,
    importance=0.95,
    confidence=1.0,
    created_at=...,
    last_used=...
)
```

This abstraction allows TokenOS to compare very different kinds of information.

---

# 7. Token Allocation

Suppose an application gives TokenOS a 4,000-token context budget.

The system cannot include every available memory.

Each memory therefore receives a utility score based on factors such as:

```text
relevance
importance
confidence
recency
token cost
```

An initial heuristic could resemble:

```text
utility =
(relevance × importance × confidence)
/
token_cost
```

TokenOS then attempts to select the highest-value collection of memories that fits within the available context budget.

The first implementation should use simple heuristics.

Later versions could investigate learned ranking or allocation policies.

---

# 8. Context Compiler

Once TokenOS selects memories, the **Context Compiler** converts them into the final prompt.

For example:

```text
TASK

Implement max_subarray(nums).


REQUIREMENTS

Runtime must be O(n).


KNOWN FACTS

Input may contain negative numbers.


EXECUTABLE MEMORY

assert max_subarray([-2, -1]) == -1
assert max_subarray([5]) == 5


CURRENT IMPLEMENTATION

...
```

The underlying LLM does not need to understand TokenOS.

From the model's perspective, it simply receives a carefully constructed prompt.

---

# 9. MVP Architecture

Initial repository:

```text
tokenos/
│
├── runtime.py
├── memory.py
├── store.py
├── allocator.py
├── compiler.py
├── tokenizer.py
│
├── patchsearch/
│   ├── extractor.py
│   ├── validator.py
│   ├── retriever.py
│   └── patches.py
│
├── agents/
│   └── coding_agent.py
│
├── benchmarks/
│   ├── runner.py
│   ├── metrics.py
│   └── baselines.py
│
└── tests/
```

---

# 10. MVP Scope

The first version should **not attempt to solve general-purpose LLM memory**.

TokenOS v0.1 should focus exclusively on coding agents.

This gives the project objective correctness measurements because generated programs can be automatically tested.

The agent receives a programming problem and gets several attempts to solve it.

After every failed attempt, different memory strategies determine what information is provided on the next attempt.

---

# 11. Experimental Setup

Use approximately **100–300 coding problems** from established coding benchmarks.

Each problem receives a maximum number of attempts, such as:

```text
Attempt 1
Attempt 2
Attempt 3
Attempt 4
Attempt 5
```

Run identical problems across several memory strategies.

### Baseline A — No Memory

Each attempt receives only the original problem.

### Baseline B — Full History

Every previous:

* solution
* failure
* test output
* error
* correction

is retained.

### Baseline C — Sliding Window

Only the most recent context is retained.

### Baseline D — Summary Memory

Old failures are summarized into natural language.

### Experiment E — PatchSearch

Failures are converted into executable constraints.

### Experiment F — TokenOS

TokenOS dynamically selects among:

* raw context
* summaries
* facts
* patches
* recent history

under a fixed token budget.

---

# 12. Metrics

TokenOS should be evaluated across four dimensions.

## Correctness

Measure:

```text
pass@1
pass@3
pass@5
overall solve rate
```

The most important requirement is that token savings do not substantially decrease correctness.

---

## Token Efficiency

Measure:

```text
input tokens
output tokens
total tokens
tokens per solved problem
```

A useful derived metric:

```text
Token Efficiency =
Solved Problems / Total Tokens
```

---

## Cost

Record the approximate API cost required by each strategy.

Example:

| Strategy     | Solve Rate | Avg Tokens | Cost |
| ------------ | ---------: | ---------: | ---: |
| No Memory    |        61% |      5,200 |    $ |
| Full History |        82% |     18,400 |  $$$ |
| Summary      |        79% |     10,100 |   $$ |
| PatchSearch  |        83% |      7,300 |    $ |
| TokenOS      |        85% |      6,500 |    $ |

The numbers above are illustrative; actual results must come from experiments.

---

## Latency

Measure:

```text
average completion latency
total task latency
number of model calls
```

Compression itself can introduce overhead, so TokenOS must demonstrate that token savings justify additional processing.

---

# 13. Important Ablations

A strong portfolio project should investigate **why** the system works rather than only reporting that it works.

Run experiments such as:

### PatchSearch without retrieval

Give every patch to every future attempt.

This tests whether retrieval actually matters.

### PatchSearch without validation

Store every generated constraint.

This tests whether incorrect memories contaminate future generations.

### Different token budgets

Test TokenOS with:

```text
1K
2K
4K
8K
```

tokens of available context.

This can produce a particularly interesting graph:

```text
Task Success
     ↑
100% |
     |
 80% |              Full History
     |        TokenOS ──────────
 60% |    ─────
     |
 40% |
     +--------------------------→
        Context Tokens
```

The ideal result would show TokenOS reaching strong performance with substantially less context.

---

# 14. Failure Cases

TokenOS should explicitly investigate cases where compression hurts performance.

Examples include:

* a patch captures the wrong lesson
* two patches contradict each other
* important reasoning cannot easily become a constraint
* retrieval selects an irrelevant memory
* old information becomes stale
* compression removes subtle information
* too many patches overwhelm the model
* model behavior changes when constraints are presented without their original explanation

These aren't weaknesses to hide.

They are some of the most interesting parts of the project.

---

# 15. Development Plan

## Phase 1 — Benchmark Infrastructure

Build the coding agent.

Implement:

* benchmark loading
* LLM calls
* code execution
* test evaluation
* token counting
* retry loop
* experiment logging

Get **No Memory** and **Full History** working first.

---

## Phase 2 — PatchSearch

Implement:

```text
failure → patch extraction → validation → storage → retrieval
```

Run:

```text
Full History
vs.
PatchSearch
```

This is the first major milestone.

If PatchSearch cannot reduce tokens while maintaining reasonable correctness, investigate why before building the larger system.

---

### Phase 2 outcome — measured, 2026-08-31

**Gate:** reduce tokens while maintaining reasonable correctness.

**HumanEval+ (n=100, three runs): passed on tokens, no correctness signal.**
PatchSearch cut tokens 12–28% per solved problem at parity solve@5 (0.96–0.97).
But the benchmark is saturated at ~0.97, so there is no headroom in which a
memory strategy could demonstrate a correctness benefit. Paired solve deltas
were 0, 0, and −4. Treat HumanEval+ as a regression guard, not evidence.

**BigCodeBench-Hard: extraction coverage is structurally zero.**
This is a property of the code, not a measurement. `extract_memory` emits an
`executable_constraint` only when evidence carries both a `function_call` and
an `expected_value` observation marked trusted. Only the HumanEval+ evaluator
ever sets `trusted`; BigCodeBench emits unittest tracebacks, which carry no
oracle `(input, expected)` pair. So every BigCodeBench failure becomes a
bounded `failure_summary`, and PatchSearch degenerates to lossy compression of
full-history — one sentence against prior code plus a full traceback.

Three defects were masking how that degraded strategy actually performs, all
fixed before the deciding run:

* Assertion messages were recorded as `actual_value`, so memory claimed the
  candidate "returned" a comparison. unittest's 80-character repr elision then
  deleted the region distinguishing expected from actual.
* Candidate subprocesses wrote to file descriptor 1, corrupting the evaluator's
  result JSON. This biased against every solution that shells out.
* `openai.APIConnectionError` was classified non-retryable, ending a whole
  trajectory on any transient network fault.

A preflight pass now excludes tasks whose own reference solution fails in the
evaluator image, so a broken task cannot be counted against a strategy.

**Result: PatchSearch loses on correctness and does not save tokens per unit
of work.** Run `20260831T133412Z-b22fb4`, all 32 preflight-viable
BigCodeBench-Hard tasks, five attempts, no infrastructure failures.

| strategy | solve@1 | solve@5 | solved | tokens | tokens/solve |
|---|---|---|---|---|---|
| full-history | 0.281 | **0.750** | 24/32 | 329,322 | **13,722** |
| patchsearch | 0.281 | 0.500 | 16/32 | 282,632 | 17,664 (+28.7%) |
| patchsearch-explained | 0.219 | 0.531 | 17/32 | 263,602 | 15,506 (+13.0%) |

PatchSearch solved eight tasks fewer than full-history and solved none that
full-history missed. Eight discordant pairs, all one direction: McNemar exact
p = 0.0078.

The token result inverts under normalization, which is the finding that matters
most. PatchSearch used 14.2% fewer raw tokens, but it also solved a third fewer
problems, so **per solved problem it cost 28.7% more**. A compression number
quoted without a correctness denominator is not a saving; it is partly a record
of giving up earlier.

**Correction, after replication.** The solve@1 spread suggested run-to-run
noise of one to two tasks. Repeating the identical configuration showed it is
larger: full-history moved 24 -> 21 and PatchSearch 16 -> 19, a three-task swing
each, and the same comparison gave p = 0.0078 in one run and p = 0.625 in the
next. A single run's p-value overstates what one run supports. Pooled across
both, the direction holds -- 11 discordant pairs to 1, p = 0.0063 -- but any
future single-run claim should be treated as provisional until replicated.

### Consequences for the thesis

The hypothesis in section 3 is too broad as stated. What the evidence supports
is narrower and more useful:

> Compact executable memory pays where an oracle exposes counterexamples.
> Where a benchmark reports only unittest diagnostics, no extraction is
> possible, and bounded summaries carry less than the history they replace.

That boundary — *when* compressed memory pays, and why — is the honest result
of Phase 2 and should be what the project claims. Section 20's "particularly
strong result" framing assumes a win that the harder benchmark does not
support; the finding is a condition, not a defeat.

### Cut from later phases

**Explained PatchSearch is discontinued** — but not because it is worse than
plain PatchSearch. The two are indistinguishable from each other: across the
two measured runs they traded places on both solve rate (17 vs 16, then 16 vs
17) and tokens per solve, within the one-to-two task noise band. The reason to
stop is that neither variant beats full-history on a summary-only benchmark, so
a second variant of a mechanism that does not work buys nothing. Keep the code
path for reproducing published numbers; do not extend it.

---

## Phase 3 — Additional Memory Types

Add:

```text
FACT
SUMMARY
DECISION
RAW
EPHEMERAL
```

Create a unified memory interface.

---

### The mechanism, corrected — 2026-09-01

PatchSearch's loss was never that summarising a failure destroys information.
It was that PatchSearch **discarded the prior implementation**, forcing the
model to rewrite a whole function from a one-sentence hint while full history
handed it code to edit. That is a defect in the strategy, not a cost inherent
to compression.

Three fixes, all in the evidence path:

* Split the assertion into `actual_value` and `expected_value`. unittest
  renders `assertEqual` as `<actual> != <expected>`, which was being carried as
  prose. Memory now states the gap a retry has to close.
* Report every failing test, not only the first. A retry that satisfied one
  assertion routinely broke the next.
* Carry the most recent completion alongside the bounded memory
  (`patchsearch-with-code`). One completion, not the four full history keeps.

**Result: parity with full history at 21.5% fewer tokens, on the hard
benchmark.** Run `20260901T024612Z-bf19df`, 32 preflight-viable tasks.

| strategy | solve@5 | solved | tokens | tokens/solve | vs full-history |
|---|---|---|---|---|---|
| no-memory | 0.312 | 10/32 | 252,095 | 25,210 | +64.7% |
| full-history | 0.688 | 22/32 | 336,779 | 15,308 | — |
| patchsearch | 0.562 | 18/32 | 276,905 | 15,384 | +0.5% |
| **patchsearch-with-code** | **0.656** | **21/32** | **264,267** | **12,584** | **-17.8%** |

21 against 22 is statistically indistinguishable (discordant 3/2, p = 1.0) at
21.5% fewer raw tokens and 17.8% less per solved problem. This satisfies
section 20's first two criteria on a benchmark that is not saturated.

The two controls are what make the mechanism legible rather than the number
lucky. `no-memory` at 10/32 shows memory is doing most of the work
(p = 0.001 against with-code). Plain `patchsearch` at 15,384 tokens per solve
shows the summary alone buys **no token advantage at all** once normalised by
correctness -- it simply solves less. Restoring one completion recovers three
of the four lost tasks *and* drops cost 18%.

Bounded failure memory plus the current implementation beats verbose history.
That is the thesis, and it now has evidence on the harder benchmark.

**Standing caveat.** One run, and noise here is about three tasks, so a
one-task gap is inside it. The claim is parity, not superiority. The 21.5%
token reduction is the robust half. Replication is required before this is
stated without qualification.

### Replication — 2026-09-25

Run `20260925T150550Z-f59c90`, identical configuration, the two strategies
only.

| run | full-history | with-code | discordant (fh / wc) | p | raw tokens | tokens/solve |
|---|---|---|---|---|---|---|
| original | 22/32 | 21/32 | 3 / 2 | 1.0 | -21.5% | -17.8% |
| replication | 21/32 | 19/32 | 3 / 1 | 0.625 | -16.7% | -7.9% |
| **pooled** | **43/64** | **40/64** | **6 / 3** | **0.508** | **-19.0%** | **-13.0%** |

**The token reduction replicates; the size of the per-solve saving does not.**
Raw tokens fell 16.7% against 21.5%, and pooled the saving is 19.0%. Per solved
problem it roughly halved, to 7.9%, because with-code solved two fewer this time
while full-history solved one fewer. Pooled, with-code is 13.0% cheaper per solve.

**Correctness is not distinguishable, but the direction is consistent.** Neither
run nor the pool is significant (pooled p = 0.508). But with-code trailed in both
runs, and full-history holds 6 of the 9 discordant tasks. The accurate claim is
narrower than "parity": *about 19% fewer tokens, at a correctness cost that is
not measurable at n = 32 and, if real, is about three tasks in 64.* Separating
the two would take roughly four times as many paired tasks, which means an
evaluator image that runs more than 32 of the 148 Hard tasks.

Both strategies moved between runs (full-history lost 2 and gained 1;
with-code lost 2 and gained 0), consistent with the three-task noise band
already recorded.

---

### Phase 4 precondition, resolved 2026-09-01

An allocator allocates a scarce resource. Under task-local memory nothing was
scarce: the retrieval budget bound on **0 of 155** attempts, the pool never
exceeded five memories, and full history peaked at 6,828 tokens against a
~200k window. A utility score would have ranked items that were all included
anyway. Compression cannot pay where nothing fails to fit.

Cross-task memory (`store.py`) and the heuristic allocator (`allocator.py`) are
now implemented, and the precondition is met: the candidate pool reaches 15,218
characters against a 3,000-character budget and **binds on 148 of 148 tasks**,
with roughly 24 of 74 memories selected per attempt.

Two constraints shaped the store. It is **frozen**, seeded from a completed
run, because online accumulation would make a task's context depend on which
other tasks a thread pool finished first. And **executable constraints never
cross task boundaries**: every BigCodeBench task defines `task_func`, so a
transferred constraint would assert against an unrelated function of the same
name and be silently false. Only summaries transfer.

**Result: allocation works; there is nothing worth allocating.**
Run `20260901T003400Z-52a4f2`, same 32 tasks.

| strategy | solve@5 | solved | tokens | tokens/solve |
|---|---|---|---|---|
| full-history | 0.656 | 21/32 | 341,439 | 16,259 |
| patchsearch | 0.594 | 19/32 | 285,390 | **15,021** |
| patchsearch-crosstask | 0.531 | 17/32 | **379,641** | 22,332 |

Cross-task retrieval is the only strategy that made the prompt *larger* than
full history while solving fewer tasks: compression became expansion. It cost
48.7% more per solved problem than plain PatchSearch. The difference in solve
rate is inside the noise band established above (p = 0.625 against PatchSearch),
so the correctness claim is weak; the token cost is not.

The cause was measurable before the run and is a property of the benchmark, not
the code. Across 186 memories from 27 tasks, every root cause was task-local --
a missing `wget` on one task, `KeyError: 'zz'` on another, a different absent
path each time. Exactly one pair of tasks shared a failure mode. BigCodeBench-
Hard is a set of deliberately independent library-composition problems, so
there is no recurring mistake for accumulated experience to catch, and the
allocator spends 2,706 characters per retry on other tasks' irrelevant history.

**What this establishes.** The allocation machinery is verified and the policy
is observable: selection is deterministic, budget-respecting, and now actually
binding. What it lacks is a task distribution with recurring failure modes.
Optimal allocation of memory that does not transfer is still worthless, and
costs more than not allocating at all. Evaluating the allocator on its merits
requires a benchmark where the same mistake recurs across tasks -- a repository
of related tasks, a multi-file agent trajectory, or a benchmark family sharing
an API surface -- not a wider search over scoring functions.

---

## Phase 4 — Token Allocator

Implement memory scoring.

Start with heuristic scoring rather than ML.

For example:

```text
score =
relevance × importance × confidence × recency
```

Normalize by token cost.

Select memories until the context budget is exhausted.

---

## Phase 5 — TokenOS Runtime

Combine:

```text
Memory Store
      +
PatchSearch
      +
Retriever
      +
Allocator
      +
Context Compiler
```

into one runtime.

Desired developer interface:

```python
from tokenos import Runtime

runtime = Runtime(
    context_budget=4000
)

result = runtime.run(task)
```

---

### Phase 5 outcome — built, 2026-09-25

The runtime exists with the interface above. `Runtime(context_budget=4000)`
solves one task with `patchsearch-with-code`, the strategy that reached parity,
and `tokenos solve` exposes it on the command line. Each attempt runs task-local
retrieval, optional cross-task allocation from a frozen store, then a new
**context compiler** (`compiler.py`), which is the piece that was missing:
retrieval bounded memory in characters, but nothing bounded the prompt.

**The compiler reproduces the measured strategy exactly.** It renders through
the same `build_prompt` the benchmark used. Replayed over run
`20260901T024612Z-bf19df`, all 194 recorded PatchSearch prompts are recompiled
byte-for-byte when the budget does not bind, so the runtime is the benchmarked
strategy rather than a lookalike of it.

**Token estimation is calibrated, not guessed.** Across 707 recorded requests
the densest text reached 3.34 UTF-8 bytes per API-reported input token; the
compiler divides by 3.0, which overestimates all 194 replayed attempts. No
tokenizer dependency was added.

**Priority under pressure follows the ablations.** Memory lifted 10/32 to 18/32
and the latest code lifted 18/32 to 21/32, so when a budget binds the compiler
drops transferred memory first, then the code, then summaries oldest first, and
keeps executable constraints longest. The code is kept or dropped whole. A task
that does not fit on its own raises `ContextBudgetError` rather than being
truncated.

**Where the budget binds** (`tokenos sweep`, same run, estimated tokens):

| budget | with-code attempts trimmed | code dropped | memories dropped | tokens vs unbounded |
|---|---|---|---|---|
| unbounded / 4000 | 0 / 90 | 0 | 0 | — |
| 1500 | 18 / 90 | 18 | 1 | -17.8% |
| 1000 | 48 / 90 | 47 | 18 | -34.0% |

At the default 4,000 tokens nothing is cut, because bounded memory already
peaks at 1,775 billed tokens. That is the Phase 2 result restated as a guarantee:
the context stays small without a limit having to enforce it. Tighter budgets
mostly remove the code first, which is the component worth three tasks, so their
correctness cost has to be **measured live, not inferred**. The sweep prices
each budget in advance so that live run can be chosen deliberately.

**A defect fixed along the way.** Parsing and evaluation are now one shared
function. Before, a `ValueError` raised *inside the evaluator* — including
`AdapterError` for an unregistered benchmark — was caught as the model's
malformed output and scored as a failed attempt. It now stops the trajectory
as an evaluator error.

**Next.** (Replication done, recorded above.) Run with-code at 1,000
tokens against unbounded to measure what dropping the code costs.

**Budgeted benchmark runs — built, 2026-10-06.** `tokenos benchmark
--context-budget N` routes every bounded-memory prompt through the compiler,
so a live run measures the Runtime's actual behaviour under a budget. Full
history is left uncompiled as the control. Each `request_started` event records
the compiled context (tokens, code kept or dropped, memories dropped); a task
that cannot fit stops its trajectory as `context_budget_exceeded` without a
model call. Unbounded runs are unchanged: replaying the replication run
`20260925T150550Z-f59c90`, all 99 with-code prompts are reproduced byte-for-byte,
and at 1,000 tokens 52 of 99 would be trimmed (code dropped on 51), with no task
too large to fit. The live run is

```bash
tokenos benchmark --dataset bigcodebench-hard --all --skip-unviable-tasks \
  --strategies full-history,patchsearch-with-code --context-budget 1000 \
  --budget-usd 4 --concurrency 4
```

which pairs with-code@1000 against full history in the same run and against
the two unbounded with-code runs (21/32, 19/32).

### Binding budget — measured, 2026-10-07

Run `20261007T033256Z-4affde`, the command above ($2.16).

| strategy | solved | tokens | tokens/solve |
|---|---|---|---|
| full-history | 24/32 | 356,281 | 14,845 |
| with-code @ 1,000 | 16/32 | 285,718 | 17,857 |
| with-code, unbounded (two earlier runs) | 21/32, 19/32 | 264,267 / 296,118 | 12,584 / 15,585 |

**A 1,000-token budget costs correctness, and saves nothing.** Against full
history in the same run, with-code@1000 lost 8 tasks and won none (McNemar
exact p = 0.008), the first significant correctness difference in the
project. Against the unbounded with-code runs it lost 5 and 3 tasks and never
solved one they missed. It used 19.8% fewer tokens than full history, but 20%
*more per solve*, and no fewer tokens than unbounded with-code: failed tasks
run all five attempts, so the attempts the budget trims are paid for again in
extra retries.

**What the budget cut was the code.** Code was dropped on 55 of 73 retries and
memories on 6. Retry prompts stayed at 534–577 input tokens against 876–980
unbounded, i.e. close to plain PatchSearch, and 16/32 is in line with plain
PatchSearch's 18/32. The ablation predicted this: the latest code is worth
about three tasks, and it is the first thing a tight budget removes.

**Consequences.** Full history scored 24/32 here against 22 and 21 before, so
part of the 8-task gap is run noise, but the zero discordant tasks in the
budget's favour across three comparisons is not. The default budget of 4,000
stays: it never binds on this benchmark. A budget below the code's size should
be treated as a different, weaker strategy rather than a cheaper version of the
same one. Whether the compiler should give up older summaries before the code
is open; plain PatchSearch (memory, no code) and with-code differ mainly in the
code, so the evidence leans that way.

### Ten attempts, uncapped output — 2026-10-07

Run `20261007T045053Z-f3837c`: `--max-attempts 10 --max-output-tokens 16384`,
three strategies ($6.45).

**The output cap was the invalid-output problem.** Every `invalid_output` in
the earlier runs (41, 38, 33 per run) had hit the 4,096-token output cap: the
reasoning model spent it before closing the JSON. At 16,384 there was 1 invalid
response in 405 calls.

| strategy | solved | input tokens | total tokens | tokens/solve |
|---|---|---|---|---|
| full-history | 26/32 | 365,602 | 829,645 | 31,909 |
| with-code | 24/32 | 122,035 (-67%) | 535,253 (-35%) | 22,302 (-30%) |
| best-code | 24/32 | 125,552 | 579,369 | 24,140 |

Mean input tokens at attempt 10: full history 7,688, with-code 1,132. With
more attempts the saving grows, as the thesis predicts. Correctness still
trails by two (2 full-history-only tasks, 0 the other way). Best-code, which
shows the attempt with the fewest failing tests instead of the latest, did not
help (4 / 2 against full history).

**Why bounded memory loses the tasks it loses: it gets stuck.** On
BigCodeBench/326 with-code failed the same three tests ten times running; on
785 it raised the same `FileNotFoundError` seven times. Retrieval deduplicates
identical failures, so the model sees one calm summary and makes another small
edit. Full history shows the repetition. `patchsearch-adaptive` restores it:
repeat counts on memories, an explicit STUCK directive after two identical
failures, and the latest valid attempt's raw test output (bounded to 1,500
characters).

### Stuck detection — measured and replicated, 2026-10-07

Runs `20261007T055643Z-cd5774` (with with-code as a control) and
`20261007T065630Z-1cba26` (replication), ten attempts, 16,384-token output
cap.

| run | full-history | adaptive | discordant (fh / ad) | tokens vs fh |
|---|---|---|---|---|
| first | 27/32 | 28/32 | 2 / 3 | -41.5% |
| replication | 24/32 | 25/32 | 2 / 3 | -32.6% |
| **pooled** | **51/64** | **53/64** | **4 / 6 (p = 0.75)** | **-36.7%** |

Per solve, adaptive cost 18,786 tokens against 30,821 (-39%); input tokens
fell 63%. In the first run, with-code repeated its usual result (24/32), and
adaptive beat it on 5 tasks while losing 1, including both stuck tasks (326,
785), solved by attempts 3 and 4. The STUCK directive fired on 17 of 113 calls
and was followed directly by a pass twice; the raw test output appears to have
done more on 326 and 785.

**This is the first configuration that does not trail full history.** The
claim is parity at substantially lower cost, not superiority: the solve
difference is well inside the noise. Caveat: the mechanism was designed after
looking at failures in run `20261007T045053Z-f3837c` on the same 32 tasks, so
held-out tasks are the next test. Adaptive is now the Runtime's default
strategy, with the default context budget raised to 8,000 (adaptive prompts
reach about 4,300 estimated tokens, so 4,000 would have dropped code) and the
default output cap to 16,384.

### Third replication — 2026-10-11

Run `20261011T171248Z-087afa`, same configuration: full history 27/32,
adaptive 24/32 (4 / 1 discordant), 36.4% fewer tokens. Pooled over three runs:
**full history 78/96, adaptive 77/96 (8 / 7 discordant, p = 1.0)**; tokens
-36.5%, input -63.7%, estimated cost -19.6% ($5.52 vs $6.88). The earlier
two-run lead was noise; the claim on these 32 tasks is parity. Twenty of the
32 tasks resolve identically every run; the variance comes from ten.

### Held-out tasks — full BigCodeBench-Hard, 2026-10-11

Run `20261011T184316Z-8039bc` on an Azure VM (Standard_D4s_v4) with the full
official evaluator (`tokenos-bigcodebench-full:0.3.0`): all 148 tasks, of which
138 pass reference preflight. Ten attempts, 16,384 output cap, one run, $15.76
estimated model spend. The 32 tasks above are where stuck detection was
designed; the other 106 viable tasks are **held out**.

| tasks | n | full history | adaptive | discordant (fh / ad) | p | tokens | cost |
|---|---|---|---|---|---|---|---|
| tuned | 32 | 26 | 25 | 3 / 2 | 1.0 | -41.4% | -20.9% |
| **held-out** | **106** | **98** | **93** | **6 / 1** | **0.125** | **-15.0%** | **-1.6%** |
| all viable | 138 | 124 | 118 | 9 / 3 | 0.146 | -24.6% | -7.8% |

**The result does not transfer cleanly.** On held-out tasks adaptive trails by
five, not significantly, and the cost saving disappears.

- **The held-out tasks are easier.** 45 of 106 were solved on the first attempt
  by both strategies; only 9 full-history trajectories reached attempt 5. The
  token saving comes from long trajectories (full history reaches 8,103 input
  tokens at attempt 10, adaptive 1,479), and there were few.
- **Fewer tokens, not less cost.** Input fell 43%, but output rose 2.8%:
  adaptive made 314 calls to full history's 271, because the tasks it failed
  ran all ten attempts. Output is 6x the price of input.
- **Half the correctness gap is first-attempt sampling.** On 3 of the 6
  full-history-only tasks (1019, 1022, 89), full history passed on attempt 1,
  where both strategies send essentially the same prompt. Conditioned on both
  failing attempt 1, the split is 3 / 1.
- **Stuck detection fires but does not rescue.** On four of the losses
  adaptive failed with the same wrong answer ten times while the STUCK
  directive was shown 6-8 times. Telling the model it is stuck is not enough to
  change its approach on these tasks.

**Consequences.** The defensible claim is narrower than "parity at a third
fewer tokens": on the 32 tasks used for development, parity at 36% fewer tokens
and 20% lower cost, replicated three times; on held-out tasks, 15% fewer tokens,
no cost saving, and a non-significant five-task deficit in one run. The saving
scales with trajectory length, so it is largest exactly where retries are long.
Next: a second held-out run to separate the deficit from noise, and a stronger
stuck intervention (e.g. withholding the looping code, or regression-aware
prompting) evaluated on held-out tasks only.

### Output cost — 2026-10-12

Output is 88–94% of estimated cost (6x the input price) and TokenOS only
shrinks input, which is why its token saving outran its cost saving. Across
the four ten-attempt adaptive runs: reasoning is 82–86% of output tokens; a
retry produces about 2.5x the output of a first attempt; and **53–66% of all
output is spent on trajectories that never pass**.

**Stopping stuck trajectories (measured exactly by replay).** Stopping only
truncates, so recorded runs give its effect without new runs
(`tokenos stopsim`, 244 adaptive trajectories, 202 solves):

| stop after K identical failures | solves | abandoned | output | cost |
|---|---|---|---|---|
| never | 202 | 0 | — | $13.37 |
| 5 | 199 (-3) | 19 | -11.5% | -11.5% |
| **4** | **198 (-4)** | 29 | **-19.2%** | **-19.1%** |
| 3 | 191 (-11) | 42 | -28.1% | -28.0% |
| 2 | 181 (-21) | 58 | -43.8% | -43.6% |

A plain attempt cap is worse at every comparable point (8 attempts: -6 solves
for -15% output). On the full 148-task run alone, K = 4 brings adaptive to
$6.35 against full history's $7.92, losing 2 solves. Caveat: K was read off
these same runs; the next live run should use K = 4 as fixed in advance.
`--stop-after-identical K` enforces it live.

**Context overflow.** From recorded prompt sizes in the full run, with a 15%
margin for tokenizer differences: on an 8K-context server 14 of 148
full-history trajectories cannot fit at all (5 of 148 at 16K); adaptive's
longest prompt is 3,095 tokens, so none overflow.

**Reasoning effort.** `gpt-5.4-mini` accepts `none`, `low`, `medium`, `high`,
`xhigh` (`minimal` is rejected). Planned runs, adaptive on the 32 development
tasks, each capped at $4, against the three existing medium-effort runs
(28, 25, 24 solved): `--reasoning-effort low`, and `low` with
`--retry-reasoning-effort medium`. **Pass criteria, fixed before running:**
cost per solve at least 15% below the baseline mean, at least 24/32 solved,
and invalid responses not above baseline.

**Reasoning effort — measured, 2026-10-12.** Adaptive, 32 development tasks,
ten attempts, one run per arm, against the three medium-effort runs:

| effort | run | solved | output tokens | reasoning tokens | cost | cost/solve vs medium | verdict |
|---|---|---|---|---|---|---|---|
| medium (3 runs) | baseline | 24–28 | 303K–432K | 247K–362K | $1.45–2.05 | — | — |
| none | `2d6e12` | 21 | 74K | 0 | $0.47 | -69.5% | fails (< 24 solved) |
| **low** | `1e6ee9` | **26** | 112K | 50K | **$0.61** | **-67.7%** | **passes** |
| low, medium on retries | `f05eb0` | 27 | 345K | 286K | $1.65 | -16.0% | passes, barely |

**Low effort keeps correctness at a third of the cost.** 26 solved sits inside
medium's 24–28 band; reasoning tokens fell about six-fold. Escalating to
medium on retries gives most of the saving back, because retries are where
the output is. Turning reasoning off saves the most but loses about four tasks.
Stopping after four identical failures on top of low effort gives 25 solved
for $0.48.

Caveats: one run per arm on the development tasks (runs vary by about three
tasks); the `low` run was interrupted by a network drop on 8 requests and
resumed with every completed attempt kept. Reasoning effort is a model
setting, not a TokenOS mechanism: it would cut full history's cost too, so it
changes the absolute bill, not the comparison between strategies. Next: the
`low` arm on held-out tasks, with full history at low effort alongside.

**Self-hosted serving and kernels (`colab/`).** Real TokenOS traffic is
exported (`tokenos export-prompts`) and replayed on a Colab T4 against vLLM
serving Qwen2.5-Coder-7B (AWQ): baseline, prefix caching, and n-gram prompt-
lookup speculative decoding (a retry mostly rewrites code already in its
prompt), plus quantization and concurrency. Hand-written Triton kernels for
fused softmax and single-query decode attention (Qwen2.5-7B shapes, single and
split-K) measure decode cost against context length as achieved bandwidth.
The kernels pass their correctness tests under Triton's CPU interpreter; no
GPU numbers exist until the notebooks are run.

---

## Phase 6 — Large Evaluation

Run all memory strategies across the complete benchmark.

Store every experiment in structured files containing:

```text
problem
strategy
attempt
success
input_tokens
output_tokens
latency
patches_generated
patches_retrieved
```

Then generate benchmark tables and graphs.

---

# 16. Portfolio Demo

The project should have an interactive demo where someone can watch context grow.

For example:

```text
CODING AGENT — ATTEMPT 4

Raw accumulated history:
8,421 tokens

TokenOS context:
2,137 tokens

Reduction:
74.6%

Memories selected:

✓ Current implementation
✓ Complexity requirement
✓ Patch: empty input
✓ Patch: negative numbers
✗ Attempt #1 reasoning
✗ Old traceback
✗ Duplicate compiler output
✗ Irrelevant tool result
```

Then show the resulting model answer and whether the tests pass.

This makes the concept immediately understandable.

---

# 17. README Results Section

The final README should lead with measurable results rather than architecture.

Something like:

> **TokenOS reduced LLM context usage by X% while maintaining Y% of full-history task performance across N coding tasks.**

Then:

```text
                    Solve Rate      Tokens/Solve

Full History           XX%             XXXXX
Sliding Window         XX%             XXXXX
Summary                XX%             XXXXX
PatchSearch            XX%             XXXXX
TokenOS                XX%             XXXXX
```

Only use real experimental numbers.

---

# 18. Stretch Goals

Once the MVP works, TokenOS could expand into more sophisticated techniques.

### Semantic retrieval

Use embeddings to identify relevant memories.

### Learned memory ranking

Train a small model to predict whether a memory will improve the next generation.

### Memory aging

Reduce priority for memories that have not been useful recently.

### Memory consolidation

Merge several related memories into one.

### Contradiction detection

Detect when stored memories conflict.

### Cross-task memory

Determine whether lessons from one problem can safely transfer to another.

### Adaptive token budgets

Give difficult tasks larger context budgets while aggressively compressing easy tasks.

### Multi-agent memory

Allow several agents to share a TokenOS memory store.

---

# 19. What NOT to Build Initially

Avoid turning TokenOS into:

* another vector database
* a generic chatbot memory system
* a prompt wrapper
* an enormous agent framework
* a custom LLM
* a new tokenizer
* an actual operating system

The interesting problem is specifically **context allocation under a token budget**.

Keep that central.

---

# 20. Definition of Success

The MVP is successful if it demonstrates:

1. **Correctness:** TokenOS maintains competitive task success.
2. **Compression:** It substantially reduces context tokens.
3. **Cost:** Reduced tokens translate into lower inference cost.
4. **Latency:** Performance remains practical.
5. **Adaptivity:** Different information is represented and retained differently.
6. **Reproducibility:** Results can be reproduced through the benchmark suite.

A particularly strong result would look something like:

> TokenOS achieved 96% of full-history coding performance while using 48% fewer input tokens.

Even better would be:

> TokenOS exceeded full-history performance while using substantially fewer tokens.

---

# 21. Portfolio Positioning

The project should be presented as an **LLM systems / inference optimization project**, not simply an AI application.

The central idea is:

> **LLMs should not have to remember everything in order to learn from everything.**

PatchSearch provides the first concrete example: verbose failures can sometimes be transformed into compact executable memories.

TokenOS generalizes the principle by treating the context window as a scarce computational resource and dynamically deciding **what deserves a token.**
