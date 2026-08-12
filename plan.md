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
