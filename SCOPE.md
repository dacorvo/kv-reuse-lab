# reagent — scope and next steps

This file is a **handoff summary** documenting what reagent owns now
that the work has been split into multiple repos, and where it goes
next. Read [README.md](README.md) for the project's full context;
this is the orientation note for picking up work.

## Repo split (May 2026)

The work that started in `hf-mount-cache-examples` has been
decomposed into independently maintained repos:

| repo | purpose | status |
|---|---|---|
| **`agentcap`** (sibling at `../agentcap`) | Capture real LLM-agent chat-completion traffic, export as Hugging Face datasets. Transparent OpenAI-compat HTTP proxy + offline manifest builder. | Active, proxy + tests done. |
| **`reagent`** (this repo) | Measure correctness of cross-request KV-cache reuse (the *consumption / splice* step), analyse recurrence in captured traces, and design the runtime cache. | Active, measurement code mature, runtime impl not started. |
| ~~`hf-mount-cache-examples`~~ | Original prototyping ground. **Deprecated** — closed out as a research log. The conclusions live in its `FINDINGS.md`; the code patterns live, refactored, in the two repos above. | Read-only. |

The two ruled-out implementation paths:

- **LMCache.** Implements only prefix caching (the byte-stable
  first-turn block). Doesn't address post-prefix chunk caching, where
  the substrate identified by the measurements lives. Not extensible
  to the chunk-cache regime without a major rewrite.
- **`llama.cpp --cache-reuse`.** Already does the same RoPE-rephased
  reuse reagent measures, so the gap with llama.cpp is **not** the
  rephasing mechanism. The gap is: (a) it can only stitch one
  contiguous run of cached tokens, not multiple disjoint fragments
  separated by new content; (b) it identifies cache hits by sliding
  byte-search over the rendered token stream rather than by chunk
  identity. Adapting it to do selective per-chunk caching would
  require invasive changes to its `kv_cache_unified` internals.

A useful runtime cache will be a fresh implementation. Reagent owns
its design.

## What we know so far

Distilled from `../hf-mount-cache-examples/FINDINGS.md`,
`../hf-mount-cache-examples/KV Cache Sliding.md`, and reagent's own
measurement runs. Read those for citations and full numbers; this
section is the load-bearing summary. Terminology (**CP**,
**post-prefix matches**) is defined in
[README.md](README.md#concepts) — read that first if either is
unfamiliar.

**Trust caveat.** Many prior splice-correctness results predate the
role-filter (`02bad94`) and first-turn-skip (`a3ce1dc`) fixes in the
matcher. Pre-fix measurements admitted assistant-token spans as splice
candidates and treated first-turn matches as splices instead of
prefix-cache hits. Both contaminate the splice-correctness signal.
Treat any pre-fix number as **indicative, not established** — including
ones cited in the reagent README that look quantitative. The honest
state is closer to "we have seen the mechanism work and we have seen
it fail, but we don't yet have a clean enough harness to report
failure rates."

### Mechanism (KV reuse with RoPE rephasing)

- **The mechanism works in clean cases.** Pasting a cached chunk into
  a new prompt at a different position, with the position-encoded part
  rotated to match the new position, produces output essentially
  indistinguishable from a fresh forward when the chunk is taken from
  a clean structural boundary and the splice is followed by enough
  suffix tokens for the model to recover from any small content
  discontinuity. Validated on six models (Gemma-4 in four sizes,
  Llama-3.1-8B, Llama-3.2-1B), L=128, position offset Δ ≤ 1000 tokens.
- **The result is suggestive, not airtight.** The validation was on a
  small handful of pairs per cell. Catastrophic failures occurred
  within that small sample even on Gemma-4 (the most shift-robust
  family). "Shifted reuse is safe" is not a closed question.
- **Failure modes observed under the pre-fix harness — none are
  cleanly established.** Two patterns appeared in older runs but both
  citations are flagged by the role-filter / first-turn-skip commits
  themselves:
  - *"Splice too close to end of turn"* (`reagent/README.md:318-334`):
    Llama-3.1-8B SWE-smith Django produced a 28% sub-0.80 failure
    rate at splice-end B-position 6-10k. **The commit `02bad94` notes
    that this run had 26% A / 18% B assistant tokens, so naive
    matches likely crossed role boundaries; the existing Django
    results need to be re-run on the input-only matcher.** Pandas-dev
    (`:399-402`) was verified valid under the role filter but the
    bad-B/bad-A verification pair has *zero* splice candidates under
    the first-turn skip (`a3ce1dc`) — meaning the "failure" was a
    pair-selection artifact, not a mechanism failure. The
    "insufficient suffix to wash out cache-conditioning mismatch"
    *mechanism* is plausible but the supporting data is no longer
    quantitatively trustworthy.
  - *Parameterised templates (OpenCode).* User-recalled hypothesis:
    OpenCode-style tool-response blocks may be shaped
    `<parameters><template body>` with parameters preceding the
    template that consumes them, and splicing across that boundary
    may break the parameter→template binding. **No measurement
    evidence for this in `FINDINGS.md`, the README, or any saved
    analysis JSON.** Treat as something to investigate if a captured
    OpenCode corpus shows sub-0.80 splices at chunk-internal
    boundaries; do not cite as established.
- **Hybrid attention bounds the floor, not the mean.** Gemma-4 hybrid
  floors at sim 0.755; dense Llama-3.1-8B can hit 0.437 despite
  similar mean.
- **The naive (un-rephased) baseline is reagent-internal only.**
  llama.cpp `--cache-reuse` always rephases. The naive comparison was
  used to quantify how much rephasing buys you, not as a description
  of any production stack.

### Methodology bias to know about

- **Caching model output is meaningless.** Some early reagent runs
  treated assistant-turn bytes as cache candidates — but the serving
  stack already has the KV from generating them, and would never
  re-prefill them. Recent commits (`02bad94`, `a3ce1dc`) added an
  input-only role filter and skip first-turn matches in
  `find_matches`. Splice-correctness numbers from before those fixes
  may be inflated.
- **N=5 minimum for cross-family claims.** N=1 probes have produced
  misleading verdicts in earlier work.

### Admission, not lookup, is the hard problem

- The matching algorithm is the easy half. The hard half is deciding
  *what to cache in the first place* at write time.
- **Byte-level admission heuristics topped out.** Best correlation
  observed (entropy at the chunk being cached, K/V perturbation
  sensitivity) is r ≈ +0.36 between the predictor and downstream
  splice safety. Useful as a ranking signal, **insufficient as a sole
  gate**. Cheap entropy proxy is roughly as good as expensive K/V
  perturbation — don't re-run perturbation experiments expecting more
  signal.
- **For OpenCode we ended up needing request semantics.** The cache
  key shipped is `tool_response:tool_name+args_hash`, supplied by a
  request-render hook on the agent side. The cache layer does no
  content classification. That's a load-bearing design decision:
  byte-pattern admission was abandoned in favour of structural keys
  the agent author hands us.

### Substrate composition (agentcap-Hermes corpus, May 2026)

First measurement on a fresh capture aimed at the target use case
(local model + same user/team + recurring tasks). 27 sessions × 4
turns through Hermes on Gemma-4-E4B-it, generic transformers-
internals planning prompts, synthesized follow-ups.

**Important framing.** Hermes accumulates memory state across
sessions in this corpus — session N inherits the memory block
written during sessions 1..N-1, so any two sessions have divergent
memory states. **This is the target regime, not a bug in the
capture.** A real shared runtime cache populates a single bucket of
K/V chunks across all sessions/users on a host (and across hosts in
a fleet); whoever populates a chunk first bakes in their local
state — memory block included. Future requests that share the
underlying bytes (tools schema, file contents, web fetch result)
but were rendered against a different memory state will have a
short CP with the cached chunk. The bytes past that CP are exactly
the post-prefix substrate the runtime cache is for. agentcap's
session-spanning memory accumulation faithfully reproduces that
multi-user / shared-bucket pattern within a single capture.

- **Apparent coverage = 86%** (mean fraction of every prompt covered
  by ≥128-token cross-session matches). Misleading on its own.
- **Decomposed:**
  - ~31.6% of matched volume is the Hermes system prefix (first ~4.7k
    tokens). Prefix-cacheable; trivial.
  - ~66.3% is the tools-schema injection (~9.5k tokens, byte-stable
    across sessions) which sits **behind** Hermes's memory block.
  - ~1.4% is `system_other` (skills index, AGENTS.md leak, memory
    fragments).
  - ~0.69% is real `tool_response` substrate — all of it
    `skills_list` enumerations.
  - ~0.04% user content + ~0.01% assistant slop (the role filter
    works).
- **The memory block ends CP at median ~4.7k tokens.** Cross-session
  CP between any two distinct sessions in this corpus = median 4,719
  tokens. The remaining ~10k tokens of byte-stable system content
  (tools schema) are **post-prefix matches** because the two sessions
  rendered against different memory states. This is exactly the kind
  of post-prefix recurrence that motivates a shared runtime cache:
  the underlying bytes (tools schema) are common across users, but
  the local-state injection (memory block) breaks CP on every pair,
  so a single-user prefix cache cannot serve the bytes past the
  break. A shared bucket of chunks plus reagent's splice mechanism
  is what makes those bytes reusable.
- **Skills are NOT dynamically fetched in this corpus.** 0 calls to
  `skill_view`. Hermes's system prompt instructs the agent to
  "MUST load it with skill_view(name)" when a skill matches; Gemma-4
  didn't comply. The `tool_response` substrate was therefore much
  smaller than it could have been — this is a model-compliance
  artifact, not a workload property. Different model driving Hermes
  could produce a very different substrate composition without
  changing the prompts.

## In scope for reagent

1. **Splice-correctness measurement** (`measure_reuse_drift.py`,
   `measure_multi_splice_b.py`). The mechanism itself, with the
   caveats above. `measure_multi_splice_b.py` is the harness to use
   for any further work — ~17× faster than the original on hybrid
   attention models, validated as bit-equivalent.

2. **Trace-recurrence analysis** ([`trace_analysis/`](trace_analysis/)).
   `analyze_trace_reuse.py` computes coverage / longest-match /
   fragments per request. `categorize_matches.py` decomposes the
   matched volume by role + content type — load-bearing for telling
   real substrate from framework boilerplate.

3. **Admission strategy design** (was: predictors, item 3 in the
   previous SCOPE). Question is no longer "find a better byte-level
   heuristic" — that path is closed at r ≈ 0.36. Question is "given
   what categorize_matches.py shows on each new corpus, what
   structural / request-semantic admission rules would actually
   capture the substrate worth caching." Pen-and-paper work informed
   by data, not new measurement code.

4. **Eventual runtime cache implementation** (not started). Design
   constraints below; do not start until the admission picture on
   representative corpora has converged.

## Out of scope for reagent

- **Trace capture.** Owned by `agentcap`. Reagent ingests the
  exported HF datasets; it does not run agents or capture HTTP.
- **Hand-crafting tasks to make recurrence appear.** Generic prompts
  are the chosen experimental philosophy — the corpus should reflect
  how a real user/team exercises the agent, not how the experimenter
  thinks substrate should arise. If recurrence is weaker than
  expected, vary the *agent* or *model*, not the task list.
- **Model serving / inference engine internals.** Reagent uses
  `transformers` for splice forwards because the harness has to
  inject K/V tensors at chosen positions, which no production serving
  stack lets you do directly. The runtime impl will be a thin layer
  *on* a serving stack, not a replacement for one.
- **Agent-framework specifics.** Reagent's measurements are
  framework-agnostic. Substrate composition observations are
  agent-specific by nature, but the harness doesn't bake any
  framework assumptions in.

## Next steps in priority order

### 1. Re-run substrate categorisation as new corpora arrive

`categorize_matches.py` on the agentcap-Hermes corpus settled the
shape of the question for that particular (agent, model) pair. The
useful next datapoints are different (agent, model) pairs:

- **Hermes on a more compliant model** (Llama-3.3-70B-Instruct,
  Claude-Sonnet-4.6, etc.) — would the `skill_view` substrate that
  Gemma-4 skipped show up? If yes, that's the genuine same-user/team
  recurrence the cache was designed for.
- **OpenCode on the same models** — re-confirm the parameterised-
  template failure mode is reproducible on the new harness.
- **Other agents** as agentcap gains drivers for them.

For each, the deliverable is a `categorize_matches.py` output saying
"X% of recurrent volume is prefix-cacheable, Y% is post-prefix
boilerplate, Z% is real tool_response substrate, here are the keys
that would address Z."

### 2. Document the admission rules each corpus implies

Once two or three corpora have been categorised, the structural rules
should converge. Probable shape:

- Always cache the tools-schema block, keyed by agent_build_id
  (because Hermes's memory injection breaks the trivial prefix-cache
  reach for it).
- Cache `role=tool` blocks past the first turn, keyed by
  `tool_response:tool_name+args_hash`.
- Don't cache anything in role=assistant (server already has it).
- Don't cache anything that touches the memory section (it's
  poison — drifts every turn).

The output is a written admission policy in this repo, not new
measurement code.

### 3. Splice-correctness gaps that the runtime cache will hit

The published "shifted reuse is essentially indistinguishable" result
does NOT bound:

- **Interior-span chunks** — caching mid-message body, not just
  structural-edge chunks. Required for any chunk in the tools-schema
  block since the block has no internal turn boundary.
- **Multi-segment composition** — splicing two or three disjoint
  cached chunks into the same prompt with re-prefilled gaps. Does
  per-chunk error add, multiply, or saturate?
- **Accumulated reuse** — same chunk re-spliced across turns 2, 3, 4
  of a session. Errors a single splice hides may compound.

These are real measurement work and need GPU time. Defer until
item 2's admission rules tell us *which* chunks to test on. Don't run
broad sweeps speculatively.

### 4. Runtime cache implementation

A fresh implementation, not a fork of `lmcache` or `llama.cpp`.
Design constraints, in order:

- **Server-side, in front of a prefill path.** Wraps a serving stack's
  chat-completion handler. First target: `transformers serve` (we
  already have a fork to extend). vLLM is the second target.
- **Cache-key strategy is non-semantic and consumer-supplied.** The
  cache layer is told "these byte ranges are cacheable, hash them
  this way" by a request-render hook on the agent side. It does not
  classify content. Two shipping key schemes:
  - `tools_block:agent_build_id` — for the tools-schema block that
    sits behind Hermes's memory injection. Post-prefix recurrence
    the trivial prefix cache cannot reach.
  - `tool_response:tool_name+args_hash` — for `role=tool` message
    blocks past the first turn. The substrate identified by the
    `agentcap` manifests.
- **Splice mechanism is the shifted reuse from
  `measure_multi_splice_b.py`.** Same mechanism llama.cpp ships in
  `--cache-reuse`. Drop reagent's un-rephased baseline — it was a
  measurement strawman, not a candidate.
- **Storage: in-memory LRU + disk-backed (parquet or sqlite).** K/V
  tensors per layer, per chunk. Reads from disk on miss-of-hot-cache;
  promotes to in-memory on hit. Eviction by LRU + size.
- **Out of scope for the implementation:** task-success metrics,
  semantic content classification, replicas / multi-host coherency,
  encryption, multi-tenant isolation. Note: multi-user-shared-Hermes
  is an open question (would memory pollution help or hurt
  cross-user recurrence?) but it's outside the cache layer's
  responsibility.

The runtime impl is the largest item on this list. Don't start it
until items 1–3 have converged on a stable admission picture and a
real consumer commits to using it.

## What this repo deliberately does NOT do

- Capture agent traffic — `agentcap` does that.
- Build a runtime cache today — see item 4; not until committed to.
- Score task success — out of scope; reagent measures the next-token
  distribution, not whether the agent's eventual answer is correct.
- Hand-craft task suites to manufacture recurrence — see "Out of
  scope" above.
- Replicate every existing serving stack's caching behaviour — the
  splice-correctness measurement targets the mechanism (shifted
  reuse), and the answer transfers to any stack that implements it.

## Reading order for someone new

1. [README.md](README.md) — project overview + the published
   measurement result + drift-mode protocol.
2. `trace_analysis/README.md` — the cache-hit opportunity question
   and how it complements the correctness measurement.
3. This file — what's owned by this repo vs `agentcap`, what's
   already known, prioritised next steps.
4. `../agentcap/AGENTS.md` — handoff notes on the capture side.
5. `../hf-mount-cache-examples/FINDINGS.md` — frozen research log
   from the prototype phase. Useful for "why did we end up here"
   questions; not a current source of truth.

## Conventions

- Splice harnesses are `transformers`-based and run on a single
  GPU per measurement. Multi-GPU is for larger models, set up via
  `--device-map` flags on each script.
- New experiments go in their own `experiment_*.py` file with a
  matching `analyze_*` script. Don't extend the main harness with
  experiment-specific options.
- Tests under `test_*.py` are the regression net for the splice
  mechanics. Run them whenever you touch `kv_cache.py`,
  `rope_shift.py`, or `drift_modes.py`.
- HF datasets consumed by `trace_analysis/` get pinned by
  `dataset_id + revision` (or by bucket parquet path). Don't depend
  on a moving `main`.
