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
| **`llama.cpp`** fork at `dacorvo/llama.cpp`, branch `feat/cache-reuse-symmetric` | Server-side splice mechanism. Two patches: (1) symmetric `--cache-reuse` that walks both `head_c` and `head_p` so chunks can recur at non-aligned positions; (2) K-shift for text-only IM-RoPE / M-RoPE so Qwen3.5/3.6 / VL-family text inputs can use the cache-reuse path at all. | Active, mechanically correct (mechanical test passes at fp32 rel_err ~ 0). End-to-end correctness on hybrid models has a known gap — see findings below. |
| ~~`hf-mount-cache-examples`~~ | Original prototyping ground. **Deprecated** — closed out as a research log. The conclusions live in its `FINDINGS.md`; the code patterns live, refactored, in the two repos above. | Read-only. |

Status of the previously ruled-out implementation paths:

- **LMCache.** Still ruled out for the same reason: prefix-cache only.
  Doesn't address post-prefix chunk caching where the substrate lives.
  Not extensible to the chunk-cache regime without a major rewrite.
- **`llama.cpp --cache-reuse`.** **Partially un-ruled-out** by the
  symmetric-search patch on the fork. The patch removes blocker (a)
  from the previous SCOPE: --cache-reuse can now stitch *multiple*
  disjoint matched fragments separated by re-prefilled gaps. Blocker
  (b) — sliding byte-search vs request-semantic chunk identity — is
  still there but is independent: the request-semantic admission
  layer can sit on top of the now-extended splice mechanism without
  the kv_cache_unified rewrite the previous SCOPE feared. A single
  llama-server with the patch + a request-rewriting hook upstream
  could be a viable shipping path.

A fresh runtime-cache implementation is still on the table; the
patched llama.cpp is now also a candidate. Whether to fork-and-ship
vs. build a new layer depends on the open correctness questions
documented under "Findings — llama.cpp side" below.

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

### Substrate composition (agentcap goose + opencode corpora, May 2026)

Captured `transformers-coding-session/{goose,opencode}.parquet` against
Qwen3.6-35B-A3B. Different shape than Hermes-Gemma:

- **Goose**: 785 requests / 101 sessions. CP frac mean 0.44; post-prefix
  frac mean 0.18. 44% of requests have ≥2 disjoint post-prefix
  fragments. Of post-prefix tool_response tokens: `tree` 74%, `shell`
  15%, `analyze` 10% — very concentrated on a few `(tool_name,
  args_hash)` buckets. Top single bucket: `tree({"path":".","depth":2})`
  → 690k tokens × 62 hits.
- **Opencode**: 1062 requests / 72 sessions. CP frac mean 0.50;
  post-prefix mean 0.10 but multi-fragment 52% (more, smaller chunks).
  `read` 65%, `grep` 20%, `glob` 15% — many specific files
  (`chat_template_utils.py`, `tokenization_utils_base.py`, `trainer.py`).

Both confirm the same redesign signal: **request-semantic admission
keyed on `(tool_name, canonical_args_hash)` would capture the bulk of
non-CP recurrence**. No byte-level admission heuristic needed. The
shape is consistent across two agents on the same model.

### Findings — llama.cpp side (`feat/cache-reuse-symmetric`)

Two patches landed on the fork:

1. **Symmetric `--cache-reuse`.** Walks both `head_c` (cache index) and
   `head_p` (recipient index) on miss. Snapshots source ranges into a
   temp seq at packed positions, applies splices just-in-time during
   the prefill batch loop so the engine's strict `Y = X + 1` batch
   constraint stays satisfied. Three integration tests pass on
   tinyllama2; the agentcap manifest test on Llama-3.2-1B fires the
   path with positive shifts (`+803`, `+860`) — exactly the case the
   legacy single-pointer scan cannot reach. Drops Llama-3.2-1B
   recipient prompt_ms from 3592 → 1443 ms in the multi-splice case.

2. **K-shift for text-only IM-RoPE / M-RoPE.** Forward IMROPE writes
   per-token positions `(t,t,t,0)` for text inputs. The matching
   K-shift writes `(δ,δ,δ,0)` and applies it via the same
   `ggml_rope_multi` op — 2D rotations compose additively. Mechanical
   test in `tests/test-rope.cpp` confirms `(t,t,t,0) + (δ,δ,δ,0) ≡
   (t+δ,t+δ,t+δ,0)` at fp32 with `rel_err ~ 0`. `get_can_shift` now
   returns true for IMROPE/MROPE under the text-only invariant
   `ext.x == ext.y == pos[i]`; image cells (h, w spatial) are
   refused.

Single-file scope (~85 LoC in `src/llama-kv-cache.cpp`, plus the test
and `kv-cells.h` shift_ext flag). Investigation doc at
[`tools/server/notes/IM_ROPE_SHIFT_INVESTIGATION.md`](https://github.com/dacorvo/llama.cpp/blob/feat/cache-reuse-symmetric/tools/server/notes/IM_ROPE_SHIFT_INVESTIGATION.md)
on the branch.

One genuinely new architectural finding the implementation surfaced:

- **Hybrid recurrent state on Qwen3.5/3.6.** `LLM_ARCH_QWEN35MOE` is
  in `llm_arch_is_hybrid` — alongside the attention KV cache there's a
  recurrent (gated delta net) state. `llama_memory_recurrent::seq_add`
  only relabels positions; the recurrent state itself was rolled
  forward from the donor's history and isn't recomputable from a
  position relabel. So while the K-shift patch removes the IMROPE
  blocker on the attention side, splicing into a hybrid model still
  produces degraded output because the recurrent state isn't
  reordered with the splice. This is *not* the same divergence reagent
  has been measuring (which is the K-only RoPE-rephasing question);
  it's an independent blocker specific to hybrid architectures.
  **Out of scope for the current branch** — would need a separate
  redesign of hybrid memory for cache-reuse-with-splice.

Output divergence between cold-prefill and shifted-reuse on
*non-hybrid* models is the question this repo has been measuring
since day one (see "Mechanism" above and the published
`measure_reuse_drift.py` results). The bound is sim ≥ 0.95 / top-1
agreement on chunks ≥ 128 within the trust caveats listed earlier;
any tinyllama-at-temp-0 character-level divergence we observe in
the llama.cpp end-to-end test is consistent with that, not a
contradiction of it.

Net: the symmetric+IMROPE patch is *mechanically correct* and ready
for non-hybrid IMROPE/MROPE models (any pure-attention transformer
using `LLAMA_ROPE_TYPE_IMROPE` / `LLAMA_ROPE_TYPE_MROPE`). For hybrid
models like Qwen3.5/3.6 it lights up the code path but the end-to-end
output is degraded by the orthogonal recurrent-state problem.

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

What's actually new since the last SCOPE: the splice mechanism is
implemented in the llama.cpp fork (symmetric search + IMROPE shift,
mechanically correct), and we have substrate categorisation on two
agentcap corpora (goose, opencode). The output-divergence-of-shifted-
reuse question is *not* new — it's the original reagent question, and
the published result already says "diverges but at sim ≥ 0.95 / top-1
agreement on chunks ≥ 128 within the trust caveats above". Items below
focus on the genuinely open questions.

### 1. Decide what to do about the hybrid-model recurrent state

For the production target (Qwen3.5/3.6), splice into hybrid models
produces degraded output regardless of the K-shift patch — the
recurrent state from the donor doesn't survive a sequence reorder
into the recipient's path. This is a llama.cpp-side architectural
issue, separate from the rephasing mechanism reagent measures.

Two reasonable actions before any upstream PR:

- **Tighten `llama_memory_hybrid::get_can_shift`** to refuse shift
  when any cell has a populated recurrent counterpart. Closes the
  door on hybrid+cache-reuse cleanly until/unless someone solves the
  recurrent-state-reorder problem.
- **Document and shelve.** The K-shift patch lights up the path; the
  recurrent state degrades the output. For non-hybrid IMROPE/MROPE
  the patch is correct as-is. Land it scoped to non-hybrid; open a
  separate issue for hybrid.

Recommendation: tighten get_can_shift on this branch, file the
recurrent-state issue separately. Pick before opening upstream PR.

### 2. Re-run substrate categorisation on a non-hybrid model

The agentcap captures we have are all on Qwen3.6-35B-A3B (hybrid).
For an honest end-to-end splice run on real workload data we'd want
captures from a non-hybrid model. Candidates with single-axis RoPE
and no recurrent counterpart:

- **Gemma-4-26B-A4B-it** — partially captured already. Hermes-Gemma
  corpus exists; goose/opencode/pi captures against this model would
  let us run the manifest test end-to-end with the patched llama.cpp
  without the recurrent-state confound.
- **Llama-3.3-70B-Instruct** — high compliance with Hermes-style
  skill loading.
- **Qwen3-32B-Instruct** (non-3.5) — closest dialect cousin to
  Qwen3.6 without the hybrid memory.

The capture+categorize+manifest pipeline already works (goose and
opencode results landed). What's missing is recapture against a
non-hybrid model so the end-to-end splice test in
`trace_analysis/test_splice_against_manifest.py` actually exercises
real (donor, recipient) recurrence on the production-target patch.

### 3. Document the admission rules the corpora imply

Already converging from goose + opencode. Probable shape:

- Always cache the tools-schema block, keyed by agent_build_id
  (Hermes's memory injection breaks the trivial prefix-cache reach
  for it; on goose/opencode the schema sits inside CP and is served
  by ordinary prefix-cache, but the rule is generic).
- Cache `role=tool` blocks past the first turn, keyed by
  `tool_response:tool_name+args_hash`. **This is the load-bearing
  rule** — goose's top 11 `(tree, args)` buckets account for ~90% of
  tree's 1.9M post-prefix tool_response tokens; opencode's top
  buckets are similarly concentrated on a handful of `read` /
  `grep` calls on hot files.
- Don't cache anything in role=assistant (server already has the KV
  from generating it).
- Don't cache anything that touches the memory section (drifts
  every turn).

Output: a written admission policy in this repo, plus a sketch of
the request-render hook on the agent side that would supply the
keys. Note that this layer can now sit on top of patched llama.cpp
(blocker (b) from the previous SCOPE turned out to be an HTTP-layer
concern, not a kv_cache_unified rewrite).

### 4. Splice-correctness gaps that the published result does NOT bound

The published "shifted reuse is essentially indistinguishable" result
covers ≥ 128-token chunks at structural edges with enough suffix to
recover. The runtime cache will hit cases the result *does not* cover:

- **Interior-span chunks** — caching mid-message body. Required for
  tools-schema chunks (no internal turn boundary).
- **Multi-segment composition** — splicing two or three disjoint
  cached chunks into the same prompt with re-prefilled gaps. The
  symmetric-cache-reuse patch *does* multi-segment splicing today;
  whether per-chunk error adds, multiplies, or saturates is
  unmeasured. The patched llama.cpp gives us a way to run this on
  real workloads via the manifest harness.
- **Accumulated reuse** — same chunk re-spliced across turns 2, 3, 4
  of a session. Errors a single splice hides may compound.

GPU time required. Defer until item 2's non-hybrid corpus exists so
the measurements are on real recurrence patterns, not synthesized
donor/recipient pairs.

### 5. Runtime cache strategy decision

The previous SCOPE assumed a fresh implementation. With the patched
llama.cpp on the table, there are now three options:

- **Fork-and-ship the patched llama.cpp** + a request-rewriting hook
  that supplies tool_response keys at the HTTP layer. Fastest to a
  shipping product. Constrained to llama.cpp-supported models, and
  to non-hybrid attention until item 1 finds a way around the
  recurrent-state issue.
- **Fresh implementation on top of `transformers serve` / vLLM.**
  More work; lets us implement chunk-level admission natively and
  potentially work around the hybrid-recurrent-state issue at the
  cache layer.
- **Both.** The patched llama.cpp ships first as a proof; the fresh
  implementation is the longer-term home.

Decide after items 1–3 are settled.

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
