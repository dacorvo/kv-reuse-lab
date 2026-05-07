# reagent — scope and next steps

Orientation note for picking up work. Read [README.md](README.md) for
project context and the **CP** / **post-prefix matches** terminology.

## Repos in this project

| repo | role |
|---|---|
| `../agentcap` | Capture LLM-agent chat-completion traffic via a transparent OpenAI-compat proxy; export as HF datasets. |
| **`reagent`** (this repo) | Measure cross-request KV-cache-reuse correctness (splice-with-shift); analyse recurrence in captured traces; design the runtime cache. |
| `dacorvo/llama.cpp` branch `feat/cache-reuse-symmetric` | Research instrument used to test whether the splice mechanism is *applicable* to specific architectures. Two patches: symmetric `--cache-reuse` (multi-segment splice) + K-shift for text-only IM-RoPE / M-RoPE. Whether it ships upstream is a downstream concern. |
| ~~`hf-mount-cache-examples`~~ | Deprecated research log. |

## What we know

### Mechanism — splice with RoPE rephasing

The mechanism works on clean structural-edge chunks of size ≥ 128 at
sim ≥ 0.95 / top-1 agreement, validated on six attention models
(Gemma-4 sizes, Llama-3.2-1B, Llama-3.1-8B). Within that envelope,
spliced output is essentially indistinguishable from cold prefill at
the next-token-distribution level. The result is **suggestive, not
airtight**: small N per cell, catastrophic outliers exist, and the
fixes in `02bad94` (input-only role filter) and `a3ce1dc` (skip
first-turn matches) postdate most of the published numbers, so any
pre-fix figure is indicative not established.

What the published result does **not** bound:

- Interior-span chunks (mid-message, no structural edge).
- Multi-segment composition — two or three disjoint cached chunks
  in one prompt with re-prefilled gaps.
- Accumulated reuse across multiple turns of one session.
- Hybrid-architecture splicing (see "Findings — llama.cpp side").

llama.cpp `--cache-reuse` does the same RoPE rephasing the splice
harness does, so the mechanism is not the differentiator with
production stacks. The naive (un-rephased) baseline is reagent-internal
only — no production stack ships it.

### Admission, not lookup, is the hard problem

The matching algorithm is the easy half; deciding what to cache at
write time is the hard half. Best byte-level admission heuristic
correlation observed (entropy at the chunk being cached, K/V
perturbation sensitivity) is r ≈ +0.36 — useful as ranking, **not as
a sole gate**. Cheap entropy proxy ≈ expensive K/V perturbation;
don't redo perturbation experiments.

For agent workloads, **request-semantic keys win**. The cache layer
is told "these byte ranges are cacheable, hash them this way" by a
request-render hook on the agent side. Two shipping key shapes:

- `tools_block:agent_build_id` for the tools-schema block (sits
  behind any session-state injection like Hermes's memory).
- `tool_response:tool_name+canonical_args_hash` for `role=tool`
  message blocks past the first turn.

### Substrate composition

Two corpora measured, both via `categorize_matches.py`:

- **Hermes-Gemma (gemma-4-E4B-it, 27 sessions × 4 turns).** ~86%
  apparent coverage decomposes to: ~32% Hermes system prefix
  (prefix-cacheable), ~66% tools-schema (post-prefix because Hermes's
  memory injection breaks CP at ~4.7k tokens), ~1.4% system_other,
  ~0.69% tool_response (skills_list only — Gemma-4 didn't comply with
  the "MUST skill_view(name)" instruction, so this is artificially
  small).
- **goose + opencode (qwen3.6-35b-a3b, ~800–1000 requests each).**
  Different shape. Post-prefix tool_response is the load-bearing
  recurrence:
  - goose: `tree` 74%, `shell` 15%, `analyze` 10%. Top single bucket
    `tree({"path":".","depth":2})` → 690k tokens × 62 hits.
  - opencode: `read` 65%, `grep` 20%, `glob` 15%. Concentrated on a
    handful of hot files (`chat_template_utils.py`, `trainer.py`, …).

Both confirm the same redesign signal: request-semantic admission on
`(tool_name, args_hash)` captures the bulk of non-CP recurrence. No
byte-level heuristic needed.

**Important framing.** agentcap captures accumulate session-spanning
state (memory blocks, tool histories) by design. This *is* the
multi-user / shared-bucket regime — bytes are common across users
(tools schema, file contents) but local-state injection breaks CP, so
those bytes become post-prefix matches. That's exactly where the
runtime cache earns its keep.

### Findings — llama.cpp side

The fork branch was used as a research instrument to test mechanism
applicability. Two findings:

- **Multi-axis-RoPE attention models** (Qwen3 VL family, Qwen3
  non-3.5): mechanism applies. The `(t,t,t,0) + (δ,δ,δ,0) ≡
  (t+δ,t+δ,t+δ,0)` IMROPE composition is mathematically correct
  (mechanical test at fp32 `rel_err ~ 0`); the symmetric search +
  K-shift end-to-end pipeline runs; output divergence is governed by
  the same attention-side dynamics reagent measures.
- **Hybrid attention+recurrent models** (Qwen3.5/3.6): mechanism does
  *not* apply. These have a recurrent state alongside the attention
  KV — a per-layer rolling aggregate of all prior tokens that's not
  position-addressable. Splicing into a hybrid model leaves the
  recurrent state stale (it carries the donor's prefix; the recipient
  needs its own). Reagent's published splice-correctness work
  measured attention-only and does **not** bound this — the
  recurrent dependency has no spatial decay and no recovery
  mechanism.

Net: the symmetric+IMROPE patch is mechanically correct and ready for
non-hybrid IMROPE/MROPE attention models. For hybrid Qwen3.5/3.6 it
lights up the code path but produces semantically degraded output —
a separate problem not solvable at the K-shift layer.

## Next steps (priority order)

### 1. Recapture corpora on a non-hybrid model

Existing agentcap captures are all on Qwen3.6-35B-A3B (hybrid). For
end-to-end splice-correctness measurements to be meaningful we need
captures from a non-hybrid model running the same agents.
Substrate composition is a workload property, not a model property,
so it should reproduce. Candidates:

- **Gemma-4-26B-A4B-it** — non-hybrid, partial captures already
  exist.
- **Llama-3.3-70B-Instruct** — non-hybrid + high compliance with
  Hermes-style skill loading (would surface the skills-substrate
  Gemma-4 didn't).
- **Qwen3-32B-Instruct** (non-3.5) — closest dialect-cousin to
  Qwen3.6 without the hybrid memory.

Deliverable: fresh `categorize_matches.py` outputs across goose,
opencode, pi, hermes on at least one of these.

### 2. Splice-correctness measurements past the published bound

Three categories the published result does not cover, listed in
priority of relevance to the runtime cache:

- **Multi-segment composition** — does per-chunk error add, multiply,
  or saturate when 2-3 chunks are spliced into one prompt? The
  symmetric-cache-reuse patch can drive this experiment on real
  traces; `measure_multi_splice_b.py` can also.
- **Interior-span chunks** — chunks not at message boundaries.
  Required for tools-schema reuse since the block has no internal
  turn boundary.
- **Accumulated reuse** — same chunk re-spliced across turns of one
  session.

Defer until item 1 has a corpus to test against.

### 3. Document admission rules

Pen-and-paper, no measurement. Already converging from goose +
opencode. Probable shape:

- Always cache the tools-schema block, keyed by `agent_build_id`.
- Cache `role=tool` blocks past the first turn, keyed by
  `tool_response:tool_name+args_hash`. **Load-bearing rule.**
- Don't cache role=assistant (server already has the KV).
- Don't cache anything touching the memory section (drifts every
  turn).

Output: a written admission policy in this repo + a sketch of the
agent-side request-render hook that would supply the keys.

### 4. Runtime cache strategy decision

Three options, decide after items 1-3:

- Reagent-design + vendored patched llama.cpp as the engine.
  Fastest. Constrained to llama.cpp-supported, non-hybrid models.
- Reagent-design + fresh runtime cache on `transformers serve` or
  vLLM. More work; not tied to llama.cpp release cadence; could
  potentially handle hybrid at the cache layer.
- Both — patched llama.cpp as the proof / research vehicle, fresh
  implementation as the longer-term home.

Decision depends on which models the eventual consumer cares about
and what items 1-2 reveal.

## In scope / out of scope

**In scope.** Splice-correctness measurement; recurrence analysis on
captured traces; admission-policy design; eventual runtime cache
implementation (not started — gated by items 1-3 above).

**Out of scope.** Capturing agent traffic (agentcap owns that);
hand-crafting tasks to manufacture recurrence (generic prompts are
the chosen experimental philosophy — vary the *agent or model* if
recurrence is weak, not the task list); inference-engine internals
beyond the splice mechanism; agent-framework specifics; task-success
metrics; semantic content classification at the cache layer.

## Reading order for someone new

1. [README.md](README.md) — project overview, drift-mode protocol,
   published measurement.
2. [`trace_analysis/README.md`](trace_analysis/README.md) — recurrence
   analysis side.
3. This file — what's owned, what's known, what's next.
4. `../agentcap/AGENTS.md` — capture-side handoff notes.
