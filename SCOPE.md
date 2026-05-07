# reagent — scope and next steps

Orientation note for picking up work. Read [README.md](README.md) for
project context and the **CP** / **post-prefix matches** terminology.

## Repos in this project

| repo | role |
|---|---|
| `../agentcap` | Capture LLM-agent chat-completion traffic via a transparent OpenAI-compat proxy; export as HF datasets. |
| **`reagent`** (this repo) | Measure cross-request KV-cache-reuse correctness (splice-with-shift); analyse recurrence in captured traces; design the runtime cache. |
| `dacorvo/llama.cpp` branch `feat/cache-reuse-symmetric` | Research instrument used to test whether the splice mechanism is *applicable* to specific architectures. Two patches: symmetric `--cache-reuse` (multi-segment splice) + K-shift for text-only IM-RoPE / M-RoPE. Whether it ships upstream is a downstream concern. SWA models (Gemma-4 family) require `--swa-full` at server launch to bypass the iswa size-mismatch shift gate. |
| ~~`hf-mount-cache-examples`~~ | Deprecated research log. |

## What we know

### Mechanism — splice with RoPE rephasing

The mechanism works on clean structural-edge chunks of size ≥ 128 at
sim ≥ 0.95 / top-1 agreement, validated on six attention models
(Gemma-4 sizes, Llama-3.2-1B, Llama-3.1-8B). Within that envelope,
spliced output is essentially indistinguishable from cold prefill at
the next-token-distribution level. End-to-end through patched
llama-server reproduces the bound on Gemma-4-E4B-it (80 pairs,
mean sim 0.92, mean KL 0.23, agree 82.5%, **zero catastrophic
outliers**).

What the result does **not** bound:

- Interior-span chunks (mid-message, no structural edge).
- Accumulated reuse across multiple turns of one session.
- Hybrid-architecture splicing (see "Findings — llama.cpp side").

Multi-segment composition is empirically clean on the patched
llama-server — symmetric search handles up to 9-segment splices with
mixed-direction shifts (±30k positions) without producing structurally
broken output.

llama.cpp `--cache-reuse` does the same RoPE rephasing the splice
harness does, so the mechanism is not the differentiator with
production stacks. The naive (un-rephased) baseline is reagent-internal
only — no production stack ships it.

### Soft donor-context bleed (Gemma-4 finding)

On the Gemma-4 sweep the bottom-quintile pairs by sim (0.65–0.75)
share a pattern: spliced model produces a response that "remembers"
turns from the donor's session that don't exist in the recipient's.
Spliced K/V cells encode attention to donor's pre-chunk context;
when transplanted into a recipient with a *shorter* pre-chunk history
than the donor's, those K/V values inject context recipient's flow
doesn't have.

Empirical signal: splice quality (sim) scales monotonically with
`recipient_turns − donor_turns`:

```
delta = -3   N=3   mean_sim 0.808
delta =  0   N=9   mean_sim 0.896
delta = +2   N=13  mean_sim 0.971
delta = +4   N=4   mean_sim 0.967
```

Pre-splice admission heuristic candidate: prefer donors whose turn
position is ≤ recipient's. Single-agent / single-model / 30-task
observation — needs cross-agent + cross-model + larger-corpus
validation before being treated as a load-bearing rule.

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
applicability. Three architecture classes:

- **Multi-axis-RoPE attention models** (Qwen3 VL family, Qwen3
  non-3.5): mechanism applies. The `(t,t,t,0) + (δ,δ,δ,0) ≡
  (t+δ,t+δ,t+δ,0)` IMROPE composition is mathematically correct
  (mechanical test at fp32 `rel_err ~ 0`); the symmetric search +
  K-shift end-to-end pipeline runs; output divergence is governed by
  the same attention-side dynamics reagent measures.
- **SWA attention models** (Gemma-4 family): mechanism applies, but
  llama.cpp's iswa cache layout sizes the SWA buffer smaller than the
  base buffer by default. The shift gate at
  `llama-kv-cache-iswa.cpp:223` requires equal-sized base/SWA caches.
  Pass `--swa-full` at server launch to equalize them; then K-shift
  forwards through both buffers correctly. Cost: the SWA memory
  saving is given back. Verified end-to-end on Gemma-4-E4B-it (80
  pairs, zero catastrophic outliers).
- **Hybrid attention+recurrent models** (Qwen3.5/3.6): mechanism does
  *not* apply. The recurrent gated-delta-net layer state is a
  per-layer rolling aggregate of all prior tokens, not
  position-addressable. Splicing into a hybrid model leaves the
  recurrent state stale (it carries the donor's prefix; the recipient
  needs its own). Empirically: 9% (8/88) of Qwen3.6-35B-A3B goose
  pairs collapse the spliced first-token onto two specific tokens —
  `</` (token 510, orphan close-tag fragment) or `<|im_end|>` (token
  248046, premature end-of-turn) — producing syntactically valid but
  semantically broken output. Recurrent state corruption signature.
  Not fixable at the K-shift layer.

Net: the symmetric+IMROPE patch is mechanically correct and ready for
non-hybrid attention models (full or SWA). For hybrid Qwen3.5/3.6 it
lights up the code path but produces semantically degraded output —
a separate problem not solvable at the K-shift layer.

## Next steps (priority order)

### 1. Disk-backed long-running cache + benchmark A/B

The splice mechanism is empirically correct on attention-only
architectures (Gemma-4 sweep above). The next thing worth measuring
is whether a long-running cache *across* many real coding tasks
delivers wall-clock and tokens-prefilled wins without breaking
correctness. Setup:

- Build a disk-backed cache layer behind patched llama-server that
  persists chunks across requests, keyed by
  `tool_response:tool_name+args_hash`.
- Run a single-repo agent task suite end-to-end. **SWE-bench Verified
  filtered to django/django (231 tasks)** is the cleanest existing
  fit: same repo across hundreds of tasks ⇒ recurring `tree`,
  `read_file`, `grep` calls populate the cache and hit each other.
- A/B: same agent, same tasks, two servers — vanilla vs
  cache-enabled. Compare task pass rate (correctness gate),
  wall-clock, total tokens prefilled, GPU-seconds.

Pass-rate equality (within noise) is the correctness gate; the rest
are the value-add metrics.

### 2. Cross-agent / cross-model validation of admission heuristics

The Gemma-4 turn-delta finding (splice quality scales with
`recipient_turns − donor_turns`) was observed on goose × Gemma-4 only.
Before treating it as a load-bearing admission rule, validate on:

- Different agents: opencode, aider, hermes (different prompt /
  tool-call structures).
- Different non-hybrid attention models: Llama-3.3-70B-Instruct,
  Qwen3-32B (non-3.5).
- Larger task pools.

### 3. Splice-correctness measurements past the published bound

Defer to item 1's corpus. The categories not yet covered:

- **Interior-span chunks** — chunks not at message boundaries.
  Required for tools-schema reuse since the block has no internal
  turn boundary.
- **Accumulated reuse** — same chunk re-spliced across turns of one
  session.

Multi-segment composition is empirically clean (handled).

### 4. Document admission rules

Pen-and-paper, no measurement. Already converging from goose +
opencode + gemma4. Current shape:

- Always cache the tools-schema block, keyed by `agent_build_id`.
- Cache `role=tool` blocks past the first turn, keyed by
  `tool_response:tool_name+args_hash`. **Load-bearing rule.**
- Don't cache role=assistant (server already has the KV; and donors
  whose assistant content gets matched produce a degenerate "same-task
  replay" regime, not a production scenario).
- Don't cache anything touching the memory section (drifts every
  turn).
- Prefer donors whose turn position is ≤ recipient's at admission
  time (gemma4 finding, pending validation per item 2).
- Floor admission at chunks ≥ ~1k tokens — below that the prefill
  saving is dwarfed by lookup overhead. The 128-token min-match in
  `categorize_matches.py` is for measurement, not production
  admission.

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
