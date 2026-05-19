# kv-reuse-lab — scope and next steps

Orientation note. Read [README.md](README.md) for the headline result
and the **CP** / **post-prefix matches** terminology.

## Repos

| repo | role |
|---|---|
| `../agentcap` | Capture LLM-agent chat-completion traffic via a transparent OpenAI-compat proxy; export as HF datasets. |
| **`kv-reuse-lab`** (this repo) | Recurrence analysis on captures + splice-correctness harness + admission-policy design. |
| `dacorvo/llama.cpp` branch `feat/cache-reuse-symmetric` | Research instrument: symmetric `--cache-reuse` (multi-segment splice) + K-shift for IM-RoPE / M-RoPE. SWA models need `--swa-full` to bypass the iswa size-mismatch shift gate. |

## What's known

### Mechanism

The splice mechanism (RoPE-rephased K/V transplant) is correct on
attention-only architectures: full attention, multi-axis RoPE, and SWA
(with `--swa-full`). Confirmed end-to-end through patched llama-server
across the full agent × model matrix:

- **Gemma-4-E4B (SWA + attention):** 574 successful splices across 8
  (agent, session) corpora — goose, hermes, opencode, pi × {tc, hh}.
  Mean agree 84.5%, mean sim 0.93, 3.3% catastrophic by generic
  threshold (top-1 mismatch + KL>1 or sim<0.5).
- **Qwen3-Coder-30B (full attention):** 396 splices across 4 (agent)
  corpora × tc. Mean agree 84.8%, mean sim 0.93, 2.0% catastrophic.
  First measurement of this architecture class; confirms the
  full-attention prediction in the README architecture table.
- **Gemma-4-26B (SWA + attention):** 4-corpus retry in progress
  (sequential on 4 GPUs each — the model + ctx=131072 KV doesn't fit
  on a 2-GPU split).
- **Drift magnitude does not predict splice quality.** Agree rate is
  flat across |drift| buckets from <100 to ≥50k tokens; bigger-drift
  buckets actually trend slightly better because they're dominated by
  larger chunks.

The mechanism is **not applicable** to hybrid attention+recurrent
models (Qwen3.5/3.6). On goose × Qwen3.6-35B-A3B, 9% of pairs
collapse the spliced first token onto either token 510 (`</`, orphan
close-tag) or token 248046 (`<|im_end|>`, premature end-of-turn) —
recurrent gated-delta-net state corruption that the K-shift cannot
reach. Detail at
[trace_analysis/results/agentcap_goose_splice_postmortem.md](trace_analysis/results/agentcap_goose_splice_postmortem.md).

Multi-segment composition (up to 9 disjoint chunks with mixed-direction
shifts ±30k positions) handled cleanly by the symmetric search.

What the result does not bound: interior-span chunks (mid-message, no
structural edge); accumulated reuse across many turns of one session.

### Soft donor-context bleed (pending validation)

Bottom-quintile pairs on Gemma-4 (sim 0.65–0.80) share a pattern: the
spliced model's response is shaped by the donor's task framing because
the chunk's K/V cells encode attention to the donor's pre-chunk
context. Splice quality scales monotonically with
`recipient_turns − donor_turns`:

```
delta = -3   N=3   mean_sim 0.808
delta =  0   N=9   mean_sim 0.896
delta = +2   N=13  mean_sim 0.971
delta = +4   N=4   mean_sim 0.967
```

Candidate pre-splice admission heuristic: prefer donors whose
turn-position ≤ recipient's. Single-corpus / single-model observation
— validation across other agents and models is in next steps.

### Substrate

Recurrence is dominated by post-prefix tool responses keyed by
`(tool_name, args_hash)`. On goose × {Qwen3.6, Gemma-4-E4B,
Gemma-4-26B} the shape is the same: `tree`, `analyze`, `shell`,
`read` produce the bulk of cacheable tokens; concentrated on a
handful of recurrent calls (e.g. `tree({"path":"."})`). Substrate
composition is a workload property, not a model property.

Implication: **request-semantic admission keys are sufficient**, no
byte-level admission heuristic needed.

### Admission rules (current shape)

- Cache the tools-schema block, keyed by `agent_build_id`.
- Cache `role=tool` blocks past the first turn, keyed by
  `tool_response:tool_name+args_hash`. Load-bearing.
- Don't cache `role=assistant` (server has the KV; and donor-assistant
  content matching across sessions corresponds to a same-task-replay
  regime, not a real production scenario).
- Don't cache anything touching session-local injection (memory
  blocks).
- Prefer donors whose turn-position ≤ recipient's (pending broader
  validation).
- Floor admission at chunks ≥ ~1k tokens — below that the prefill
  saving is dwarfed by lookup overhead.

## Next steps

### 1. Disk-backed cache + benchmark A/B

Build a disk-backed cache behind patched llama-server, keyed by
`tool_response:tool_name+args_hash`. Run a single-repo agent suite
through it. Best existing fit: **SWE-bench Verified filtered to
django/django (231 tasks)** — same repo across hundreds of tasks,
recurring `tree` / `read_file` / `grep` calls, deterministic
test-based pass/fail.

A/B same agent, same tasks, two servers — vanilla vs cache-enabled.
Pass-rate equality is the correctness gate; total wall-clock, tokens
prefilled, GPU-seconds are the value-add metrics.

### 2. Cross-agent / cross-model heuristic validation

**Splice-correctness generalization — done.** Confirmed on
goose/hermes/opencode/pi × {Gemma-4-E4B, Qwen3-Coder-30B} (Gemma-4-26B
retry pending), see headline. Splice quality is agent-invariant on
attention-only architectures; the catastrophic-outlier signatures
remain bounded to the hybrid Qwen3.5/3.6 class.

**Turn-delta admission heuristic — still pending.** The
`recipient_turns − donor_turns` regression (item §1.3) is computable
from the new sweep data ([/tmp/cm_verify/splice_metrics/](file:///tmp/cm_verify/splice_metrics/))
across agents — replicate the goose × Gemma-4 finding before treating
it as load-bearing.

### 3. Past-the-bound splice correctness (deferred)

Interior-span chunks; accumulated reuse across many turns of one
session. Defer until item 1's corpus is running.

## In scope / out of scope

**In.** Recurrence analysis; splice-correctness measurement;
admission-policy design; runtime cache implementation (gated on
item 1).

**Out.** Capturing agent traffic (agentcap owns that); hand-crafting
tasks to manufacture recurrence (generic prompts are the chosen
philosophy — vary the agent or model if recurrence is weak, not the
task list); inference-engine internals beyond the splice mechanism;
agent-framework specifics; task-success metrics beyond pass-rate
equality; semantic content classification at the cache layer.

## Reading order

1. [README.md](README.md) — pipeline + headline result.
2. [`trace_analysis/README.md`](trace_analysis/README.md) — script-level methodology.
3. This file — known state + next steps.
4. `../agentcap/AGENTS.md` — capture-side handoff notes.
