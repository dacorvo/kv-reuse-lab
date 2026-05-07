# reagent — KV cache reuse correctness for agent workloads

Reagent characterises whether server-side KV-cache reuse preserves
task behaviour when a cached block appears at a drifted absolute
position in a new request, and analyses how often such reusable
chunks appear in real agent traces. It does **not** ship a runtime
cache; it characterises the splice mechanism and the substrate, with
an eye to what a runtime cache would need to handle.

For the orientation note (what reagent owns vs sibling repos, what's
known, what's next), read [SCOPE.md](SCOPE.md).

## What reagent does, in three pieces

1. **Recurrence analysis** in [`trace_analysis/`](trace_analysis/).
   For real agent corpora captured by `../agentcap`, how much of a
   request can be byte-matched against earlier requests, what
   fraction is post-prefix (vs CP), and what request-semantic
   structure does the recurring content share. Output includes a
   per-request CP/post-prefix breakdown
   (`analyze_trace_reuse.py`), a match-category and
   `(tool_name, args_hash)` decomposition (`categorize_matches.py`),
   and an emitted **splice-candidate manifest** of (donor, recipient,
   range) triples for downstream correctness measurement.
2. **Splice-correctness measurement** end-to-end through a patched
   llama-server. `trace_analysis/test_splice_against_manifest.py`
   takes the manifest, fetches each pair's request body from the
   source parquet, posts donor + recipient sequentially to the
   server, and parses the server's stderr to verify the splice
   actually fires plus measure the resulting output divergence.
3. **Mechanism-applicability research instrument** at
   `dacorvo/llama.cpp` branch `feat/cache-reuse-symmetric`. Two
   patches that extend `--cache-reuse` (multi-segment splice;
   K-shift for text-only IM-RoPE / M-RoPE) so we can test which
   architectures the splice mechanism is *applicable* to. See
   `tools/server/notes/IM_ROPE_SHIFT_INVESTIGATION.md` on the
   branch for the findings.

## Concepts

Two terms do most of the work. Defined precisely so the rest of the
docs stop drifting.

### Common Prefix (CP)

For two requests `A` and `B`, the **Common Prefix** is the byte-stable
token region they share starting at position 0:

```
A:  [t0, t1, t2, t3, t4, t5, ...]
B:  [t0, t1, t2, X,  Y,  Z,  ...]
                 ↑
                 CP ends here (length 3)
```

CP is what a **standard prefix cache** can serve: bytes 0 through
`CP-1` are byte-identical to a previously-prefilled request, the
cache hits; at byte `CP` the bytes diverge and from there on it's
fresh prefill. Every production serving stack ships some form of
prefix cache, so CP coverage is the *baseline* — what you get
without any of the work this project measures.

### Post-prefix matches

A **post-prefix match** is any byte-exact run of ≥ N tokens between
two requests that starts at a position past either side's CP. By
construction it cannot be served by a prefix cache: at least one of
the two requests already diverged before this match begins.

Post-prefix matches come from two sources:

- **Mid-prefix injection breaks CP early.** Hermes injects a
  `MEMORY (your personal notes)` block between the system prompt
  and the tools schema. The system-prompt bytes are identical
  across two sessions, but as soon as memory differs CP ends —
  even though ~10k bytes of byte-identical tools-schema sit right
  after, those bytes are now post-prefix on every cross-session
  pair.
- **Recurring content past the first turn.** Tool responses (file
  reads, web fetches, command outputs) that recur when the same
  user/team repeats similar work.

Both sources need a cache that can match against the *content* of a
chunk, not just the offset. llama.cpp's `--cache-reuse` is the only
production stack that does this today, with two limitations: it
historically could only stitch one contiguous post-prefix run (the
fork patch removes that), and it identifies hits by sliding
byte-search rather than by chunk identity.

## Splice mechanism

The splice mechanism: when chunk X appears in request A at position
p_A and in request B at position p_B ≠ p_A, take A's K/V cells for
X out of cache, re-rotate K by the position delta to match B's
RoPE phases, paste it into B's prefill at p_B, and continue
decoding. This matches what llama.cpp's `--cache-reuse` does in
the engine. The patched fork extends it to multi-segment (multiple
disjoint chunks per request) and to multi-axis-RoPE models.

Whether splicing produces the same output as cold prefill is the
core correctness question. End-to-end measurements through the
patched llama-server (see "Splice-correctness measurements" below)
confirm clean behaviour on attention-only architectures (Gemma-4
family) and a 9% catastrophic failure rate on hybrid
attention+recurrent architectures (Qwen3.6 family). See
[SCOPE.md](SCOPE.md) for what's bounded vs unbounded.

## Recurrence analysis — what's known so far

Three corpora measured via `categorize_matches.py`:

**Hermes-Gemma** (`gemma-4-E4B-it`, 27 sessions × 4 turns,
agentcap-captured). Apparent coverage 86% decomposes to: ~32%
Hermes system prefix (prefix-cacheable), ~66% tools-schema
(post-prefix because Hermes's memory injection breaks CP at
~4.7k tokens), ~1.4% system_other, ~0.69% tool_response (only
skills_list — Gemma-4 didn't comply with the "MUST skill_view(name)"
instruction, so this is artificially small).

**goose + opencode** (`qwen3.6-35b-a3b`, ~800-1000 requests each).
Different shape. Post-prefix tool_response is the load-bearing
recurrence:

- goose: `tree` 74%, `shell` 15%, `analyze` 10%. Top single bucket
  `tree({"path":".","depth":2})` → 690k tokens × 62 hits.
- opencode: `read` 65%, `grep` 20%, `glob` 15%. Concentrated on a
  handful of hot files (`chat_template_utils.py`, `trainer.py`, …).

**goose × gemma-4-E4B-it** (`google/gemma-4-E4B-it`, 68 sessions,
310 requests). Same shape as goose × qwen3.6 — `tree` and `analyze`
dominate post-prefix tool_response:

- `tree({"path":"."})` 1.78M tokens × 238 matches (top bucket)
- `tree({"path":"./"})` 227k × 33; `analyze({"path":"."})` 164k × 39
- 84% match coverage, 57% post-prefix.

Substrate composition reproduces across model families — confirms
recurrence is a workload property, not a model property.

All three confirm the same redesign signal: **request-semantic
admission on `(tool_name, args_hash)` captures the bulk of non-CP
recurrence**. No byte-level admission heuristic needed.

## Splice-correctness measurements

End-to-end through patched llama-server, both corpora measured:

| corpus | model | N | agree | mean KL | mean sim | catastrophic |
|---|---|---|---|---|---|---|
| goose × Qwen3.6 | qwen3.6-35B-A3B (hybrid) | 88 | 75% | 4.40 nats | 0.91 | 8 (9%) |
| goose × Gemma-4 | gemma-4-E4B-it (attention) | 80 | 82.5% | 0.23 nats | 0.92 | 0 |

Hybrid splice produces well-formed but divergent output 9% of the
time, collapsing to either `</` (token 510, orphan close-tag stream
like `</parameter>\n</function>\n</tool_call>`) or `<|im_end|>`
(token 248046, premature end-of-turn). These are recurrent-state
corruption signatures — the K-shift can rephase attention K/V but
cannot rewrite the recurrent gated-delta-net layer's compressed
state. Documented at
[trace_analysis/results/agentcap_goose_splice_postmortem.md](trace_analysis/results/agentcap_goose_splice_postmortem.md).

Attention-only models splice cleanly. Bottom-quintile divergence on
Gemma-4 is "soft donor-context bleed": the spliced model produces a
response shaped by the donor's task framing rather than the
recipient's. Magnitude correlates with `recipient_turns − donor_turns`
(see [SCOPE.md](SCOPE.md) for the candidate admission heuristic).

See [`trace_analysis/README.md`](trace_analysis/README.md) for
methodology and [SCOPE.md](SCOPE.md) for the corpus-derived
admission policy.

## Mechanism applicability — llama.cpp fork

Branch `feat/cache-reuse-symmetric` on `dacorvo/llama.cpp` carries
two patches used as a research instrument:

1. **Symmetric `--cache-reuse`.** Walks both `head_c` (cache index)
   and `head_p` (recipient index) on miss, snapshots source ranges
   into a temp seq, applies splices just-in-time during the
   prefill batch loop. Fixes the "single contiguous run"
   limitation of the upstream algorithm.
2. **K-shift for text-only IM-RoPE / M-RoPE.** Forward IMROPE
   writes per-token positions `(t,t,t,0)` for text inputs; the
   matching K-shift writes `(δ,δ,δ,0)` and applies it via
   `ggml_rope_multi`. Mechanically tested at fp32 with
   `rel_err ~ 0`.

Findings (see `tools/server/notes/IM_ROPE_SHIFT_INVESTIGATION.md`
on the branch):

- The splice mechanism is **applicable** to full-attention models
  (Llama-3.x, Qwen3 non-3.5) and to multi-axis-RoPE attention
  models (Qwen3 VL family).
- It is **applicable** to SWA models (Gemma-4 family), but
  llama.cpp's iswa cache layout sizes the SWA buffer smaller than
  the base buffer by default, blocking the shift gate at
  `llama-kv-cache-iswa.cpp:223`. Pass `--swa-full` at server launch
  to equalize the buffers; then K-shift forwards correctly. Cost is
  the SWA memory savings.
- It is **not applicable** to hybrid attention+recurrent models
  (Qwen3.5/3.6) — the recurrent state is contextual drift made
  explicit and unrecoverable, and the splice-correctness work
  reagent has done does not bound it.

## Running

All scripts use [PEP 723 inline dependencies](https://peps.python.org/pep-0723/)
executed via `uv run --script`. Install `uv` once:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Recurrence analysis

```bash
uv run --script trace_analysis/analyze_trace_reuse.py \
    --dataset agentcap \
    --agentcap-source 'hf://buckets/dacorvo/agentcap-traces/...' \
    --n-sessions 100 --min-match 128 \
    --output trace_analysis/results/<name>.json

uv run --script trace_analysis/categorize_matches.py \
    --source 'hf://buckets/dacorvo/agentcap-traces/...' \
    --min-match 128 \
    --output trace_analysis/results/<name>_categories.json
```

`categorize_matches.py` also emits
`<name>_categories.splice_candidates.jsonl` next to the JSON — the
splice-pair manifest the correctness harness consumes.

### End-to-end splice correctness

A patched llama-server with `feat/cache-reuse-symmetric` checked out
and built (`cd llama.cpp && cmake -B build && cmake --build build
--target llama-server`):

```bash
uv run --script trace_analysis/test_splice_against_manifest.py \
    --manifest trace_analysis/results/<name>_categories.splice_candidates.jsonl \
    --gguf /path/to/model.gguf \
    --top 5
```

For each manifest pair: posts donor + recipient to the server,
captures the splice events from server stderr (`scheduled splice`
and `reusing chunk` lines), reports per-pair size scheduled, size
applied, position shift, and `cache_n` attribution.

Optional flags for specific architectures:

- `--tensor-split 1` — pin to a single GPU (default uses pipeline
  parallelism across all visible GPUs; small models like
  Gemma-4-E4B trip ggml's `GGML_SCHED_MAX_SPLIT_INPUTS` limit on
  multi-GPU split).
- `--swa-full` — required for SWA models (Gemma-4 family) to bypass
  the iswa size-mismatch shift gate.

A foundational pre-flight test on a tiny model:

```bash
uv run --script trace_analysis/test_cache_reuse_smoke.py
```

Spins up llama-server on a free port with a Llama-3.2-1B GGUF and
verifies the splice mechanism fires on a tailored donor/recipient
pair. Pre-requisite: a Llama-3.2-1B GGUF at the path baked into
the script (see the script's header comments).

## Contributing

```bash
bash scripts/setup.sh
```

Installs a pre-commit hook that runs `uvx ruff format` +
`uvx ruff check --fix` on staged Python files.

## Data

Agent traces captured via `../agentcap` and stored at
`hf://buckets/dacorvo/agentcap-traces/`. Hermes / goose / opencode
/ pi runs against `gemma-4-E4B-it` and `qwen3.6-35b-a3b`. Used by
`trace_analysis/` for substrate composition and recurrence
analysis, and as the source of (donor, recipient) pairs for the
splice-correctness harness.

## Relation to CacheSlide

[CacheSlide (USENIX FAST '26)](https://www.usenix.org/system/files/fast26-liu-yang.pdf)
introduces the phase-drift bound (§3.3), the layer-wise
amplification result (§5.3), and a production reuse policy based on
boundary-token recompute (§1.4). Reagent's prior splice-correctness
measurements borrowed the core idea (KL on next-token distribution
as a function of Δ) but used a full-conversation in-context
baseline rather than chunk-alone, and shifted the prompt by
duplicating the system prompt's own content (semantically
redundant) so any divergence was attributable to position rather
than to changed downstream task expectations. The current
agentcap-driven pipeline keeps the in-context principle and
extends it to multi-segment splices on real workload data.
