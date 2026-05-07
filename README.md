# reagent — KV cache reuse correctness for agent workloads

Reagent measures whether server-side KV-cache reuse preserves task
behaviour when a cached block appears at a drifted absolute position
in a new request, and analyses how often such reusable chunks appear
in real agent traces. It does **not** ship a runtime cache; it
characterises the splice mechanism and the substrate, with an eye to
what a runtime cache would need to handle.

For the orientation note (what reagent owns vs. sibling repos, what's
known, what's next), read [SCOPE.md](SCOPE.md). This README documents
the published measurement and the current toolset.

## What reagent does, in three pieces

1. **Splice-correctness measurement.** Given a cached chunk, does
   reusing it at a new position produce the same model output as a
   fresh forward? Two harnesses: `measure_reuse_drift.py` (single
   chunk, controlled drift) and `measure_multi_splice_b.py`
   (multi-segment splice on real agent traces). Both compare *naive*
   reuse (splice cached K/V at original RoPE phases) against
   *shifted* reuse (re-rotate K to the new positions, matching
   llama.cpp's `llama_memory_seq_add`).
2. **Recurrence analysis** in [`trace_analysis/`](trace_analysis/).
   For real agent corpora (captured by `../agentcap`), how much of a
   request can be byte-matched against earlier requests, what
   fraction is post-prefix, and what request-semantic structure
   does the recurring content share?
3. **Mechanism-applicability research instrument** at
   `dacorvo/llama.cpp` branch `feat/cache-reuse-symmetric`. Two
   patches that extend `--cache-reuse` (multi-segment splice;
   K-shift for text-only IM-RoPE / M-RoPE) so we can test which
   architectures the splice mechanism is *applicable* to. See
   `tools/server/notes/IM_ROPE_SHIFT_INVESTIGATION.md` on the
   branch for the findings.

## Concepts

Two terms do most of the work. Defined precisely so the rest of
the docs stop drifting.

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
  `MEMORY (your personal notes)` block between the system prompt and
  the tools schema. The system-prompt bytes are identical across two
  sessions, but as soon as memory differs CP ends — even though
  ~10k bytes of byte-identical tools-schema sit right after, those
  bytes are now post-prefix on every cross-session pair.
- **Recurring content past the first turn.** Tool responses (file
  reads, web fetches, command outputs) that recur when the same
  user/team repeats similar work.

Both sources need a cache that can match against the *content* of a
chunk, not just the offset. llama.cpp's `--cache-reuse` is the only
production stack that does this today, with two limitations: it
historically could only stitch one contiguous post-prefix run (the
fork patch removes that), and it identifies hits by sliding
byte-search rather than by chunk identity.

## Splice-correctness — the published result

### Single-chunk, system-duplicate drift

Headline measurement on Hermes function-calling traces, L=128 tokens,
N=20 examples per cell, embedder `bge-small-en-v1.5`.
`sim(fresh, reused)` = cosine similarity of greedy continuations.
`naive → shifted`:

| model | Δ=0 | Δ=100 | Δ=500 | Δ=1000 |
|---|---|---|---|---|
| `gemma-4-E4B` | 1.00→1.00 | 0.96→1.00 | 0.93→0.99 | 0.92→0.99 |
| `gemma-4-31B` | 1.00→1.00 | 0.99→0.99 | 0.95→0.98 | 0.91→0.98 |
| `gemma-4-26B-A4B` | 0.99→0.99 | 0.98→0.98 | 0.86→0.98 | 0.85→0.99 |
| `Llama-3.1-8B` | 0.98→0.98 | 0.86→0.94 | 0.79→0.97 | 0.81→0.96 |
| `gemma-4-E2B` | 0.99→0.99 | 0.96→0.98 | 0.78→0.98 | 0.60→**0.99** |
| `Llama-3.2-1B` | 0.96→0.96 | 0.79→0.95 | 0.73→0.97 | 0.68→**0.95** |

Shifted reuse lifts every model to ≥ 0.94 sim across the drift
range. The phase error is essentially the whole story for clean,
structural-edge chunks.

![reagent panel](reagent_panel.png)

### Multi-segment shifted reuse on real agent traces

`measure_multi_splice_b.py` (Scheme B — chunked prefill with cache
injection, ~17× faster than Scheme A on hybrid models) splices
multiple disjoint matched chunks into a single prompt with shifted
RoPE. Same workload as the analysis side (SWE-smith / Nemotron
trajectories grouped by repo).

Llama-3.1-8B head-to-head on django, N=20, 190 ordered pairs:

| metric | Llama-3.2-1B | Gemma-4 E4B | Llama-3.1-8B |
|---|---|---|---|
| sim mean | 0.974 | 0.981 | 0.975 |
| **bit-exact** | 29% | 34% | **46%** |
| sim < 0.95 | 20% | 15% | **9%** |
| sim < 0.80 | 4% | **1%** | 5% |
| **min sim** | 0.71 | **0.755** | **0.437** |

Hybrid Gemma-4 has the higher floor (0.755), dense Llama-8B the
lower floor (0.437) but the best typical case (46% bit-exact).
Failures cluster at splices landing 6-10k tokens into short B
trajectories — short enough that the suffix can't recover from the
cache-conditioning mismatch, long enough that pure RoPE phase isn't
the cause.

### Trust caveat

Most of the published numbers above predate two matcher fixes
(`02bad94` input-only role filter; `a3ce1dc` skip first-turn
matches). Pre-fix runs let assistant-token spans count as splice
candidates and treated first-turn matches as splices instead of
prefix-cache hits — both contaminate the splice-correctness signal.
Treat any pre-fix figure as **indicative, not established**. The
honest framing: we have seen the mechanism work and we have seen it
fail, but we don't yet have a clean enough harness to report
failure rates with confidence.

[SCOPE.md](SCOPE.md) has the full list of unbounded cases (interior-
span chunks, accumulated multi-turn reuse, hybrid-architecture
splicing) that the published result does **not** cover.

### Predicting splice safety at write time

On the 56-pair Gemma-4 / Nemotron-pandas-dev run, three candidate
predictors were Pearson-correlated against measured `sim_fresh_reused`:

| predictor | r vs sim |
|---|---|
| attention locality (W=256) | +0.04 (none) |
| attention entropy at chunk (B) | **+0.36** |
| K/V perturbation cosine (A) | -0.26 |

Cheap entropy proxy ≈ expensive K/V perturbation; both top out at
r ≈ ±0.30. **Byte-level admission heuristics topped out** —
useful as a soft ranking, **not as a sole gate**. For agent
workloads, request-semantic admission (`tool_response:tool_name+
args_hash`) won out as the load-bearing rule. See
[SCOPE.md](SCOPE.md) for the corpus-derived admission policy.

## Recurrence analysis

[`trace_analysis/`](trace_analysis/) ingests agentcap-captured
corpora and reports:

- Per-request **CP** vs **post-prefix** coverage
  (`analyze_trace_reuse.py`).
- Match-category breakdown — what *kind* of bytes recur (Hermes
  system prefix, tools schema, memory section, tool response, user
  content) — `categorize_matches.py`.
- A `(tool_name, args_hash)` drilldown for the `tool_response`
  category, plus an emitted **splice-candidate manifest** that the
  `test_splice_against_manifest.py` harness feeds into a llama-server
  for end-to-end verification.

Headline findings (qwen3.6-35b-a3b captures, May 2026): goose's
post-prefix tool_response bytes are 74% `tree`, 15% `shell`, 10%
`analyze` — top single bucket `tree({"path":".","depth":2})` →
690k tokens × 62 hits. Opencode is 65% `read`, 20% `grep`, 15%
`glob`, concentrated on a handful of hot files. Both confirm
that request-semantic admission keyed on `(tool_name, args_hash)`
captures the bulk of non-CP recurrence — no byte-level
heuristic needed.

See [`trace_analysis/README.md`](trace_analysis/README.md) for
detailed methodology.

## Mechanism applicability — llama.cpp fork

Branch `feat/cache-reuse-symmetric` on `dacorvo/llama.cpp` carries
two patches used as a research instrument:

1. **Symmetric `--cache-reuse`.** Walks both `head_c` (cache index)
   and `head_p` (recipient index) on miss, snapshots source ranges
   into a temp seq, applies splices just-in-time during the prefill
   batch loop. Fixes the "single contiguous run" limitation of the
   upstream algorithm.
2. **K-shift for text-only IM-RoPE / M-RoPE.** Forward IMROPE writes
   per-token positions `(t,t,t,0)` for text inputs; the matching
   K-shift writes `(δ,δ,δ,0)` and applies it via `ggml_rope_multi`.
   Mechanically tested at fp32 with `rel_err ~ 0`.

Findings (see `tools/server/notes/IM_ROPE_SHIFT_INVESTIGATION.md`
on the branch):

- The splice mechanism is **applicable** to multi-axis-RoPE
  attention models (Qwen3 VL family, Qwen3 non-3.5).
- It is **not applicable** to hybrid attention+recurrent models
  (Qwen3.5/3.6) — the recurrent state is contextual drift made
  explicit and unrecoverable, and the published splice-correctness
  result does not bound it.

## Running

All scripts use [PEP 723 inline dependencies](https://peps.python.org/pep-0723/)
executed via `uv run --script`. Install `uv` once:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Single-chunk drift sweep:

```bash
./run.sh                                        # full panel
MODELS="meta-llama/Llama-3.2-1B-Instruct" ./run.sh   # one model
```

Multi-segment splice on real traces:

```bash
uv run --script measure_multi_splice_b.py \
    --model google/gemma-4-E4B-it \
    --repo django --n-sessions 10 --min-match 128 \
    --output results/multi_splice_b_django_gemma4.json
```

Trace recurrence analysis:

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

End-to-end against a llama-server with the patched cache-reuse:

```bash
uv run --script trace_analysis/test_splice_against_manifest.py \
    --manifest trace_analysis/results/<name>_categories.splice_candidates.jsonl \
    --gguf /path/to/model.gguf \
    --top 5
```

Tests:

```bash
./test_drift_modes.py        # 59 parametrized tests on drift_modes
./test_kv_cache.py           # 5 tests on write_kv_span routing
./test_multi_splice_b.py --model meta-llama/Llama-3.2-1B-Instruct \
    --device-map cuda:0
uv run --script trace_analysis/test_cache_reuse_smoke.py
```

## Contributing

```bash
bash scripts/setup.sh
```

Installs a pre-commit hook that runs `uvx ruff format` +
`uvx ruff check --fix` on staged Python files.

## Data

Two corpora drive the work:

- `NousResearch/hermes-function-calling-v1` (`func_calling`) —
  used for the original single-chunk drift measurements. Multi-turn
  agent traces with system prompt + tools schema + tool calls.
- **agentcap captures** at `hf://buckets/dacorvo/agentcap-traces/`
  — captured via the `../agentcap` proxy. Real agent runs (Hermes,
  goose, opencode, pi) against `gemma-4-E4B-it` and
  `qwen3.6-35b-a3b`. Used by `trace_analysis/` for substrate
  composition and recurrence analysis.

## Relation to CacheSlide

[CacheSlide (USENIX FAST '26)](https://www.usenix.org/system/files/fast26-liu-yang.pdf)
introduces the phase-drift bound (§3.3), the layer-wise amplification
result (§5.3), and a production reuse policy based on boundary-token
recompute (§1.4). We borrow the core measurement idea (KL on the
next-token distribution as a function of Δ) but make two corrections
for agent-era serving:

1. **In-context baseline.** CacheSlide extracts the cached chunk
   from a prefill of the chunk alone. That conflates position drift
   with "chunk-alone vs chunk-in-context" drift. We baseline from
   the full agent conversation.
2. **Semantically valid drift.** We shift only by duplicating the
   system prompt's own content — redundant by construction — so any
   divergence is attributable to position rather than to changed
   downstream task expectations.
