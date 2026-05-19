# kv-reuse-lab — KV cache reuse correctness for agent workloads

Measures whether server-side KV-cache reuse preserves task behaviour
when a cached chunk appears at a different absolute position in a new
request. Pipeline:

1. **Recurrence analysis** ([`trace_analysis/categorize_matches.py`](trace_analysis/categorize_matches.py))
   over an agentcap-captured corpus. Emits a per-bucket category
   breakdown + a `(donor, recipient, tok_range)` splice-candidate
   manifest keyed by `(tool_name, args_hash)`.
2. **Splice-correctness measurement** ([`trace_analysis/test_splice_against_manifest.py`](trace_analysis/test_splice_against_manifest.py))
   end-to-end through patched llama-server. Per pair: warm donor → run
   recipient with cache-reuse, run a cold recipient in a fresh server,
   compare first-token KL + top-1 + 64-token continuation similarity.
3. **Mechanism research instrument** at `dacorvo/llama.cpp` branch
   `feat/cache-reuse-symmetric` — symmetric `--cache-reuse` (multi-segment
   splice) + K-shift for IM-RoPE.

Orientation note for current state and next steps: [SCOPE.md](SCOPE.md).

## Concepts

### Common Prefix (CP)

For two requests `A` and `B`, the **CP** is the byte-stable token
region they share starting at position 0. CP is what a standard
prefix cache serves; at byte `CP` the bytes diverge and from there on
it's fresh prefill.

### Post-prefix matches

A **post-prefix match** is any byte-exact run of ≥ N tokens between
two requests starting past either side's CP. Cannot be served by a
prefix cache: at least one of the two requests already diverged before
the match begins. Comes from recurring tool responses (file reads,
directory listings, command outputs) that the same user/team produces
across sessions.

llama.cpp's `--cache-reuse` is the only production stack today that
serves these via byte-search-and-rephase; the patched fork extends it
to multi-segment composition and to additional architectures.

## Splice mechanism

When chunk X appears in donor at position p_A and recipient at
position p_B ≠ p_A, take donor's K/V cells for X out of cache, RoPE-
rephase K by the position delta, paste into recipient's prefill at
p_B, continue.

## Headline result

End-to-end measurement across **16 (agent, model, session) corpora**:
four agents (goose, hermes, opencode, pi), three splice-applicable
model families, two task substrates (`tc` = transformers-coding-session,
`hh` = hf-hub-session). Per-model aggregates:

| model | corpora | N pairs | agree | mean KL | mean sim | catastrophic |
|---|---|---|---|---|---|---|
| Gemma-4-E4B (SWA + attention) | 8 (4 agents × {tc, hh}) | 574 | 84.5% | 0.16 | 0.93 | 19 (3.3%) |
| Qwen3-Coder-30B (full attention) | 4 (4 agents × tc) | 396 | 84.8% | 0.36 | 0.93 | 8 (2.0%) |
| Gemma-4-26B (SWA + attention) | 4 (4 agents × tc) | _retry running_ | — | — | — | — |
| Qwen3.6-35B (hybrid, off-table) | 1 (goose × tc) | 88 | 75.0% | 4.40 | 0.91 | **8 (9%)** |

Per-corpus breakdown + drift-bucket regression: [/tmp/cm_verify/sweep_summary_full.md](file:///tmp/cm_verify/sweep_summary_full.md) (auto-generated when the 26B retry completes).

Attention-only architectures splice cleanly across all four agents:
agree-rate spread on Gemma-4-E4B is 67–91% (hh_opencode lowest at N=6,
others 73–91%); Qwen3-Coder-30B 82–90%. Splice quality does **not**
degrade with drift magnitude — |drift| < 100 and |drift| ≥ 50k token
buckets both clean. Hybrid attention+recurrent (Qwen3.5/3.6) remains
the only architecture class producing the catastrophic orphan-token
signatures (`</`, `<|im_end|>`): K-shift rephases attention K/V but
cannot rewrite the gated-delta-net layer's compressed state. Detail
at [trace_analysis/results/agentcap_goose_splice_postmortem.md](trace_analysis/results/agentcap_goose_splice_postmortem.md).

Median prefill speedup on the original goose × Gemma-4-26B sweep:
**2.0×** wall-clock, p75 3.3× (cross-agent timings pending the 26B retry).

## Architecture applicability

| class | example | works | flags |
|---|---|---|---|
| full attention | Llama-3.x, Qwen3 (non-3.5) | yes | — |
| multi-axis-RoPE | Qwen3-VL | yes | — |
| SWA + attention | Gemma-4 family | yes | `--swa-full` |
| hybrid attention+recurrent | Qwen3.5/3.6 | **no** | — |

The `--swa-full` requirement bypasses llama.cpp's iswa size-mismatch
shift gate at `llama-kv-cache-iswa.cpp:223`. Cost: the SWA memory
savings are given back. Hybrid models are not fixable at the K-shift
layer.

## Running

Install `uv` once:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Recurrence analysis

```bash
uv run --script trace_analysis/categorize_matches.py \
    --source 'hf://buckets/dacorvo/agentcap-traces/<corpus>.parquet' \
    --min-match 128 \
    --output trace_analysis/results/<name>_match_categories.json
```

Emits `<output>.splice_candidates.jsonl` next to the JSON.

### Splice-correctness end-to-end

Patched llama-server with `feat/cache-reuse-symmetric` checked out and
built (`cd llama.cpp && cmake -B build && cmake --build build --target
llama-server`):

```bash
uv run --script trace_analysis/test_splice_against_manifest.py \
    --manifest trace_analysis/results/<name>.splice_candidates.jsonl \
    --gguf /path/to/model.gguf \
    --top 99 \
    --output trace_analysis/results/<name>_splice_metrics.json
```

For SWA models add `--swa-full`. For small models on a multi-GPU host
add `--tensor-split 1` (single GPU).

Smoke test on a Llama-3.2-1B GGUF:

```bash
uv run --script trace_analysis/test_cache_reuse_smoke.py
```

## Contributing

```bash
bash scripts/setup.sh
```

Installs a pre-commit hook that runs `uvx ruff format` +
`uvx ruff check --fix` on staged Python files.

## Data

Agent traces captured via `../agentcap` and stored at
`hf://buckets/dacorvo/agentcap-traces/`. The recurrence analysis and
splice-correctness harness both stream from these parquets.
