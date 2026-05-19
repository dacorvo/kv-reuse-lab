# trace_analysis

Two scripts driving the recurrence + splice-correctness pipeline against
agentcap-captured corpora. Terminology (CP, post-prefix matches) lives
in [../README.md](../README.md).

## `categorize_matches.py`

Scans an agentcap parquet for byte-exact cross-session matches past CP,
tags each match by role + content sub-type, aggregates by `(tool_name,
args_hash)`, and emits:

- a category breakdown JSON (CP volume, post-prefix volume, per-tool
  decomposition);
- a splice-candidate manifest `<output>.splice_candidates.jsonl`
  consumed by the correctness harness — one (donor, recipient,
  tok_range) entry per recurring tool-response chunk.

```bash
uv run --script trace_analysis/categorize_matches.py \
    --source 'hf://buckets/dacorvo/agentcap-traces/<corpus>.parquet' \
    --min-match 4096 \
    --output trace_analysis/results/<name>_match_categories.json
```

`--min-match` is in **characters** (≈ ¼ of a token for English). 4096 chars ≈
a thousand tokens — below that, prefill savings are dwarfed by lookup
overhead. The script is fork-Pool parallel across `os.cpu_count()` and
scales linearly with `--workers`; on a 192-vCPU host a typical
corpus runs in seconds.

## `test_splice_against_manifest.py`

Drives a patched llama-server end-to-end against the manifest. Per
pair: posts donor (warm-up) then recipient through the cache-reuse
engine, runs a cold recipient in a fresh server for comparison,
captures splice events from server stderr, and reports first-token KL
+ top-1 agreement + 64-token continuation cosine similarity.

```bash
uv run --script trace_analysis/test_splice_against_manifest.py \
    --manifest trace_analysis/results/<name>_match_categories.splice_candidates.jsonl \
    --gguf /path/to/model.gguf \
    --top 99 \
    --output trace_analysis/results/<name>_splice_metrics.json
```

Architecture-specific flags:
- `--tensor-split 1` — pin to a single GPU (default uses pipeline
  parallelism; small models on a multi-GPU host can trip ggml's
  `GGML_SCHED_MAX_SPLIT_INPUTS` limit).
- `--swa-full` — required for SWA models (Gemma-4 family) to bypass
  the iswa size-mismatch shift gate at server-context.cpp.

## `test_cache_reuse_smoke.py`

Pre-flight smoke test on a tailored donor/recipient pair against a
small Llama-3.2-1B GGUF. Verifies the splice mechanism fires; useful
sanity check before running the full harness.

## Recurrence sweep — 2026-05-19

Categorization ran against the full
`hf://buckets/dacorvo/agentcap-traces/` snapshot (22 parquets across two
session corpora, four agents, six model classes) with
`--min-match 4096` chars (~1024 tokens). Columns are derived from
`<corpus>_match_categories.json`; **manifest pairs** is the row count of
the sibling `splice_candidates.jsonl`.

- **match%** — fraction of requests that have at least one post-CP
  byte-stable run ≥ 4096 chars against any prior session.
- **CP%** — share of input bytes that a vanilla prefix cache would
  serve (common-prefix coverage).
- **post-input%** — share of input bytes recurring past CP — the
  splice-cache target. Bigger = more headroom over a prefix cache.
- **manifest pairs** — splice candidates emitted (top 20 buckets × 5
  largest pairs per bucket, max 100). Drives the downstream
  splice-correctness harness.

### Gemma-4-E4B-it (SWA)

| corpus | sessions | requests | match% | CP% | post-input% | manifest pairs |
|---|---:|---:|---:|---:|---:|---:|
| hh/goose | 102 | 324 | 0.0% | 69.7% | 0.00% | 1 |
| hh/hermes | 31 | 297 | 56.2% | 44.8% | 33.59% | 85 |
| hh/opencode | 61 | 377 | 0.8% | 87.9% | 0.37% | 6 |
| hh/pi | 30 | 275 | 53.8% | 34.1% | 39.68% | 74 |
| tc/goose | 68 | 310 | 75.8% | 21.8% | 48.13% | 92 |
| tc/hermes | 27 | 299 | 7.7% | 72.2% | 4.24% | 20 |
| tc/opencode | 58 | 310 | 12.9% | 39.6% | 12.72% | 60 |
| tc/pi | 38 | 304 | 18.4% | 31.0% | 14.55% | 54 |

### Gemma-4-26B-A4B-it (SWA)

| corpus | sessions | requests | match% | CP% | post-input% | manifest pairs |
|---|---:|---:|---:|---:|---:|---:|
| tc/goose | 169 | 1,139 | 52.4% | 15.2% | 33.53% | 100 |
| tc/hermes | 127 | 1,592 | 33.0% | 38.8% | 21.80% | 100 |
| tc/opencode | 32 | 562 | 36.5% | 22.5% | 21.51% | 97 |
| tc/pi | 58 | 567 | 42.2% | 20.8% | 26.57% | 98 |

### Qwen3-Coder-30B-A3B-Instruct (full attention)

| corpus | sessions | requests | match% | CP% | post-input% | manifest pairs |
|---|---:|---:|---:|---:|---:|---:|
| tc/goose | 333 | 1,379 | 58.4% | 11.6% | 64.69% | 100 |
| tc/hermes | 122 | 1,838 | 57.8% | 35.5% | 22.94% | 100 |
| tc/opencode | 44 | 1,134 | 33.6% | 28.8% | 16.26% | 100 |
| tc/pi | 51 | 1,004 | 48.8% | 33.0% | 20.13% | 100 |

### GLM-4.5-Air (dense GLM4MoE)

| corpus | sessions | requests | match% | CP% | post-input% | manifest pairs |
|---|---:|---:|---:|---:|---:|---:|
| tc/hermes | 70 | 2,061 | 35.6% | 32.2% | 9.30% | 100 |

### Qwen3.6-35B-A3B (hybrid attention + gated DeltaNet)

Captured for reference only — splice-correctness is known-broken on this
class (recurrent-state bleed per the goose × Qwen3.6 postmortem).

| corpus | sessions | requests | match% | CP% | post-input% | manifest pairs |
|---|---:|---:|---:|---:|---:|---:|
| tc/goose | 101 | 785 | 40.4% | 16.4% | 24.91% | 97 |
| tc/hermes | 36 | 1,006 | 46.3% | 29.7% | 21.33% | 100 |
| tc/opencode | 72 | 1,062 | 30.8% | 16.5% | 13.09% | 97 |
| tc/pi | 52 | 1,426 | 45.0% | 18.5% | 16.17% | 100 |
| tc/hermes (via hf-router) | 30 | 924 | 36.5% | 29.8% | 6.78% | 98 |

### Patterns

- **Substrate effect** holds across all attention-only models: `goose`
  and `hermes` are strong reuse substrates (post-input% 22-65%);
  `opencode` is structurally weak (every request re-randomizes the
  request preamble — post-input% 16-29% on `tc`, near-zero on `hh`);
  `pi` is middling.
- **Session corpus** matters as much as agent: `hh-session` (hf-hub
  exercises) inverts the picture for `goose` — only 0% match — because
  the corpus's task seeds barely overlap. `tc-session`
  (transformers-coding) drives much more cross-session recurrence.
- **CP vs post-input trade**: a high CP% almost always pairs with a
  low post-input%. CP captures the agent's stable system+tools head;
  what's left to splice past CP is what differs in the head and
  recurs only in tool outputs.
- **Architecture** is orthogonal to recurrence — the numbers above are
  function of (agent, corpus, model-tokenizer), not attention class.
  Splice-correctness (next stage) is where architecture starts to
  matter (SWA, hybrid recurrent, etc.).
