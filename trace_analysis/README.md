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
    --min-match 128 \
    --output trace_analysis/results/<name>_match_categories.json
```

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

## Caveats

- The 128-token `--min-match` is a measurement floor. For a production
  cache, the meaningful admission threshold is closer to 1k tokens —
  below that, the prefill saving is dwarfed by lookup overhead.
- `--max-tokens-per-session` (default 20000) caps the indexing window;
  byte-stable regions extending past that are missed by the manifest
  but found by the engine at runtime.
- `split_match_by_role` slices matches at every role boundary; the
  intent is to exclude model-generated content, but the current
  implementation also splits at user/tool boundaries inside
  non-model-generated runs. The engine ignores the manifest at runtime
  and matches whatever's actually byte-stable, so this is a
  manifest-bookkeeping bug rather than a correctness one.
