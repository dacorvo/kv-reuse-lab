# trace_analysis — KV-cache-reuse opportunity in real agent traces

Reagent's main harness (`../measure_reuse_drift.py`) measures the
*correctness* of naive vs shifted KV reuse *given* that a cache hit
has occurred. This subdirectory measures a complementary,
data-analytical question: **how often do cache hits actually occur
in real agent workloads, and are they single contiguous spans or
multi-segment?**

## Why this matters

A serving engine can only benefit from cache reuse if the new
prompt's tokens actually match something already in the cache.
llama.cpp's `--cache-reuse N` finds a *single* contiguous ≥N-token
run of byte-exact matches starting from where the prefix diverges,
re-rotates its RoPE phases (the "shifted" reuse reagent measures),
and stops. It cannot stitch together multiple cached fragments
separated by new content.

The two numbers this analysis computes per request:

- `coverage_frac` — fraction of the request's tokens contained in
  *any* ≥N-token byte-exact match against earlier requests. Upper
  bound on what any reuse engine could ever save.
- `longest_frac` — largest single contiguous match as a fraction of
  the request. This is roughly what llama.cpp's current algorithm
  can actually exploit.

Their difference — the **gap** — is the upside a hypothetical
gap-stitching engine could capture that llama.cpp cannot today.

## Method

For each target request in a streamed dataset, we:

1. Tokenize it via a neutral tokenizer (default Llama-3.2-1B).
2. Scan the target left-to-right, looking up each N-token window in a
   hash-indexed pool of windows from *earlier* requests (from other
   sessions, so we don't count the trivial prefix overlap that a
   continuing conversation produces).
3. On hit, extend greedily to find the longest match; on miss,
   advance by one token. Non-overlapping matches.
4. Separately, count intra-request duplicates: same ≥N-token span
   appearing twice within a single prompt (the "same file read twice
   in one session" case).

Algorithm is `O(total_tokens)` with hash-based indexing;
`--max-tokens-per-session` caps the per-session memory footprint.
See `analyze_trace_reuse.py` for full details.

## Datasets

| key | source | what it represents |
|---|---|---|
| `hermes` | `NousResearch/hermes-function-calling-v1` `func_calling` | synthetic function-calling traces, different tool definitions per example |
| `swe-smith` | `SWE-bench/SWE-smith-trajectories` (tool split) | one-shot SWE-agent runs on SWE-bench tasks |
| `claude-hs` | `archit11/claude_code_traces_hs` | real Claude Code sessions on the `juspay/hyperswitch` repo |

`swe-smith` supports a `--repo-filter PREFIX` flag that restricts
the session pool to a single repository (e.g. `django` →
`django__django-*`), which approximates "many bug-fix sessions on
the same codebase" — as close as public data gets to "one developer
across many sessions on one repo".

## Running

```bash
# Hermes, 100 sessions, min-match N=32 (low threshold picks up short shared boilerplate)
./analyze_trace_reuse.py --dataset hermes --n-sessions 100 --min-match 32 \
    --output results/hermes_N100_min32.json

# SWE-smith, same N=128 threshold llama.cpp uses, no repo filter
./analyze_trace_reuse.py --dataset swe-smith --n-sessions 30 --min-match 128 \
    --output results/swe_ungrouped_N30_min128.json

# SWE-smith, django-only — same-repo cross-session
./analyze_trace_reuse.py --dataset swe-smith --repo-filter django \
    --n-sessions 20 --min-match 128 \
    --output results/swe_django_N20_min128.json

# Claude Code on hyperswitch
./analyze_trace_reuse.py --dataset claude-hs --n-sessions 10 --min-match 128 \
    --output results/claude_hyperswitch_N10_min128.json
```

Numbers are tokenizer-dependent; pick one representative of your
deployment model. Default is `meta-llama/Llama-3.2-1B-Instruct`.

## Results

All runs below use `--min-match 128` (llama.cpp's typical default)
except the first Hermes row, which uses `min-match 32` to show what
can be found at a lower threshold.

| dataset | config | coverage mean | coverage p90 | longest mean | fragments mean | ≥2 fragments | intra-dup |
|---|---|---|---|---|---|---|---|
| `hermes` | N=100, min=32 | 0.15 | 0.22 | 0.08 | 2.5 | 100% | — |
| `hermes` | N=100, min=128 | **0.00** | 0.00 | 0.00 | 0.0 | 0% | — |
| `swe-smith` ungrouped | N=30 | 0.06 | 0.15 | 0.05 | 1.2 | 11% | 27% |
| `swe-smith` django-only | N=20 | **0.29** | 0.77 | 0.15 | 3.5 | 73% | 41% |
| `claude-hs` (hyperswitch) | N=10 | **0.28** | 0.33 | 0.16 | 4.0 | 90% | 45% |

### What these numbers say

1. **Without repo-scoping, cross-session reuse is modest.** On
   synthetic Hermes (different tool defs per example) cross-session
   matches at llama.cpp's production threshold are ≈ 0. On SWE-smith
   across random bug-fix tasks, coverage is ≈ 6% — mostly the
   SWE-agent boilerplate shared across all tasks.
2. **With repo-scoping, reuse is substantial and genuinely
   multi-segment.** Both django-filtered SWE-smith and Claude Code
   hyperswitch sessions show coverage ≈ 29% with ≈ 4 distinct match
   fragments per request on average. More than 70% of requests have
   multi-segment matches that llama.cpp's `--cache-reuse` could not
   assemble.
3. **The gap matters.** `coverage − longest ≈ 13–14%` on
   repo-grouped workloads — that fraction of every prompt sits in
   *interior* matched spans that today's prefix-plus-one-slide
   engines cannot exploit.

### Where the matches come from

A short look at Hermes matches (`--min-match 32`): the two recurring
spans are (1) the Llama-3 chat template header + Hermes
system-prompt intro (~92 tokens) and (2) the standard tool-call
instruction block + `<|eot_id|>` → user header boundary (~63
tokens). Both are framework boilerplate. Below 128 tokens
individually → invisible to llama.cpp at its default threshold.

On repo-grouped traces, the matches are larger and content-driven:
identical chunks of `models/base.py`, common import blocks, shared
utility functions that recur across different bug-fix attempts on
the same codebase.

## Caveats

- **Numbers are tokenizer-dependent.** We use Llama-3.2-1B as a
  neutral reference; switching to a different tokenizer will shift
  the absolute numbers (but not the cross-dataset relative
  ordering).
- **Cross-session only.** Intra-session prefix reuse (consecutive
  requests in one session — the trivial case llama.cpp handles
  every time) is excluded. Intra-request duplicates are reported
  separately.
- **Token cap.** Very long trajectories are truncated to
  `--max-tokens-per-session` (default 20k). SWE-smith tasks can run
  longer than this; we effectively analyze the first 20k tokens of
  each session.
- **Repo-grouping is a proxy.** Multiple bug-fix attempts on the
  same repo ≠ "one developer over many sessions", but it's the
  closest proxy public data offers. The Claude Code hyperswitch
  dataset is closer to the target workload, and its numbers match
  the django-filtered SWE-smith ones — we take that as evidence the
  proxy is reasonable.

## Relation to reagent's main harness

Reagent measures the **correctness cost** of cache reuse
(`sim(fresh, reused)` falling below 1.0 when the cached chunk sits
at a drifted position). This subdirectory measures the
**availability of reuse opportunities** in the first place. Both
numbers are needed to judge whether a gap-stitching cache engine is
worth building.

The numbers above show that **real same-codebase agent workloads
produce substantial, genuinely fragmented reuse opportunities that
llama.cpp's current algorithm cannot exploit**. On django-filtered
SWE-smith and Claude Code on hyperswitch, ~29% of every prompt is
covered by byte-exact ≥128-token matches against earlier sessions,
split across ~4 disjoint fragments, with ~14% of target tokens
sitting in interior matches that the prefix-plus-one-slide scan
structurally cannot reach (see the llama.cpp algorithm trace in the
main README). The pessimistic framing ("if coverage were
concentrated in a single contiguous match, llama.cpp would handle
it") only applies to random-task or single-example workloads like
ungrouped SWE-smith or Hermes, where cache opportunity is near
zero anyway.

The upshot: on realistic workloads the coverage/gap question is
settled — multi-segment reuse *would* pay. The remaining open
question is the one reagent's main harness addresses: when an
engine implements gap-stitching (splicing multiple cached segments
into a new prompt with re-prefilled gaps between them), what is
the *correctness* penalty at each splice, and how do the per-chunk
errors compose? That is the load-bearing question for whether such
an engine can ship.
