# reagent — tool-result KV cache reuse sensitivity

**reagent** measures whether a language model's next-token
distribution changes when a serving stack reuses a previously computed
KV tensor for a tool result — the text the agent pastes in after a
tool call — at a position that has drifted because the surrounding
system prompt or earlier turns grew between the two requests.

## What we are testing

Agent servers would like to cache tool results (file contents,
retrieved documents, API responses) and splice the cached KV into
later requests where the same content recurs. The problem is
positional: the cached tensor has RoPE phases baked for the tokens'
original absolute positions, and the model might produce different
next-token predictions when those phases don't match the current
layout.

The experiment sets the chunk at two different positions in
otherwise-equivalent prompts, runs the forward *with* and *without*
reuse, and compares the resulting next-token distributions plus the
subsequent 64-token greedy continuations. If the reused variant is
indistinguishable from a fresh forward, naive cache reuse is safe for
that model at that drift. If it's not, the model needs a correction
scheme (boundary recompute, re-prefill, or skipping reuse past a
threshold).

## Scenario

An agent request typically looks like
`[system][user][assistant: tool_call][tool: tool_result]`, ending
right before the model decodes the next assistant turn. The **tool
result** is the largest, most cache-worthy chunk — hundreds to
thousands of tokens of agent-pasted content the server had to prefill
but never generated.

Across requests, the same tool result may appear at different absolute
positions because the system prompt gained a tool definition, or an
earlier turn was rewritten, or a prior summary was injected. The
serving stack wants to skip re-prefilling that chunk; the question is
whether doing so silently degrades the response.

## Protocol

1. Take a real agent trace from `NousResearch/hermes-function-calling-v1`
   (`func_calling` config). Render it through the model's chat
   template with `add_generation_prompt=True` so the prompt ends at
   the position the model is about to decode from.
2. Run a full in-context prefill of the baseline prompt and snapshot
   the last `L = 128` tokens of the tool-result turn's KV. That's the
   **baseline cached chunk**.
3. Build a **drifted prompt** by appending verbatim copies of the
   system prompt's own content until the rendered length has grown
   by ≥ Δ tokens. Same instructions, same tools — semantically
   redundant, so the model's expected output is unchanged and any
   divergence is attributable to position.
4. For each Δ ∈ {0, 50, 100, 200, 500, 1000}, run two forwards:
   - **Fresh**: prefill the drifted prompt end-to-end, record the
     next-token distribution and greedy-decode 64 more tokens.
   - **Reused**: prefill the drifted prompt up to the chunk, overwrite
     the last `L` KV entries per layer in place with the baseline
     cached chunk, forward the remaining suffix with explicit
     `cache_position` so the trigger token's RoPE phase is correct,
     record the distribution and greedy-decode 64 more tokens.
5. Aggregate over N=20 examples.

## Metrics

| Metric | What it answers |
|---|---|
| `mean_kl ± stdev_kl` | KL(fresh ∥ reused) on the first generated token, in nats. |
| `agree_rate` | Fraction of examples where fresh and reused pick the same top-1 token. |
| `mean_sim_fresh_reused` | Cosine similarity of sentence-embedded (`bge-small-en-v1.5`) fresh and reused continuations. **Headline**: does reuse change what the agent says? |
| `mean_sim_reused_reference` | Cosine similarity of the reused continuation against the dataset's gold assistant reply. Tracks whether reuse degrades task quality, not just self-consistency. |
| `mean_fresh_entropy` | Entropy of the fresh distribution. Values near zero mean the trigger is near-deterministic; the KL column has limited dynamic range there. |
| `mean_actual_delta` | Achieved token drift. Δ targets are floors; inflation may overshoot. |

**KL scale** (nats, log-e): near zero = indistinguishable; ~1 =
sampling would frequently diverge; ≥ 5 = the distributions disagree
on almost everything.

**Cosine-similarity scale** (under `bge-small-en-v1.5`):
≥ 0.95 = same answer; 0.80–0.95 = same topic, rewording; 0.50–0.80 =
different content (possibly different action); < 0.50 = different
meaning.

## Reading a row

```
                          Δ=0         Δ=100       Δ=500       Δ=1000
llama-3.2-1b-instruct
  mean_kl ± stdev         0.00±0.01   0.12±0.05   0.31±0.12   0.48±0.18
  top-1 agree             1.00        0.95        0.85        0.75
  sim(fresh, reused)      1.00        0.98        0.92        0.88
  sim(reused, reference)  0.74        0.73        0.70        0.68
```

- **Δ=0** is the sanity floor. Non-zero KL here means a harness bug,
  not a model insight.
- **`sim(fresh, reused)`** is the headline. Above 0.9, naive reuse
  produces a semantically-equivalent response. Below 0.8, reuse
  rewrites the answer.
- **KL** flags first-token disagreement; **sim** tells you whether
  the disagreement persists or the two sides converge on similar
  content after a few tokens. Use both.
- **`sim(reused, reference)`** gives a quality floor: if it drops
  faster than `sim(fresh, reused)` as Δ grows, reuse is degrading
  task quality.
- **Large stdev matters.** `±6` on an `8` KL mean means the outcome
  is prompt-dependent and production quality would be unpredictable.

### Drift magnitudes

| Δ | Realistic analogue |
|---|---|
| 0 | identical prompts — sanity check |
| 100 | slightly longer system prompt |
| 500 | extra earlier exchange, or system prompt doubled |
| 1000 | conversation grew significantly, or system prompt tripled |

### Passing bar

A model is safe for naive cross-prompt cache reuse at a given drift
when mean KL < ~1 nat, stdev small relative to the mean, top-1 agree
≥ 0.9, and `sim(fresh, reused)` ≥ 0.95. Anything worse means the
serving stack needs a correction policy (boundary recompute,
re-prefill, or skip-reuse past a Δ threshold).

## Running

### Prerequisite: `uv`

All scripts here are [PEP 723 inline-dependency](https://peps.python.org/pep-0723/)
Python files executed via `uv run --script`, so `uv` is the only thing
you need to install once:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh     # Linux / macOS
# or https://docs.astral.sh/uv/getting-started/installation/ for Windows
```

Everything else (torch, transformers, sentence-transformers, …) is
pinned inline in each script's header and is pulled into a cached
per-script virtualenv by `uv` automatically on first invocation.

### Commands

```bash
./run.sh                             # full panel
MODELS="meta-llama/Llama-3.2-1B-Instruct" \
    ./run.sh                         # single model
N_EXAMPLES=10 ./run.sh               # faster, noisier
FORCE=1 ./run.sh                     # re-run models with existing JSON
```

The wrapper calls `measure_reuse_drift.py` via `uv run --script`, skips
models that already have a non-empty `results/<model>.json`
(unless `FORCE=1`), and ends with a per-drift comparison table. Each
model writes a JSON with both aggregated statistics and per-example
rows.

`visualize.py` renders a 4-panel figure
(`sim(fresh, reused)`, top-1 agreement, KL ± stdev, and
`sim(reused, reference)`) from whatever JSONs are present.

## Contributing

If you plan to push changes, install the pre-commit hook once per
clone:

```bash
bash scripts/setup.sh
```

The script double-checks `uv` is on your PATH, warms the `uvx ruff`
cache, and symlinks `scripts/pre-commit` into the repo's
`.git/hooks/`. Every commit thereafter runs `uvx ruff format` and
`uvx ruff check --fix` on staged Python files and re-stages the
result. No additional dependencies — the hook uses ephemeral
`uvx`-managed envs.

## Relation to CacheSlide

[CacheSlide (USENIX FAST '26)](https://www.usenix.org/system/files/fast26-liu-yang.pdf)
introduces the phase-drift bound (§3.3), the layer-wise amplification
result (§5.3), and a production reuse policy based on boundary-token
recompute (§1.4). We borrow the core measurement idea — KL on the
next-token distribution as a function of Δ — but make two corrections
for agent-era serving:

1. **In-context baseline.** CacheSlide extracts the cached chunk from
   a prefill of the chunk alone. That conflates position drift with
   "chunk-alone vs chunk-in-context" drift. We baseline from a full
   in-context prefill of the real agent conversation.
2. **Semantically valid drift.** Common shift methods (truncating
   earlier turns, inserting unrelated content) either produce
   malformed conversations or change the model's expected output,
   muddying the measurement. We shift only by duplicating the system
   prompt's own content — redundant by construction — so the model's
   expected reply is unchanged and any divergence is attributable to
   position.

## Data

`NousResearch/hermes-function-calling-v1`, `func_calling` config.
Multi-turn agent traces with a system prompt defining tools, a user
question, an assistant tool call, and a tool-result turn (mean ≈ 400
tokens — comfortably larger than our 128-token cache chunk). We take
the first 20 examples that pass a minimum-length filter; each is
measured independently and the aggregate is reported.
