# reagent — tool-result KV cache reuse sensitivity

**reagent** measures whether server-side KV cache reuse preserves
task behaviour when the cached block appears at a drifted absolute
position in a new request. It targets the *read/consumption* step of
cross-request cache reuse — the splice-the-cached-K-into-the-new-prefix
path — and compares two strategies at that step: *naive* (splice with
original RoPE phases) and *shifted* (re-rotate K to the new positions,
matching llama.cpp's `llama_memory_seq_add`).

## Scope

- **Cached blocks are semantically aligned.** We cache the last
  `L=128` tokens of a Hermes tool-result turn — a long, self-contained,
  deterministic block at a clean structural edge. This is the best-case
  input to the splice path.
- **The read path is the same across cache policies.** Whether a server
  caches only whole semantic units (hypothetical "semantic caching") or
  any contiguous N+ token span (llama.cpp's `--cache-reuse`), the
  consumption step is identical: greedy byte-exact match → overwrite
  K/V slots → continue decoding. The two policies differ only in what
  enters the cache; once a match is found, the splice mechanism we
  measure is shared.
- **Cache policy is server-side.** The client sends the prompt; what
  recurs and what matches is decided by the server. Reagent measures a
  serving-stack property, not an agent-framework property.
- What we **do not** test yet: length-based caching that can match
  arbitrary interior spans (fragments that cross turn boundaries,
  non-structural subsequences). See "Future directions".

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

**Cosine-similarity scale** (under `bge-small-en-v1.5`, matching the
green / yellow / red bands in the figure): ≥ 0.95 = same answer
(green); 0.80–0.95 = same topic, rewording (yellow); < 0.80 =
different content or meaning (red).

## Results

N = 20 Hermes examples, L = 128, `bge-small-en-v1.5` embedder.

![reagent panel](reagent_panel.png)

### sim(fresh, reused) — headline

Semantic similarity between the greedy continuation produced without
reuse and with reuse. Green band (≥ 0.95) means naive reuse produces
a semantically-equivalent response; yellow (0.80–0.95) means same
topic but rewording; red (< 0.80) means reuse rewrites the answer.

| model | Δ=0 | Δ=50 | Δ=100 | Δ=200 | Δ=500 | Δ=1000 |
|---|---|---|---|---|---|---|
| `gemma-4-E4B` | 1.00 | 0.99 | 0.96 | 0.96 | 0.93 | 0.92 |
| `gemma-4-31B` | 1.00 | 0.99 | 0.99 | 0.97 | 0.95 | 0.91 |
| `gemma-4-26B-A4B` | 0.99 | 0.97 | 0.98 | 0.96 | 0.86 | 0.85 |
| `Llama-3.1-8B` | 0.98 | 0.92 | 0.86 | 0.82 | 0.79 | 0.81 |
| `Llama-3.2-1B` | 0.96 | 0.84 | 0.79 | 0.74 | 0.73 | 0.68 |
| `gemma-4-E2B` | 0.99 | 0.97 | 0.96 | 0.91 | 0.78 | 0.60 |

### top-1 agreement

Fraction of examples where fresh and reused produce the same argmax
token at the trigger position. Green (≥ 0.90), yellow (0.70–0.90),
red (< 0.70) in the figure.

| model | Δ=0 | Δ=50 | Δ=100 | Δ=200 | Δ=500 | Δ=1000 |
|---|---|---|---|---|---|---|
| `gemma-4-E4B` | 1.00 | 1.00 | 0.90 | 0.90 | 0.75 | 0.80 |
| `gemma-4-31B` | 1.00 | 1.00 | 0.95 | 0.85 | 0.75 | 0.55 |
| `Llama-3.1-8B` | 0.95 | 0.80 | 0.65 | 0.50 | 0.50 | 0.45 |
| `gemma-4-26B-A4B` | 1.00 | 0.95 | 0.80 | 0.80 | 0.45 | 0.40 |
| `gemma-4-E2B` | 1.00 | 0.85 | 0.90 | 0.70 | 0.35 | 0.15 |
| `Llama-3.2-1B` | 0.95 | 0.40 | 0.20 | 0.10 | 0.20 | 0.10 |

### mean KL(fresh ∥ reused) (nats)

| model | Δ=0 | Δ=50 | Δ=100 | Δ=200 | Δ=500 | Δ=1000 |
|---|---|---|---|---|---|---|
| `gemma-4-E4B` | 0.00±0.00 | 0.01±0.02 | 0.03±0.04 | 0.09±0.09 | 0.62±1.29 | 0.69±1.41 |
| `Llama-3.1-8B` | 0.00±0.00 | 0.08±0.04 | 0.21±0.20 | 1.02±1.45 | 1.94±2.46 | 2.09±2.39 |
| `gemma-4-31B` | 0.00±0.00 | 0.03±0.06 | 0.15±0.58 | 0.40±1.41 | 1.98±4.61 | 2.65±5.41 |
| `gemma-4-26B-A4B` | 0.01±0.01 | 0.06±0.14 | 0.09±0.14 | 0.36±1.07 | 2.91±3.36 | 2.74±3.14 |
| `Llama-3.2-1B` | 0.00±0.00 | 2.25±1.67 | 3.47±2.32 | 5.50±3.03 | 5.44±3.02 | 5.93±3.26 |
| `gemma-4-E2B` | 0.01±0.05 | 1.91±5.72 | 0.14±0.40 | 2.49±4.59 | 3.99±4.74 | 7.01±5.56 |

### sim(reused, reference)

Cosine similarity of the reused continuation against the dataset's
gold assistant reply — a task-quality floor independent of fresh.

| model | Δ=0 | Δ=50 | Δ=100 | Δ=200 | Δ=500 | Δ=1000 |
|---|---|---|---|---|---|---|
| `gemma-4-E4B` | 0.91 | 0.90 | 0.89 | 0.89 | 0.88 | 0.88 |
| `gemma-4-31B` | 0.91 | 0.91 | 0.90 | 0.91 | 0.89 | 0.87 |
| `gemma-4-26B-A4B` | 0.90 | 0.89 | 0.90 | 0.89 | 0.84 | 0.83 |
| `Llama-3.1-8B` | 0.87 | 0.83 | 0.82 | 0.79 | 0.74 | 0.74 |
| `gemma-4-E2B` | 0.88 | 0.88 | 0.88 | 0.83 | 0.77 | 0.58 |
| `Llama-3.2-1B` | 0.76 | 0.76 | 0.73 | 0.67 | 0.67 | 0.65 |

### How to read the tables

- **Δ=0** is the sanity floor. Non-zero KL here means a harness bug,
  not a model insight.
- **`sim(fresh, reused)`** is the headline. ≥ 0.95 (green) means
  naive reuse is indistinguishable; 0.80–0.95 (yellow) means same
  topic but rewording; < 0.80 (red) means reuse rewrites the answer.
- **KL** flags first-token disagreement; **sim** tells you whether
  the disagreement persists or the two sides converge on similar
  content after a few tokens. Use both.
- **`sim(reused, reference)`** gives a quality floor: if it drops
  faster than `sim(fresh, reused)` as Δ grows, reuse is degrading
  task quality.
- **Large stdev matters.** Gemma-4-E2B at Δ=1000 shows `7.01±5.56`:
  the outcome is prompt-dependent and production quality would be
  unpredictable.
- **Agreement vs similarity diverge.** Top-1 agreement drops fast
  (even the best models are at 0.55–0.80 by Δ=1000), while
  `sim(fresh, reused)` holds ≥ 0.90 for the Gemma-4 ≥ 4B panel —
  greedy decoding recovers to semantically-equivalent output
  despite first-token disagreement.

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

`visualize.py` renders a 2-panel figure (`sim(fresh, reused)` and
top-1 agreement) from whatever JSONs are present, with green /
yellow / red safety bands drawn from the Passing Bar thresholds.

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

## Future directions

- **Length-based cache reuse.** llama.cpp's `--cache-reuse N` greedily
  finds any N+ token byte-exact match regardless of structural
  alignment. At low N this splices fragments that cross turn
  boundaries or sit mid-paragraph — K vectors whose attention was
  conditioned on context that is *not* present in the new request.
  Our harness always caches a semantically-aligned slice, so this
  failure mode is out of scope. Testing it would mean adding a
  harness setting that picks arbitrary interior cache boundaries
  (e.g. spans that cross the `assistant(tool_call) → tool(result)`
  boundary) and rerunning the naive/shifted comparison.
- **Small cache chunks.** As `L` shrinks, a larger fraction of the
  cached K vectors attend to pre-block tokens rather than to other
  block-internal tokens. Shift correction handles positions but
  cannot repair attention to a different external neighbourhood, so
  the shifted-reuse ceiling is expected to drop below the near-1.00
  level we see at `L=128`.
- **Non-agentic cached content.** Retrieved documents in RAG, source
  code chunks, long user-pasted files. Same splice mechanism but
  different attention-conditioning profiles; worth replicating on a
  non-Hermes dataset.
- **Accumulated reuse across many turns.** Our measurement is single
  hop: one request writes the cache, the next reads it. A realistic
  agent session reuses the same cached chunk across many turns; if
  errors compound, a setup that is lossless at N=1 may not be at
  N=10.
