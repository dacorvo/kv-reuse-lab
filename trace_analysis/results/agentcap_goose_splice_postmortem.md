# goose × Qwen3.6 splice-correctness post-mortem

End-to-end splice harness on all 99 splice-candidate pairs from
`agentcap_goose_match_categories.splice_candidates.jsonl` against
Qwen3.6-35B-A3B (hybrid attention+recurrent) running through patched
llama-server with `feat/cache-reuse-symmetric` + IM-RoPE K-shift.

Per pair: spliced run (donor pre-warm → recipient with cache-reuse) vs
cold run (recipient alone in a fresh server). Metrics: top-1
agreement at first generated token, top-20 jaccard, approximate first-
token KL, and 64-token continuation cosine similarity (`bge-small-en`).

## Headline numbers

- **N measured: 88** (11 pairs skipped — recipient or donor exceeded
  65k ctx; chunk-size sweep was largest-first so smaller pairs are
  fully covered).
- **Top-1 agreement: 75.0%** (66 / 88).
- **Mean continuation similarity: 0.906**; median 0.940.
- **Mean first-token KL: 4.4 nats** (skewed by catastrophic outliers).
- **Severity distribution:**
  - 63 green (sim ≥ 0.85, KL < 30): 72%
  - 15 yellow (0.7 ≤ sim < 0.85): 17%
  - 2 red (sim < 0.7, no recurrent-state-bleed): 2%
  - **8 catastrophic (KL > 30 OR sim = None): 9%**

Severity is **bimodal** — KL is either < 5 or > 40, almost no middle.
The catastrophic tail is sharp, not smooth.

## The catastrophic failure mode is exactly two tokens

All 8 catastrophic pairs collapse the spliced first-token onto one of
two specific Qwen3.6 tokens:

| token id | surface form | meaning | catastrophic count |
|---|---|---|---|
| 510 | `</` | start of a closing tag | 6 |
| 248046 | `<\|im_end\|>` | end-of-turn | 2 |

The spliced continuations are diagnostic:

```
pair  5 (510): "</parameter>\n<parameter=depth>\n1\n</parameter>\n</function>\n</tool_call>"
pair 10 (510): "</parameter>\n</function>\n</tool_call>"
pair 28 (510): "</parameter>\n<parameter=depth>\n2\n</parameter>\n</function>\n</tool_call>"
pair 30 (510): "</parameter>\n</function>\n</tool_call>"
pair 49 (510): "</parameter>\n</function>\n</tool_call>"
pair 54 (510): "</parameter>\n</function>\n</tool_call>"
pair 74 (248046): ""    (empty — immediate end-of-turn)
pair 87 (248046): ""
```

In every Type-1 (`</`) failure the model emits a **syntactically valid
closing-tag sequence with no matching openings in the response**. The
donor's recurrent state was mid-`<parameter>` body when its prefill
crossed chunk-end; that state is what the spliced K/V transports into
the recipient's prefill, so when the recipient asks for the next
token the model writes the close-tag the donor was about to write.
Type 2 fires when the donor's recurrent state was at end-of-turn —
the spliced state immediately closes the turn.

Cold continuations for every catastrophic pair start cleanly with a
preamble or fresh `<tool_call>`. Cold prefill is fine; splice is
corrupting state.

This is the recurrent-state-bleed signature `SCOPE.md` flagged for
hybrid Qwen3.5/3.6 — **the recurrent state is not position-addressable
and the K-shift cannot rewrite it**, so the donor's mid-stream summary
leaks into the recipient. Reagent's published splice-correctness work
measured pure attention and explicitly did not bound this; the present
data measures it.

## What predicts catastrophic — recipient post-chunk tail

For each pair we computed `post_r = recipient_total_tokens -
recipient.chunk_end_position`. This is **how many fresh recipient
tokens are prefilled after the spliced cells** — i.e. how much
opportunity the recurrent state has to absorb the recipient's actual
context before generation begins.

| post_r range | N | catastrophic | non-catastrophic |
|---|---|---|---|
| ≤ 10 | 6 | **5** (83%) | 1 (yellow, KL=3.15) |
| 11–365 | 26 | 1 (pair 54, splice_ratio=0.30) | 25 |
| ≥ 366 | 56 | 2 (pairs 28, 30 — splice_ratio ≥ 0.76) | 54 |

`post_r ≤ 10` is essentially **just the chat-template tail**
(`<\|im_end\|>\n<\|im_start\|>assistant\n` is ~5 tokens). When the
recipient's chunk ends at the last-tool-message boundary and the
spliced cells extend to within 5 tokens of generation start, the donor
footprint dominates the recurrent state and **5 of 6 such pairs fail
catastrophically**.

The 2 catastrophic pairs at `post_r ≥ 30000` (28, 30) are explained by
a second factor: their applied splice covers 76–80% of the recipient's
total tokens. When *most* of the prefill comes from spliced cells
regardless of where chunk-end sits, the donor footprint dominates
throughout. Pair 54 is at `post_r = 366` and 30% splice ratio — the
sole catastrophic that doesn't fit either threshold cleanly.

A combined heuristic `post_r ≤ 10 OR splice_ratio ≥ 0.7` catches **7
of 8 catastrophics** at the cost of 17 false-positives in the yellow
band.

The cleaner production rule is just **`post_r ≤ 10` ⇒ refuse splice**.
Computable at admission/match time from manifest data. Catches the
bulk of catastrophics with one false-positive in the yellow band.

## What does NOT predict catastrophic

- **Chunk size**: catastrophics span 1.5k to 40k tokens.
- **Position drift**: catastrophics span -2395 to +7004.
- **Single- vs multi-segment**: 6 single + 2 multi-segment among
  catastrophics. Multi-segment splices (up to 9 disjoint segments
  with mixed-direction shifts on pairs 94-98) handled cleanly by
  the symmetric-cache-reuse patch.
- **Tool category**: 5 in `tree:`, 3 in `analyze:`. Not specific to
  any tool.
- **Bucket identity**: most buckets had 5/5 clean (`load_skill`,
  `analyze:a57136ea3e`, `shell:a1db44eaa4`, `shell:f2aed7a6ee`,
  `shell:cbe0dc1fc5`, `tree:dde522f96c`, `tree:849f1f0159`). The
  bucket alone does not determine outcome — within a single bucket
  the 5 pairs hit different post-chunk-tail lengths and catch the
  failure independently.

## Best pairs (for contrast)

The 6 lowest-KL pairs all have sim ≥ 0.961 and span every tool family:

| pair | bucket | KL | sim | top1 |
|---|---|---|---|---|
| 52 | analyze:886dbab073 | 0.005 | 1.000 | `<tool_call>` |
| 23 | tree:dde522f96c | 0.007 | 1.000 | `<tool_call>` |
| 62 | shell:cbe0dc1fc5 | 0.011 | 1.000 | "Now" |
| 93 | load_skill:12bf83673f | 0.012 | 1.000 | `<tool_call>` |
| 9 | tree:d6b975e432 | 0.014 | 1.000 | `<tool_call>` |
| 76 | analyze:380fdbaa53 | 0.016 | 1.000 | `<tool_call>` |

When splice works, it works cleanly — the spliced 64-token
continuation is **bit-for-bit similar to cold** (sim = 1.0). The
mechanism is mathematically correct on the attention side; what fails
is hybrid recurrent-state correctness at chunk-end.

## What this means

For Qwen3.6 (and presumably Qwen3.5):

1. **Splice geometry** (multi-segment, mixed-direction shifts up to
   ±30k) is robust. The symmetric-cache-reuse + IM-RoPE K-shift patch
   handles every geometry tested without producing "broken nonsense";
   when output diverges it does so on specific recurrent-state-bleed
   tokens, not random corruption.

2. **75% agreement on the next move** + 90% mean sim is a workable
   regime for some workloads but not for production correctness — the
   9% catastrophic tail produces syntactically-valid orphan
   close-tags and silent end-of-turn, which would corrupt downstream
   tool-execution.

3. **The catastrophic mode is gateable pre-splice**: refuse splice
   when the recipient's post-chunk tail is too short to wash out the
   donor's recurrent footprint. Prevents 5/8 of the catastrophics in
   our sample with one false positive. A second gate on splice ratio
   (≥ 0.7) catches 2 more.

4. **The remaining catastrophic** (pair 54, post_r=366, ratio=0.30)
   suggests there's residual risk even with both gates active — at
   least for hybrid models. This argues for a non-hybrid model
   recapture (Llama-3.3-70B or similar) before declaring the
   splice-with-rephasing approach correct enough for production.

For non-hybrid attention-only models the recurrent-state-bleed is
absent by construction; reagent's published bound applies. Hybrid
splice with the heuristic gate above is **acceptable for
research-grade reuse but not safe by default**.
