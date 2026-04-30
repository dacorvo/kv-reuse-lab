#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch>=2.4,<2.11",
#   "datasets>=2.20",
#   "transformers>=5.5",
#   "sentencepiece",
#   "sentence-transformers>=3.0",
# ]
# ///
"""Test two candidate "splice safety" predictors against measured sim.

For every (A, B) pair with at least one match span, compute:

    pre_ctx_sim  = cosine sim of last-W tokens before A's match-start
                   and last-W tokens before B's match-start, embedded
                   via the harness's bge-small-en-v1.5 model
    div_gap      = b_start - prev_b_end (tokens of non-matching content
                   between the previous match span's end and this
                   match's start)

For multi-match pairs, aggregate per pair as
    min(pre_ctx_sim)  — the worst-case match boundary
    max(div_gap)      — the largest divergence the model has to cross

Then correlate each predictor with the measured ``sim_fresh_reused``
across all pairs, and print bucketed crosstabs so the relationship is
visible at a glance.

Usage:
    uv run --script analyze_splice_predictors.py \\
        --model google/gemma-4-E4B-it \\
        --dataset nemotron-swe --repo pandas-dev \\
        --n-sessions 20 --max-tokens-per-session 12000 \\
        results/multi_splice_b_pandas_gemma4_e4b.json
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent))

from measure_multi_splice import find_matches, load_sessions  # noqa: E402
from similarity import load_embedder_and_cos_sim  # noqa: E402


_DIGIT_RUN = re.compile(r"(\d)\1{9,}")


def _is_degenerate(r: dict) -> bool:
    return bool(
        _DIGIT_RUN.search(r.get("fresh_text", "") or "")
        or _DIGIT_RUN.search(r.get("reused_text", "") or "")
    )


def _load_rows(path: Path) -> List[dict]:
    text = path.read_text()
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return json.loads(text).get("per_pair", [])


def _pearson(xs: List[float], ys: List[float]) -> float:
    n = len(xs)
    if n < 3:
        return float("nan")
    mx, my = statistics.mean(xs), statistics.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = sum((x - mx) ** 2 for x in xs) ** 0.5
    den_y = sum((y - my) ** 2 for y in ys) ** 0.5
    if den_x == 0 or den_y == 0:
        return float("nan")
    return num / (den_x * den_y)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("path", type=Path)
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", default="swe-smith")
    p.add_argument("--repo", default="django")
    p.add_argument("--n-sessions", type=int, default=20)
    p.add_argument("--max-tokens-per-session", type=int, default=12000)
    p.add_argument("--min-match", type=int, default=128)
    p.add_argument(
        "--ctx-window",
        type=int,
        default=256,
        help="How many tokens before each match start to include in the "
        "pre-context similarity computation.",
    )
    p.add_argument(
        "--use-gap-content",
        action="store_true",
        help="Instead of pre-match window, embed the divergent gap content "
        "(tokens between previous match end and this match start) on each "
        "side. Targets the actually-different text directly.",
    )
    p.add_argument(
        "--metric",
        choices=("embed", "jaccard"),
        default="embed",
        help="Similarity metric for the chunk comparison. "
        "embed = bge-small cosine. "
        "jaccard = token 3-gram Jaccard similarity (faster, no model).",
    )
    p.add_argument("--embedder", default="BAAI/bge-small-en-v1.5")
    args = p.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    print(f"[info] loading {args.n_sessions} {args.repo}* sessions...")
    sessions = load_sessions(
        tok,
        args.n_sessions,
        args.repo,
        args.max_tokens_per_session,
        dataset=args.dataset,
    )
    iid_to_toks = {iid: ids for iid, ids in sessions}
    if args.metric == "embed":
        cos_sim = load_embedder_and_cos_sim(args.embedder)

        def chunk_sim(a_text: str, b_text: str, a_toks=None, b_toks=None) -> float:
            return cos_sim(a_text, b_text) if a_text and b_text else float("nan")
    else:

        def chunk_sim(a_text: str, b_text: str, a_toks=None, b_toks=None) -> float:
            if not a_toks or not b_toks:
                return float("nan")
            n = 3
            a_grams = {tuple(a_toks[i : i + n]) for i in range(len(a_toks) - n + 1)}
            b_grams = {tuple(b_toks[i : i + n]) for i in range(len(b_toks) - n + 1)}
            if not a_grams or not b_grams:
                return float("nan")
            return len(a_grams & b_grams) / len(a_grams | b_grams)

    rows = _load_rows(args.path)
    rows = [r for r in rows if not _is_degenerate(r)]
    print(f"[info] {len(rows)} non-degenerate pairs")

    enriched = []
    for r in rows:
        a_toks = iid_to_toks.get(r["a_id"])
        b_toks = iid_to_toks.get(r["b_id"])
        if a_toks is None or b_toks is None:
            continue
        spans = r.get("match_spans") or [
            list(s) for s in find_matches(b_toks, a_toks, args.min_match)
        ]
        if not spans:
            continue
        spans = sorted(spans, key=lambda s: s[0])
        per_match_ctx_sim = []
        per_match_div_gap = []
        prev_b_end = 0
        prev_a_end = 0
        for bs, be, as_, ae in spans:
            div_gap = bs - prev_b_end
            if args.use_gap_content:
                # Embed the divergent text BETWEEN matches: the slice
                # of A from prev_a_end..as_ vs slice of B from
                # prev_b_end..bs. These are the unique-to-each-session
                # bug-description chunks, with no boilerplate dilution.
                a_slice = a_toks[prev_a_end:as_] if as_ > prev_a_end else []
                b_slice = b_toks[prev_b_end:bs] if bs > prev_b_end else []
            else:
                W = args.ctx_window
                a_slice = a_toks[max(0, as_ - W) : as_]
                b_slice = b_toks[max(0, bs - W) : bs]
            a_chunk = tok.decode(a_slice) if a_slice else ""
            b_chunk = tok.decode(b_slice) if b_slice else ""
            sim = chunk_sim(a_chunk, b_chunk, a_slice, b_slice)
            per_match_ctx_sim.append(sim)
            per_match_div_gap.append(div_gap)
            prev_b_end = be
            prev_a_end = ae
        enriched.append(
            {
                **r,
                "min_pre_ctx_sim": min(
                    s for s in per_match_ctx_sim if s == s
                ),  # NaN-safe: NaN check
                "max_div_gap": max(per_match_div_gap),
                "per_match_ctx_sim": per_match_ctx_sim,
                "per_match_div_gap": per_match_div_gap,
            }
        )

    print(f"\n=== {len(enriched)} pairs analysed ===")

    # Pearson correlations
    sims = [r["sim_fresh_reused"] for r in enriched]
    ctx = [r["min_pre_ctx_sim"] for r in enriched]
    gap = [float(r["max_div_gap"]) for r in enriched]
    r_ctx = _pearson(ctx, sims)
    r_gap = _pearson(gap, sims)
    print("\nPearson correlation vs sim_fresh_reused:")
    print(f"  min(pre_ctx_sim)  ↔ sim_fresh_reused : r = {r_ctx:+.3f}")
    print(f"  max(div_gap)      ↔ sim_fresh_reused : r = {r_gap:+.3f}")

    def _bucket_sim(s: float) -> str:
        if s >= 0.99:
            return "≥0.99"
        if s >= 0.95:
            return "0.95-0.99"
        if s >= 0.80:
            return "0.80-0.95"
        return "<0.80"

    sim_keys = ["≥0.99", "0.95-0.99", "0.80-0.95", "<0.80"]

    # ctx-sim crosstab
    ctx_buckets = [
        ("≥0.95", lambda x: x >= 0.95),
        ("0.85-0.95", lambda x: 0.85 <= x < 0.95),
        ("0.70-0.85", lambda x: 0.70 <= x < 0.85),
        ("<0.70", lambda x: x < 0.70),
    ]
    grid: Dict[str, Dict[str, int]] = {
        b[0]: {s: 0 for s in sim_keys} for b in ctx_buckets
    }
    for r in enriched:
        for name, pred in ctx_buckets:
            if pred(r["min_pre_ctx_sim"]):
                grid[name][_bucket_sim(r["sim_fresh_reused"])] += 1
                break
    print("\nmin(pre_ctx_sim) × sim_fresh_reused crosstab:")
    print("  " + " " * 12 + "  ".join(f"{s:>10s}" for s in sim_keys) + "  total")
    for name, _ in ctx_buckets:
        row = grid[name]
        total = sum(row.values())
        cells = "  ".join(f"{row[s]:>10d}" for s in sim_keys)
        print(f"  {name:<12s}{cells}  {total:>5d}")

    # div-gap crosstab
    gap_buckets = [
        ("0", lambda x: x == 0),
        ("1-100", lambda x: 0 < x <= 100),
        ("100-500", lambda x: 100 < x <= 500),
        ("500+", lambda x: x > 500),
    ]
    grid2: Dict[str, Dict[str, int]] = {
        b[0]: {s: 0 for s in sim_keys} for b in gap_buckets
    }
    for r in enriched:
        for name, pred in gap_buckets:
            if pred(r["max_div_gap"]):
                grid2[name][_bucket_sim(r["sim_fresh_reused"])] += 1
                break
    print("\nmax(div_gap) × sim_fresh_reused crosstab:")
    print("  " + " " * 12 + "  ".join(f"{s:>10s}" for s in sim_keys) + "  total")
    for name, _ in gap_buckets:
        row = grid2[name]
        total = sum(row.values())
        cells = "  ".join(f"{row[s]:>10d}" for s in sim_keys)
        print(f"  {name:<12s}{cells}  {total:>5d}")

    # Bad pair detail with both predictors
    print("\nworst pairs (sim < 0.95):")
    print(f"  {'sim':>5}  {'ctx_sim':>7}  {'div_gap':>7}  pair")
    bad = sorted(
        (r for r in enriched if r["sim_fresh_reused"] < 0.95),
        key=lambda r: r["sim_fresh_reused"],
    )
    for r in bad:
        print(
            f"  {r['sim_fresh_reused']:>5.2f}  "
            f"{r['min_pre_ctx_sim']:>7.3f}  "
            f"{r['max_div_gap']:>7d}  "
            f"{r['b_id'][-22:]} ← {r['a_id'][-22:]}"
        )


if __name__ == "__main__":
    main()
