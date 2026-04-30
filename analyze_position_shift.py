#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch>=2.4,<2.11",
#   "datasets>=2.20",
#   "transformers>=5.5",
#   "sentencepiece",
# ]
# ///
"""Recompute match positions for an existing multi_splice result file
and correlate position-shift (b_start - a_start) with sim/KL.

Older result files (pre `feat: persist per-pair match_spans`) don't
have positions in the per-pair rows. This script reloads the source
sessions from the dataset (via measure_multi_splice.load_sessions)
and replays find_matches per pair to recover positions, then prints:

    - per-pair: shift / span / cov / sim / kl  (sortable)
    - cross-tab: shift bucket × sim bucket
    - bad-pair detail (sim < 0.95): every match's [b_start, a_start, len, shift]

Usage:
    uv run --script analyze_position_shift.py \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --repo django --n-sessions 20 \\
        --max-tokens-per-session 12000 \\
        --min-match 128 \\
        results/multi_splice_b_django_llama8b.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent))

from measure_multi_splice import find_matches, load_sessions  # noqa: E402


def _load_rows(path: Path) -> List[dict]:
    text = path.read_text()
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    obj = json.loads(text)
    return obj.get("per_pair", [])


def _bucket_shift(s: int) -> str:
    s = abs(s)
    if s < 500:
        return "<0.5k"
    if s < 2000:
        return "0.5-2k"
    if s < 5000:
        return "2-5k"
    if s < 10000:
        return "5-10k"
    return "10k+"


def _bucket_sim(s: float) -> str:
    if s >= 0.99:
        return "≥0.99"
    if s >= 0.95:
        return "0.95-0.99"
    if s >= 0.80:
        return "0.80-0.95"
    return "<0.80"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("path", type=Path)
    p.add_argument("--model", required=True)
    p.add_argument(
        "--dataset",
        default="swe-smith",
        help="Source dataset key as passed to load_sessions "
        "(swe-smith / nemotron-swe).",
    )
    p.add_argument("--repo", default="django")
    p.add_argument("--n-sessions", type=int, default=20)
    p.add_argument("--max-tokens-per-session", type=int, default=12000)
    p.add_argument("--min-match", type=int, default=128)
    p.add_argument(
        "--show-bad",
        type=float,
        default=0.95,
        help="Print full match detail for pairs with sim below this threshold.",
    )
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
    print(f"[info] loaded {len(sessions)} sessions")

    rows = _load_rows(args.path)
    print(f"[info] {len(rows)} pairs in {args.path.name}")

    # For each row, recompute matches and tag with position info.
    enriched: List[dict] = []
    for r in rows:
        a_toks = iid_to_toks.get(r["a_id"])
        b_toks = iid_to_toks.get(r["b_id"])
        if a_toks is None or b_toks is None:
            continue
        if r.get("match_spans"):
            spans = [tuple(s) for s in r["match_spans"]]
        else:
            spans = find_matches(b_toks, a_toks, args.min_match)
        # Aggregate per-pair: median shift, max shift, mean shift, max span pos in B.
        if not spans:
            continue
        shifts = [bs - as_ for bs, _be, as_, _ae in spans]
        max_b_pos = max(be for _bs, be, _, _ in spans)
        enriched.append(
            {
                **r,
                "spans": spans,
                "max_shift": max(shifts, key=abs),
                "mean_shift": sum(shifts) / len(shifts),
                "max_b_pos": max_b_pos,
            }
        )

    # Cross-tab |max_shift| × sim
    shift_keys = ["<0.5k", "0.5-2k", "2-5k", "5-10k", "10k+"]
    sim_keys = ["≥0.99", "0.95-0.99", "0.80-0.95", "<0.80"]
    grid: Dict[str, Dict[str, int]] = {s: {k: 0 for k in sim_keys} for s in shift_keys}
    for r in enriched:
        grid[_bucket_shift(r["max_shift"])][_bucket_sim(r["sim_fresh_reused"])] += 1

    print("\n=== |max_shift| × sim crosstab ===")
    print("  " + " " * 8 + "  ".join(f"{s:>10s}" for s in sim_keys) + "  total")
    for s in shift_keys:
        row = grid[s]
        total = sum(row.values())
        cells = "  ".join(f"{row[k]:>10d}" for k in sim_keys)
        print(f"  {s:<8s}{cells}  {total:>5d}")

    # Cross-tab max_b_pos × sim — tests "early splices fail more".
    pos_keys = ["<1k", "1-3k", "3-6k", "6-10k", "10k+"]

    def _bucket_pos(p: int) -> str:
        if p < 1000:
            return "<1k"
        if p < 3000:
            return "1-3k"
        if p < 6000:
            return "3-6k"
        if p < 10000:
            return "6-10k"
        return "10k+"

    pos_grid: Dict[str, Dict[str, int]] = {
        p: {k: 0 for k in sim_keys} for p in pos_keys
    }
    for r in enriched:
        pos_grid[_bucket_pos(r["max_b_pos"])][_bucket_sim(r["sim_fresh_reused"])] += 1

    print("\n=== max_b_pos × sim crosstab ===")
    print("  " + " " * 8 + "  ".join(f"{s:>10s}" for s in sim_keys) + "  total")
    for p_key in pos_keys:
        row = pos_grid[p_key]
        total = sum(row.values())
        cells = "  ".join(f"{row[k]:>10d}" for k in sim_keys)
        print(f"  {p_key:<8s}{cells}  {total:>5d}")

    # Bad pair detail — show full match info, sorted by sim asc.
    bad = sorted(
        (r for r in enriched if r["sim_fresh_reused"] < args.show_bad),
        key=lambda r: r["sim_fresh_reused"],
    )
    if bad:
        print(f"\n=== bad pairs (sim < {args.show_bad}, n={len(bad)}) ===")
        print("  sim     kl    shift   max_b_pos  cov     n  b_id ← a_id")
        for r in bad:
            spans_str = " | ".join(
                f"[B={bs:>5d}+{be - bs} A={as_:>5d}]"
                for bs, be, as_, _ae in r["spans"][:3]
            )
            extra = "..." if len(r["spans"]) > 3 else ""
            print(
                f"  {r['sim_fresh_reused']:.2f}  {r['kl']:.2f}  "
                f"{r['mean_shift']:>+6.0f}  {r['max_b_pos']:>5d}  "
                f"{r['coverage_frac']:.2f}  {r['n_matches']}  "
                f"{r['b_id'][-30:]} ← {r['a_id'][-30:]}"
            )
            print(f"      spans: {spans_str}{extra}")


if __name__ == "__main__":
    main()
