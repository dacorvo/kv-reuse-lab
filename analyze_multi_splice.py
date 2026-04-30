#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Cross-run analyzer for multi_splice / multi_splice_b results.

Loads one or more aggregate JSON files (or per-pair .jsonl), and
prints:

    - headline distribution per file (n, mean/median/p10 sim,
      bit-exact count, sim<0.95 count, sim<0.80 count)
    - "bad B" sessions: any B where ≥ 40% of reuses sit below 0.95
    - coverage × sim cross-tab (0-5 / 5-20 / 20-50 / 50+ vs
      sim ≥0.99 / 0.95-0.99 / 0.80-0.95 / <0.80)

Designed to support comparing Llama-1B (Scheme A, N=10), Gemma-4
(Scheme B, N=20), and Llama-8B (Scheme B, N=20) on the same
django SWE-smith workload, but works on any combination.

Usage:
    uv run --script analyze_multi_splice.py \\
        results/multi_splice_b_django_gemma4_e4b.json \\
        results/multi_splice_b_django_llama8b.json
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


_DIGIT_RUN = re.compile(r"(\d)\1{9,}")


def _is_degenerate(text: str) -> bool:
    """Spot generation collapses to a token-repetition attractor.

    Triggered by 10+ identical digits in a row (e.g. ``1111111111``,
    seen on Gemma-4 / Nemotron pandas-dev when prompt ends with
    ``(#NNNNN)\\n``). Doesn't flag long runs of spaces or alpha
    characters since those legitimately occur in code indentation
    and decorative dividers.
    """
    return bool(text and _DIGIT_RUN.search(text))


def _load(path: Path) -> tuple[str, List[dict]]:
    """Return (label, per_pair_rows) from either an aggregate .json
    or a per-pair .jsonl file.
    """
    text = path.read_text()
    label = path.stem
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        return label, rows
    obj = json.loads(text)
    rows = obj.get("per_pair", [])
    if "model" in obj:
        label = f"{label}  ({obj['model']}, scheme={obj.get('scheme', 'A')})"
    return label, rows


def _bucket_cov(c: float) -> str:
    if c < 0.05:
        return "0-5%"
    if c < 0.20:
        return "5-20%"
    if c < 0.50:
        return "20-50%"
    return "50+%"


def _bucket_sim(s: float) -> str:
    if s >= 0.99:
        return "≥0.99"
    if s >= 0.95:
        return "0.95-0.99"
    if s >= 0.80:
        return "0.80-0.95"
    return "<0.80"


def _print_headline(label: str, rows: List[dict]) -> None:
    """Print headline distribution. Auto-filters degenerate-generation
    pairs (e.g. token-repetition attractors) and notes the count.
    """
    n_total = len(rows)
    clean = [
        r
        for r in rows
        if not _is_degenerate(r.get("fresh_text", "") or "")
        and not _is_degenerate(r.get("reused_text", "") or "")
    ]
    n_degen = n_total - len(clean)
    sims = [r["sim_fresh_reused"] for r in clean]
    n = len(sims)
    deg_note = f" — dropped {n_degen} degenerate-generation pair(s)" if n_degen else ""
    print(f"\n=== {label}  (n={n}/{n_total}{deg_note}) ===")
    if n == 0:
        return
    sims_sorted = sorted(sims)
    p10 = sims_sorted[max(0, int(0.10 * (n - 1)))]
    print(
        f"  sim:  mean={statistics.mean(sims):.4f}  "
        f"median={statistics.median(sims):.4f}  p10={p10:.4f}"
    )
    bit_exact = sum(1 for s in sims if s == 1.0)
    sub95 = sum(1 for s in sims if s < 0.95)
    sub80 = sum(1 for s in sims if s < 0.80)
    print(
        f"  bit-exact={bit_exact}/{n} ({bit_exact / n:.0%})  "
        f"sim<0.95={sub95}/{n} ({sub95 / n:.0%})  "
        f"sim<0.80={sub80}/{n} ({sub80 / n:.0%})"
    )


def _print_bad_bs(label: str, rows: List[dict]) -> None:
    by_b: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        by_b[r["b_id"]].append(r["sim_fresh_reused"])
    flagged = []
    for b, sims in by_b.items():
        if len(sims) < 3:
            continue
        bad = sum(1 for s in sims if s < 0.95)
        if bad / len(sims) >= 0.40:
            flagged.append(
                (
                    bad / len(sims),
                    b,
                    bad,
                    len(sims),
                    statistics.mean(sims),
                    min(sims),
                )
            )
    if not flagged:
        return
    flagged.sort(reverse=True)
    print(f"\n  bad-B (≥40% reuses sim<0.95) [{label}]:")
    for frac, b, bad, n, mean, mn in flagged:
        print(f"    {b:60s}  bad={bad}/{n} ({frac:.0%})  mean={mean:.3f}  min={mn:.3f}")


def _print_crosstab(label: str, rows: List[dict]) -> None:
    cov_keys = ["0-5%", "5-20%", "20-50%", "50+%"]
    sim_keys = ["≥0.99", "0.95-0.99", "0.80-0.95", "<0.80"]
    grid: Dict[str, Dict[str, int]] = {c: {s: 0 for s in sim_keys} for c in cov_keys}
    for r in rows:
        grid[_bucket_cov(r["coverage_frac"])][_bucket_sim(r["sim_fresh_reused"])] += 1
    print(f"\n  coverage × sim crosstab [{label}]:")
    header = "  " + " " * 8 + "  ".join(f"{s:>10s}" for s in sim_keys) + "  total"
    print(header)
    for c in cov_keys:
        row = grid[c]
        total = sum(row.values())
        cells = "  ".join(f"{row[s]:>10d}" for s in sim_keys)
        print(f"  {c:<8s}{cells}  {total:>5d}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("paths", nargs="+", type=Path)
    args = p.parse_args()

    raw_loaded = [(label, rows) for label, rows in (_load(p) for p in args.paths)]
    # Pre-filter degenerate-generation pairs once; pass the cleaned
    # rows to bad-B detection and crosstab so they all use the same
    # set as the headline.
    loaded = []
    for label, rows in raw_loaded:
        clean = [
            r
            for r in rows
            if not _is_degenerate(r.get("fresh_text", "") or "")
            and not _is_degenerate(r.get("reused_text", "") or "")
        ]
        loaded.append((label, rows, clean))

    for label, rows, clean in loaded:
        _print_headline(label, rows)
        _print_bad_bs(label, clean)
        _print_crosstab(label, clean)


if __name__ == "__main__":
    main()
