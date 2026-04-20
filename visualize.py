#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "matplotlib>=3.8",
# ]
# ///
"""Plot reagent panel results.

Usage:
    uv run --script visualize.py                 # reads results/*.json
    uv run --script visualize.py --out out.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _short(model_id: str) -> str:
    tail = model_id.split("/")[-1]
    return tail.replace("-Instruct", "").replace("-it", "")


def load_results(dir_path: Path):
    rows = []
    for f in sorted(dir_path.glob("*.json")):
        data = json.loads(f.read_text())
        drifts = sorted(int(d) for d in data["per_drift"])
        series = {
            "model": data["model"],
            "drifts": drifts,
            "n_examples": data.get("n_examples", "?"),
        }
        for key in (
            "mean_kl",
            "stdev_kl",
            "agree_rate",
            "mean_top5_overlap",
            "mean_fresh_entropy",
            "mean_sim_fresh_reused",
            "mean_sim_reused_reference",
            "mean_actual_delta",
        ):
            series[key] = [
                data["per_drift"][str(d)].get(key, float("nan")) for d in drifts
            ]
        rows.append(series)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="results")
    p.add_argument("--out", default="results/reagent_panel.png")
    args = p.parse_args()

    rows = load_results(Path(args.results_dir))
    if not rows:
        raise SystemExit(f"no *.json files in {args.results_dir}")

    # Color each model consistently across subplots.
    cmap = plt.get_cmap("tab10")
    colors = {r["model"]: cmap(i % 10) for i, r in enumerate(rows)}

    n_examples = rows[0].get("n_examples", "?")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=True)
    fig.suptitle(
        f"reagent — tool-result KV cache reuse sensitivity  "
        f"(N={n_examples} Hermes examples, L=128 chunk)",
        fontsize=14,
    )

    # Safety bands. Green = safe to reuse, yellow = borderline, red = unsafe.
    # Thresholds come from the Passing Bar section of the README.
    # (lower, mid, upper): [lower, mid) = red, [mid, upper) = yellow,
    # [upper, top] = green.
    SAFE_GREEN = "#c8e6c9"
    SAFE_YELLOW = "#f5deb3"
    SAFE_RED = "#f4a6a6"

    panels = [
        (
            "mean_sim_fresh_reused",
            "Cosine similarity (fresh vs reused)",
            "cos-sim  (1.0 = identical meaning)",
            (0.0, 1.02),
            (0.80, 0.95, 1.02),
        ),
        (
            "agree_rate",
            "Top-1 agreement rate",
            "fraction of examples where argmax token matches",
            (0.0, 1.05),
            (0.70, 0.90, 1.05),
        ),
    ]

    for (key, title, ylabel, ylim, bands), ax in zip(panels, axes):
        red_hi, yellow_hi, green_hi = bands
        ax.axhspan(ylim[0], red_hi, color=SAFE_RED, alpha=0.6, zorder=0)
        ax.axhspan(red_hi, yellow_hi, color=SAFE_YELLOW, alpha=0.6, zorder=0)
        ax.axhspan(yellow_hi, green_hi, color=SAFE_GREEN, alpha=0.6, zorder=0)
        for r in rows:
            xs = r["mean_actual_delta"]
            ys = r[key]
            c = colors[r["model"]]
            label = _short(r["model"])
            ax.plot(xs, ys, marker="o", linewidth=2.0, color=c, label=label, zorder=3)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3, zorder=1)
        ax.set_ylim(*ylim)
        ax.set_xlabel("actual drift Δ (tokens)")

    # Put a single legend underneath all plots.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(len(rows), 6),
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
