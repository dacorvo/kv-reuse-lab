#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch>=2.4,<2.11",
#   "accelerate>=1.0",
#   "transformers>=5.5",
#   "kernels>=0.5",
#   "datasets>=2.20",
#   "sentencepiece",
#   "jinja2>=3.0",
# ]
# ///
"""Attention-locality experiment: write-time predictor for splice safety.

Hypothesis: a chunk's K/V is intrinsically cacheable if the model's
attention at the chunk positions is concentrated locally (within a
recency window) rather than spread across the entire prefix. When
attention is local, K/V is content-determined and approximately
prefix-invariant; when attention is broad, K/V depends on the full
prior context and splicing across sessions transfers stale conditioning.

For each session in the dataset, we:
    1. Run Gemma-4 forward with output_attentions=True, truncating
       to ``--max-positions`` tokens (default 2800 — covers the body
       match band we care about; saves memory).
    2. On *full-attention* layers only (sliding-window layers are
       trivially local by construction), compute per-position
       locality(p) = sum_h sum_{q in [p-W, p]} attn[h, p, q] / n_heads
       where W = ``--locality-window`` (default 256).
    3. Average across full layers → per-position scalar.
    4. Save the per-position vector to a sidecar JSON.

After all sessions are processed, correlate each pair's locality
(averaged over A's match positions and B's match positions) with the
measured ``sim_fresh_reused`` from an existing results file.

Usage:
    uv run --script experiment_attention_locality.py \\
        --model google/gemma-4-E4B-it \\
        --dataset nemotron-swe --repo pandas-dev \\
        --n-sessions 20 --max-tokens-per-session 12000 \\
        --max-positions 2800 \\
        --results-json results/multi_splice_b_pandas_gemma4_e4b.json \\
        --output results/attention_locality_pandas_gemma4_e4b.json
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch

sys.path.insert(0, str(Path(__file__).parent))

from measure_multi_splice import load_sessions  # noqa: E402
from model_loading import add_model_args, load_model, load_tokenizer  # noqa: E402


_DIGIT_RUN = re.compile(r"(\d)\1{9,}")


def _is_degenerate(r: dict) -> bool:
    return bool(
        _DIGIT_RUN.search(r.get("fresh_text", "") or "")
        or _DIGIT_RUN.search(r.get("reused_text", "") or "")
    )


def _full_layer_mask(model) -> List[bool]:
    """Return a list, one bool per layer, True iff layer is a
    full-attention layer (False for sliding-window). Falls back to
    "all full" if config doesn't expose layer_types.
    """
    for root in (getattr(model, "model", None), model):
        if root is None:
            continue
        for attr in ("language_model", ""):
            inner = getattr(root, attr, None) if attr else root
            if inner is None:
                continue
            cfg = getattr(inner, "config", None)
            types = getattr(cfg, "layer_types", None) if cfg is not None else None
            if types:
                return [t != "sliding_attention" for t in types]
    return None  # type: ignore


@torch.no_grad()
def _compute_locality(
    model,
    input_ids: torch.Tensor,
    full_layer_mask: List[bool] | None,
    locality_window: int,
    band_start: int,
    band_end: int,
) -> List[float]:
    """Run a forward pass and return per-position locality scores for
    positions in [band_start, band_end). Locality is averaged across
    full-attention layers and across heads.

    Locality(p) = sum_{q = max(0, p-W) .. p} P_attn(p attends to q),
    averaged across heads and full layers. Lies in [0, 1]; 1 means
    "all attention is within the last W tokens", 0 means "attention is
    entirely beyond the recency window".
    """
    device = next(model.parameters()).device
    ids = input_ids.unsqueeze(0).to(device)
    out = model(input_ids=ids, output_attentions=True, use_cache=False)
    attns = out.attentions  # tuple of (1, heads, seq, seq)
    seq_len = ids.shape[1]
    band_end = min(band_end, seq_len)
    if band_end <= band_start:
        return []
    n_layers = len(attns)
    if full_layer_mask is None:
        full_layer_mask = [True] * n_layers
    full_idx = [i for i, b in enumerate(full_layer_mask) if b]
    if not full_idx:
        full_idx = list(range(n_layers))  # fall back

    n_pos = band_end - band_start
    locality = torch.zeros(n_pos, device=device)
    entropy = torch.zeros(n_pos, device=device)
    eff_support = torch.zeros(
        n_pos, device=device
    )  # 1 / sum(p^2): "effective # of attended tokens"
    for li in full_idx:
        a = attns[li]  # (1, h, seq, seq)
        a_band = a[0, :, band_start:band_end, :].float()  # (h, n_pos, seq) for numerics
        # local mass with locality_window
        positions = torch.arange(band_start, band_end, device=device)
        q_indices = torch.arange(seq_len, device=device).unsqueeze(0)
        lo = (positions - locality_window).clamp(min=0).unsqueeze(1)
        hi = positions.unsqueeze(1) + 1
        mask = (q_indices >= lo) & (q_indices < hi)
        masked = a_band * mask.unsqueeze(0).to(a_band.dtype)
        local_per_head = masked.sum(dim=-1)  # (h, n_pos)
        locality += local_per_head.mean(dim=0)
        # entropy per position: -sum(p log p) summed over q, averaged over heads
        # (well-defined on attention probs which sum to 1)
        eps = 1e-12
        ent_per_head = -(a_band * (a_band + eps).log()).sum(dim=-1)  # (h, n_pos)
        entropy += ent_per_head.mean(dim=0)
        # 1 / sum(p^2) — effective support size; tells how many tokens
        # the attention is spread over (independent of vocab size)
        eff_per_head = 1.0 / ((a_band**2).sum(dim=-1) + eps)  # (h, n_pos)
        eff_support += eff_per_head.mean(dim=0)
    locality /= len(full_idx)
    entropy /= len(full_idx)
    eff_support /= len(full_idx)
    # Sync all devices BEFORE freeing — async attention kernels on
    # other GPUs may still be writing into the very tensors we're
    # about to release. Same fix as in measure_multi_splice.
    if torch.cuda.is_available():
        for d in range(torch.cuda.device_count()):
            torch.cuda.synchronize(d)
    result = {
        "locality": locality.detach().float().cpu().tolist(),
        "entropy": entropy.detach().float().cpu().tolist(),
        "eff_support": eff_support.detach().float().cpu().tolist(),
    }
    del attns, out, locality, entropy, eff_support
    return result


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
    add_model_args(p)
    p.add_argument("--dataset", default="swe-smith")
    p.add_argument("--repo", default="django")
    p.add_argument("--n-sessions", type=int, default=20)
    p.add_argument("--max-tokens-per-session", type=int, default=12000)
    p.add_argument(
        "--max-positions",
        type=int,
        default=2800,
        help="Truncate each session to this many tokens before the "
        "forward. Saves attention memory; must cover the band we care "
        "about (typically 2300-2800 for nemotron-pandas).",
    )
    p.add_argument(
        "--band-start",
        type=int,
        default=1900,
        help="Lowest position to record locality for.",
    )
    p.add_argument(
        "--band-end",
        type=int,
        default=2800,
        help="Highest position (exclusive) to record locality for.",
    )
    p.add_argument(
        "--locality-window",
        type=int,
        default=256,
        help="Recency window W in tokens.",
    )
    p.add_argument(
        "--results-json",
        type=Path,
        required=True,
        help="Existing multi_splice_b results JSON to correlate against.",
    )
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    print("[info] loading tokenizer + sessions", flush=True)
    tok = load_tokenizer(args)
    sessions = load_sessions(
        tok,
        args.n_sessions,
        args.repo,
        args.max_tokens_per_session,
        dataset=args.dataset,
    )
    print(
        f"[info] {len(sessions)} sessions; truncating each to {args.max_positions} tokens for forward"
    )
    # Eager attention so output_attentions=True works.
    args.attn_impl = "eager"
    print("[info] loading model with attn_implementation=eager", flush=True)
    model = load_model(args)
    model.eval()
    full_mask = _full_layer_mask(model)
    if full_mask is None:
        print("[warn] could not detect layer_types; treating all layers as full")
    else:
        n_full = sum(full_mask)
        print(f"[info] detected {n_full}/{len(full_mask)} full-attention layers")

    # Compute per-session locality
    per_session_locality: Dict[str, Dict[str, List[float]]] = {}
    t0 = time.time()
    for idx, (iid, ids) in enumerate(sessions):
        clipped = ids[: args.max_positions]
        if len(clipped) < args.band_end:
            print(
                f"[warn] {iid} only {len(clipped)} tokens; band may be partial",
                flush=True,
            )
        try:
            ten = torch.tensor(clipped, dtype=torch.long)
            metrics = _compute_locality(
                model,
                ten,
                full_mask,
                args.locality_window,
                args.band_start,
                args.band_end,
            )
            per_session_locality[iid] = metrics
            loc = metrics["locality"]
            ent = metrics["entropy"]
            elapsed = time.time() - t0
            print(
                f"[info] [{idx + 1}/{len(sessions)}]  {iid[-25:]:25s}  "
                f"loc mean={sum(loc) / max(1, len(loc)):.3f}  "
                f"ent mean={sum(ent) / max(1, len(ent)):.2f}  "
                f"elapsed={elapsed:.0f}s",
                flush=True,
            )
        except torch.cuda.OutOfMemoryError as e:
            print(f"[err] OOM on {iid}: {e}", flush=True)
            torch.cuda.empty_cache()
            per_session_locality[iid] = {
                "locality": [],
                "entropy": [],
                "eff_support": [],
            }

        # Save partial results every iteration so a crash doesn't waste work.
        partial = {
            "model": args.model,
            "dataset": args.dataset,
            "repo": args.repo,
            "band_start": args.band_start,
            "band_end": args.band_end,
            "locality_window": args.locality_window,
            "max_positions": args.max_positions,
            "per_session": per_session_locality,
        }
        args.output.write_text(json.dumps(partial, indent=2))

    # Correlate with results
    results = json.loads(args.results_json.read_text())
    rows = [r for r in results.get("per_pair", []) if not _is_degenerate(r)]
    print(f"\n[info] {len(rows)} non-degenerate pairs to correlate")

    pair_records = []
    for r in rows:
        spans = r.get("match_spans") or []
        if not spans:
            continue
        bs, be, as_, ae = spans[-1]
        a_metrics = per_session_locality.get(
            r["a_id"], {"locality": [], "entropy": [], "eff_support": []}
        )
        b_metrics = per_session_locality.get(
            r["b_id"], {"locality": [], "entropy": [], "eff_support": []}
        )

        def mean_slice(vec, start, end, band_start):
            if not vec:
                return float("nan")
            i = max(0, start - band_start)
            j = min(len(vec), end - band_start)
            if j <= i:
                return float("nan")
            return sum(vec[i:j]) / (j - i)

        rec = {
            "a_id": r["a_id"],
            "b_id": r["b_id"],
            "sim_fresh_reused": r["sim_fresh_reused"],
            "coverage_frac": r["coverage_frac"],
            "n_matches": r["n_matches"],
            "last_span_b": [bs, be],
            "last_span_a": [as_, ae],
        }
        for metric in ("locality", "entropy", "eff_support"):
            rec[f"a_{metric}"] = mean_slice(a_metrics[metric], as_, ae, args.band_start)
            rec[f"b_{metric}"] = mean_slice(b_metrics[metric], bs, be, args.band_start)
        pair_records.append(rec)

    out = {
        "model": args.model,
        "dataset": args.dataset,
        "repo": args.repo,
        "band_start": args.band_start,
        "band_end": args.band_end,
        "locality_window": args.locality_window,
        "max_positions": args.max_positions,
        "per_session": per_session_locality,
        "pair_records": pair_records,
    }
    args.output.write_text(json.dumps(out, indent=2))
    print(f"[info] wrote {args.output}", flush=True)

    # Print correlation summary across all metrics
    def isnum(v):
        return v == v and v is not None

    valid = [
        r for r in pair_records if isnum(r["a_locality"]) and isnum(r["b_locality"])
    ]
    print(f"\n=== correlation summary ({len(valid)} pairs with valid metrics) ===")
    if valid:
        sims = [r["sim_fresh_reused"] for r in valid]
        for metric in ("locality", "entropy", "eff_support"):
            a_vals = [r[f"a_{metric}"] for r in valid]
            b_vals = [r[f"b_{metric}"] for r in valid]
            mn_vals = [min(a, b) for a, b in zip(a_vals, b_vals)]
            mx_vals = [max(a, b) for a, b in zip(a_vals, b_vals)]
            print(f"  {metric}:")
            print(f"    a_{metric:<11s} ↔ sim : r = {_pearson(a_vals, sims):+.3f}")
            print(f"    b_{metric:<11s} ↔ sim : r = {_pearson(b_vals, sims):+.3f}")
            print(
                f"    min({metric:<8s})    ↔ sim : r = {_pearson(mn_vals, sims):+.3f}"
            )
            print(
                f"    max({metric:<8s})    ↔ sim : r = {_pearson(mx_vals, sims):+.3f}"
            )

    print(f"[info] total elapsed {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
