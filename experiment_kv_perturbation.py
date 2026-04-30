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
"""K/V prefix-perturbation: ground-truth measure of cacheability.

For a candidate body-match chunk in session A, the question "is this
chunk's K/V intrinsically cacheable" can be answered directly by:

    1. Forward A's full sequence → record K/V at the chunk positions.
       Call this K_real.
    2. Construct a perturbed sequence: keep A[chunk_start:chunk_end]
       in place at the same absolute position, but replace the prefix
       A[0:chunk_start] with the prefix from a DIFFERENT session.
       Forward and record K/V at the (same) chunk positions. Call
       this K_perturbed.
    3. The relative L2 distance ||K_real - K_perturbed|| / ||K_real||
       measures how much the chunk's K/V depends on the specific
       prefix vs the chunk content alone. A small distance means the
       K/V is intrinsically cacheable; large means context-dependent.

This is the actual property the user proposed: "identify cacheable
content when we have the full encoded prefix." Run on each session,
correlate the per-session K/V perturbation distance with each pair's
measured ``sim_fresh_reused`` (averaged across the perturbing-prefix
sessions, as a per-session score).

We use a *small* prefix-perturbation budget (3 alternative prefixes
per chunk) to keep cost manageable: 4 forwards per session.
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

from kv_cache import layer_kv as _layer_kv_tensors  # noqa: E402
from kv_cache import num_layers as _num_layers  # noqa: E402
from measure_multi_splice import load_sessions  # noqa: E402
from model_loading import add_model_args, load_model, load_tokenizer  # noqa: E402


_DIGIT_RUN = re.compile(r"(\d)\1{9,}")


def _is_degenerate(r: dict) -> bool:
    return bool(
        _DIGIT_RUN.search(r.get("fresh_text", "") or "")
        or _DIGIT_RUN.search(r.get("reused_text", "") or "")
    )


def _sync_all() -> None:
    if torch.cuda.is_available():
        for d in range(torch.cuda.device_count()):
            torch.cuda.synchronize(d)


@torch.no_grad()
def _kv_at_chunk(model, input_ids: torch.Tensor, chunk_start: int, chunk_end: int):
    """Forward and return per-layer K/V slices at [chunk_start, chunk_end),
    as a list of (K, V) pairs on CPU.
    """
    device = next(model.parameters()).device
    ids = input_ids.unsqueeze(0).to(device)
    out = model(input_ids=ids, use_cache=True)
    past = out.past_key_values
    _sync_all()
    n_layers = _num_layers(past)
    kvs = []
    for li in range(n_layers):
        k, v = _layer_kv_tensors(past, li)
        # k, v shape: (1, n_heads, seq, head_dim) or with sliding window
        # truncation. We slice the chunk band and copy to CPU.
        if k.shape[-2] >= chunk_end:
            k_slice = k[..., chunk_start:chunk_end, :].detach().cpu().float()
            v_slice = v[..., chunk_start:chunk_end, :].detach().cpu().float()
        else:
            # On sliding-window layers the cache may be shorter than
            # the full sequence; skip these (they truncate distant
            # history anyway, so prefix-perturbation only meaningfully
            # affects full layers).
            k_slice = None
            v_slice = None
        kvs.append((k_slice, v_slice))
    del past, out
    return kvs


def _full_layer_mask(model) -> List[bool] | None:
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
    return None


def _kv_relative_distance(kvs_a, kvs_b, full_mask=None) -> float:
    """Average relative L2 distance ||a-b||/||a|| over full-attention
    layers (or all if no mask), keys and values combined.
    """
    n_layers = len(kvs_a)
    rel = []
    for li in range(n_layers):
        if full_mask is not None and not full_mask[li]:
            continue
        ka, va = kvs_a[li]
        kb, vb = kvs_b[li]
        if ka is None or kb is None:
            continue
        for x, y in ((ka, kb), (va, vb)):
            num = (x - y).flatten().norm().item()
            den = max(x.flatten().norm().item(), 1e-9)
            rel.append(num / den)
    if not rel:
        return float("nan")
    return sum(rel) / len(rel)


def _pearson(xs: List[float], ys: List[float]) -> float:
    n = len(xs)
    if n < 3:
        return float("nan")
    mx, my = statistics.mean(xs), statistics.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    return num / (dx * dy) if dx * dy else float("nan")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_model_args(p)
    p.add_argument("--dataset", default="swe-smith")
    p.add_argument("--repo", default="django")
    p.add_argument("--n-sessions", type=int, default=20)
    p.add_argument("--max-tokens-per-session", type=int, default=12000)
    p.add_argument("--max-positions", type=int, default=4500)
    p.add_argument(
        "--n-perturb",
        type=int,
        default=3,
        help="How many alternative-prefix perturbations per session.",
    )
    p.add_argument("--results-json", type=Path, required=True)
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
    iid_to_toks = {iid: ids for iid, ids in sessions}

    args.attn_impl = "eager"
    print("[info] loading model", flush=True)
    model = load_model(args)
    model.eval()
    full_mask = _full_layer_mask(model)
    if full_mask:
        print(f"[info] {sum(full_mask)}/{len(full_mask)} full-attention layers")

    # Read existing results to get the body-match span per session
    # (use the LAST span of any pair where this session is B).
    results = json.loads(args.results_json.read_text())
    rows = [r for r in results.get("per_pair", []) if not _is_degenerate(r)]
    # Build per-B "canonical body-match span" by picking the one with
    # the most matches found (consistent across pairs).
    span_by_b: Dict[str, tuple] = {}
    for r in rows:
        spans = r.get("match_spans") or []
        if not spans:
            continue
        bs, be, *_ = spans[-1]
        if (
            r["b_id"] not in span_by_b
            or be - bs > span_by_b[r["b_id"]][1] - span_by_b[r["b_id"]][0]
        ):
            span_by_b[r["b_id"]] = (bs, be)
    print(f"[info] {len(span_by_b)} sessions appear as B with a body-match span")

    # For each session, compute K_real and K under each of n_perturb
    # alternative prefixes (taken from the n_perturb subsequent
    # sessions in the load order, modulo wrap-around).
    per_session_pert: Dict[str, Dict] = {}
    iids_in_order = [iid for iid, _ in sessions]
    t0 = time.time()
    for idx, iid in enumerate(iids_in_order):
        if iid not in span_by_b:
            continue
        bs, be = span_by_b[iid]
        a_toks = iid_to_toks[iid]
        # Bound by max_positions
        if be > args.max_positions:
            print(
                f"[warn] {iid} body match ends at {be} > max_positions; skipping",
                flush=True,
            )
            continue
        a_clip = a_toks[: args.max_positions]

        try:
            kv_real = _kv_at_chunk(
                model, torch.tensor(a_clip, dtype=torch.long), bs, be
            )
        except torch.cuda.OutOfMemoryError as e:
            print(f"[err] OOM on real {iid}: {e}", flush=True)
            torch.cuda.empty_cache()
            continue

        rel_dists = []
        for k_pert in range(1, args.n_perturb + 1):
            alt_idx = (idx + k_pert) % len(iids_in_order)
            alt_iid = iids_in_order[alt_idx]
            alt_toks = iid_to_toks[alt_iid]
            if len(alt_toks) < bs:
                continue
            # Build perturbed sequence: alt's first bs tokens + a's
            # chunk content [bs:be] (so chunk content stays the same
            # at the same positions, only the prefix changes).
            pert = list(alt_toks[:bs]) + list(a_clip[bs:be])
            try:
                kv_alt = _kv_at_chunk(
                    model, torch.tensor(pert, dtype=torch.long), bs, be
                )
                d = _kv_relative_distance(kv_real, kv_alt, full_mask)
                rel_dists.append(d)
            except torch.cuda.OutOfMemoryError as e:
                print(f"[err] OOM on pert {k_pert}: {e}", flush=True)
                torch.cuda.empty_cache()

        per_session_pert[iid] = {
            "span": [bs, be],
            "rel_kv_distances": rel_dists,
            "mean_rel_kv_dist": (
                sum(rel_dists) / len(rel_dists) if rel_dists else float("nan")
            ),
        }
        elapsed = time.time() - t0
        print(
            f"[info] [{idx + 1}/{len(iids_in_order)}]  {iid[-25:]:25s}  "
            f"span=[{bs}:{be}]  n_pert={len(rel_dists)}  "
            f"mean_rel_dist={per_session_pert[iid]['mean_rel_kv_dist']:.3f}  "
            f"elapsed={elapsed:.0f}s",
            flush=True,
        )
        # Save partial.
        args.output.write_text(
            json.dumps({"model": args.model, "per_session": per_session_pert}, indent=2)
        )

    # Correlate: for each pair, look up B's mean_rel_kv_dist (the
    # write-time predictor for B's chunk). Higher distance = less
    # cacheable = expect lower sim_fresh_reused.
    pair_records = []
    for r in rows:
        b_id = r["b_id"]
        a_id = r["a_id"]
        if b_id not in per_session_pert:
            continue
        b_dist = per_session_pert[b_id]["mean_rel_kv_dist"]
        a_dist = per_session_pert.get(a_id, {}).get("mean_rel_kv_dist", float("nan"))
        pair_records.append(
            {
                "a_id": a_id,
                "b_id": b_id,
                "sim_fresh_reused": r["sim_fresh_reused"],
                "coverage_frac": r["coverage_frac"],
                "n_matches": r["n_matches"],
                "a_kv_dist": a_dist,
                "b_kv_dist": b_dist,
            }
        )

    out = {
        "model": args.model,
        "dataset": args.dataset,
        "repo": args.repo,
        "n_perturb": args.n_perturb,
        "max_positions": args.max_positions,
        "per_session": per_session_pert,
        "pair_records": pair_records,
    }
    args.output.write_text(json.dumps(out, indent=2))
    print(f"\n[info] wrote {args.output}", flush=True)

    def isnum(v):
        return v == v and v is not None

    valid = [r for r in pair_records if isnum(r["a_kv_dist"]) and isnum(r["b_kv_dist"])]
    print(f"\n=== correlation summary ({len(valid)} pairs) ===")
    if valid:
        sims = [r["sim_fresh_reused"] for r in valid]
        a_d = [r["a_kv_dist"] for r in valid]
        b_d = [r["b_kv_dist"] for r in valid]
        mx = [max(x, y) for x, y in zip(a_d, b_d)]
        mn = [min(x, y) for x, y in zip(a_d, b_d)]
        print(f"  a_kv_dist  ↔ sim_fresh_reused : r = {_pearson(a_d, sims):+.3f}")
        print(f"  b_kv_dist  ↔ sim_fresh_reused : r = {_pearson(b_d, sims):+.3f}")
        print(f"  max(a,b)   ↔ sim_fresh_reused : r = {_pearson(mx, sims):+.3f}")
        print(f"  min(a,b)   ↔ sim_fresh_reused : r = {_pearson(mn, sims):+.3f}")

    print(f"[info] total elapsed {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
