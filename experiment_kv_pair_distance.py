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
"""Per-pair K/V mismatch: measure exactly the K/V delta each splice
sees, rather than averaged "perturbation sensitivity".

For each (A, B) pair from an existing results file:
    1. Forward A's full sequence → get K_A at the body-match's
       a-side positions [as_, ae].
    2. Forward "B's prefix B[0:bs] + A's body content B[bs:be]"
       (i.e., put A's chunk content at B's actual position with B's
       actual prefix in front) → get K_pair at positions [bs, be].
    3. Distance(K_A, K_pair) is the literal K/V drift the splice
       transfers — the cache substitutes K_A in place of K_pair.

Caveat: when shift = bs - as_ ≠ 0, the chunk content's RoPE phases
differ between A's K (computed at positions as_..ae) and the
"target" K (computed at positions bs..be). The splicing harness
shifts the K's RoPE by `shift_k_rope`, which is what makes the
naive comparison a bit off. Here we just measure raw K-K distance,
so the shift contribution is part of the signal.

Hypothesis: pair-level K/V distance correlates with sim_fresh_reused
*per pair* with much higher r than the random-perturbation
"per-session cacheability score" did (which gave r ≈ -0.25 because it
was an aggregate, not the actual distance the pair sees).
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path
from typing import List

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
def _kv_at_chunk(model, input_ids: torch.Tensor, lo: int, hi: int):
    device = next(model.parameters()).device
    ids = input_ids.unsqueeze(0).to(device)
    out = model(input_ids=ids, use_cache=True)
    past = out.past_key_values
    _sync_all()
    n_layers = _num_layers(past)
    kvs = []
    for li in range(n_layers):
        k, v = _layer_kv_tensors(past, li)
        if k.shape[-2] >= hi:
            kvs.append(
                (
                    k[..., lo:hi, :].detach().cpu().float(),
                    v[..., lo:hi, :].detach().cpu().float(),
                )
            )
        else:
            kvs.append((None, None))
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


def _kv_distances(kvs_a, kvs_b, full_mask=None) -> dict:
    rel_l2 = []
    cos_dis = []
    for li in range(len(kvs_a)):
        if full_mask is not None and not full_mask[li]:
            continue
        ka, va = kvs_a[li]
        kb, vb = kvs_b[li]
        if ka is None or kb is None:
            continue
        for x, y in ((ka, kb), (va, vb)):
            xf = x.flatten().float()
            yf = y.flatten().float()
            num = (xf - yf).norm().item()
            den = max(xf.norm().item(), 1e-9)
            rel_l2.append(num / den)
            dot = (xf * yf).sum().item()
            denom = max(xf.norm().item() * yf.norm().item(), 1e-9)
            cos_dis.append(1.0 - dot / denom)
    if not rel_l2:
        return {"rel_l2": float("nan"), "cos_dis": float("nan")}
    return {
        "rel_l2": sum(rel_l2) / len(rel_l2),
        "cos_dis": sum(cos_dis) / len(cos_dis),
    }


def _pearson(xs, ys):
    if len(xs) < 3:
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

    results = json.loads(args.results_json.read_text())
    rows = [r for r in results.get("per_pair", []) if not _is_degenerate(r)]

    # Cache K_A per session, computed once per A on demand.
    a_kv_cache: dict = {}

    def get_kv_a(iid, lo, hi):
        key = (iid, lo, hi)
        if key in a_kv_cache:
            return a_kv_cache[key]
        toks = iid_to_toks[iid]
        if hi > args.max_positions:
            return None
        clip = toks[: args.max_positions]
        kv = _kv_at_chunk(model, torch.tensor(clip, dtype=torch.long), lo, hi)
        a_kv_cache[key] = kv
        return kv

    pair_records = []
    t0 = time.time()
    for ridx, r in enumerate(rows):
        spans = r.get("match_spans") or []
        if not spans:
            continue
        bs, be, as_, ae = spans[-1]
        L = be - bs
        if L != ae - as_:
            continue
        if be > args.max_positions or ae > args.max_positions:
            continue
        a_toks = iid_to_toks.get(r["a_id"])
        b_toks = iid_to_toks.get(r["b_id"])
        if a_toks is None or b_toks is None:
            continue

        kv_a = get_kv_a(r["a_id"], as_, ae)
        if kv_a is None:
            continue
        # Build "B's prefix + A's body content" sequence:
        # tokens[0:bs] = B's prefix
        # tokens[bs:be] = A's body content (which equals B's body content
        # by byte-exact match, so practically this == B's full prefix
        # truncated). Use B's actual tokens — same content.
        target_seq = list(b_toks[: args.max_positions])
        if be > len(target_seq):
            continue
        kv_target = _kv_at_chunk(
            model, torch.tensor(target_seq, dtype=torch.long), bs, be
        )
        d = _kv_distances(kv_a, kv_target, full_mask)
        pair_records.append(
            {
                "a_id": r["a_id"],
                "b_id": r["b_id"],
                "sim_fresh_reused": r["sim_fresh_reused"],
                "coverage_frac": r["coverage_frac"],
                "n_matches": r["n_matches"],
                "shift": bs - as_,
                "rel_l2": d["rel_l2"],
                "cos_dis": d["cos_dis"],
            }
        )
        elapsed = time.time() - t0
        if (ridx + 1) % 5 == 0 or ridx == 0:
            print(
                f"[info] [{ridx + 1}/{len(rows)}] sim={r['sim_fresh_reused']:.2f} "
                f"l2={d['rel_l2']:.3f} cos={d['cos_dis']:.3f}  elapsed={elapsed:.0f}s",
                flush=True,
            )
        # Save partial.
        args.output.write_text(json.dumps({"pairs": pair_records}, indent=2))

    args.output.write_text(json.dumps({"pairs": pair_records}, indent=2))
    print(f"[info] wrote {args.output}  n_pairs={len(pair_records)}", flush=True)

    # Correlation.
    sims = [r["sim_fresh_reused"] for r in pair_records]
    l2 = [r["rel_l2"] for r in pair_records]
    cos = [r["cos_dis"] for r in pair_records]
    print("\n=== per-pair K/V mismatch correlation ===")
    print(f"  rel_l2  ↔ sim : r = {_pearson(l2, sims):+.3f}")
    print(f"  cos_dis ↔ sim : r = {_pearson(cos, sims):+.3f}")
    # Also break by coverage
    high_cov = [r for r in pair_records if r["coverage_frac"] >= 0.20]
    if high_cov:
        sims_h = [r["sim_fresh_reused"] for r in high_cov]
        l2_h = [r["rel_l2"] for r in high_cov]
        cos_h = [r["cos_dis"] for r in high_cov]
        print(
            f"\nhigh-cov (≥20%) only: n={len(high_cov)}  "
            f"r(l2)={_pearson(l2_h, sims_h):+.3f}  "
            f"r(cos)={_pearson(cos_h, sims_h):+.3f}"
        )


if __name__ == "__main__":
    main()
