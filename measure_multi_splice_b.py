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
#   "sentence-transformers>=3.0",
#   "jinja2>=3.0",
# ]
# ///
"""Multi-segment shifted KV reuse — Scheme B (chunked prefill with skip).

Companion to ``measure_multi_splice.py`` (Scheme A). Same dataset, same
match-finding, same metrics; the difference is *where* and *how* the
splice lands in the prefill timeline.

Scheme A (post-prefill overwrite):
    1. Batched prefill of B up to the end of its last match.
    2. Overwrite each matched span with shifted A K/V.
    3. *One-token-at-a-time* forward the suffix tokens.

Scheme B (this script) — variable-chunk chunked prefill with skip:
    Walk B left-to-right.
      For each match span, in order:
        a. Batched prefill the gap B[prev_end : b_start] (real model
           forward).
        b. Manually extend the cache by (b_end - b_start) entries
           WITHOUT a model forward:
             - Full-attention layers: append shifted A K/V (true skip,
               saves the full attention compute on the match span).
             - Sliding-attention layers: append zeros. The sliding
               cache only retains the last sliding_window-1 tokens
               anyway, so the placeholder is evicted within the next
               sliding-window-1 tokens of subsequent gap prefill. The
               only window where the zeros affect attention output is
               the immediate next ~511 tokens of the next gap on the
               sliding layers — bounded noise.
      After the last match, batched prefill the suffix B[last_b_end : T].
      Read logits at the last position; greedy-decode.

Why bother:
  * The gap forward in Scheme A is *per-token* on Gemma-family hybrid
    attention (heterogeneous per-layer cache lengths force one-token
    forwards), and dominates wallclock — ~10 min/pair on Gemma-4 E4B.
    Scheme B replaces it with batched prefills of every gap segment
    AND skips compute on match spans entirely.
  * The gap tokens in Scheme A were prefilled BEFORE the splice, so
    their K/V condition on the original B context — the splice
    arrives underneath them. In Scheme B the splice lands first, and
    each subsequent gap prefills under attention to the already-shifted
    match content, which is what a real gap-stitching engine would do.

The fixes from Scheme A (cross-GPU sync after every prefill, no
empty_cache in the hot path, KV writes via `kv_cache.write_kv_span`)
are carried over.

Usage:
    ./measure_multi_splice_b.py --model google/gemma-4-E4B-it \\
        --repo django --n-sessions 10 --min-match 128 \\
        --output results/multi_splice_b_django.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F

from aggregation import percentile as _percentile
from generation import greedy_continue
from kv_cache import num_layers as _num_layers
from measure_multi_splice import (
    _sliding_window,
    find_matches,
    fresh_forward,
    load_sessions,
    prefill_and_snapshot,
)
from model_loading import add_model_args, load_model, load_tokenizer
from rope_shift import shift_k_rope
from similarity import load_embedder_and_cos_sim


def _sync_all() -> None:
    if torch.cuda.is_available():
        for d in range(torch.cuda.device_count()):
            torch.cuda.synchronize(d)


def _inject_cached_span(past, model, b_start, b_end, a_start, a_end, cached_a_kvs):
    """Manually extend each layer's cache by appending shifted A K/V at
    absolute positions [b_start, b_end). No model forward — true skip.

    cached_a_kvs is the full-sequence A snapshot (every layer has the
    full T_A entries, including sliding layers — chunked prefill in
    prefill_and_snapshot ensures this). For each layer we slice
    [a_start, a_end), shift K's RoPE phase by (b_start - a_start), and
    call ``past.layers[li].update`` so the cache class handles
    truncation / cumulative_length bookkeeping consistently for both
    DynamicLayer and DynamicSlidingWindowLayer.
    """
    shift = b_start - a_start
    span_len = b_end - b_start
    for li in range(_num_layers(past)):
        layer = past.layers[li]
        kA = cached_a_kvs[li][0][..., a_start:a_end, :]
        vA = cached_a_kvs[li][1][..., a_start:a_end, :]
        if kA.shape[-2] != span_len:
            # Defensive: cached_a's slice somehow shorter than expected.
            continue
        # Move to the same device/dtype the layer's cache lives on.
        # If the cache hasn't been initialised yet (first segment is a
        # match at position 0), fall back to the model's parameter
        # device.
        if layer.is_initialized:
            tgt_device = layer.keys.device
            tgt_dtype = layer.keys.dtype
        else:
            tgt_device = next(model.parameters()).device
            tgt_dtype = next(model.parameters()).dtype
        kA_dev = kA.to(tgt_device, tgt_dtype)
        vA_dev = vA.to(tgt_device, tgt_dtype)
        if shift != 0:
            kA_dev = shift_k_rope(kA_dev, model, li, shift)
        layer.update(kA_dev, vA_dev)
    _sync_all()


@torch.no_grad()
def multi_splice_b_forward(
    model,
    session_b_ids: torch.Tensor,
    cached_a_kvs,
    matches: List[Tuple[int, int, int, int]],
    gen_tokens: int,
    stop_ids: set,
):
    """Scheme B: chunked prefill of gaps + manual cache injection for
    matches. See module docstring.
    """
    device = next(model.parameters()).device
    T = session_b_ids.shape[0]
    matches = sorted(matches, key=lambda m: m[0])
    sw = _sliding_window(model)
    chunk = (sw - 1) if sw is not None else max(T, 1)

    past = None
    last_logits = None

    def prefill(start: int, end: int):
        """Chunked-prefill B[start:end) (chunks aligned on sliding window
        when present). Updates ``past`` and ``last_logits``.
        """
        nonlocal past, last_logits
        s = start
        while s < end:
            e = min(s + chunk, end)
            ids = session_b_ids[s:e].unsqueeze(0).to(device)
            positions = torch.arange(s, e, device=device)
            out = model(
                input_ids=ids,
                past_key_values=past,
                position_ids=positions.unsqueeze(0),
                cache_position=positions,
                use_cache=True,
                logits_to_keep=1,
            )
            past = out.past_key_values
            last_logits = out.logits[0, -1].detach()
            # Sync BEFORE `del out`: prefill kernels may still be writing
            # to per-layer KV across cuda:0..N async, and we don't want
            # any downstream teardown to race with them.
            _sync_all()
            del out
            s = e

    cur = 0
    for b_start, b_end, a_start, a_end in matches:
        # Gap before this match.
        prefill(cur, b_start)
        # Match span — directly inject the cached A K/V at positions
        # [b_start, b_end), no model forward. If past is still None
        # (rare: first segment is a match starting at 0), prefill the
        # match span instead so the cache gets initialised the normal
        # way; subsequent matches go through the fast path.
        if past is None:
            prefill(b_start, b_end)
        else:
            _inject_cached_span(
                past, model, b_start, b_end, a_start, a_end, cached_a_kvs
            )
        cur = b_end
    # Suffix after the last match.
    prefill(cur, T)

    log = last_logits.float().cpu()
    first = int(log.argmax().item())
    if gen_tokens > 0:
        gen_ids = greedy_continue(
            model,
            past,
            first_token=first,
            start_pos=T,
            max_new=gen_tokens,
            stop_token_ids=stop_ids,
        )
    else:
        gen_ids = [first]
    return log, gen_ids


def run(args):
    tok = load_tokenizer(args)

    print(f"[info] loading up to {args.n_sessions} {args.repo}* sessions", flush=True)
    sessions = load_sessions(
        tok, args.n_sessions, args.repo, args.max_tokens_per_session
    )
    print(
        f"[info] collected {len(sessions)} sessions "
        f"(avg {sum(len(s[1]) for s in sessions) // len(sessions)} tokens)",
        flush=True,
    )

    model = load_model(args)

    cos_sim = load_embedder_and_cos_sim(args.embedder)

    stop_ids = set()
    for attr in ("eos_token_id", "pad_token_id"):
        tid = getattr(tok, attr, None)
        if tid is not None:
            if isinstance(tid, (list, tuple)):
                stop_ids.update(int(x) for x in tid)
            else:
                stop_ids.add(int(tid))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_path.with_suffix(".jsonl")
    done_pairs = set()
    if jsonl_path.exists():
        with jsonl_path.open() as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                done_pairs.add((r["a_id"], r["b_id"]))
        print(
            f"[info] resume: {len(done_pairs)} pairs already in {jsonl_path}",
            flush=True,
        )

    a_cache = {}
    rows: List[dict] = []

    for bi, (b_id, b_toks) in enumerate(sessions):
        candidate_a_ids = [a_id for a_id, _ in sessions[:bi]]
        if candidate_a_ids and all(
            (a_id, b_id) in done_pairs for a_id in candidate_a_ids
        ):
            print(f"[info] skip B={bi:02d} (all pairs cached)", flush=True)
            continue

        session_b_ids = torch.tensor(b_toks, dtype=torch.long)
        fresh_log, fresh_gen, fresh_past = fresh_forward(
            model, session_b_ids, args.gen_tokens, stop_ids
        )
        fresh_text = tok.decode(fresh_gen)
        del fresh_past
        torch.cuda.empty_cache()

        for ai, (a_id, a_toks) in enumerate(sessions[:bi]):
            if (a_id, b_id) in done_pairs:
                continue
            matches = find_matches(b_toks, a_toks, args.min_match)
            if not matches:
                continue
            if a_id not in a_cache:
                a_cache[a_id] = prefill_and_snapshot(
                    model, torch.tensor(a_toks, dtype=torch.long)
                )
            cached_a = a_cache[a_id]
            try:
                reused_log, reused_gen = multi_splice_b_forward(
                    model,
                    session_b_ids,
                    cached_a,
                    matches,
                    args.gen_tokens,
                    stop_ids,
                )
            except Exception as e:  # noqa: BLE001
                print(
                    f"[warn] multi_splice_b failed for {a_id}→{b_id}: {e}",
                    flush=True,
                )
                continue
            reused_text = tok.decode(reused_gen)
            log_p = F.log_softmax(fresh_log, dim=-1)
            log_q = F.log_softmax(reused_log, dim=-1)
            kl = (log_p.exp() * (log_p - log_q)).sum().item()
            top1_fresh = int(fresh_log.argmax().item())
            top1_reused = int(reused_log.argmax().item())
            sim = cos_sim(fresh_text, reused_text)
            covered = sum(b_end - b_start for b_start, b_end, _, _ in matches)
            row = {
                "a_id": a_id,
                "b_id": b_id,
                "b_len": len(b_toks),
                "a_len": len(a_toks),
                "n_matches": len(matches),
                "match_spans": [
                    [int(bs), int(be), int(as_), int(ae)] for bs, be, as_, ae in matches
                ],
                "covered_tokens": covered,
                "coverage_frac": covered / max(1, len(b_toks)),
                "longest_match": max(
                    b_end - b_start for b_start, b_end, _, _ in matches
                ),
                "kl": kl,
                "top1_fresh": top1_fresh,
                "top1_reused": top1_reused,
                "agree": int(top1_fresh == top1_reused),
                "sim_fresh_reused": sim,
                "fresh_text": fresh_text[:256],
                "reused_text": reused_text[:256],
            }
            rows.append(row)
            with jsonl_path.open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(
                f"[info] B={bi:02d} A={ai:02d}  matches={len(matches):>2}  "
                f"cov={row['coverage_frac']:.2f}  KL={kl:.2f}  sim={sim:.2f}",
                flush=True,
            )

        if args.dispose_a_caches_every and (bi + 1) % args.dispose_a_caches_every == 0:
            a_cache.clear()
            torch.cuda.empty_cache()

    # Aggregate. Read JSONL (includes any prior-resumed rows).
    all_rows: List[dict] = []
    if jsonl_path.exists():
        with jsonl_path.open() as f:
            for line in f:
                try:
                    all_rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    summary = {}
    for key in ("coverage_frac", "kl", "agree", "sim_fresh_reused", "n_matches"):
        xs = [r[key] for r in all_rows]
        if xs:
            xs_sorted = sorted(xs)
            summary[key] = {
                "n": len(xs),
                "mean": statistics.mean(xs),
                "median": statistics.median(xs),
                "p10": _percentile(xs_sorted, 0.10),
                "p90": _percentile(xs_sorted, 0.90),
            }
    aggregate = {
        "model": args.model,
        "scheme": "B",
        "dataset_repo": args.repo,
        "min_match": args.min_match,
        "n_sessions": args.n_sessions,
        "n_pairs": len(all_rows),
        **{k: v for k, v in summary.items()},
        "per_pair": all_rows,
    }
    out_path.write_text(json.dumps(aggregate, indent=2))
    print(f"[info] wrote {out_path}  ({len(all_rows)} pairs)", flush=True)

    print()
    print("=== summary ===")
    for key in ("coverage_frac", "kl", "agree", "sim_fresh_reused", "n_matches"):
        s = summary.get(key)
        if s and s.get("n"):
            print(
                f"{key:<20}  mean={s['mean']:.2f}  median={s['median']:.2f}  "
                f"p10={s['p10']:.2f}  p90={s['p90']:.2f}"
            )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    add_model_args(p)
    p.add_argument("--repo", default="django")
    p.add_argument("--n-sessions", type=int, default=10)
    p.add_argument("--max-tokens-per-session", type=int, default=15000)
    p.add_argument("--min-match", type=int, default=128)
    p.add_argument("--gen-tokens", type=int, default=64)
    p.add_argument("--embedder", default="BAAI/bge-small-en-v1.5")
    p.add_argument(
        "--dispose-a-caches-every",
        type=int,
        default=3,
        help="Clear the cached per-A KV snapshot dict every N B iterations "
        "to bound memory. Set 0 to keep all in memory.",
    )
    p.add_argument("--output", required=True)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
