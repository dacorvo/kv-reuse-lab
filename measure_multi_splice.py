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
"""Multi-segment shifted KV reuse on real same-codebase agent traces.

Reagent's ``measure_reuse_drift.py`` splices ONE cached chunk into a
drifted prompt. This script extends that to MULTIPLE shifted chunks
spliced into a single new prompt — the situation a gap-stitching
serving engine would face on same-codebase agent workloads where
(per ``trace_analysis/``) each new request has several byte-exact
matches scattered through its prompt.

Scheme A (implemented here): post-prefill overwrite.
    1. Prefill session B up to the end of its last matching span
       (native / fresh for all tokens).
    2. Overwrite each matched span of B's K/V with session A's cached
       K (RoPE-shifted by the per-span position delta) and V.
    3. Forward the remaining B tokens one at a time with explicit
       cache_position so they attend to the overwritten spans.
    4. Compare logits + greedy continuation to a fully-fresh B forward.

This does not test Scheme B, where the gap tokens between splices
are themselves prefilled AFTER the splices are in place so that
their K/V reflects the shifted-chunk context. Scheme B might
compensate for the context-conditioning mismatch that Scheme A
amplifies; that's a follow-up.

Dataset source: ``SWE-bench/SWE-smith-trajectories`` filtered to one
repo (default ``django``). Each trajectory's final chat-template
rendering is one "session"; each ordered pair (A, B) of trajectories
produces 0+ matching byte-exact spans of length ≥ ``--min-match``.
Pairs with no matches are skipped.

Usage:
    ./measure_multi_splice.py --model google/gemma-4-E4B-it \\
        --repo django --n-sessions 10 --min-match 128 \\
        --output results/multi_splice_django.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from generation import forward_with_cache, greedy_continue
from kv_cache import layer_kv as _layer_kv_tensors
from kv_cache import num_layers as _num_layers
from kv_cache import write_kv_span
from aggregation import percentile as _percentile
from model_loading import add_model_args, load_model, load_tokenizer
from rope_shift import shift_k_rope
from similarity import load_embedder_and_cos_sim


@torch.no_grad()
def prefill_and_snapshot(model, input_ids: torch.Tensor, offload: str = "cpu"):
    """Run a full prefill and return per-layer (K, V) snapshots.
    By default snapshots are moved to CPU so that accumulating multiple
    session caches doesn't exhaust GPU memory; the splice path moves
    slices back to the target device just-in-time.
    """
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    out = model(input_ids=input_ids.unsqueeze(0), use_cache=True, logits_to_keep=1)
    pkv = out.past_key_values
    kvs = []
    for li in range(_num_layers(pkv)):
        k, v = _layer_kv_tensors(pkv, li)
        kvs.append((k.detach().to(offload), v.detach().to(offload)))
    del out, pkv
    torch.cuda.empty_cache()
    return kvs


# ---------------------------------------------------------------------------
# Dataset: django-filtered SWE-smith trajectories
# ---------------------------------------------------------------------------


def _chatml_msgs(row) -> List[Dict[str, str]]:
    raw_msgs = row.get("messages")
    if isinstance(raw_msgs, str):
        msgs = json.loads(raw_msgs)
    else:
        msgs = raw_msgs or []
    norm = []
    for m in msgs:
        role = m.get("role", "")
        content = m.get("content") or ""
        tc = m.get("tool_calls")
        if role == "tool":
            role = "user"
        if tc and not content:
            content = json.dumps(tc, ensure_ascii=False)
        if role in ("system", "user", "assistant"):
            norm.append({"role": role, "content": str(content)})
    return norm


def load_sessions(
    tokenizer, n_sessions: int, repo_prefix: str, max_tokens: int
) -> List[Tuple[str, List[int]]]:
    """Return a list of (instance_id, token_ids) for up to n_sessions
    django trajectories. Each token_ids is the FULL chat-rendered
    prefix of the trajectory with add_generation_prompt=True, capped
    at max_tokens.
    """
    from datasets import load_dataset

    ds = load_dataset("SWE-bench/SWE-smith-trajectories", split="tool", streaming=True)
    sessions: List[Tuple[str, List[int]]] = []
    for row in ds:
        if len(sessions) >= n_sessions:
            break
        iid = row.get("instance_id", "")
        if not iid.startswith(repo_prefix):
            continue
        msgs = _chatml_msgs(row)
        if not msgs:
            continue
        try:
            enc = tokenizer.apply_chat_template(
                msgs,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
            )
            ids = enc["input_ids"]
            if hasattr(ids, "tolist"):
                ids = ids.tolist()
            if ids and isinstance(ids[0], list):
                ids = ids[0]
            ids = list(ids)[:max_tokens]
        except Exception:
            continue
        if len(ids) < 512:
            continue
        sessions.append((iid, ids))
    if len(sessions) < n_sessions:
        raise RuntimeError(
            f"only {len(sessions)}/{n_sessions} {repo_prefix}* sessions found "
            f"(consider widening --max-tokens-per-session)"
        )
    return sessions


# ---------------------------------------------------------------------------
# Match finding
# ---------------------------------------------------------------------------


def find_matches(
    target: List[int], prior: List[int], min_n: int
) -> List[Tuple[int, int, int, int]]:
    """Return non-overlapping (b_start, b_end, a_start, a_end) match
    spans where prior[a_start:a_end] == target[b_start:b_end],
    each of length ≥ min_n. Target-side greedy.
    """
    index: Dict[int, List[int]] = {}
    if len(prior) < min_n:
        return []
    for p in range(len(prior) - min_n + 1):
        key = hash(tuple(prior[p : p + min_n]))
        index.setdefault(key, []).append(p)
    matches: List[Tuple[int, int, int, int]] = []
    T = len(target)
    i = 0
    while i + min_n <= T:
        window = target[i : i + min_n]
        cands = index.get(hash(tuple(window)))
        if not cands:
            i += 1
            continue
        best_len = 0
        best_p = -1
        for p in cands:
            if prior[p : p + min_n] != window:
                continue
            ext = min_n
            limit = min(T - i, len(prior) - p)
            while ext < limit and target[i + ext] == prior[p + ext]:
                ext += 1
            if ext > best_len:
                best_len = ext
                best_p = p
        if best_len < min_n:
            i += 1
            continue
        matches.append((i, i + best_len, best_p, best_p + best_len))
        i += best_len
    return matches


# ---------------------------------------------------------------------------
# Experiment core
# ---------------------------------------------------------------------------


@torch.no_grad()
def fresh_forward(model, input_ids: torch.Tensor, gen_tokens: int, stop_ids: set):
    """Full fresh prefill of input_ids; return (log_last, first_token,
    greedy_text_ids, past_kv)."""
    device = next(model.parameters()).device
    out = model(
        input_ids=input_ids.unsqueeze(0).to(device),
        use_cache=True,
        logits_to_keep=1,
    )
    past = out.past_key_values
    log = out.logits[0, -1].detach().float().cpu()
    first = int(log.argmax().item())
    if gen_tokens > 0:
        gen_ids = greedy_continue(
            model,
            past,
            first_token=first,
            start_pos=input_ids.shape[0],
            max_new=gen_tokens,
            stop_token_ids=stop_ids,
        )
    else:
        gen_ids = [first]
    del out
    torch.cuda.empty_cache()
    return log, gen_ids, past


@torch.no_grad()
def multi_splice_forward(
    model,
    session_b_ids: torch.Tensor,
    cached_a_kvs,
    matches: List[Tuple[int, int, int, int]],
    gen_tokens: int,
    stop_ids: set,
):
    """Scheme A. Prefill B up to the end of its last match, overwrite
    every matched span with shifted cached K from A (V copied as-is),
    then forward the remaining B tokens one at a time and greedy-decode.
    """
    device = next(model.parameters()).device
    T = session_b_ids.shape[0]
    last_b_end = max(b_end for _, b_end, _, _ in matches)
    prefill_ids = session_b_ids[:last_b_end].unsqueeze(0).to(device)
    out = model(input_ids=prefill_ids, use_cache=True, logits_to_keep=1)
    past = out.past_key_values
    del out
    torch.cuda.empty_cache()
    # Overwrite each matched span per-layer with shifted A K + A V.
    for b_start, b_end, a_start, a_end in matches:
        shift = b_start - a_start
        span_len = b_end - b_start
        for li in range(_num_layers(past)):
            k_tensor, _ = _layer_kv_tensors(past, li)
            if k_tensor.shape[-2] < b_end:
                # Sliding-window cache: the target span might not exist at
                # this layer. Skip silently.
                continue
            kA = cached_a_kvs[li][0][..., a_start:a_end, :]
            vA = cached_a_kvs[li][1][..., a_start:a_end, :]
            kA_dev = kA.to(k_tensor.device, k_tensor.dtype)
            if shift != 0:
                kA_dev = shift_k_rope(kA_dev, model, li, shift)
            # Guard on span length for sliding-window where a_end - a_start
            # might have been clipped in cached_a_kvs.
            if kA_dev.shape[-2] != span_len:
                continue
            write_kv_span(past, model, li, slice(b_start, b_end), kA_dev, vA)
    # Forward the remaining tokens of B one at a time so they attend
    # to the (partially overwritten) cache.
    remaining = session_b_ids[last_b_end:T]
    if remaining.shape[0] > 0:
        logits = forward_with_cache(model, remaining, past, trigger_position=last_b_end)
        log = logits[-1].float().cpu()
    else:
        # The last match reaches the very end of B; re-forward the
        # final token to produce fresh logits that see the overwritten K/V.
        last_tok = session_b_ids[last_b_end - 1 : last_b_end]
        logits = forward_with_cache(
            model, last_tok, past, trigger_position=last_b_end - 1
        )
        log = logits[-1].float().cpu()
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


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


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

    # Cache per session A: lazy-compute the first time A is used as a
    # donor, reuse thereafter. Sessions are small enough (15k tokens)
    # to keep a few caches alive simultaneously.
    a_cache: Dict[str, list] = {}

    # Incremental per-pair JSONL — appended on each completed pair so a
    # crash mid-run only loses the currently-in-flight pair. On restart
    # we read the JSONL and skip (a_id, b_id) pairs already recorded.
    jsonl_path = Path(args.output).with_suffix(".jsonl")
    done_pairs: set = set()
    results: List[dict] = []
    if jsonl_path.exists():
        with jsonl_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                done_pairs.add((row["a_id"], row["b_id"]))
                results.append(row)
        print(
            f"[info] resume: {len(done_pairs)} pairs already in {jsonl_path}",
            flush=True,
        )
    jsonl_f = jsonl_path.open("a")

    for bi, (b_id, b_toks) in enumerate(sessions):
        # Skip this B entirely if every earlier-A pair is already done.
        candidate_a_ids = [a_id for a_id, _ in sessions[:bi]]
        if candidate_a_ids and all(
            (a_id, b_id) in done_pairs for a_id in candidate_a_ids
        ):
            print(f"[info] skip B={bi:02d} (all pairs cached)", flush=True)
            continue

        session_b_ids = torch.tensor(b_toks, dtype=torch.long)
        # Fresh forward once per B (needed for KL/sim against any A).
        fresh_log, fresh_gen, fresh_past = fresh_forward(
            model, session_b_ids, args.gen_tokens, stop_ids
        )
        fresh_text = tok.decode(fresh_gen)
        del fresh_past
        torch.cuda.empty_cache()
        # Try every earlier session as A.
        for ai, (a_id, a_toks) in enumerate(sessions[:bi]):
            if (a_id, b_id) in done_pairs:
                continue
            matches = find_matches(b_toks, a_toks, args.min_match)
            if not matches:
                continue
            # Compute / fetch A's full KV snapshot.
            if a_id not in a_cache:
                a_cache[a_id] = prefill_and_snapshot(
                    model, torch.tensor(a_toks, dtype=torch.long)
                )
            cached_a = a_cache[a_id]
            try:
                reused_log, reused_gen = multi_splice_forward(
                    model,
                    session_b_ids,
                    cached_a,
                    matches,
                    args.gen_tokens,
                    stop_ids,
                )
            except Exception as e:
                print(f"[warn] multi_splice failed for {a_id}→{b_id}: {e}", flush=True)
                continue
            reused_text = tok.decode(reused_gen)
            # Metrics
            log_p = F.log_softmax(fresh_log, dim=-1)
            log_q = F.log_softmax(reused_log, dim=-1)
            kl = (log_p.exp() * (log_p - log_q)).sum().item()
            top1_fresh = int(fresh_log.argmax().item())
            top1_reused = int(reused_log.argmax().item())
            sim = cos_sim(fresh_text, reused_text)
            covered = sum(e - s for s, e, _, _ in matches)
            longest = max(e - s for s, e, _, _ in matches)
            row = {
                "a_id": a_id,
                "b_id": b_id,
                "b_len": len(b_toks),
                "a_len": len(a_toks),
                "n_matches": len(matches),
                "covered_tokens": covered,
                "coverage_frac": covered / max(len(b_toks), 1),
                "longest_match": longest,
                "kl": kl,
                "top1_fresh": top1_fresh,
                "top1_reused": top1_reused,
                "agree": int(top1_fresh == top1_reused),
                "sim_fresh_reused": sim,
                "fresh_text": fresh_text,
                "reused_text": reused_text,
            }
            results.append(row)
            jsonl_f.write(json.dumps(row) + "\n")
            jsonl_f.flush()
            done_pairs.add((a_id, b_id))
            print(
                f"[info] B={bi:02d} A={ai:02d}  matches={len(matches):2d}  "
                f"cov={covered / max(len(b_toks), 1):.2f}  "
                f"KL={kl:.2f}  sim={sim:.2f}",
                flush=True,
            )
        # Free A caches periodically if memory is tight.
        if args.dispose_a_caches_every > 0 and bi % args.dispose_a_caches_every == 0:
            a_cache.clear()
            torch.cuda.empty_cache()

    jsonl_f.close()
    if not results:
        raise SystemExit("no pairs with matches found")

    def _summ(key, pct=False):
        xs = [
            r[key]
            for r in results
            if r[key] is not None
            and not (
                isinstance(r[key], float) and (r[key] != r[key])  # NaN
            )
        ]
        if not xs:
            return {"n": 0}
        return {
            "n": len(xs),
            "mean": statistics.mean(xs),
            "median": statistics.median(xs),
            "p10": _percentile(xs, 0.10),
            "p90": _percentile(xs, 0.90),
        }

    summary = {
        "model": args.model,
        "dataset_repo": args.repo,
        "min_match": args.min_match,
        "n_sessions": len(sessions),
        "n_pairs": len(results),
        "coverage_frac": _summ("coverage_frac"),
        "kl": _summ("kl"),
        "agree_rate": _summ("agree"),
        "sim_fresh_reused": _summ("sim_fresh_reused"),
        "n_matches": _summ("n_matches"),
        "per_pair": results,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[info] wrote {out_path}  ({len(results)} pairs)")
    print()
    print("=== summary ===")
    for key in ("coverage_frac", "kl", "agree_rate", "sim_fresh_reused", "n_matches"):
        s = summary[key]
        if s.get("n"):
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
