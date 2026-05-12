#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "transformers>=5.5",
#   "huggingface_hub>=0.25",
#   "pyarrow>=15",
#   "jinja2>=3.0",
# ]
# ///
"""Find cross-session cache-reuse matches in an agentcap corpus.

For each pair of captured requests, compute:
  - CP (common prefix — what a standard prefix cache would serve)
  - post-CP byte-stable matches against any prior session

Model-generated tokens (role=assistant) are excluded — the serving stack
already has their KV from generating them, and they're not what cache-
reuse needs to handle. Every non-assistant post-CP match becomes a
splice-candidate manifest entry.

Usage:
    ./categorize_matches.py \\
        --source hf://buckets/dacorvo/agentcap-traces/<corpus>.parquet \\
        --min-match 128 \\
        --output trace_analysis/results/<name>_match_categories.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


_WANTED_COLS = (
    "request_id", "model", "captured_at", "request",
    "n_tokens", "sections", "token_role", "rendered_tokens",
)


def _select_cols(pf) -> list[str]:
    have = set(pf.schema_arrow.names)
    return [c for c in _WANTED_COLS if c in have]


def stream_rows(source: str) -> Iterable[dict]:
    import pyarrow.parquet as pq

    if source.startswith("hf://"):
        from huggingface_hub import HfFileSystem

        fs = HfFileSystem()
        bare = source[len("hf://") :].rstrip("/")
        try:
            info = fs.info(bare)
        except FileNotFoundError:
            info = None
        if info and info.get("type") == "file":
            files = [bare]
        else:
            files = sorted(fs.glob(bare + "/**/*.parquet"))
            if not files:
                files = sorted(fs.glob(bare + "/*.parquet"))
        if not files:
            raise FileNotFoundError(f"no parquet files under {source!r}")
        for path in files:
            with fs.open(path, "rb") as fh:
                pf = pq.ParquetFile(fh)
                cols = _select_cols(pf)
                for batch in pf.iter_batches(batch_size=16, columns=cols):
                    for row in batch.to_pylist():
                        yield row
        return

    p = Path(source)
    if p.is_dir() and (p / "dataset_info.json").exists():
        from datasets import load_from_disk

        for row in load_from_disk(str(p)):
            yield row
        return
    if p.is_file() and p.suffix == ".parquet":
        files = [p]
    elif p.is_dir():
        files = sorted(p.glob("*.parquet"))
    else:
        raise FileNotFoundError(f"unrecognised source {source!r}")
    for path in files:
        pf = pq.ParquetFile(str(path))
        cols = _select_cols(pf)
        for batch in pf.iter_batches(batch_size=16, columns=cols):
            for row in batch.to_pylist():
                yield row


def _decode_request(request) -> dict:
    """Newer exports store ``request`` as a JSON string. Normalize."""
    if request is None:
        return {}
    if isinstance(request, str):
        try:
            return json.loads(request)
        except json.JSONDecodeError:
            return {}
    return request


def _render_tokens(row: dict, tok) -> List[int]:
    """Render token ids on the fly when the row doesn't carry
    ``rendered_tokens``. Mirrors what the serving engine prefilled at
    capture time (with ``add_generation_prompt=True``)."""
    req = _decode_request(row.get("request"))
    msgs = req.get("messages") or []
    if not msgs:
        return []
    try:
        out = tok.apply_chat_template(
            msgs,
            tools=req.get("tools"),
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
        )
    except Exception:
        return []
    ids = out["input_ids"]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return list(ids)


def session_id_for(row: dict) -> str | None:
    req = _decode_request(row.get("request"))
    msgs = req.get("messages") or []
    first_user = next(
        (m.get("content") for m in msgs if m.get("role") == "user"), None
    )
    if not first_user:
        return None
    return hashlib.sha1(str(first_user).encode("utf-8")).hexdigest()[:12]


def add_to_index(
    index: Dict[int, List[Tuple[int, int]]],
    seq: List[int],
    seq_idx: int,
    min_n: int,
) -> None:
    if len(seq) < min_n:
        return
    for p in range(len(seq) - min_n + 1):
        index.setdefault(hash(tuple(seq[p : p + min_n])), []).append((seq_idx, p))


def max_common_prefix(
    target: List[int],
    prior_seqs: List[List[int]],
    prior_sessions: List[str],
    exclude_session: str,
) -> int:
    """Longest leading-token run target shares with any allowed prior."""
    cp = 0
    for si, prior in enumerate(prior_seqs):
        if prior_sessions[si] == exclude_session:
            continue
        L = min(len(prior), len(target))
        if L == 0 or prior[0] != target[0]:
            continue
        k = 1
        while k < L and prior[k] == target[k]:
            k += 1
        if k > cp:
            cp = k
    return cp


def find_matches(
    target: List[int],
    prior_seqs: List[List[int]],
    prior_sessions: List[str],
    index: Dict[int, List[Tuple[int, int]]],
    min_n: int,
    exclude_session: str,
    skip_to: int = 0,
) -> List[Tuple[int, int, int, int]]:
    """Return (b_start, b_end, donor_seq_idx, donor_start) for every
    non-overlapping ≥``min_n``-token match. ``donor_seq_idx`` indexes
    into ``prior_seqs``; together with the donor's request_id (kept
    parallel by the caller), this is what the splice manifest needs to
    identify both ends of the candidate splice pair."""
    matches: List[Tuple[int, int, int, int]] = []
    i = max(0, skip_to)
    T = len(target)
    while i + min_n <= T:
        window = target[i : i + min_n]
        cands = index.get(hash(tuple(window)))
        if not cands:
            i += 1
            continue
        best_len = 0
        best_si = -1
        best_p = -1
        for si, p in cands:
            if prior_sessions[si] == exclude_session:
                continue
            seq = prior_seqs[si]
            if seq[p : p + min_n] != window:  # hash collision guard
                continue
            ext = min_n
            limit = min(T - i, len(seq) - p)
            while ext < limit and target[i + ext] == seq[p + ext]:
                ext += 1
            if ext > best_len:
                best_len = ext
                best_si = si
                best_p = p
        if best_len < min_n:
            i += 1
            continue
        matches.append((i, i + best_len, best_si, best_p))
        i += best_len
    return matches


def split_match_by_role(
    start: int, end: int, token_role: List[str]
) -> List[Tuple[int, int, str]]:
    """A match span may straddle role boundaries. Return contiguous
    (sub_start, sub_end, role) sub-segments."""
    out: List[Tuple[int, int, str]] = []
    i = start
    while i < end:
        r = token_role[i] if i < len(token_role) else "unknown"
        j = i + 1
        while j < end and (
            (token_role[j] if j < len(token_role) else "unknown") == r
        ):
            j += 1
        out.append((i, j, r))
        i = j
    return out


def _sample_splice_candidates(
    candidates: List[dict],
    *,
    top_buckets: int,
    per_bucket: int,
) -> List[dict]:
    """Cap manifest size. Rank buckets by total recurrence-token volume,
    keep the top ``top_buckets``, then keep the largest ``per_bucket``
    candidate pairs per bucket (bigger chunks dominate splice savings)."""
    by_bucket: Dict[str, List[dict]] = defaultdict(list)
    bucket_volume: Dict[str, int] = defaultdict(int)
    for c in candidates:
        by_bucket[c["bucket_id"]].append(c)
        bucket_volume[c["bucket_id"]] += c["chunk_n_tokens"]
    ranked_buckets = sorted(bucket_volume.items(), key=lambda kv: -kv[1])
    out: List[dict] = []
    for bid, _ in ranked_buckets[:top_buckets]:
        cands = sorted(by_bucket[bid], key=lambda c: -c["chunk_n_tokens"])
        out.extend(cands[:per_bucket])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True,
                    help="Local parquet/dataset path or hf://buckets/... URI.")
    ap.add_argument("--min-match", type=int, default=128)
    ap.add_argument("--max-tokens-per-session", type=int, default=20000,
                    help="Cap each request's token sequence to bound index memory.")
    ap.add_argument("--splice-top-buckets", type=int, default=20,
                    help="Splice manifest: keep this many top buckets "
                    "(byte-identical recurring chunks) ranked by volume.")
    ap.add_argument("--splice-per-bucket", type=int, default=5,
                    help="Splice manifest: keep this many largest-chunk "
                    "candidates per kept bucket.")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    print(f"[info] streaming rows from {args.source}", flush=True)
    grouped: Dict[str, List[Tuple[int, dict]]] = defaultdict(list)
    model_id = None
    for row in stream_rows(args.source):
        sid = session_id_for(row)
        if not sid:
            continue
        if model_id is None:
            model_id = row.get("model")
        grouped[sid].append((int(row.get("captured_at", 0)), row))

    if not grouped:
        raise SystemExit("no rows yielded")

    ordered: List[Tuple[str, dict]] = []
    sids = sorted(grouped.keys(), key=lambda s: min(t for t, _ in grouped[s]))
    for sid in sids:
        rows = sorted(
            grouped[sid], key=lambda x: (x[0], int(x[1].get("n_tokens", 0)))
        )
        for _, r in rows:
            ordered.append((sid, r))
    print(
        f"[info] {len(grouped)} sessions, {len(ordered)} requests, "
        f"model={model_id}",
        flush=True,
    )

    print(f"[info] loading tokenizer {model_id}", flush=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)

    prior_seqs: List[List[int]] = []
    prior_sessions: List[str] = []
    prior_request_ids: List[str] = []
    index: Dict[int, List[Tuple[int, int]]] = {}

    splice_candidates: List[dict] = []
    total_target_tokens = 0
    total_cp_tokens = 0
    total_post_prefix_input_tokens = 0
    total_post_prefix_model_output_tokens = 0
    n_requests_with_post_prefix = 0

    for sid, row in ordered:
        rendered = row.get("rendered_tokens")
        ids = list(rendered) if rendered else _render_tokens(row, tok)
        ids = ids[: args.max_tokens_per_session]
        if not ids:
            continue
        token_role: List[str] = list(row.get("token_role") or [])[: args.max_tokens_per_session]
        total_target_tokens += len(ids)

        cp = max_common_prefix(ids, prior_seqs, prior_sessions, sid)
        if cp >= args.min_match:
            total_cp_tokens += cp

        spans = find_matches(
            ids, prior_seqs, prior_sessions, index, args.min_match, sid,
            skip_to=cp,
        )
        if spans:
            n_requests_with_post_prefix += 1
        for s, e, donor_si, donor_p in spans:
            donor_request_id = (
                prior_request_ids[donor_si] if 0 <= donor_si < len(prior_request_ids) else ""
            )
            for sub_s, sub_e, role in split_match_by_role(s, e, token_role):
                length = sub_e - sub_s
                if role == "assistant":
                    total_post_prefix_model_output_tokens += length
                    continue
                total_post_prefix_input_tokens += length
                snippet_ids = ids[sub_s : min(sub_e, sub_s + 80)]
                snippet = tok.decode(snippet_ids, skip_special_tokens=False)
                chunk_bytes = b",".join(str(t).encode() for t in ids[sub_s:sub_e])
                content_hash = hashlib.sha1(chunk_bytes).hexdigest()[:10]
                donor_sub_start = donor_p + (sub_s - s)
                donor_sub_end = donor_sub_start + length
                splice_candidates.append({
                    "bucket_id": f"{role}:{content_hash}",
                    "role": role,
                    "snippet": snippet,
                    "model": model_id,
                    "source_parquet": args.source,
                    "donor": {
                        "request_id": donor_request_id,
                        "tok_range": [donor_sub_start, donor_sub_end],
                    },
                    "recipient": {
                        "request_id": row["request_id"],
                        "tok_range": [sub_s, sub_e],
                    },
                    "chunk_n_tokens": length,
                    "position_drift": sub_s - donor_sub_start,
                })

        add_to_index(index, ids, len(prior_seqs), args.min_match)
        prior_seqs.append(ids)
        prior_sessions.append(sid)
        prior_request_ids.append(row["request_id"])

    pct = lambda x: x / max(total_target_tokens, 1)
    print()
    print(f"requests:                            {len(ordered)}")
    print(f"requests with ≥1 post-prefix match:  {n_requests_with_post_prefix}")
    print(f"total target tokens:                 {total_target_tokens:,}")
    print(f"  CP (prefix-cache served):          {total_cp_tokens:,}  ({pct(total_cp_tokens):.2%})")
    print(f"  post-prefix input (splice cand.):  {total_post_prefix_input_tokens:,}  ({pct(total_post_prefix_input_tokens):.2%})")
    print(f"  post-prefix model output (skipped):{total_post_prefix_model_output_tokens:,}  ({pct(total_post_prefix_model_output_tokens):.2%})")

    out = {
        "source": args.source,
        "model": model_id,
        "min_match": args.min_match,
        "n_sessions": len(grouped),
        "n_requests": len(ordered),
        "n_requests_with_post_prefix_match": n_requests_with_post_prefix,
        "total_target_tokens": total_target_tokens,
        "total_cp_tokens": total_cp_tokens,
        "total_post_prefix_input_tokens": total_post_prefix_input_tokens,
        "total_post_prefix_model_output_tokens": total_post_prefix_model_output_tokens,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(f"\n[info] wrote {args.output}")

    if splice_candidates:
        manifest_path = Path(args.output).with_suffix(".splice_candidates.jsonl")
        sampled = _sample_splice_candidates(
            splice_candidates,
            top_buckets=args.splice_top_buckets,
            per_bucket=args.splice_per_bucket,
        )
        with manifest_path.open("w") as fh:
            for c in sampled:
                fh.write(json.dumps(c) + "\n")
        print(
            f"[info] wrote splice manifest: {len(sampled)} pairs across "
            f"{len({c['bucket_id'] for c in sampled})} buckets → {manifest_path}"
        )


if __name__ == "__main__":
    main()
