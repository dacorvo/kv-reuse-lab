#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "huggingface_hub>=0.25",
#   "pyarrow>=15",
# ]
# ///
"""Find cross-session cache-reuse matches in an agentcap corpus.

Operates on the captured *messages* directly — no chat-template render,
no tokenizer. The recurring chunks we care about (tool outputs, file
contents, command results) live inside ``message.content`` as plain
strings; that's the search corpus. Position drift is measured in
characters and is reported as a hint to the splice test, which
re-tokenizes server-side and finds the matching span in token space
itself.

For each pair of captured requests, compute:
  - CP (common prefix — what a standard prefix cache would serve, in
    character units of the linearized message text)
  - post-CP byte-stable matches against any prior session

Model-generated tokens (role=assistant) are excluded — the serving
stack already has their KV from generating them, and they're not what
cache-reuse needs to handle. Every non-assistant post-CP match becomes
a splice-candidate manifest entry.

Usage:
    ./categorize_matches.py \\
        --source hf://buckets/dacorvo/agentcap-traces/<corpus>.parquet \\
        --output trace_analysis/results/<name>_match_categories.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


# Only need request_id + request body + a couple of metadata fields.
# Everything else (rendered_tokens, sections, etc.) is gone from the
# new agentcap export; this script no longer reads it.
_WANTED_COLS = ("request_id", "model", "captured_at", "request")


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
    """``request`` is JSON-stringified in the parquet (the new export
    format). Decode it. Older Dataset.to_dict captures may already be
    dicts; pass those through."""
    if request is None:
        return {}
    if isinstance(request, str):
        try:
            return json.loads(request)
        except json.JSONDecodeError:
            return {}
    return request


def _flatten_content(content) -> str:
    """OpenAI content can be a list of typed parts (multimodal-style).
    Reduce to the concatenated text. ``None`` and non-string types
    become ``""`` / their ``str()`` representation respectively."""
    if isinstance(content, list):
        return "".join(
            p.get("text", "") for p in content
            if isinstance(p, dict) and p.get("type") == "text"
        )
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    return str(content)


def linearize_request(req_body: dict) -> Tuple[str, List[Tuple[int, int, str, int]]]:
    """Project ``request.messages`` onto a single search string. Returns
    ``(text, spans)`` where ``spans`` is a list of
    ``(start_char, end_char, role, msg_idx)`` so a found match position
    can be looked up to discover the role of the chunk it falls in.

    Messages are concatenated with a single ``\\n`` separator that
    *doesn't* belong to any span — recurrence matches that straddle
    the boundary still find the substantive text on either side."""
    msgs = req_body.get("messages") or []
    parts: list[str] = []
    spans: list[tuple[int, int, str, int]] = []
    pos = 0
    for i, m in enumerate(msgs):
        if not isinstance(m, dict):
            continue
        role = m.get("role", "?")
        content = _flatten_content(m.get("content"))
        if content:
            spans.append((pos, pos + len(content), role, i))
        parts.append(content)
        parts.append("\n")
        pos += len(content) + 1
    return "".join(parts), spans


def role_at(spans: List[Tuple[int, int, str, int]], pos: int) -> str:
    """Look up the role of the message that contains ``pos`` in the
    linearized text. Returns ``"unknown"`` if ``pos`` falls outside
    any span (e.g. on a separator newline)."""
    for s, e, role, _ in spans:
        if s <= pos < e:
            return role
    return "unknown"


def split_match_by_role(
    start: int, end: int, spans: List[Tuple[int, int, str, int]]
) -> List[Tuple[int, int, str]]:
    """A match span may straddle role boundaries. Return contiguous
    (sub_start, sub_end, role) sub-segments by stepping through
    ``spans``."""
    out: List[Tuple[int, int, str]] = []
    i = start
    while i < end:
        r = role_at(spans, i)
        j = i + 1
        while j < end and role_at(spans, j) == r:
            j += 1
        out.append((i, j, r))
        i = j
    return out


def session_id_for(row: dict) -> str | None:
    req = _decode_request(row.get("request"))
    msgs = req.get("messages") or []
    first_user = next(
        (m.get("content") for m in msgs if m.get("role") == "user"), None
    )
    if not first_user:
        return None
    # ``first_user`` may be a list (content parts); flatten for hashing.
    if isinstance(first_user, list):
        first_user = _flatten_content(first_user)
    return hashlib.sha1(str(first_user).encode("utf-8")).hexdigest()[:12]


def add_to_index(
    index: Dict[int, List[Tuple[int, int]]],
    text: str,
    seq_idx: int,
    min_n: int,
) -> None:
    if len(text) < min_n:
        return
    for p in range(len(text) - min_n + 1):
        index.setdefault(hash(text[p : p + min_n]), []).append((seq_idx, p))


def max_common_prefix(
    target: str,
    all_texts: List[str],
    all_sessions: List[str],
    exclude_session: str,
    donor_idx_lt: int,
) -> int:
    """Longest leading-character run target shares with any allowed
    donor. ``donor_idx_lt`` excludes donors whose seq_idx is ``>=``
    that bound — i.e. only earlier (chronologically prior) donors
    count, matching the original sequential semantics."""
    cp = 0
    for si in range(donor_idx_lt):
        if all_sessions[si] == exclude_session:
            continue
        prior = all_texts[si]
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
    target: str,
    all_texts: List[str],
    all_sessions: List[str],
    index: Dict[int, List[Tuple[int, int]]],
    min_n: int,
    exclude_session: str,
    donor_idx_lt: int,
    skip_to: int = 0,
) -> List[Tuple[int, int, int, int]]:
    """Return (b_start, b_end, donor_seq_idx, donor_start) for every
    non-overlapping ≥``min_n``-char match. ``donor_seq_idx`` indexes
    into ``all_texts``; together with the donor's request_id (kept
    parallel by the caller), this is what the splice manifest needs to
    identify both ends of the candidate splice pair.

    ``donor_idx_lt`` excludes candidates with ``seq_idx >= donor_idx_lt``
    so the global index can be searched by any receiver while still
    honouring the chronological constraint."""
    matches: List[Tuple[int, int, int, int]] = []
    i = max(0, skip_to)
    T = len(target)
    while i + min_n <= T:
        window = target[i : i + min_n]
        cands = index.get(hash(window))
        if not cands:
            i += 1
            continue
        best_len = 0
        best_si = -1
        best_p = -1
        for si, p in cands:
            if si >= donor_idx_lt:
                continue
            if all_sessions[si] == exclude_session:
                continue
            seq = all_texts[si]
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


def _sample_splice_candidates(
    candidates: List[dict],
    *,
    top_buckets: int,
    per_bucket: int,
) -> List[dict]:
    """Cap manifest size. Rank buckets by total recurrence-char volume,
    keep the top ``top_buckets``, then keep the largest ``per_bucket``
    candidate pairs per bucket (bigger chunks dominate splice savings)."""
    by_bucket: Dict[str, List[dict]] = defaultdict(list)
    bucket_volume: Dict[str, int] = defaultdict(int)
    for c in candidates:
        by_bucket[c["bucket_id"]].append(c)
        bucket_volume[c["bucket_id"]] += c["chunk_n_chars"]
    ranked_buckets = sorted(bucket_volume.items(), key=lambda kv: -kv[1])
    out: List[dict] = []
    for bid, _ in ranked_buckets[:top_buckets]:
        cands = sorted(by_bucket[bid], key=lambda c: -c["chunk_n_chars"])
        out.extend(cands[:per_bucket])
    return out


# --- Globals populated in main(), inherited read-only by Pool workers
# via fork. Kept module-level on purpose: passing them as worker args
# would re-pickle the 100s-of-MB index per task and kill the speedup.
_TEXTS: List[str] = []
_SESSIONS: List[str] = []
_REQUEST_IDS: List[str] = []
_SPANS: List[List[Tuple[int, int, str, int]]] = []
_INDEX: Dict[int, List[Tuple[int, int]]] = {}
_ARGS: argparse.Namespace | None = None
_MODEL_ID: str | None = None


def _process_one(seq_idx: int) -> dict:
    """Per-receiver worker. Computes CP + post-CP matches against the
    global index, returning the receiver's aggregates + raw splice
    candidates. Uses the module globals (fork-inherited) so the index
    isn't pickled into each task."""
    text = _TEXTS[seq_idx]
    if not text:
        return {"seq_idx": seq_idx, "empty": True}

    spans = _SPANS[seq_idx]
    sid = _SESSIONS[seq_idx]
    rid = _REQUEST_IDS[seq_idx]
    assert _ARGS is not None  # narrow for type-checkers
    args = _ARGS

    cp = max_common_prefix(text, _TEXTS, _SESSIONS, sid, donor_idx_lt=seq_idx)
    cp_counted = cp if cp >= args.min_match else 0

    spans_matches = find_matches(
        text, _TEXTS, _SESSIONS, _INDEX, args.min_match, sid,
        donor_idx_lt=seq_idx, skip_to=cp,
    )

    post_prefix_input = 0
    post_prefix_model_output = 0
    splice_candidates: List[dict] = []
    for s, e, donor_si, donor_p in spans_matches:
        donor_request_id = (
            _REQUEST_IDS[donor_si] if 0 <= donor_si < len(_REQUEST_IDS) else ""
        )
        for sub_s, sub_e, role in split_match_by_role(s, e, spans):
            length = sub_e - sub_s
            if role == "assistant":
                post_prefix_model_output += length
                continue
            post_prefix_input += length
            snippet = text[sub_s : min(sub_e, sub_s + 80)]
            chunk_bytes = text[sub_s:sub_e].encode("utf-8")
            content_hash = hashlib.sha1(chunk_bytes).hexdigest()[:10]
            donor_sub_start = donor_p + (sub_s - s)
            donor_sub_end = donor_sub_start + length
            splice_candidates.append({
                "bucket_id": f"{role}:{content_hash}",
                "role": role,
                "snippet": snippet,
                "model": _MODEL_ID,
                "source_parquet": args.source,
                "donor": {
                    "request_id": donor_request_id,
                    "char_range": [donor_sub_start, donor_sub_end],
                },
                "recipient": {
                    "request_id": rid,
                    "char_range": [sub_s, sub_e],
                },
                "chunk_n_chars": length,
                "position_drift": sub_s - donor_sub_start,
            })

    return {
        "seq_idx": seq_idx,
        "text_len": len(text),
        "cp_counted": cp_counted,
        "has_post_prefix_match": bool(spans_matches),
        "post_prefix_input": post_prefix_input,
        "post_prefix_model_output": post_prefix_model_output,
        "splice_candidates": splice_candidates,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True,
                    help="Local parquet/dataset path or hf://buckets/... URI.")
    ap.add_argument("--min-match", type=int, default=4096,
                    help="Minimum match length in CHARACTERS (was tokens in "
                    "the old format). Default 4096 ≈ a thousand tokens "
                    "for English; tool outputs / file contents recur at "
                    "much larger sizes than this.")
    ap.add_argument("--max-chars-per-session", type=int, default=80_000,
                    help="Cap each request's linearized text length to bound "
                    "index memory.")
    ap.add_argument("--splice-top-buckets", type=int, default=20,
                    help="Splice manifest: keep this many top buckets "
                    "(byte-identical recurring chunks) ranked by volume.")
    ap.add_argument("--splice-per-bucket", type=int, default=5,
                    help="Splice manifest: keep this many largest-chunk "
                    "candidates per kept bucket.")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1,
                    help="Number of worker processes for the per-receiver "
                    "match search. Defaults to os.cpu_count(). Set to 1 to "
                    "run sequentially (useful for debugging).")
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
        rows = sorted(grouped[sid], key=lambda x: x[0])
        for _, r in rows:
            ordered.append((sid, r))
    print(
        f"[info] {len(grouped)} sessions, {len(ordered)} requests, "
        f"model={model_id}",
        flush=True,
    )

    # Stage 1 (sequential): linearize every receiver up-front so the
    # workers can read by seq_idx without needing the raw row.
    print("[info] linearising requests…", flush=True)
    texts: List[str] = []
    sessions: List[str] = []
    request_ids: List[str] = []
    spans_per_req: List[List[Tuple[int, int, str, int]]] = []
    for sid, row in ordered:
        req = _decode_request(row.get("request"))
        text, spans = linearize_request(req)
        text = text[: args.max_chars_per_session]
        trimmed_spans = [
            (s, min(e, len(text)), r, idx)
            for (s, e, r, idx) in spans
            if s < len(text)
        ]
        texts.append(text)
        sessions.append(sid)
        request_ids.append(row["request_id"])
        spans_per_req.append(trimmed_spans)

    # Stage 2 (sequential): build the global hash index from every
    # text. The chronological constraint (donor.seq_idx < receiver
    # .seq_idx) is enforced inside each worker via ``donor_idx_lt``.
    print(
        f"[info] building global index (min_match={args.min_match} chars)…",
        flush=True,
    )
    index: Dict[int, List[Tuple[int, int]]] = {}
    for si, text in enumerate(texts):
        add_to_index(index, text, si, args.min_match)
    n_index_entries = sum(len(v) for v in index.values())
    print(
        f"[info] index: {len(index):,} keys, {n_index_entries:,} entries",
        flush=True,
    )

    # Stage 3: per-receiver match search, parallel over receivers.
    # Globals are populated *before* fork so children inherit the
    # built index + texts at zero serialisation cost.
    global _TEXTS, _SESSIONS, _REQUEST_IDS, _SPANS, _INDEX, _ARGS, _MODEL_ID
    _TEXTS = texts
    _SESSIONS = sessions
    _REQUEST_IDS = request_ids
    _SPANS = spans_per_req
    _INDEX = index
    _ARGS = args
    _MODEL_ID = model_id

    n_workers = max(1, args.workers)
    print(
        f"[info] searching {len(texts)} receivers with {n_workers} workers…",
        flush=True,
    )
    seq_idxs = list(range(len(texts)))
    import time
    t0 = time.time()
    n_total = len(seq_idxs)
    per_receiver: List[dict] = []

    def _emit_progress(done: int) -> None:
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 else 0.0
        eta = (n_total - done) / rate if rate > 0 else float("inf")
        print(
            f"[progress] {done}/{n_total} receivers ({100 * done / n_total:.1f}%) "
            f"  elapsed={elapsed:.1f}s  rate={rate:.1f}/s  eta={eta:.0f}s",
            flush=True,
        )

    # Print progress every ~2% or every 60s of wall time, whichever
    # comes first; both gives steady output on short and long runs.
    progress_every = max(1, n_total // 50)
    last_print_t = t0

    if n_workers == 1:
        for done, i in enumerate(seq_idxs, start=1):
            per_receiver.append(_process_one(i))
            if done % progress_every == 0 or time.time() - last_print_t > 60:
                _emit_progress(done)
                last_print_t = time.time()
    else:
        # ``fork`` is the linux default and the whole point of keeping
        # the index module-global: workers share it via copy-on-write
        # without any pickling. Set explicitly so future Python
        # default-changes don't silently regress to ``spawn``.
        ctx = mp.get_context("fork")
        chunksize = max(1, len(seq_idxs) // (n_workers * 4) or 1)
        with ctx.Pool(n_workers) as pool:
            for done, r in enumerate(
                pool.imap_unordered(_process_one, seq_idxs, chunksize=chunksize),
                start=1,
            ):
                per_receiver.append(r)
                if done % progress_every == 0 or time.time() - last_print_t > 60:
                    _emit_progress(done)
                    last_print_t = time.time()
    _emit_progress(len(per_receiver))

    # Stage 4 (sequential): reconcile per-receiver results.
    splice_candidates: List[dict] = []
    total_target_chars = 0
    total_cp_chars = 0
    total_post_prefix_input_chars = 0
    total_post_prefix_model_output_chars = 0
    n_requests_with_post_prefix = 0
    n_nonempty = 0
    for r in per_receiver:
        if r.get("empty"):
            continue
        n_nonempty += 1
        total_target_chars += r["text_len"]
        total_cp_chars += r["cp_counted"]
        if r["has_post_prefix_match"]:
            n_requests_with_post_prefix += 1
        total_post_prefix_input_chars += r["post_prefix_input"]
        total_post_prefix_model_output_chars += r["post_prefix_model_output"]
        splice_candidates.extend(r["splice_candidates"])

    pct = lambda x: x / max(total_target_chars, 1)
    print()
    print(f"requests:                            {n_nonempty}")
    print(f"requests with ≥1 post-prefix match:  {n_requests_with_post_prefix}")
    print(f"total target chars:                  {total_target_chars:,}")
    print(f"  CP (prefix-cache served):          {total_cp_chars:,}  ({pct(total_cp_chars):.2%})")
    print(f"  post-prefix input (splice cand.):  {total_post_prefix_input_chars:,}  ({pct(total_post_prefix_input_chars):.2%})")
    print(f"  post-prefix model output (skipped):{total_post_prefix_model_output_chars:,}  ({pct(total_post_prefix_model_output_chars):.2%})")

    out = {
        "source": args.source,
        "model": model_id,
        "min_match": args.min_match,
        "unit": "chars",
        "n_sessions": len(grouped),
        "n_requests": len(ordered),
        "n_requests_with_post_prefix_match": n_requests_with_post_prefix,
        "total_target_chars": total_target_chars,
        "total_cp_chars": total_cp_chars,
        "total_post_prefix_input_chars": total_post_prefix_input_chars,
        "total_post_prefix_model_output_chars": total_post_prefix_model_output_chars,
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
