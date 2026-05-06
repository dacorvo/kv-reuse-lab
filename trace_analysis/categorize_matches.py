#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "transformers>=5.5",
#   "huggingface_hub>=0.25",
#   "pyarrow>=15",
# ]
# ///
"""Classify cross-session cache-reuse matches in an agentcap corpus.

`analyze_trace_reuse.py` answers "how much of every prompt is covered by
byte-exact cross-session matches". This answers the next question:
*which kind of bytes recur*. That's the load-bearing input for cache
admission design — if the recurrence is dominated by framework
boilerplate (system prompt, tool schemas) the cache is uninteresting;
if it's dominated by user/team substrate (tool responses, project
context) it earns its keep.

For each match span we identify in a target request, we tag it by:
  - role (from the row's per-token role labels)
  - content sub-type (string-pattern match against distinctive markers
    in the decoded text), e.g. "Hermes system prefix", "tool schema
    injection", "memory section", "tool response", "user content".

Output is a per-category breakdown of matched-token volume + sample
snippets per category for verification.

Usage:
    ./categorize_matches.py \\
        --source hf://buckets/dacorvo/agentcaps-traces/hermes-gemma-4-E4B-it \\
        --min-match 128 \\
        --output trace_analysis/results/agentcap_match_categories.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


CATEGORY_RULES: List[Tuple[str, re.Pattern]] = [
    # Order matters — first match wins. More specific patterns first.
    ("hermes_system_prefix", re.compile(r"You are Hermes Agent")),
    ("tools_schema", re.compile(r"<\|tool\|*>\s*declaration:|<\|*tool>declaration:")),
    ("project_context_agentsmd", re.compile(r"# Project Context\b|## AGENTS\.md\b")),
    ("memory_section", re.compile(r"MEMORY \(your personal notes\)|MEMORY\s+\(your")),
    ("skills_index", re.compile(r"<available_skills>")),
    ("hermes_skill_view", re.compile(r"skill_view\(name=")),
    # Generic fallbacks (role-based) handled in classify().
]


def classify(role: str, text: str) -> str:
    if role == "user":
        return "user_content"
    if role == "assistant":
        return "assistant_output_should_not_be_cached"
    if role == "tool":
        return "tool_response"
    # role == "system" or unknown: pattern-match the text.
    for label, pat in CATEGORY_RULES:
        if pat.search(text):
            return label
    return "system_other"


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
) -> List[Tuple[int, int]]:
    matches: List[Tuple[int, int]] = []
    i = max(0, skip_to)
    T = len(target)
    while i + min_n <= T:
        window = target[i : i + min_n]
        cands = index.get(hash(tuple(window)))
        if not cands:
            i += 1
            continue
        best_len = 0
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
        if best_len < min_n:
            i += 1
            continue
        matches.append((i, i + best_len))
        i += best_len
    return matches


def _tool_call_args_by_id(request: dict) -> Dict[str, str]:
    """Walk a request body's assistant messages to recover the JSON
    string passed as ``arguments`` to each tool call, keyed by
    ``tool_call_id``. Used by the tool_response category drilldown to
    join a captured ``role=tool`` token range back to the (tool_name,
    args) pair that produced it."""
    out: Dict[str, str] = {}
    for m in request.get("messages") or []:
        if m.get("role") != "assistant":
            continue
        for tc in m.get("tool_calls") or []:
            tcid = tc.get("id")
            fn = tc.get("function") or {}
            if tcid:
                args = fn.get("arguments")
                out[tcid] = args if isinstance(args, str) else json.dumps(args, sort_keys=True)
    return out


def _section_for_token(sections: List[dict], pos: int) -> dict | None:
    """Return the section whose [tok_range[0], tok_range[1]) covers
    ``pos``. Sections are stored in token order so a linear scan is
    fine for the lengths we deal with (≤ a few dozen sections)."""
    for s in sections:
        tr = s.get("tok_range") or [0, 0]
        if tr[0] <= pos < tr[1]:
            return s
    return None


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


def _build_detail(
    cat: str,
    tool_detail_tokens: Dict[Tuple[str, str], int],
    tool_detail_count: Dict[Tuple[str, str], int],
    tool_detail_args: Dict[Tuple[str, str], str],
    tool_detail_snippet: Dict[Tuple[str, str], str],
    tool_only_tokens: Dict[str, int],
    tool_only_count: Dict[str, int],
) -> dict | None:
    """Per-category drilldown attached as ``by_category[*].detail`` in
    the output JSON. Today only ``tool_response`` has one — buckets the
    matched tokens by ``(tool_name, args_hash)`` so a reader can tell
    whether the recurrence is a few hot tools (cacheable by request
    semantics) or scattered across many one-offs (admission would need
    to be content-driven instead). Returns ``None`` for categories
    without a drilldown."""
    if cat != "tool_response" or not tool_detail_tokens:
        return None
    total = sum(tool_detail_tokens.values())
    by_tool = [
        {
            "tool_name": name,
            "tokens": toks,
            "frac_of_tool_response": toks / total,
            "n_matches": tool_only_count[name],
        }
        for name, toks in sorted(tool_only_tokens.items(), key=lambda kv: -kv[1])
    ]
    by_tool_args = [
        {
            "tool_name": name,
            "args_hash": ah,
            "args_sample": tool_detail_args[(name, ah)],
            "tokens": toks,
            "frac_of_tool_response": toks / total,
            "n_matches": tool_detail_count[(name, ah)],
            "snippet": tool_detail_snippet[(name, ah)],
        }
        for (name, ah), toks in sorted(tool_detail_tokens.items(), key=lambda kv: -kv[1])
    ]
    return {"by_tool": by_tool, "by_tool_and_args": by_tool_args}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True,
                    help="Local parquet/dataset path or hf://buckets/... URI.")
    ap.add_argument("--min-match", type=int, default=128)
    ap.add_argument("--max-tokens-per-session", type=int, default=20000,
                    help="Cap each request's token sequence to bound index memory.")
    ap.add_argument("--n-samples-per-category", type=int, default=4,
                    help="How many decoded snippets to keep per category.")
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

    # Order sessions by earliest capture, requests within session by capture+len.
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

    # Incremental cross-session matching, same as analyze_trace_reuse.
    prior_seqs: List[List[int]] = []
    prior_sessions: List[str] = []
    index: Dict[int, List[Tuple[int, int]]] = {}

    cat_tokens: Dict[str, int] = defaultdict(int)
    cat_match_count: Dict[str, int] = defaultdict(int)
    cat_lengths: Dict[str, List[int]] = defaultdict(list)
    cat_samples: Dict[str, List[dict]] = defaultdict(list)

    # tool_response drilldown: group recurrence by (tool_name, args_hash)
    # so we can answer "which (tool, arguments) pairs account for the
    # bulk of post-prefix tool_response matches". Args are joined via
    # tool_call_id from the section that owns the matched tokens.
    tool_detail_tokens: Dict[Tuple[str, str], int] = defaultdict(int)
    tool_detail_count: Dict[Tuple[str, str], int] = defaultdict(int)
    tool_detail_args: Dict[Tuple[str, str], str] = {}
    tool_detail_snippet: Dict[Tuple[str, str], str] = {}

    total_matched_tokens = 0
    total_target_tokens = 0
    total_cp_tokens = 0
    total_post_prefix_tokens = 0
    n_requests_with_post_prefix = 0

    CP_CATEGORY = "common_prefix_served_by_prefix_cache"

    for r_idx, (sid, row) in enumerate(ordered):
        rendered = row.get("rendered_tokens")
        if rendered:
            ids = list(rendered)
        else:
            ids = _render_tokens(row, tok)
        ids = ids[: args.max_tokens_per_session]
        if not ids:
            continue
        token_role: List[str] = list(row.get("token_role") or [])[: args.max_tokens_per_session]
        total_target_tokens += len(ids)

        # Per-row tool-call args lookup for the tool_response drilldown.
        row_request = _decode_request(row.get("request"))
        row_args_by_id = _tool_call_args_by_id(row_request)
        row_sections = list(row.get("sections") or [])

        # CP first — that's what a standard prefix cache would serve.
        cp = max_common_prefix(ids, prior_seqs, prior_sessions, sid)
        if cp >= args.min_match:
            total_cp_tokens += cp
            cat_tokens[CP_CATEGORY] += cp
            cat_match_count[CP_CATEGORY] += 1
            cat_lengths[CP_CATEGORY].append(cp)
            total_matched_tokens += cp
            if len(cat_samples[CP_CATEGORY]) < args.n_samples_per_category:
                snippet_ids = ids[: min(cp, 80)]
                snippet = tok.decode(snippet_ids, skip_special_tokens=False)
                cat_samples[CP_CATEGORY].append({
                    "request_id": row["request_id"],
                    "session_id": sid,
                    "tok_range": [0, cp],
                    "tokens": cp,
                    "role": "system",  # CP always starts at system
                    "snippet": snippet,
                })

        # Post-prefix matches: scan only past CP.
        spans = find_matches(
            ids, prior_seqs, prior_sessions, index, args.min_match, sid,
            skip_to=cp,
        )
        if spans:
            n_requests_with_post_prefix += 1
        post_prefix_in_request = sum(e - s for s, e in spans)
        total_post_prefix_tokens += post_prefix_in_request
        for s, e in spans:
            for sub_s, sub_e, role in split_match_by_role(s, e, token_role):
                # Decode a leading snippet of the sub-match (cap for sample storage).
                snippet_ids = ids[sub_s : min(sub_e, sub_s + 80)]
                snippet = tok.decode(snippet_ids, skip_special_tokens=False)
                full_for_classify = tok.decode(
                    ids[sub_s : min(sub_e, sub_s + 200)],
                    skip_special_tokens=False,
                )
                cat = classify(role, full_for_classify)
                length = sub_e - sub_s
                cat_tokens[cat] += length
                cat_match_count[cat] += 1
                cat_lengths[cat].append(length)
                total_matched_tokens += length
                if len(cat_samples[cat]) < args.n_samples_per_category:
                    cat_samples[cat].append({
                        "request_id": row["request_id"],
                        "session_id": sid,
                        "tok_range": [sub_s, sub_e],
                        "tokens": length,
                        "role": role,
                        "snippet": snippet,
                    })
                if cat == "tool_response":
                    sec = _section_for_token(row_sections, sub_s)
                    if sec is not None and sec.get("role") == "tool":
                        tool_name = sec.get("tool_name") or "<unknown>"
                        tcid = sec.get("tool_call_id") or ""
                        args_str = row_args_by_id.get(tcid, "")
                        # Hash the args to bucket recurrences. Hash on
                        # the canonicalized JSON so whitespace-only
                        # differences don't fragment a real recurrence.
                        ah = hashlib.sha1(args_str.encode("utf-8")).hexdigest()[:10]
                        key = (tool_name, ah)
                        tool_detail_tokens[key] += length
                        tool_detail_count[key] += 1
                        if key not in tool_detail_args:
                            tool_detail_args[key] = args_str[:200]
                        if key not in tool_detail_snippet:
                            tool_detail_snippet[key] = snippet

        # Add to prior pool.
        add_to_index(index, ids, len(prior_seqs), args.min_match)
        prior_seqs.append(ids)
        prior_sessions.append(sid)

    # Build report.
    cats_sorted = sorted(cat_tokens.items(), key=lambda kv: -kv[1])
    print()
    print(f"requests:                          {len(ordered)}")
    print(f"requests with ≥1 post-prefix match:{n_requests_with_post_prefix}")
    print(f"total target tokens:               {total_target_tokens:,}")
    print(f"total matched tokens:              {total_matched_tokens:,}  "
          f"({total_matched_tokens / max(total_target_tokens,1):.2%} of target)")
    print(f"  CP (prefix-cache served):        {total_cp_tokens:,}  "
          f"({total_cp_tokens / max(total_target_tokens,1):.2%} of target)")
    print(f"  post-prefix (non-trivial cache): {total_post_prefix_tokens:,}  "
          f"({total_post_prefix_tokens / max(total_target_tokens,1):.2%} of target)")
    print()
    print(f"{'category':<48} {'tokens':>12} {'frac_of_match':>14} "
          f"{'n_matches':>10} {'mean_len':>10}")
    for cat, toks in cats_sorted:
        n = cat_match_count[cat]
        mean_len = statistics.mean(cat_lengths[cat]) if cat_lengths[cat] else 0
        print(f"{cat:<48} {toks:>12,} {toks/max(total_matched_tokens,1):>14.2%} "
              f"{n:>10} {mean_len:>10.0f}")

    # Tool-response drilldown: rank (tool_name, args_hash) buckets by
    # tokens. Two passes — by tool only, then by (tool, args).
    tool_only_tokens: Dict[str, int] = defaultdict(int)
    tool_only_count: Dict[str, int] = defaultdict(int)
    for (tool_name, _ah), toks in tool_detail_tokens.items():
        tool_only_tokens[tool_name] += toks
        tool_only_count[tool_name] += tool_detail_count[(tool_name, _ah)]

    total_tr_tokens = sum(tool_detail_tokens.values()) or 1
    if tool_detail_tokens:
        print()
        print("tool_response drilldown — by tool name:")
        print(f"  {'tool':<28} {'tokens':>10} {'frac_of_tr':>12} {'n_matches':>10}")
        for tool_name, toks in sorted(tool_only_tokens.items(), key=lambda kv: -kv[1])[:10]:
            print(f"  {tool_name:<28} {toks:>10,} {toks/total_tr_tokens:>12.2%} "
                  f"{tool_only_count[tool_name]:>10}")
        print()
        print("tool_response drilldown — top (tool, args_hash) buckets:")
        print(f"  {'tool':<20} {'args_hash':<12} {'tokens':>10} {'frac':>8} {'n':>5}")
        ranked = sorted(tool_detail_tokens.items(), key=lambda kv: -kv[1])
        for (tool_name, ah), toks in ranked[:10]:
            print(f"  {tool_name:<20} {ah:<12} {toks:>10,} "
                  f"{toks/total_tr_tokens:>7.2%} {tool_detail_count[(tool_name, ah)]:>5}")

    out = {
        "source": args.source,
        "model": model_id,
        "min_match": args.min_match,
        "n_sessions": len(grouped),
        "n_requests": len(ordered),
        "n_requests_with_post_prefix_match": n_requests_with_post_prefix,
        "total_target_tokens": total_target_tokens,
        "total_matched_tokens": total_matched_tokens,
        "total_cp_tokens": total_cp_tokens,
        "total_post_prefix_tokens": total_post_prefix_tokens,
        "by_category": [
            {
                "category": cat,
                "tokens": toks,
                "frac_of_matched": toks / max(total_matched_tokens, 1),
                "n_matches": cat_match_count[cat],
                "mean_match_length": statistics.mean(cat_lengths[cat]),
                "median_match_length": statistics.median(cat_lengths[cat]),
                "max_match_length": max(cat_lengths[cat]),
                "samples": cat_samples[cat],
                "detail": _build_detail(
                    cat,
                    tool_detail_tokens,
                    tool_detail_count,
                    tool_detail_args,
                    tool_detail_snippet,
                    tool_only_tokens,
                    tool_only_count,
                ),
            }
            for cat, toks in cats_sorted
        ],
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(f"\n[info] wrote {args.output}")


if __name__ == "__main__":
    main()
