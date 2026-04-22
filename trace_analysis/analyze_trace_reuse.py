#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "datasets>=2.20",
#   "transformers>=5.5",
#   "jinja2>=3.0",
#   "matplotlib>=3.8",
# ]
# ///
"""Measure cross-request KV-cache-reuse *opportunity* in real agent
traces.

For every request in a dataset, this script computes — against the
pool of all *earlier* requests — how much of the request's prompt is
covered by byte-exact token-span matches of at least ``--min-match``
tokens. Three numbers per request:

* ``coverage``   — fraction of target tokens that live inside some
  matching span. Upper bound on anything a reuse engine could ever
  save from prefill.
* ``longest``    — length of the single longest matching span as a
  fraction of target tokens. This is what a "prefix-plus-one-slide"
  engine (llama.cpp's current ``--cache-reuse``) can actually
  exploit from the start of the prompt.
* ``fragments``  — number of disjoint matching spans that contribute
  to coverage. ``fragments > 1`` with coverage meaningfully larger
  than ``longest`` is the gap-stitching upside llama.cpp does not
  capture today.

The algorithm is simple: build an index of all (min_n)-length token
windows from prior requests, scan the target left-to-right, and for
each window lookup extend greedily against the best matching prior.
Non-overlapping matches on the target side. O(N * L_avg) with a
dict-based index; runs in minutes on a few hundred sessions.

Usage:
    ./analyze_trace_reuse.py --dataset hermes --n-sessions 50
    ./analyze_trace_reuse.py --dataset swe-smith --n-sessions 20 --min-match 64
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


# ---------------------------------------------------------------------------
# Dataset adapters: yield (session_id, [request_token_ids, ...]) tuples.
# Each request_token_ids list is a prompt as it would be prefilled by a
# serving engine at that point in the conversation.
# ---------------------------------------------------------------------------


def _tokenize_chat(tokenizer, msgs) -> List[int]:
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
    return list(ids)


def _hermes_sessions(
    tokenizer, n_sessions: int
) -> Iterable[Tuple[str, List[List[int]]]]:
    """Stream Hermes func_calling sessions, yielding the sequence of
    prefix requests (one per tool-result boundary) for each.
    """
    from datasets import load_dataset

    ds = load_dataset(
        "NousResearch/hermes-function-calling-v1",
        "func_calling",
        split="train",
        streaming=True,
    )
    role_map = {
        "system": "system",
        "human": "user",
        "user": "user",
        "gpt": "assistant",
        "assistant": "assistant",
        "tool": "user",
        "function": "user",
    }

    def _to_msgs(convs):
        out = []
        for t in convs:
            role = t.get("from") or t.get("role") or ""
            content = t.get("value") or t.get("content") or ""
            r = role_map.get(role, role)
            if r:
                out.append({"role": r, "content": content})
        return out

    yielded = 0
    for row in ds:
        if yielded >= n_sessions:
            break
        convs = row.get("conversations") or []
        # Build request prefixes at each tool-result boundary.
        prefixes: List[List[int]] = []
        running: List[dict] = []
        for t in convs:
            running.append(t)
            if (t.get("from") or t.get("role")) == "tool":
                msgs = _to_msgs(running)
                try:
                    ids = _tokenize_chat(tokenizer, msgs)
                except Exception:
                    continue
                prefixes.append(ids)
        if not prefixes:
            continue
        yield (f"hermes-{yielded:04d}", prefixes)
        yielded += 1


def _swe_smith_sessions(
    tokenizer, n_sessions: int, repo_filter: str | None = None
) -> Iterable[Tuple[str, List[List[int]]]]:
    from datasets import load_dataset

    ds = load_dataset(
        "SWE-bench/SWE-smith-trajectories",
        split="tool",
        streaming=True,
    )
    yielded = 0
    for row in ds:
        if yielded >= n_sessions:
            break
        instance_id = row.get("instance_id", "")
        if repo_filter and not instance_id.startswith(repo_filter):
            continue
        raw_msgs = row.get("messages")
        if isinstance(raw_msgs, str):
            try:
                msgs = json.loads(raw_msgs)
            except Exception:
                continue
        else:
            msgs = raw_msgs
        if not msgs:
            continue
        # Normalize tool-call / tool-response content to plain text
        # chat-template-compatible shape (role=user with inlined tool
        # response content, since Llama/Gemma templates don't accept
        # role=tool directly).
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
        # Build request prefixes at each boundary where the NEXT turn
        # would be an assistant generation. In practice that's after
        # every tool response and after every user turn.
        prefixes: List[List[int]] = []
        running: List[dict] = []
        for i, m in enumerate(norm):
            running.append(m)
            nxt = norm[i + 1] if i + 1 < len(norm) else None
            if nxt is not None and nxt["role"] == "assistant":
                try:
                    ids = _tokenize_chat(tokenizer, running)
                except Exception:
                    continue
                prefixes.append(ids)
        if not prefixes:
            continue
        # Use instance_id as part of the session id so repo grouping
        # is visible in downstream analysis.
        tag = instance_id.split("-", 1)[0] if instance_id else "unknown"
        yield (f"swe-{tag}-{yielded:04d}", prefixes)
        yielded += 1


def _parse_chatml(text: str) -> List[Dict[str, str]]:
    """Parse a ChatML string into a list of {role, content} messages."""
    import re

    out: List[Dict[str, str]] = []
    # Match <|im_start|>role\ncontent<|im_end|> blocks. Be tolerant of
    # trailing whitespace and missing final <|im_end|>.
    pattern = re.compile(
        r"<\|im_start\|>(\w+)\n(.*?)(?:<\|im_end\|>|(?=<\|im_start\|>)|\Z)",
        re.DOTALL,
    )
    for m in pattern.finditer(text):
        role = m.group(1).strip()
        content = m.group(2).rstrip()
        if role == "tool":
            role = "user"
        if role in ("system", "user", "assistant"):
            out.append({"role": role, "content": content})
    return out


def _claude_hyperswitch_sessions(
    tokenizer, n_sessions: int
) -> Iterable[Tuple[str, List[List[int]]]]:
    """``archit11/claude_code_traces_hs`` — real Claude Code sessions on the
    juspay/hyperswitch repo. Each row is one task trajectory in ChatML
    form. Same-repo across all sessions, so the cross-session analysis
    captures file re-reads / shared source content naturally.
    """
    from datasets import load_dataset

    ds = load_dataset("archit11/claude_code_traces_hs", split="train", streaming=True)
    yielded = 0
    for row in ds:
        if yielded >= n_sessions:
            break
        text = row.get("chatml") or ""
        if not text:
            continue
        msgs = _parse_chatml(text)
        if not msgs:
            continue
        prefixes: List[List[int]] = []
        running: List[dict] = []
        for i, m in enumerate(msgs):
            running.append(m)
            nxt = msgs[i + 1] if i + 1 < len(msgs) else None
            if nxt is not None and nxt["role"] == "assistant":
                try:
                    ids = _tokenize_chat(tokenizer, running)
                except Exception:
                    continue
                prefixes.append(ids)
        if not prefixes:
            continue
        prefix_id = row.get("prefix") or f"hs-{yielded:04d}"
        yield (f"hs-{prefix_id}", prefixes)
        yielded += 1


DATASETS = {
    "hermes": _hermes_sessions,
    "swe-smith": _swe_smith_sessions,
    "claude-hs": _claude_hyperswitch_sessions,
}


# ---------------------------------------------------------------------------
# Match-finding
# ---------------------------------------------------------------------------


def build_window_index(
    sequences: List[List[int]], min_n: int
) -> Dict[Tuple[int, ...], List[Tuple[int, int]]]:
    """Return {window: [(seq_idx, start_pos), ...]} for every
    ``min_n``-token window across the given sequences.
    """
    index: Dict[Tuple[int, ...], List[Tuple[int, int]]] = {}
    for si, seq in enumerate(sequences):
        if len(seq) < min_n:
            continue
        for p in range(len(seq) - min_n + 1):
            key = tuple(seq[p : p + min_n])
            index.setdefault(key, []).append((si, p))
    return index


def find_matches(
    target: List[int],
    prior_sequences: List[List[int]],
    index: Dict[Tuple[int, ...], List[Tuple[int, int]]],
    min_n: int,
) -> List[Tuple[int, int]]:
    """Return list of non-overlapping (target_start, target_end)
    spans where target[start:end] is a byte-exact match of length
    ≥ min_n against *some* prior sequence. For each anchor position,
    pick the longest extension available across all prior matches.
    """
    matches: List[Tuple[int, int]] = []
    i = 0
    T = len(target)
    while i + min_n <= T:
        key = tuple(target[i : i + min_n])
        cands = index.get(key)
        if not cands:
            i += 1
            continue
        best_len = min_n
        for si, p in cands:
            seq = prior_sequences[si]
            ext = min_n
            tmax = T - i
            smax = len(seq) - p
            limit = min(tmax, smax)
            while ext < limit and target[i + ext] == seq[p + ext]:
                ext += 1
            if ext > best_len:
                best_len = ext
        matches.append((i, i + best_len))
        i += best_len
    return matches


def _add_sequence_to_index(
    index: Dict[int, List[Tuple[int, int]]],
    seq: List[int],
    seq_idx: int,
    min_n: int,
) -> None:
    """Append one sequence's windows to a tagged index so subsequent
    lookups can filter by sequence index (e.g. to exclude same-session
    candidates).

    Keys are ``hash(tuple(window))`` rather than the tuple itself — a
    ~100× memory saving at the cost of a per-candidate verification on
    lookup in case of hash collisions.
    """
    if len(seq) < min_n:
        return
    for p in range(len(seq) - min_n + 1):
        key = hash(tuple(seq[p : p + min_n]))
        index.setdefault(key, []).append((seq_idx, p))


def intra_request_duplicates(
    target: List[int], min_n: int
) -> List[Tuple[int, int, int]]:
    """Find non-overlapping ≥``min_n``-token spans in ``target`` that
    occur more than once *within the same sequence*. Returns
    ``(second_start, second_end, first_start)`` triples: the second
    (or later) occurrence is the interior duplicate a gap-stitching
    engine could serve from the cache of the same request's earlier
    KV.

    Greedy: left-to-right scan, for each candidate window compute its
    hash, and if the same hash appeared at an earlier position within
    the sequence (and verifies on direct compare), record the
    duplicate and skip past it.
    """
    if len(target) < min_n:
        return []
    seen: Dict[int, List[int]] = {}
    duplicates: List[Tuple[int, int, int]] = []
    i = 0
    T = len(target)
    while i + min_n <= T:
        window = target[i : i + min_n]
        h = hash(tuple(window))
        earlier = seen.get(h, [])
        best_len = 0
        best_first = -1
        for p in earlier:
            if p + min_n > i:  # overlaps with i on the target side
                continue
            if target[p : p + min_n] != window:
                continue
            ext = min_n
            limit = min(T - i, i - p)  # don't let ext cross into i
            while ext < limit and target[i + ext] == target[p + ext]:
                ext += 1
            if ext > best_len:
                best_len = ext
                best_first = p
        if best_len >= min_n and best_first >= 0:
            duplicates.append((i, i + best_len, best_first))
            # Record the duplicate's range in the index so later
            # positions can also reference it.
            for q in range(best_len):
                seen.setdefault(hash(tuple(target[i + q : i + q + min_n])), []).append(
                    i + q
                )
            i += best_len
        else:
            seen.setdefault(h, []).append(i)
            i += 1
    return duplicates


def _find_matches_filtered(
    target: List[int],
    prior_sequences: List[List[int]],
    prior_sessions: List[str],
    index: Dict[int, List[Tuple[int, int]]],
    min_n: int,
    exclude_session: str,
) -> List[Tuple[int, int]]:
    """Find non-overlapping ≥``min_n``-token matches of ``target`` in
    any prior sequence with ``prior_sessions[seq_idx] !=
    exclude_session``. Each candidate is verified on the first
    ``min_n`` tokens (hash collisions can produce spurious hits).
    """
    matches: List[Tuple[int, int]] = []
    i = 0
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
            seq = prior_sequences[si]
            # Guard against hash collisions by verifying the first
            # min_n tokens directly.
            if seq[p : p + min_n] != window:
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


def per_request_stats(
    target: List[int],
    prior_sequences: List[List[int]],
    index: Dict[Tuple[int, ...], List[Tuple[int, int]]],
    min_n: int,
) -> Dict[str, float]:
    spans = find_matches(target, prior_sequences, index, min_n)
    covered = sum(end - start for start, end in spans)
    longest = max((end - start for start, end in spans), default=0)
    return {
        "target_len": len(target),
        "covered_tokens": covered,
        "longest_match": longest,
        "n_fragments": len(spans),
        "coverage_frac": covered / max(len(target), 1),
        "longest_frac": longest / max(len(target), 1),
    }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run(
    dataset: str,
    n_sessions: int,
    min_n: int,
    tokenizer_id: str,
    out_path: Path,
    max_tokens_per_session: int,
    repo_filter: str | None = None,
):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    fetch = DATASETS[dataset]
    label = dataset + (f" (repo={repo_filter})" if repo_filter else "")
    print(f"[info] loading up to {n_sessions} {label} sessions", flush=True)

    # Collect all requests across sessions, keeping provenance. Cap each
    # request to max_tokens_per_session to bound the index memory on
    # datasets with very long trajectories (SWE-smith in particular).
    if dataset == "swe-smith":
        source = fetch(tokenizer, n_sessions, repo_filter=repo_filter)
    else:
        if repo_filter:
            print(
                f"[warn] --repo-filter ignored for dataset={dataset}",
                flush=True,
            )
        source = fetch(tokenizer, n_sessions)
    requests: List[Tuple[str, int, List[int]]] = []  # (session_id, req_idx, ids)
    for session_id, prefixes in source:
        for ri, ids in enumerate(prefixes):
            capped = ids[:max_tokens_per_session]
            requests.append((session_id, ri, capped))
    print(
        f"[info] collected {len(requests)} requests "
        f"across {len({r[0] for r in requests})} sessions  "
        f"(avg {sum(len(r[2]) for r in requests) // max(len(requests), 1)} tokens)",
        flush=True,
    )

    # Build the prior pool incrementally: each request is analyzed
    # against the global window index accumulated so far, then its own
    # windows are added to the index (tagged by session so we can
    # exclude same-session matches at lookup time — same-session
    # priors are trivial prefix extensions and would dominate the
    # coverage metric without being informative).
    prior_sequences: List[List[int]] = []
    prior_sessions: List[str] = []
    tagged_index: Dict[Tuple[int, ...], List[Tuple[int, int]]] = {}
    per_request: List[dict] = []
    for idx, (session_id, req_idx, ids) in enumerate(requests):
        # Lookup: find matches against priors with a session_id != ours.
        spans = _find_matches_filtered(
            ids, prior_sequences, prior_sessions, tagged_index, min_n, session_id
        )
        covered = sum(end - start for start, end in spans)
        longest = max((end - start for start, end in spans), default=0)
        # Intra-request duplicates: same ≥min_n span appearing twice
        # inside this prompt. Gap-stitching could serve the second
        # occurrence from the KV of the first — the "same file read
        # twice in one session" scenario.
        dups = intra_request_duplicates(ids, min_n)
        dup_covered = sum(end - start for start, end, _ in dups)
        dup_longest = max((end - start for start, end, _ in dups), default=0)
        stats = {
            "target_len": len(ids),
            "covered_tokens": covered,
            "longest_match": longest,
            "n_fragments": len(spans),
            "coverage_frac": covered / max(len(ids), 1),
            "longest_frac": longest / max(len(ids), 1),
            "intra_dup_tokens": dup_covered,
            "intra_dup_longest": dup_longest,
            "intra_n_duplicates": len(dups),
            "intra_dup_frac": dup_covered / max(len(ids), 1),
            "session_id": session_id,
            "request_idx_in_session": req_idx,
        }
        per_request.append(stats)
        # Now add this request's windows to the global index.
        _add_sequence_to_index(tagged_index, ids, len(prior_sequences), min_n)
        prior_sequences.append(ids)
        prior_sessions.append(session_id)

    if not per_request:
        raise SystemExit("no requests analyzed (not enough sessions?)")

    def _summarize(xs, key):
        vs = [r[key] for r in xs]
        return {
            "mean": statistics.mean(vs),
            "median": statistics.median(vs),
            "p90": _percentile(vs, 0.90),
            "max": max(vs),
        }

    summary = {
        "dataset": dataset,
        "n_sessions": n_sessions,
        "min_match_tokens": min_n,
        "tokenizer": tokenizer_id,
        "n_requests_analyzed": len(per_request),
        "mean_target_len": statistics.mean(r["target_len"] for r in per_request),
        "coverage_frac": _summarize(per_request, "coverage_frac"),
        "longest_frac": _summarize(per_request, "longest_frac"),
        "n_fragments": _summarize(per_request, "n_fragments"),
        "intra_dup_frac": _summarize(per_request, "intra_dup_frac"),
        "intra_n_duplicates": _summarize(per_request, "intra_n_duplicates"),
        "frac_with_multi_segment": sum(1 for r in per_request if r["n_fragments"] >= 2)
        / len(per_request),
        "frac_with_intra_dup": sum(
            1 for r in per_request if r["intra_n_duplicates"] >= 1
        )
        / len(per_request),
        "per_request": per_request,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(
        f"[info] wrote {out_path}  ({len(per_request)} requests analyzed)",
        flush=True,
    )
    print("\n=== summary ===")
    _print_summary(summary)


def _percentile(xs: List[float], q: float) -> float:
    if not xs:
        return float("nan")
    xs_sorted = sorted(xs)
    k = int(round(q * (len(xs_sorted) - 1)))
    return xs_sorted[k]


def _print_summary(s: dict):
    print(f"dataset:   {s['dataset']}")
    print(
        f"requests:  {s['n_requests_analyzed']}  (mean target tokens = {s['mean_target_len']:.0f})"
    )
    print(f"min match: {s['min_match_tokens']} tokens")
    print()
    print(f"                     {'mean':>8} {'median':>8} {'p90':>8} {'max':>8}")
    for key in (
        "coverage_frac",
        "longest_frac",
        "n_fragments",
        "intra_dup_frac",
        "intra_n_duplicates",
    ):
        v = s[key]
        if key in ("n_fragments", "intra_n_duplicates"):
            print(
                f"{key:<20} {v['mean']:>8.2f} {v['median']:>8.0f} "
                f"{v['p90']:>8.0f} {v['max']:>8.0f}"
            )
        else:
            print(
                f"{key:<20} {v['mean']:>8.2f} {v['median']:>8.2f} "
                f"{v['p90']:>8.2f} {v['max']:>8.2f}"
            )
    mult = s["frac_with_multi_segment"]
    intra = s["frac_with_intra_dup"]
    print()
    print(f"fraction of requests with ≥2 disjoint cross-session fragments: {mult:.2f}")
    print(f"fraction of requests with ≥1 intra-request duplicate:          {intra:.2f}")
    print()
    print("coverage_frac      = target tokens inside SOME cross-session matching span")
    print("longest_frac       = longest single contiguous cross-session match / T")
    print("intra_dup_frac     = target tokens covered by an intra-request duplicate")
    print("intra_n_duplicates = # duplicated ≥min_n spans within the same prompt")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dataset",
        choices=sorted(DATASETS.keys()),
        default="hermes",
        help="Trace source to analyze.",
    )
    p.add_argument(
        "--n-sessions",
        type=int,
        default=30,
        help="How many sessions to pull from the streamed dataset.",
    )
    p.add_argument(
        "--min-match",
        type=int,
        default=128,
        help="Minimum contiguous match length that counts as reusable "
        "(corresponds to llama.cpp's --cache-reuse N).",
    )
    p.add_argument(
        "--tokenizer",
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="HF tokenizer id used for rendering. Numbers are tokenizer-dependent; "
        "pick one representative of the deployment target.",
    )
    p.add_argument(
        "--max-tokens-per-session",
        type=int,
        default=20000,
        help="Cap each session's token length before indexing. Keeps "
        "memory bounded on long trajectories (SWE-smith).",
    )
    p.add_argument(
        "--repo-filter",
        default=None,
        help="For swe-smith: keep only trajectories whose instance_id "
        "starts with this prefix (e.g. 'django' → 'django__django-*'). "
        "Ensures the session pool is same-codebase so file-recurrence "
        "is measurable.",
    )
    p.add_argument(
        "--output",
        required=True,
        help="Path to write the summary JSON.",
    )
    args = p.parse_args()
    run(
        args.dataset,
        args.n_sessions,
        args.min_match,
        args.tokenizer,
        Path(args.output),
        args.max_tokens_per_session,
        repo_filter=args.repo_filter,
    )


if __name__ == "__main__":
    main()
