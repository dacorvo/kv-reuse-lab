#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "datasets>=2.20",
#   "transformers>=5.5",
#   "jinja2>=3.0",
#   "matplotlib>=3.8",
#   "pyarrow>=15",
#   "huggingface_hub>=0.25",
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


_AGENTCAP_REQUIRED_COLS = ("request_id", "model", "captured_at", "request", "n_tokens")
_AGENTCAP_OPTIONAL_COLS = ("rendered_tokens",)


def _select_agentcap_cols(pf) -> list[str]:
    """Return the intersection of the parquet's actual columns with the
    set this analyzer cares about. Older exports carry
    ``rendered_tokens`` baked in; newer exports drop it and let the
    analyzer recompute on the fly. ``request`` may be either a struct
    (legacy) or a JSON string (post-streaming-export refactor).
    """
    have = set(pf.schema_arrow.names)
    cols = [c for c in _AGENTCAP_REQUIRED_COLS if c in have]
    cols += [c for c in _AGENTCAP_OPTIONAL_COLS if c in have]
    return cols


def _stream_agentcap_rows(source: str):
    """Iterate dict rows from one of:
      - ``hf://buckets/<owner>/<name>[/<prefix>]``  (streamed via HfFileSystem)
      - a directory of ``*.parquet``               (streamed via pyarrow)
      - a single ``*.parquet`` file                (streamed via pyarrow)
      - an HF Dataset folder produced by ``agentcap export --format hf``

    No file download; bucket reads stream straight from the Hub via fsspec.
    """
    import pyarrow.parquet as pq

    if source.startswith("hf://"):
        from huggingface_hub import HfFileSystem

        fs = HfFileSystem()
        # ``source`` may point at a single .parquet file, a directory, or
        # a bucket prefix. fs.info() resolves which.
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
                cols = _select_agentcap_cols(pf)
                for batch in pf.iter_batches(batch_size=32, columns=cols):
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
        if not files:
            raise FileNotFoundError(f"no parquet files in {source!r}")
    else:
        raise FileNotFoundError(f"agentcap source not recognised: {source!r}")
    for path in files:
        pf = pq.ParquetFile(str(path))
        cols = _select_agentcap_cols(pf)
        for batch in pf.iter_batches(batch_size=32, columns=cols):
            for row in batch.to_pylist():
                yield row


def _decode_request(request) -> dict:
    """Newer exports serialize ``request`` as a JSON string (works
    around polymorphic tool schemas blowing up Arrow schema inference).
    Older ones leave it as a struct/dict. Normalize to dict."""
    if request is None:
        return {}
    if isinstance(request, str):
        try:
            return json.loads(request)
        except json.JSONDecodeError:
            return {}
    return request


def _render_tokens_for_row(row: dict, _tok_cache: dict) -> List[int]:
    """Render token ids for a row that lacks ``rendered_tokens``. Uses
    the row's own ``model`` to load (and memoize) the right tokenizer,
    then applies its chat template to ``request.messages`` (with tools
    if present), with ``add_generation_prompt=True`` to match what the
    serving engine would actually prefill at request time.

    Returns ``[]`` if rendering fails (silent skip — the caller treats
    short/empty token lists as no-op rows)."""
    from transformers import AutoTokenizer

    model = row.get("model")
    if not model:
        return []
    tok = _tok_cache.get(model)
    if tok is None:
        tok = AutoTokenizer.from_pretrained(model)
        _tok_cache[model] = tok
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


def _agentcap_sessions(
    tokenizer,
    n_sessions: int,
    *,
    source: str,
    model_filter: str | None = None,
) -> Iterable[Tuple[str, List[List[int]]]]:
    """Stream agentcap-exported rows, group by session, yield prefilled
    request token lists in capture order.

    Session id = stable hash of the first user message's content. Every
    chat-completion the agent makes on one task shares that first user
    message verbatim — Hermes's internal skill_view / memory / tool-call
    loops, plus the multi-turn follow-ups — so this groups all of them
    together as one session.

    Tokens come from the row's ``rendered_tokens`` (model-specific,
    baked at export time). The ``tokenizer`` parameter is accepted for
    interface parity with the other adapters but ignored.
    """
    import hashlib

    del tokenizer  # the rendered tokens are already model-specific

    grouped: Dict[str, List[Tuple[int, int, List[int]]]] = {}
    skipped_no_user = 0
    skipped_model = 0
    skipped_no_tokens = 0
    tok_cache: dict = {}
    for row in _stream_agentcap_rows(source):
        if model_filter and row.get("model") != model_filter:
            skipped_model += 1
            continue
        req = _decode_request(row.get("request"))
        msgs = req.get("messages") or []
        first_user = next(
            (m.get("content") for m in msgs if m.get("role") == "user"),
            None,
        )
        if not first_user:
            skipped_no_user += 1
            continue
        rendered = row.get("rendered_tokens")
        if rendered:
            ids = list(rendered)
        else:
            ids = _render_tokens_for_row(row, tok_cache)
        if not ids:
            skipped_no_tokens += 1
            continue
        sid = hashlib.sha1(str(first_user).encode("utf-8")).hexdigest()[:12]
        grouped.setdefault(sid, []).append(
            (
                int(row.get("captured_at", 0)),
                int(row.get("n_tokens", 0) or len(ids)),
                ids,
            )
        )

    if not grouped:
        raise SystemExit(
            f"no agentcap rows yielded from {source!r} "
            f"(skipped: {skipped_no_user} without a user message, "
            f"{skipped_model} not matching --model-filter, "
            f"{skipped_no_tokens} that failed to tokenize)"
        )

    # Stable session order: by earliest captured_at within each group.
    sids = sorted(grouped.keys(), key=lambda s: min(t for t, _, _ in grouped[s]))
    yielded = 0
    for sid in sids:
        if yielded >= n_sessions:
            break
        items = grouped[sid]
        # Within a session: capture-time then prompt-length tiebreak.
        items.sort(key=lambda x: (x[0], x[1]))
        prefixes = [ids for _, _, ids in items]
        yield (f"agentcap-{sid}", prefixes)
        yielded += 1


DATASETS = {
    "hermes": _hermes_sessions,
    "swe-smith": _swe_smith_sessions,
    "claude-hs": _claude_hyperswitch_sessions,
    "agentcap": _agentcap_sessions,
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


def _max_common_prefix(
    target: List[int],
    prior_sequences: List[List[int]],
    prior_sessions: List[str],
    exclude_session: str,
) -> int:
    """Return the longest leading-token run that ``target`` shares
    with any allowed prior — i.e., ``max_{prior} k s.t.
    prior[:k] == target[:k]``. This is the byte-stable common
    prefix (CP) and also the upper bound on what a standard prefix
    cache could serve for ``target``. Returns 0 if no prior shares
    so much as the first token.
    """
    cp = 0
    for si, prior in enumerate(prior_sequences):
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


def _find_matches_filtered(
    target: List[int],
    prior_sequences: List[List[int]],
    prior_sessions: List[str],
    index: Dict[int, List[Tuple[int, int]]],
    min_n: int,
    exclude_session: str,
    skip_to: int = 0,
) -> List[Tuple[int, int]]:
    """Find non-overlapping ≥``min_n``-token matches of ``target`` in
    any prior sequence with ``prior_sessions[seq_idx] !=
    exclude_session``. Each candidate is verified on the first
    ``min_n`` tokens (hash collisions can produce spurious hits).

    ``skip_to`` lets the caller start the target-side scan past a
    fixed offset — typically the CP boundary, so the matcher returns
    only post-prefix matches.
    """
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
    agentcap_source: str | None = None,
    model_filter: str | None = None,
):
    from transformers import AutoTokenizer

    if dataset == "agentcap":
        if not agentcap_source:
            raise SystemExit(
                "--dataset agentcap requires --agentcap-source "
                "(local parquet path or hf://buckets/<owner>/<name>[/<prefix>])"
            )
        tokenizer = None  # rendered_tokens already encode the right tokenizer
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)

    fetch = DATASETS[dataset]
    if dataset == "agentcap":
        label = f"agentcap ({agentcap_source})"
    else:
        label = dataset + (f" (repo={repo_filter})" if repo_filter else "")
    print(f"[info] loading up to {n_sessions} {label} sessions", flush=True)

    # Collect all requests across sessions, keeping provenance. Cap each
    # request to max_tokens_per_session to bound the index memory on
    # datasets with very long trajectories (SWE-smith in particular).
    if dataset == "swe-smith":
        source = fetch(tokenizer, n_sessions, repo_filter=repo_filter)
    elif dataset == "agentcap":
        if repo_filter:
            print("[warn] --repo-filter ignored for dataset=agentcap", flush=True)
        source = fetch(
            tokenizer,
            n_sessions,
            source=agentcap_source,
            model_filter=model_filter,
        )
    else:
        if repo_filter:
            print(
                f"[warn] --repo-filter ignored for dataset={dataset}",
                flush=True,
            )
        if model_filter:
            print(
                f"[warn] --model-filter ignored for dataset={dataset}",
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
        # CP: longest leading-token run target shares with any allowed
        # prior. A standard prefix cache could serve bytes [0, cp) for
        # this target. Anything past cp is post-prefix territory — the
        # regime a non-trivial cache (llama.cpp --cache-reuse, reagent's
        # mechanism) is for.
        cp = _max_common_prefix(
            ids, prior_sequences, prior_sessions, session_id
        )
        # Find post-prefix matches only (skip the byte-stable region).
        spans = _find_matches_filtered(
            ids, prior_sequences, prior_sessions, tagged_index, min_n,
            session_id, skip_to=cp,
        )
        post_covered = sum(end - start for start, end in spans)
        post_longest = max((end - start for start, end in spans), default=0)
        # Total = CP + post-prefix matches (non-overlapping by construction).
        # Note: cp counts as "coverage" only if cp >= min_n, to stay
        # comparable with the existing min-n floor. Below min_n the
        # prefix is too short to be meaningfully cached anyway.
        cp_covered = cp if cp >= min_n else 0
        total_covered = cp_covered + post_covered
        total_longest = max(cp_covered, post_longest)
        # Intra-request duplicates: same ≥min_n span appearing twice
        # inside this prompt. Gap-stitching could serve the second
        # occurrence from the KV of the first — the "same file read
        # twice in one session" scenario.
        dups = intra_request_duplicates(ids, min_n)
        dup_covered = sum(end - start for start, end, _ in dups)
        dup_longest = max((end - start for start, end, _ in dups), default=0)
        stats = {
            "target_len": len(ids),
            # Total recurrence (CP + post-prefix). Comparable to old
            # coverage_frac numbers in prior reports.
            "covered_tokens": total_covered,
            "longest_match": total_longest,
            "n_fragments": len(spans) + (1 if cp_covered else 0),
            "coverage_frac": total_covered / max(len(ids), 1),
            "longest_frac": total_longest / max(len(ids), 1),
            # CP: served by a standard prefix cache.
            "cp_tokens": cp,
            "cp_frac": cp / max(len(ids), 1),
            # Post-prefix: only served by a non-trivial cache.
            "post_prefix_covered_tokens": post_covered,
            "post_prefix_longest_match": post_longest,
            "post_prefix_n_fragments": len(spans),
            "post_prefix_coverage_frac": post_covered / max(len(ids), 1),
            "post_prefix_longest_frac": post_longest / max(len(ids), 1),
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
        "cp_frac": _summarize(per_request, "cp_frac"),
        "post_prefix_coverage_frac": _summarize(
            per_request, "post_prefix_coverage_frac"
        ),
        "post_prefix_longest_frac": _summarize(
            per_request, "post_prefix_longest_frac"
        ),
        "post_prefix_n_fragments": _summarize(
            per_request, "post_prefix_n_fragments"
        ),
        "intra_dup_frac": _summarize(per_request, "intra_dup_frac"),
        "intra_n_duplicates": _summarize(per_request, "intra_n_duplicates"),
        "frac_with_multi_segment_post_prefix": sum(
            1 for r in per_request if r["post_prefix_n_fragments"] >= 2
        ) / len(per_request),
        "frac_with_any_post_prefix": sum(
            1 for r in per_request if r["post_prefix_covered_tokens"] > 0
        ) / len(per_request),
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
    print(f"                              {'mean':>8} {'median':>8} {'p90':>8} {'max':>8}")
    for key in (
        "coverage_frac",
        "cp_frac",
        "post_prefix_coverage_frac",
        "post_prefix_longest_frac",
        "post_prefix_n_fragments",
        "longest_frac",
        "n_fragments",
        "intra_dup_frac",
        "intra_n_duplicates",
    ):
        v = s[key]
        if key in ("n_fragments", "intra_n_duplicates", "post_prefix_n_fragments"):
            print(
                f"{key:<29} {v['mean']:>8.2f} {v['median']:>8.0f} "
                f"{v['p90']:>8.0f} {v['max']:>8.0f}"
            )
        else:
            print(
                f"{key:<29} {v['mean']:>8.2f} {v['median']:>8.2f} "
                f"{v['p90']:>8.2f} {v['max']:>8.2f}"
            )
    mult_post = s["frac_with_multi_segment_post_prefix"]
    any_post = s["frac_with_any_post_prefix"]
    intra = s["frac_with_intra_dup"]
    print()
    print(f"fraction of requests with ≥1 post-prefix match:                {any_post:.2f}")
    print(f"fraction of requests with ≥2 disjoint post-prefix fragments:   {mult_post:.2f}")
    print(f"fraction of requests with ≥1 intra-request duplicate:          {intra:.2f}")
    print()
    print("coverage_frac              = target tokens matching SOMETHING earlier (CP + post-prefix)")
    print("cp_frac                    = target tokens served by a standard prefix cache")
    print("post_prefix_coverage_frac  = target tokens past CP that recur — only a non-trivial cache serves these")
    print("post_prefix_longest_frac   = longest single post-prefix match / T")
    print("post_prefix_n_fragments    = # disjoint post-prefix matches per request")
    print("longest_frac / n_fragments = total view (CP + post-prefix). Kept for back-compat with prior reports.")
    print("intra_dup_frac             = target tokens covered by an intra-request duplicate")


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
        "--agentcap-source",
        default=None,
        help="For dataset=agentcap: source URI. Either a local parquet "
        "file/dir, an HF Dataset folder, or hf://buckets/<owner>/<name>"
        "[/<prefix>]. Bucket reads stream via fsspec — no download.",
    )
    p.add_argument(
        "--model-filter",
        default=None,
        help="For dataset=agentcap: only keep rows whose captured "
        "request.model equals this. Useful when one bucket prefix mixes "
        "multiple models (the canonical agentcap layout doesn't, but a "
        "user-curated prefix might).",
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
        agentcap_source=args.agentcap_source,
        model_filter=args.model_filter,
    )


if __name__ == "__main__":
    main()
