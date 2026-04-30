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

Datasets (``--dataset``):
    - ``swe-smith`` (default): ``SWE-bench/SWE-smith-trajectories``,
      per-message JSON with optional tool_calls. ``--repo`` filters
      by ``instance_id`` prefix (e.g. ``django``).
    - ``nemotron-swe``: ``nvidia/Nemotron-RL-Agentic-SWE-Pivot-v1``,
      OpenAI Responses-API flat items (message / reasoning /
      function_call / function_call_output) under
      ``responses_create_params.input``. ``--repo`` filters by
      ``metadata.instance_id`` prefix (e.g. ``pandas-dev``).

Each trajectory's final chat-template rendering is one "session";
each ordered pair (A, B) of trajectories produces 0+ matching
byte-exact spans of length ≥ ``--min-match``. Pairs with no
matches are skipped.

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
def _sliding_window(model) -> int | None:
    for root in (getattr(model, "model", None), model):
        if root is None:
            continue
        for attr in ("language_model", ""):
            inner = getattr(root, attr, None) if attr else root
            if inner is None:
                continue
            cfg = getattr(inner, "config", None)
            sw = getattr(cfg, "sliding_window", None) if cfg is not None else None
            types = getattr(cfg, "layer_types", None) if cfg is not None else None
            if (
                sw is not None
                and types is not None
                and any(t == "sliding_attention" for t in types)
            ):
                return int(sw)
    return None


@torch.no_grad()
def prefill_and_snapshot(model, input_ids: torch.Tensor, offload: str = "cpu"):
    """Chunked prefill that returns per-layer FULL-sequence (K, V) snapshots.

    On hybrid models with sliding-window attention (Gemma-3/3n/4), the
    sliding cache truncates K/V to the last ``sliding_window-1`` tokens,
    so a single full-sequence forward followed by a snapshot loses the
    pre-truncation history. Splice schemes (especially scheme B in
    measure_multi_splice_b.py) need the full-length K/V on every layer
    to inject a cached match span at any absolute position.

    The fix is chunked prefill aligned on the sliding-window length:
    each chunk of length ``sliding_window-1`` fits in the sliding cache
    without eviction, so reading the cache's tail after each chunk
    yields exactly that chunk's K/V. Concatenating across chunks gives
    the full per-layer history. Snapshots are offloaded to CPU.

    Models without sliding attention (Llama etc.) do a single forward.
    """
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    T = input_ids.shape[0]
    sw = _sliding_window(model)
    chunk = (sw - 1) if sw is not None else T

    past = None
    n_layers: int | None = None
    accum_k: List[List[torch.Tensor]] | None = None
    accum_v: List[List[torch.Tensor]] | None = None

    for s in range(0, T, chunk):
        e = min(s + chunk, T)
        chunk_ids = input_ids[s:e].unsqueeze(0)
        positions = torch.arange(s, e, device=device)
        out = model(
            input_ids=chunk_ids,
            past_key_values=past,
            position_ids=positions.unsqueeze(0),
            cache_position=positions,
            use_cache=True,
            logits_to_keep=1,
        )
        past = out.past_key_values
        # Sync before reading per-layer KV — see comment in
        # multi_splice_forward for the full rationale.
        if torch.cuda.is_available():
            for d in range(torch.cuda.device_count()):
                torch.cuda.synchronize(d)
        if n_layers is None:
            n_layers = _num_layers(past)
            accum_k = [[] for _ in range(n_layers)]
            accum_v = [[] for _ in range(n_layers)]
        chunk_len = e - s
        for li in range(n_layers):
            k_tensor, v_tensor = _layer_kv_tensors(past, li)
            # chunk_len <= sliding_window-1 guarantees that the last
            # `chunk_len` entries of the cache are exactly this chunk's
            # K/V, on both sliding and full layers.
            new_k = k_tensor[..., -chunk_len:, :].detach().to(offload)
            new_v = v_tensor[..., -chunk_len:, :].detach().to(offload)
            accum_k[li].append(new_k)
            accum_v[li].append(new_v)
        del out

    kvs: List[Tuple[torch.Tensor, torch.Tensor]] = []
    assert n_layers is not None and accum_k is not None and accum_v is not None
    for li in range(n_layers):
        kvs.append((torch.cat(accum_k[li], dim=-2), torch.cat(accum_v[li], dim=-2)))
    del past
    return kvs


# ---------------------------------------------------------------------------
# Datasets — see ``_DATASETS`` below for the dispatch table.
# ---------------------------------------------------------------------------


def _strify(c) -> str:
    if c is None:
        return ""
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        out = []
        for x in c:
            if isinstance(x, dict):
                out.append(x.get("text") or x.get("content") or "")
            else:
                out.append(str(x))
        return " ".join(out)
    return str(c)


def _chatml_msgs(row) -> List[Dict[str, str]]:
    """SWE-smith trajectory → normalized [{role, content}, ...]."""
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


def _nemotron_msgs(row) -> List[Dict[str, str]]:
    """Nemotron-RL-Agentic-SWE-Pivot-v1 trajectory (OpenAI Responses-API
    flat-item list) → normalized [{role, content}, ...].

    Item types:
        - plain {role, content}: kept as-is for system/user/assistant
        - type=message: assistant text response (content is list of
          {type:'output_text', text:...} dicts)
        - type=reasoning: assistant chain-of-thought (summary list)
        - type=function_call: assistant tool call (name + arguments)
        - type=function_call_output: tool result, mapped to user role
          (consistent with the swe-smith path: tool→user)
    """
    items = (row.get("responses_create_params") or {}).get("input") or []
    norm: List[Dict[str, str]] = []
    for m in items:
        t = m.get("type")
        if t is None:
            role = m.get("role", "")
            if role in ("system", "user", "assistant"):
                norm.append({"role": role, "content": _strify(m.get("content"))})
        elif t == "message":
            role = m.get("role", "assistant")
            if role not in ("system", "user", "assistant"):
                role = "assistant"
            norm.append({"role": role, "content": _strify(m.get("content"))})
        elif t == "reasoning":
            text = _strify(m.get("summary"))
            if text:
                norm.append({"role": "assistant", "content": f"<think>{text}</think>"})
        elif t == "function_call":
            payload = json.dumps(
                {"name": m.get("name"), "arguments": m.get("arguments")},
                ensure_ascii=False,
            )
            norm.append({"role": "assistant", "content": payload})
        elif t == "function_call_output":
            norm.append(
                {
                    "role": "user",
                    "content": "Tool result: " + _strify(m.get("output")),
                }
            )
    return norm


_DATASETS: Dict[str, Dict] = {
    # SWE-smith trajectories: per-message JSON with tool_calls.
    "swe-smith": {
        "id": "SWE-bench/SWE-smith-trajectories",
        "split": "tool",
        "id_field": "instance_id",
        "to_msgs": _chatml_msgs,
    },
    # Nemotron-RL-Agentic-SWE-Pivot-v1: OpenAI Responses-API flat
    # items at responses_create_params.input. instance_id lives in
    # metadata.instance_id.
    "nemotron-swe": {
        "id": "nvidia/Nemotron-RL-Agentic-SWE-Pivot-v1",
        "split": "train",
        "id_field": ("metadata", "instance_id"),
        "to_msgs": _nemotron_msgs,
    },
}


def _id_of(row, field) -> str:
    if isinstance(field, tuple):
        cur = row
        for k in field:
            if cur is None:
                return ""
            cur = cur.get(k) if isinstance(cur, dict) else None
        return cur or ""
    return row.get(field, "") or ""


def load_sessions(
    tokenizer,
    n_sessions: int,
    repo_prefix: str,
    max_tokens: int,
    dataset: str = "swe-smith",
    with_roles: bool = False,
) -> List[Tuple[str, List[int]]] | List[Tuple[str, List[int], List[str]]]:
    """Return up to n_sessions session tuples from the selected dataset,
    filtered to ids starting with ``repo_prefix``, rendered through
    the model's chat template, capped at max_tokens.

    With ``with_roles=False`` (default): list of ``(iid, ids)``.
    With ``with_roles=True``: list of ``(iid, ids, roles)`` where
    ``roles[i]`` is the message role string for token ``ids[i]``,
    suitable to pass into ``find_matches`` for input-only matching.

    Datasets available: see ``_DATASETS``.
    """
    from datasets import load_dataset

    cfg = _DATASETS.get(dataset)
    if cfg is None:
        raise ValueError(f"unknown dataset {dataset!r}; choose from {list(_DATASETS)}")
    ds = load_dataset(cfg["id"], split=cfg["split"], streaming=True)
    sessions: List[Tuple[str, List[int]]] = []
    for row in ds:
        if len(sessions) >= n_sessions:
            break
        iid = _id_of(row, cfg["id_field"])
        if not iid.startswith(repo_prefix):
            continue
        msgs = cfg["to_msgs"](row)
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
        if with_roles:
            roles = _role_per_token(tokenizer, msgs)[: len(ids)]
            # Pad if rendering quirks left roles shorter than ids
            while len(roles) < len(ids):
                roles.append("template")
            sessions.append((iid, ids, roles))
        else:
            sessions.append((iid, ids))
    if len(sessions) < n_sessions:
        raise RuntimeError(
            f"only {len(sessions)}/{n_sessions} {repo_prefix}* sessions found "
            f"in {cfg['id']!r} "
            f"(consider widening --max-tokens-per-session)"
        )
    return sessions


# ---------------------------------------------------------------------------
# Match finding
# ---------------------------------------------------------------------------


def _role_per_token(tokenizer, msgs) -> List[str]:
    """Render messages incrementally and label each token with the
    role of the message it belongs to. Returns a list of role strings
    of length == len(full-rendered tokens). Tokens added by the chat
    template AFTER the last message (e.g. add_generation_prompt
    trailers) are labelled "template".
    """
    roles: List[str] = []
    prev_len = 0
    for i in range(len(msgs)):
        ids = tokenizer.apply_chat_template(
            msgs[: i + 1], tokenize=True, return_dict=True
        )["input_ids"]
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        cur_len = len(ids)
        roles.extend([msgs[i]["role"]] * (cur_len - prev_len))
        prev_len = cur_len
    return roles


def _is_input_role(role: str) -> bool:
    """Input-only K/V cache rule: cache K/V from prefilled input
    tokens (system / user / tool messages) but NEVER from model-
    generated tokens (assistant turns, including any reasoning the
    agent emitted as part of the trajectory). The trajectories in
    swe-smith / nemotron-swe come from real agent runs, so each
    "assistant" message in the rendered conversation is in fact
    *generated text* the model produced — its K/V is context-bound
    to that specific session's upstream and CANNOT be safely spliced
    across sessions even when the surface text matches.
    """
    return role in ("system", "user", "tool")


def find_matches(
    target: List[int],
    prior: List[int],
    min_n: int,
    target_roles: List[str] | None = None,
    prior_roles: List[str] | None = None,
) -> List[Tuple[int, int, int, int]]:
    """Return non-overlapping (b_start, b_end, a_start, a_end) match
    spans where prior[a_start:a_end] == target[b_start:b_end],
    each of length ≥ min_n. Target-side greedy.

    If ``target_roles`` and ``prior_roles`` are provided (per-token
    role strings of the same length as ``target`` and ``prior``), any
    match span that contains a non-input role token on EITHER side is
    rejected — see _is_input_role for the rule. Cache reuse only
    splices K/V from input tokens; matches that cross into a model-
    generated assistant turn would transfer K/V the model produced
    while attending to a session-specific upstream and is unsafe by
    construction.
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
        # Trim the match if either side enters a non-input role.
        if target_roles is not None and prior_roles is not None:
            allowed = best_len
            for k in range(best_len):
                if k < len(target_roles) - i and not _is_input_role(
                    target_roles[i + k]
                ):
                    allowed = k
                    break
                if k < len(prior_roles) - best_p and not _is_input_role(
                    prior_roles[best_p + k]
                ):
                    allowed = k
                    break
            if allowed < min_n:
                i += 1
                continue
            best_len = allowed
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
    # Sync ALL devices BEFORE any teardown / cache-free: device_map="balanced"
    # spreads the prefill across cuda:0..N async, and `torch.cuda.empty_cache()`
    # below could otherwise reclaim memory still being written to by the
    # prefill kernels on a different device, leaving stale pointers that
    # surface as illegal memory access on the next CUDA op.
    if torch.cuda.is_available():
        for d in range(torch.cuda.device_count()):
            torch.cuda.synchronize(d)
    del out
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
            # Pre-read sync on this layer's GPU: ensures the prefill's
            # writes here are visible before we slice / .to into it.
            if k_tensor.is_cuda:
                torch.cuda.synchronize(k_tensor.device)
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
    # device_map="balanced" puts cache layers on different GPUs. The
    # cross-GPU `kA.to(target_device)` + `copy_` sequence inside
    # `write_kv_span` is async; without a sync, the next forward can
    # read torn / stale K/V on whichever device finished last and crash
    # with an illegal memory access. Sync every visible CUDA device once
    # after the overwrite loop completes.
    if torch.cuda.is_available():
        for d in range(torch.cuda.device_count()):
            torch.cuda.synchronize(d)
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

    print(
        f"[info] loading up to {args.n_sessions} {args.repo}* sessions "
        f"from {args.dataset!r}",
        flush=True,
    )
    sessions = load_sessions(
        tok,
        args.n_sessions,
        args.repo,
        args.max_tokens_per_session,
        dataset=args.dataset,
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
                "match_spans": [
                    [int(bs), int(be), int(as_), int(ae)] for bs, be, as_, ae in matches
                ],
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
    p.add_argument(
        "--dataset",
        default="swe-smith",
        choices=sorted(_DATASETS),
        help="Source dataset for trajectories. swe-smith uses "
        "SWE-bench/SWE-smith-trajectories (per-message JSON); "
        "nemotron-swe uses nvidia/Nemotron-RL-Agentic-SWE-Pivot-v1 "
        "(OpenAI Responses-API flat items). Both render through the "
        "model's chat template.",
    )
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
