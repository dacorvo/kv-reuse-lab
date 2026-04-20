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
# ]
# ///
"""Logit deviation of naive cross-prompt KV cache reuse on agent prompts.

Scenario — cached tool-*result* reused across sessions
-------------------------------------------------------
Only client/agent-side tokens are worth caching: anything the model
generated, the model already has the KV for. The interesting cacheable
block is a *tool result* — a chunk of text the agent pastes into the
prompt (file contents, API response, retrieval payload) that the model
must prefill but did not produce.

A realistic agent request ends at the tool-result turn; the model is
then asked to generate the next assistant turn. Hermes `func_calling`
examples have exactly this shape:

    [system][user][assistant: tool_call][tool: tool_response]  ← prompt
                                                               ← model
                                                                 decodes
                                                                 the next
                                                                 assistant
                                                                 summary

The cacheable block is the `tool` turn (~400 tokens on average). If
the same file is read again in a later session whose system prompt /
earlier turns differ in length, the tool-result block sits at a
different absolute position.

  baseline_msgs = [system   , user, asst_call, tool_result]
  drifted_msgs  = [system_xN, user, asst_call, tool_result]   (same
                                                               tool-result
                                                               content)

We render both with `add_generation_prompt=True` so the last prompt
token is the one that queries "what does the assistant say next?" —
exactly what the LLM is asked to predict in a real inference call.

  baseline_KV   = prefill(render(baseline, no-gen-prompt))
                  → KV of the last L tokens of the tool turn.
  fresh_logits  = prefill(render(drifted_msgs, with gen-prompt))
                  → logits at the last position (first generation
                  token distribution).
                  Then greedy-decode `gen_tokens` more tokens to get
                  `fresh_text`.
  reused_logits = native prefill of drifted through tool turn; overwrite
                  the last L KV entries per layer with baseline chunk
                  KV; forward the generation-prompt suffix with explicit
                  cache_position → logits at the last position.
                  Then greedy-decode `gen_tokens` more tokens with the
                  same (overwritten + advanced) cache to get
                  `reused_text`.

Only the system prompt is inflated, and only by appending semantically
redundant copies of its own content. The expected next-token
distribution is therefore unchanged; any fresh-vs-reused divergence is
attributable to the cached chunk's RoPE phase / context-conditioning
mismatch at the drifted position.

Per-Δ aggregated over N examples:
  - mean_kl                   : KL(fresh || reused) at first generation
                                token (nats)
  - stdev_kl
  - agree_rate                : top-1(fresh) == top-1(reused)
  - mean_top5_overlap         : top-5 set overlap (robust to ties)
  - mean_fresh_entropy        : entropy of fresh distribution (nats)
  - mean_sim_fresh_reused     : cosine similarity of sentence-embedded
                                fresh_text and reused_text. Headline
                                "does reuse change what the agent says?"
  - mean_sim_reused_reference : cosine similarity of reused_text and
                                the dataset's gold assistant reply
                                (quality-vs-reference proxy).
  - mean_actual_delta         : achieved token drift.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


ROLE_MAP = {
    "system": "system",
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    # Tool responses are carried as user-role turns with
    # <tool_response>…</tool_response> markers in the content.
    # Gemma/Llama chat templates reject a distinct `tool` role.
    "tool": "user",
    "function": "user",
}


def hermes_to_messages(ex) -> List[Dict[str, str]]:
    out = []
    for t in ex.get("conversations") or ex.get("messages") or []:
        role = t.get("from") or t.get("role") or ""
        content = t.get("value") or t.get("content") or ""
        r = ROLE_MAP.get(role, role)
        if not r:
            continue
        out.append({"role": r, "content": content})
    return out


def render_messages(
    tokenizer, msgs, add_generation_prompt: bool = False
) -> torch.Tensor:
    enc = tokenizer.apply_chat_template(
        msgs,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        return_tensors="pt",
        return_dict=True,
    )
    ids = enc["input_ids"] if hasattr(enc, "__getitem__") else enc
    if ids.ndim == 2:
        ids = ids[0]
    return ids


def reference_assistant_response(ex) -> str:
    """Return the text of the assistant turn that follows the tool turn
    in a Hermes example — the "gold" response we'd compare against.
    Returns an empty string if the example has no such turn.
    """
    convs = ex.get("conversations") or []
    saw_tool = False
    for t in convs:
        role = t.get("from") or t.get("role") or ""
        if saw_tool and role in ("gpt", "assistant"):
            return t.get("value") or t.get("content") or ""
        if role == "tool":
            saw_tool = True
    return ""


def prompt_msgs_through_tool(ex) -> List[Dict[str, str]]:
    """Return the Hermes conversation truncated up to and including the
    first tool-result turn. Drops the trailing assistant summary so the
    prompt ends where a real agent would — before decoding the next
    assistant turn.

    Works on the raw `conversations` field so we can see `from=tool`
    before role mapping folds it into `user`.
    """
    convs = ex.get("conversations") or []
    truncated = []
    saw_tool = False
    for t in convs:
        truncated.append(t)
        if (t.get("from") or t.get("role")) == "tool":
            saw_tool = True
            break
    if not saw_tool:
        return []
    return hermes_to_messages({"conversations": truncated})


def load_hermes_examples(
    n_base: int, min_tokens_fn, config: str = "func_calling"
) -> List[dict]:
    """Stream Hermes `config` subset, keep examples with a tool turn
    that pass the per-tokenizer length check."""
    from datasets import load_dataset

    ds = load_dataset(
        "NousResearch/hermes-function-calling-v1", config, split="train", streaming=True
    )
    base: List[dict] = []
    for ex in ds:
        if len(base) >= n_base:
            break
        convs = ex.get("conversations") or []
        if not any((t.get("from") or t.get("role")) == "tool" for t in convs):
            continue
        try:
            if min_tokens_fn(ex):
                base.append(ex)
        except Exception:
            continue
    if len(base) < n_base:
        raise RuntimeError(f"only {len(base)}/{n_base} base examples found")
    return base


def inflate_system_turn(
    tokenizer, base_msgs, target_delta: int
) -> Tuple[List[Dict[str, str]], int]:
    """Append a tokenized prefix of the system content back onto itself
    until the rendered stream has grown by ≥ target_delta tokens.

    Token-level granularity: we tokenize the original system content
    once, pick K tokens from it via binary search, decode back to text,
    and append. Re-rendering the full message list gives the true drift.

    Returns (drifted_msgs, actual_delta_tokens).
    """
    T0 = render_messages(tokenizer, base_msgs).shape[0]
    if target_delta <= 0:
        return base_msgs, 0

    system_idx = next(
        (i for i, m in enumerate(base_msgs) if m["role"] == "system"), None
    )
    if system_idx is None:
        return base_msgs, 0

    original = base_msgs[system_idx]["content"]
    sys_tokens = tokenizer(original, add_special_tokens=False)["input_ids"]
    if not sys_tokens:
        return base_msgs, 0

    def build(k: int) -> Tuple[List[Dict[str, str]], int]:
        """Append k tokens' worth of redundant system content."""
        # Wrap around if k exceeds the original length.
        repeats = (k + len(sys_tokens) - 1) // len(sys_tokens)
        repeated = sys_tokens * max(1, repeats)
        append_ids = repeated[:k]
        append_text = tokenizer.decode(append_ids)
        trial_msgs = list(base_msgs)
        trial_msgs[system_idx] = {
            "role": "system",
            "content": original + "\n" + append_text,
        }
        trial_ids = render_messages(tokenizer, trial_msgs)
        return trial_msgs, trial_ids.shape[0] - T0

    # Binary search for the smallest k such that achieved delta >= target.
    lo, hi = 1, max(target_delta * 4, 4 * len(sys_tokens))
    # Grow hi until achievable delta exceeds target (handles BOS/template overhead).
    msgs_hi, d_hi = build(hi)
    while d_hi < target_delta:
        hi *= 2
        if hi > 1_000_000:
            return msgs_hi, d_hi
        msgs_hi, d_hi = build(hi)
    while lo < hi:
        mid = (lo + hi) // 2
        _, d_mid = build(mid)
        if d_mid < target_delta:
            lo = mid + 1
        else:
            hi = mid
    return build(lo)


def _stack_kv(pkv, li):
    if hasattr(pkv, "layers"):
        return pkv.layers[li].keys, pkv.layers[li].values
    if hasattr(pkv, "key_cache"):
        return pkv.key_cache[li], pkv.value_cache[li]
    return pkv[li][0], pkv[li][1]


def _num_layers(pkv):
    if hasattr(pkv, "layers"):
        return len(pkv.layers)
    if hasattr(pkv, "key_cache"):
        return len(pkv.key_cache)
    return len(pkv)


@torch.no_grad()
def prefill_full(
    model, input_ids: torch.Tensor
) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    out = model(input_ids=input_ids.unsqueeze(0), use_cache=True)
    pkv = out.past_key_values
    kvs = [
        (
            _stack_kv(pkv, li)[0].detach(),
            _stack_kv(pkv, li)[1].detach(),
        )
        for li in range(_num_layers(pkv))
    ]
    logits = out.logits[0].detach()
    del out, pkv
    return kvs, logits


def slice_tail(kvs, n: int):
    """Take the last `n` entries along the sequence dim.

    Works with sliding-window caches (HybridCache) where per-layer
    cache length can be smaller than the raw sequence length.
    """
    out = []
    for k, v in kvs:
        take = min(n, k.shape[-2])
        out.append((k[..., -take:, :].contiguous(), v[..., -take:, :].contiguous()))
    return out


@torch.no_grad()
def greedy_continue(
    model, past_kv, first_token: int, start_pos: int, max_new: int, stop_token_ids: set
) -> List[int]:
    """Greedy-decode starting from `first_token` at position `start_pos`,
    one token at a time, using and advancing `past_kv`. Stops when a
    stop token is produced or after `max_new` tokens. Returns the
    generated token ids INCLUDING `first_token` as the first element.
    """
    device = next(model.parameters()).device
    generated = [first_token]
    cur_tok = first_token
    cur_pos = start_pos
    for _ in range(max_new - 1):
        pos = torch.tensor([cur_pos], device=device)
        out = model(
            input_ids=torch.tensor([[cur_tok]], device=device),
            past_key_values=past_kv,
            position_ids=pos.unsqueeze(0),
            cache_position=pos,
            use_cache=True,
        )
        nxt = int(out.logits[0, -1].argmax().item())
        if nxt in stop_token_ids:
            break
        generated.append(nxt)
        cur_tok = nxt
        cur_pos += 1
    return generated


@torch.no_grad()
def forward_with_cache(
    model, input_ids: torch.Tensor, past_kv, trigger_position: int
) -> torch.Tensor:
    """Forward tokens one at a time using an injected cache.

    Gemma-family (hybrid sliding-window) attention rejects multi-token
    forwards when past_kv has heterogeneous per-layer lengths, so we
    loop single-token-at-a-time. Returns the logits for all new tokens
    stacked (shape [n, vocab]).

    `trigger_position` is the absolute position of the FIRST new token;
    subsequent tokens get `trigger_position + i` so RoPE phases match
    the drifted stream's absolute positions.
    """
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    n = input_ids.shape[0]
    all_logits: List[torch.Tensor] = []
    for i in range(n):
        pos = torch.tensor([trigger_position + i], device=device)
        out = model(
            input_ids=input_ids[i : i + 1].unsqueeze(0),
            past_key_values=past_kv,
            position_ids=pos.unsqueeze(0),
            cache_position=pos,
            use_cache=True,
        )
        all_logits.append(out.logits[0, -1].detach())
    return torch.stack(all_logits)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", required=True)
    p.add_argument(
        "--drifts", type=int, nargs="+", default=[0, 50, 100, 200, 500, 1000]
    )
    p.add_argument("--chunk-tokens", type=int, default=128)
    p.add_argument("--n-examples", type=int, default=20)
    p.add_argument(
        "--gen-tokens",
        type=int,
        default=64,
        help="Max new tokens to greedy-decode from fresh and "
        "reused, for the embedding-similarity comparison.",
    )
    p.add_argument(
        "--embedder",
        default="BAAI/bge-small-en-v1.5",
        help="Sentence-transformers model for embedding "
        "similarity between fresh and reused generations.",
    )
    p.add_argument(
        "--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16"
    )
    p.add_argument(
        "--attn-impl",
        default="sdpa",
        help="Use 'sdpa' for Llama / Gemma 4. Gemma 3 requires "
        "'eager' (its SDPA implementation in transformers "
        "5.5 produces wrong logits when a forward is split "
        "into prefill + per-token decode).",
    )
    p.add_argument(
        "--device-map",
        default="balanced",
        help="Weight placement strategy passed to "
        "from_pretrained. 'balanced' spreads across all "
        "visible GPUs; use 'cuda:0' to pin a small model.",
    )
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    L = args.chunk_tokens
    # Base examples only need enough room for the chunk + a real prior.
    min_base = L + 128

    print(f"[info] loading tokenizer {args.model}", flush=True)
    tok = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=args.trust_remote_code
    )

    def base_long_enough(ex) -> bool:
        msgs = prompt_msgs_through_tool(ex)
        if not msgs:
            return False
        # msgs always ends with the tool turn; drop it to measure tool_len.
        through_asst_ids = render_messages(tok, msgs[:-1])
        through_tool_ids = render_messages(tok, msgs)
        tool_len = through_tool_ids.shape[0] - through_asst_ids.shape[0]
        if tool_len < L:
            return False
        full_ids = render_messages(tok, msgs, add_generation_prompt=True)
        return full_ids.shape[0] >= min_base

    print(
        f"[info] collecting {args.n_examples} Hermes base examples "
        f"(min base tokens: {min_base})",
        flush=True,
    )
    t0 = time.time()
    base_examples = load_hermes_examples(args.n_examples, base_long_enough)
    print(
        f"[info] loaded {len(base_examples)} base examples in {time.time() - t0:.1f}s",
        flush=True,
    )

    print(
        f"[info] loading model {args.model}  device_map={args.device_map}",
        flush=True,
    )
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        device_map=args.device_map,
        attn_implementation=args.attn_impl,
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()
    print(f"[info] model loaded in {time.time() - t0:.1f}s", flush=True)

    # Embedder runs on CPU; keep off the GPU the model is using.
    print(f"[info] loading embedder {args.embedder}", flush=True)
    from sentence_transformers import SentenceTransformer

    embedder = SentenceTransformer(args.embedder, device="cpu")
    import numpy as np

    def cos_sim(a: str, b: str) -> float:
        if not a.strip() or not b.strip():
            return float("nan")
        ea, eb = embedder.encode(
            [a, b], convert_to_numpy=True, normalize_embeddings=True
        )
        return float(np.dot(ea, eb))

    # Stop-token set for greedy decoding: any EOS-like token.
    stop_ids = set()
    for attr in ("eos_token_id", "pad_token_id"):
        tid = getattr(tok, attr, None)
        if tid is not None:
            if isinstance(tid, (list, tuple)):
                stop_ids.update(int(x) for x in tid)
            else:
                stop_ids.add(int(tid))

    per_drift_raw: Dict[str, List[dict]] = {str(d): [] for d in args.drifts}

    for ex_idx, base_ex in enumerate(base_examples):
        base_msgs = prompt_msgs_through_tool(base_ex)
        reference_text = reference_assistant_response(base_ex)
        # Baseline: prompt ends at the tool-result turn.
        through_tool_ids = render_messages(tok, base_msgs)
        tool_end_base = through_tool_ids.shape[0]
        baseline_full_ids = render_messages(tok, base_msgs, add_generation_prompt=True)
        T_baseline = baseline_full_ids.shape[0]

        # Chunk: last L tokens of the tool-turn content.

        # In-context baseline: prefill render(base_msgs) up to tool_end_base.
        baseline_for_chunk = through_tool_ids[:tool_end_base].to(
            next(model.parameters()).device
        )
        baseline_kvs, _ = prefill_full(model, baseline_for_chunk)
        baseline_chunk_kv = slice_tail(baseline_kvs, L)
        del baseline_kvs
        torch.cuda.empty_cache()

        ex_kls = {}
        for target_delta in args.drifts:
            if target_delta == 0:
                drifted_msgs = base_msgs
                actual_delta = 0
            else:
                drifted_msgs, actual_delta = inflate_system_turn(
                    tok, base_msgs, target_delta
                )
            # Render drifted with add_generation_prompt=True → model's
            # next decode position is the final token.
            drifted_stream = render_messages(
                tok, drifted_msgs, add_generation_prompt=True
            )
            # Locate the tool-turn's end within the drifted stream.
            drifted_through_tool = render_messages(tok, drifted_msgs)
            tool_end_drift = drifted_through_tool.shape[0]
            T_d = drifted_stream.shape[0]

            # Fresh: prefill whole drifted stream, keep cache alive so
            # we can greedy-decode continuation from it. Ask
            # transformers to return only the last position's logits to
            # skip materialising the full [1, T, vocab] tensor — critical
            # for big models + big vocabs.
            device = next(model.parameters()).device
            with torch.no_grad():
                out_fresh = model(
                    input_ids=drifted_stream.unsqueeze(0).to(device),
                    use_cache=True,
                    logits_to_keep=1,
                )
            fresh_past = out_fresh.past_key_values
            log_fresh = out_fresh.logits[0, -1].detach().float().cpu()
            fresh_first = int(log_fresh.argmax().item())
            del out_fresh  # release the full-sequence logits immediately
            torch.cuda.empty_cache()
            if args.gen_tokens > 0:
                fresh_gen_ids = greedy_continue(
                    model,
                    fresh_past,
                    first_token=fresh_first,
                    start_pos=T_d,
                    max_new=args.gen_tokens,
                    stop_token_ids=stop_ids,
                )
                fresh_text = tok.decode(fresh_gen_ids)
            else:
                fresh_text = ""
            del fresh_past
            torch.cuda.empty_cache()

            # Reused: native-cache prefill through the tool turn (so
            # the cache is in the model's expected layout), then
            # overwrite the last L KV entries per layer with baseline
            # chunk KV (which carries the baseline's RoPE phases / H
            # states — the "reused" cache). Then forward the remaining
            # prompt tokens one at a time with explicit cache_position.
            device = next(model.parameters()).device
            with torch.no_grad():
                out = model(
                    input_ids=drifted_stream[:tool_end_drift].unsqueeze(0).to(device),
                    use_cache=True,
                    logits_to_keep=1,
                )
            past_reused = out.past_key_values
            del out  # free the prefill logits
            torch.cuda.empty_cache()
            for li, (kb, vb) in enumerate(baseline_chunk_kv):
                if hasattr(past_reused, "layers"):
                    lyr = past_reused.layers[li]
                    k_tensor, v_tensor = lyr.keys, lyr.values
                else:
                    k_tensor = past_reused.key_cache[li]
                    v_tensor = past_reused.value_cache[li]
                if k_tensor.shape[-2] < L:
                    continue
                k_tensor[..., -L:, :].copy_(kb.to(k_tensor.device, k_tensor.dtype))
                v_tensor[..., -L:, :].copy_(vb.to(v_tensor.device, v_tensor.dtype))

            remaining = drifted_stream[tool_end_drift:T_d]
            if remaining.shape[0] > 0:
                reused_logits = forward_with_cache(
                    model, remaining, past_reused, trigger_position=tool_end_drift
                )
                log_reused = reused_logits[-1].float().cpu()
            else:
                # Tool-end == prompt-end case: re-forward the last tool
                # token to get logits at that position.
                last_tok = drifted_stream[tool_end_drift - 1 : tool_end_drift]
                reused_logits = forward_with_cache(
                    model, last_tok, past_reused, trigger_position=tool_end_drift - 1
                )
                log_reused = reused_logits[-1].float().cpu()
            reused_first = int(log_reused.argmax().item())
            if args.gen_tokens > 0:
                reused_gen_ids = greedy_continue(
                    model,
                    past_reused,
                    first_token=reused_first,
                    start_pos=T_d,
                    max_new=args.gen_tokens,
                    stop_token_ids=stop_ids,
                )
                reused_text = tok.decode(reused_gen_ids)
            else:
                reused_text = ""
            del past_reused
            torch.cuda.empty_cache()

            log_p = F.log_softmax(log_fresh, dim=-1)
            log_q = F.log_softmax(log_reused, dim=-1)
            kl = (log_p.exp() * (log_p - log_q)).sum().item()
            fresh_entropy = -(log_p.exp() * log_p).sum().item()
            top1_fresh = int(log_fresh.argmax().item())
            top1_reused = int(log_reused.argmax().item())
            agree = int(top1_fresh == top1_reused)
            top5_fresh = set(log_fresh.topk(5).indices.tolist())
            top5_reused = set(log_reused.topk(5).indices.tolist())
            top5_overlap = len(top5_fresh & top5_reused) / 5.0

            # Embedding-similarity of the two generated continuations,
            # and each against the dataset's reference assistant reply.
            sim_fresh_reused = cos_sim(fresh_text, reused_text)
            sim_fresh_ref = (
                cos_sim(fresh_text, reference_text) if reference_text else float("nan")
            )
            sim_reused_ref = (
                cos_sim(reused_text, reference_text) if reference_text else float("nan")
            )

            per_drift_raw[str(target_delta)].append(
                {
                    "example_idx": ex_idx,
                    "T_drifted": T_d,
                    "tool_end_drift": tool_end_drift,
                    "target_delta": target_delta,
                    "actual_delta": actual_delta,
                    "kl": kl,
                    "fresh_entropy": fresh_entropy,
                    "agree": agree,
                    "top5_overlap": top5_overlap,
                    "top1_fresh": top1_fresh,
                    "top1_reused": top1_reused,
                    "sim_fresh_reused": sim_fresh_reused,
                    "sim_fresh_ref": sim_fresh_ref,
                    "sim_reused_ref": sim_reused_ref,
                    "fresh_text": fresh_text,
                    "reused_text": reused_text,
                }
            )
            ex_kls[target_delta] = (kl, actual_delta, fresh_entropy, sim_fresh_reused)
            torch.cuda.empty_cache()

        fresh_H = ex_kls[args.drifts[0]][2]
        summary = "  ".join(
            f"Δ={d}(~{ex_kls[d][1]}):KL={ex_kls[d][0]:.2f},sim={ex_kls[d][3]:.2f}"
            for d in args.drifts
            if d in ex_kls
        )
        tool_len = tool_end_base - render_messages(tok, base_msgs[:-1]).shape[0]
        print(
            f"[info] ex{ex_idx:02d}  T_base={T_baseline}  "
            f"tool_len={tool_len}  H(fresh@Δ=0)={fresh_H:.2f}  "
            f"{summary}",
            flush=True,
        )

    import math

    def _mean_skip_nan(xs):
        xs = [x for x in xs if isinstance(x, float) and not math.isnan(x)]
        return statistics.mean(xs) if xs else float("nan")

    agg = {}
    for d, rows in per_drift_raw.items():
        kls = [r["kl"] for r in rows]
        agrees = [r["agree"] for r in rows]
        entropies = [r["fresh_entropy"] for r in rows]
        top5s = [r["top5_overlap"] for r in rows]
        actuals = [r["actual_delta"] for r in rows]
        sims_fr = [r["sim_fresh_reused"] for r in rows]
        sims_fr_ref = [r["sim_fresh_ref"] for r in rows]
        sims_re_ref = [r["sim_reused_ref"] for r in rows]
        agg[d] = {
            "n": len(rows),
            "mean_kl": statistics.mean(kls) if kls else float("nan"),
            "stdev_kl": statistics.stdev(kls) if len(kls) > 1 else 0.0,
            "agree_rate": statistics.mean(agrees) if agrees else float("nan"),
            "agree_count": sum(agrees),
            "mean_fresh_entropy": _mean_skip_nan(entropies),
            "mean_top5_overlap": _mean_skip_nan(top5s),
            "mean_sim_fresh_reused": _mean_skip_nan(sims_fr),
            "mean_sim_fresh_reference": _mean_skip_nan(sims_fr_ref),
            "mean_sim_reused_reference": _mean_skip_nan(sims_re_ref),
            "mean_actual_delta": statistics.mean(actuals) if actuals else 0,
            "per_example": rows,
        }

    results = {
        "model": args.model,
        "chunk_tokens": L,
        "dtype": args.dtype,
        "attn_impl": args.attn_impl,
        "prompt_source": "hermes_chat_template_in_context",
        "n_examples": len(base_examples),
        "per_drift": agg,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2))
    print(f"[info] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
