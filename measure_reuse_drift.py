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
from drift_modes import (
    DRIFT_MODES,
    inflate,
    load_hermes_examples,
    prompt_msgs_through_tool,
    reference_assistant_response,
)
from generation import forward_with_cache, greedy_continue, render_messages
from kv_cache import layer_kv as _stack_kv
from kv_cache import num_layers as _num_layers
from kv_cache import slice_tail, write_kv_span  # noqa: F401
from model_loading import add_model_args, load_model, load_tokenizer
from rope_shift import shift_k_rope
from similarity import load_embedder_and_cos_sim


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


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_model_args(p)
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
        "--reuse-mode",
        choices=("naive", "shifted"),
        default="naive",
        help="KV reuse strategy. 'naive' splices cached K/V at the "
        "drifted position without correction (the baseline experiment). "
        "'shifted' applies a RoPE delta rotation to the cached K so "
        "its phases match the new positions — the same correction that "
        "llama.cpp's --cache-reuse performs via llama_memory_seq_add.",
    )
    p.add_argument(
        "--drift-mode",
        choices=DRIFT_MODES,
        default="system-duplicate",
        help="How to construct the drifted prompt. 'system-duplicate' "
        "appends duplicates of the base's own system content. "
        "'system-instructions' appends real instruction content from "
        "other Hermes examples. 'turn-insert' splices donor "
        "[user, asst, tool] triples between the original user and the "
        "assistant tool call. 'prior-tool-exchange' splices donor "
        "triples between the system and the original user.",
    )
    p.add_argument(
        "--n-donors",
        type=int,
        default=20,
        help="How many donor examples to harvest past the base set "
        "for non-trivial drift modes.",
    )
    p.add_argument("--output", required=True)
    args = p.parse_args()

    L = args.chunk_tokens
    # Base examples only need enough room for the chunk + a real prior.
    min_base = L + 128

    tok = load_tokenizer(args)

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
    need_donors = args.drift_mode != "system-duplicate"
    n_donors = args.n_donors if need_donors else 0
    base_examples, donor_examples = load_hermes_examples(
        args.n_examples, base_long_enough, n_donors=n_donors
    )
    print(
        f"[info] loaded {len(base_examples)} base + {len(donor_examples)} donor "
        f"examples in {time.time() - t0:.1f}s",
        flush=True,
    )

    model = load_model(args)

    # Embedder runs on CPU; keep off the GPU the model is using.
    cos_sim = load_embedder_and_cos_sim(args.embedder)

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
                drifted_msgs, actual_delta = inflate(
                    args.drift_mode,
                    tok,
                    base_msgs,
                    target_delta,
                    donor_examples=donor_examples,
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
            kv_shift = tool_end_drift - tool_end_base
            for li, (kb, vb) in enumerate(baseline_chunk_kv):
                k_tensor, _ = _stack_kv(past_reused, li)
                if k_tensor.shape[-2] < L:
                    continue
                cache_len = k_tensor.shape[-2]
                dst = slice(cache_len - L, cache_len)
                kb_shifted = kb
                if args.reuse_mode == "shifted" and kv_shift != 0:
                    kb_shifted = shift_k_rope(
                        kb.to(k_tensor.device, k_tensor.dtype), model, li, kv_shift
                    )
                write_kv_span(past_reused, model, li, dst, kb_shifted, vb)

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
        "reuse_mode": args.reuse_mode,
        "drift_mode": args.drift_mode,
        "prompt_source": "hermes_chat_template_in_context",
        "n_examples": len(base_examples),
        "per_drift": agg,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2))
    print(f"[info] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
