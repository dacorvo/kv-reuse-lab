"""Chat-template rendering and single-token forward helpers.

Both reagent harnesses need to:
  - render messages with/without a generation prompt
  - greedy-decode a continuation from an already-populated KV cache
  - forward new tokens one-at-a-time against an injected cache,
    setting explicit `cache_position` (required for correctness when
    the cache was mutated after prefill and for Gemma-family hybrid
    attention which rejects multi-token forwards over heterogeneous
    per-layer cache lengths).
"""

from __future__ import annotations

from typing import List

import torch


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


def _has_sliding_layers(model) -> bool:
    """True if the model mixes sliding-window and full attention (Gemma
    3/3n/4) — in that case the KV cache is heterogeneous across layers
    and multi-token forwards are rejected, so forward_with_cache must
    loop per-token.
    """
    for root in (getattr(model, "model", None), model):
        if root is None:
            continue
        for attr in ("language_model", ""):
            inner = getattr(root, attr, None) if attr else root
            if inner is None:
                continue
            cfg = getattr(inner, "config", None)
            types = getattr(cfg, "layer_types", None) if cfg is not None else None
            if types is not None:
                return any(t == "sliding_attention" for t in types)
    return False


@torch.no_grad()
def forward_with_cache(
    model, input_ids: torch.Tensor, past_kv, trigger_position: int
) -> torch.Tensor:
    """Forward tokens against an injected cache, returning per-token logits.

    On Gemma-family hybrid (sliding + full) attention, the cache is
    heterogeneous across layers and multi-token forwards are rejected
    — we fall back to a per-token loop. On homogeneous caches (Llama
    etc.) we submit a single batched forward, which is ~n× faster.

    `trigger_position` is the absolute position of the FIRST new token;
    subsequent tokens get `trigger_position + i` so RoPE phases match
    the drifted stream's absolute positions.
    """
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    n = input_ids.shape[0]
    if n == 0:
        return torch.empty((0,), device="cpu")

    if not _has_sliding_layers(model):
        positions = torch.arange(trigger_position, trigger_position + n, device=device)
        out = model(
            input_ids=input_ids.unsqueeze(0),
            past_key_values=past_kv,
            position_ids=positions.unsqueeze(0),
            cache_position=positions,
            use_cache=True,
        )
        return out.logits[0].detach()

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
