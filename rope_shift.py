"""RoPE phase-shift primitives for cache reuse.

Both reagent harnesses rotate cached K tensors by a uniform position
delta so they decode correctly at drifted absolute positions. This is
the same correction `llama_memory_seq_add` performs in llama.cpp.
"""

from __future__ import annotations

import torch


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """HF-convention RoPE half-rotation: [a | b] -> [-b | a]."""
    d = x.shape[-1] // 2
    return torch.cat((-x[..., d:], x[..., :d]), dim=-1)


def find_text_backbone(model):
    """Return (text_model, rotary_emb) for Llama-style text-only models
    and for multimodal Gemma 4 checkpoints where the text backbone is
    nested one deeper under `model.model.language_model`.
    """
    inner = getattr(model, "model", model)
    if hasattr(inner, "rotary_emb"):
        return inner, inner.rotary_emb
    lm = getattr(inner, "language_model", None)
    if lm is not None and hasattr(lm, "rotary_emb"):
        return lm, lm.rotary_emb
    raise AttributeError("could not locate rotary_emb in model")


def shift_k_rope(k: torch.Tensor, model, layer_idx: int, shift: int) -> torch.Tensor:
    """Apply a uniform RoPE delta of `shift` positions to a cached K tensor.

    The cached K was rotated at original positions [p_base..p_base+L).
    Composing an additional rotation by `shift` produces K as if it had
    been rotated at [p_base+shift..p_base+shift+L), which is what we
    need to splice it at drifted positions without phase mismatch. This
    is the same correction that `llama_memory_seq_add` applies in
    llama.cpp.

    k: [batch, num_kv_heads, seq_len, head_dim] (standard HF cache layout).

    We compute cos/sin directly from `inv_freq` rather than routing
    through `rotary.forward`, because HF's rotary modules have device
    bookkeeping (`dynamic_rope_update`, per-layer-type buffers) that
    breaks under `device_map="balanced"` when `inv_freq` and `k` live
    on different GPUs.
    """
    if shift == 0:
        return k
    text_model, rotary = find_text_backbone(model)
    # Gemma-4 registers `{layer_type}_inv_freq` per hybrid layer type;
    # Llama registers a single `inv_freq`.
    if hasattr(rotary, "inv_freq"):
        inv_freq = rotary.inv_freq
        attn_scaling = float(getattr(rotary, "attention_scaling", 1.0))
    else:
        layer_type = text_model.config.layer_types[layer_idx]
        inv_freq = getattr(rotary, f"{layer_type}_inv_freq")
        attn_scaling = float(getattr(rotary, f"{layer_type}_attention_scaling", 1.0))
    inv_freq = inv_freq.to(device=k.device, dtype=torch.float32)
    freq = inv_freq * float(shift)
    emb = torch.cat([freq, freq], dim=-1)
    cos = (emb.cos() * attn_scaling).to(dtype=k.dtype)
    sin = (emb.sin() * attn_scaling).to(dtype=k.dtype)
    # cos/sin: [head_dim] → broadcasts over [batch, heads, seq, head_dim].
    return k * cos + rotate_half(k) * sin
