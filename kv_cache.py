"""KV cache layout helpers for cross-HF-version access.

Transformers moved from the legacy `(key_cache, value_cache)` lists to
`Cache.layers[i].keys / .values` between releases. Both reagent harnesses
need to read and overwrite per-layer cache tensors, so these helpers
paper over the two layouts plus the even older tuple-of-tuples form.

This module also encapsulates the Gemma-3n / Gemma-4 KV-sharing
routing: the last `num_kv_shared_layers` attention modules don't own
their own K/V cache — they read from `past_key_values.shared_layers`,
populated by the last non-shared layer of each type. Writing to
`layers[i]` on a sharing layer lands in storage attention never reads,
which was the silent-wrongness / illegal-memory-access source in both
harnesses on Gemma. `write_kv_span` routes writes to the location
attention actually reads from.
"""

from __future__ import annotations

from typing import List, Tuple

import torch


def num_layers(pkv) -> int:
    """Layer count of a past_key_values object across HF cache layouts."""
    if hasattr(pkv, "layers"):
        return len(pkv.layers)
    if hasattr(pkv, "key_cache"):
        return len(pkv.key_cache)
    return len(pkv)


def layer_kv(pkv, li: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return mutable (K, V) tensors for layer `li` of the cache."""
    if hasattr(pkv, "layers"):
        lyr = pkv.layers[li]
        return lyr.keys, lyr.values
    if hasattr(pkv, "key_cache"):
        return pkv.key_cache[li], pkv.value_cache[li]
    return pkv[li][0], pkv[li][1]


def slice_tail(
    kvs: List[Tuple[torch.Tensor, torch.Tensor]], n: int
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Take the last `n` entries along the sequence dim.

    Works with sliding-window caches (HybridCache) where per-layer
    cache length can be smaller than the raw sequence length.
    """
    out = []
    for k, v in kvs:
        take = min(n, k.shape[-2])
        out.append((k[..., -take:, :].contiguous(), v[..., -take:, :].contiguous()))
    return out


def _text_layers(model):
    """Return the ModuleList of decoder layers (or None if introspection
    fails). Handles text-only models and multimodal-nested backbones.
    """
    for root in (getattr(model, "model", None), model):
        if root is None:
            continue
        for attr in ("language_model", ""):
            inner = getattr(root, attr) if attr else root
            if inner is None:
                continue
            layers = getattr(inner, "layers", None)
            if layers is not None:
                return layers
    return None


def attn_module(model, layer_idx: int):
    """Return the self-attention module for `layer_idx`, or None if
    the backbone layout isn't recognised.
    """
    layers = _text_layers(model)
    if layers is None or layer_idx >= len(layers):
        return None
    return getattr(layers[layer_idx], "self_attn", None)


def is_kv_shared_layer(model, layer_idx: int) -> bool:
    """True for Gemma-3n/Gemma-4 layers whose K/V come from an earlier
    layer's cache via `past_key_values.shared_layers[src]`.
    """
    attn = attn_module(model, layer_idx)
    return bool(getattr(attn, "is_kv_shared_layer", False))


def write_kv_span(
    past_kv,
    model,
    layer_idx: int,
    dst_range: slice,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
) -> None:
    """Overwrite `(layer_idx, dst_range)` in the cache with `k_new, v_new`.

    Routes writes so the attention read sees the update:
      - KV-sharing layer (Gemma)  -> `past_kv.shared_layers[src]`
      - `store_full_length_kv` src -> writes to `layers[li]` AND
        `shared_layers[li]` (belt-and-suspenders; usually they alias,
        but sliding-window caches may diverge).
      - plain layer -> `layers[li]`.

    `k_new`, `v_new` are moved to the cache tensor's device+dtype.
    """
    attn = attn_module(model, layer_idx)
    shared = getattr(past_kv, "shared_layers", None)

    if attn is not None and getattr(attn, "is_kv_shared_layer", False):
        src = int(attn.kv_shared_layer_index)
        if shared is not None and src in shared:
            sk, sv = shared[src]
            sk[..., dst_range, :].copy_(k_new.to(sk.device, sk.dtype))
            sv[..., dst_range, :].copy_(v_new.to(sv.device, sv.dtype))
        return

    k_t, v_t = layer_kv(past_kv, layer_idx)
    k_t[..., dst_range, :].copy_(k_new.to(k_t.device, k_t.dtype))
    v_t[..., dst_range, :].copy_(v_new.to(v_t.device, v_t.dtype))
    if (
        attn is not None
        and getattr(attn, "store_full_length_kv", False)
        and shared is not None
        and layer_idx in shared
    ):
        sk, sv = shared[layer_idx]
        if sk.data_ptr() != k_t.data_ptr():
            sk[..., dst_range, :].copy_(k_new.to(sk.device, sk.dtype))
            sv[..., dst_range, :].copy_(v_new.to(sv.device, sv.dtype))
