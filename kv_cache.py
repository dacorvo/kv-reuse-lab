"""KV cache layout helpers for cross-HF-version access.

Transformers moved from the legacy `(key_cache, value_cache)` lists to
`Cache.layers[i].keys / .values` between releases. Both reagent harnesses
need to read and overwrite per-layer cache tensors, so these helpers
paper over the two layouts plus the even older tuple-of-tuples form.
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
