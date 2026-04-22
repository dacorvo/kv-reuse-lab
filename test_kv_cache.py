#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "pytest>=8.0",
#   "torch>=2.4,<2.11",
# ]
# ///
"""Unit tests for `kv_cache.write_kv_span`.

Covers the three routing paths:
  - plain layer (Llama): writes to `past_kv.layers[i]`.
  - KV-sharing layer (Gemma): writes to `past_kv.shared_layers[src]`
    and NOT to `past_kv.layers[i]`.
  - source layer with `store_full_length_kv`: writes to BOTH
    `past_kv.layers[i]` and `past_kv.shared_layers[i]` when they do
    not alias; a single write when they do alias.

Run with:
    uv run --script test_kv_cache.py
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from kv_cache import write_kv_span


class _LayerEntry:
    def __init__(self, k: torch.Tensor, v: torch.Tensor):
        self.keys = k
        self.values = v


class _FakeCache:
    """Minimal stand-in for HF's `Cache` object: `layers[i].keys/.values`
    and an optional `shared_layers` dict.
    """

    def __init__(self, n_layers: int, seq: int, head_dim: int):
        self.layers = [
            _LayerEntry(
                torch.zeros(1, 1, seq, head_dim),
                torch.zeros(1, 1, seq, head_dim),
            )
            for _ in range(n_layers)
        ]


def _mk_model(n_layers: int, *, sharing_from: int = None, store_full_at: int = None):
    """Build a minimal model skeleton that `kv_cache.attn_module` can
    introspect: `model.model.layers[i].self_attn` with the three flags
    `is_kv_shared_layer`, `kv_shared_layer_index`, `store_full_length_kv`.
    """
    layers = []
    for i in range(n_layers):
        attn = SimpleNamespace(
            is_kv_shared_layer=(sharing_from is not None and i >= sharing_from),
            kv_shared_layer_index=(store_full_at if store_full_at is not None else 0),
            store_full_length_kv=(store_full_at is not None and i == store_full_at),
        )
        layers.append(SimpleNamespace(self_attn=attn))
    inner = SimpleNamespace(layers=layers)
    return SimpleNamespace(model=inner)


def test_plain_layer_writes_to_layers_entry():
    """Llama path: no shared_layers, no is_kv_shared_layer flag."""
    model = _mk_model(n_layers=4)  # no sharing, no store_full flags
    cache = _FakeCache(n_layers=4, seq=10, head_dim=8)
    k_new = torch.full((1, 1, 3, 8), 7.0)
    v_new = torch.full((1, 1, 3, 8), 9.0)

    write_kv_span(cache, model, 1, slice(4, 7), k_new, v_new)

    assert torch.equal(cache.layers[1].keys[..., 4:7, :], k_new)
    assert torch.equal(cache.layers[1].values[..., 4:7, :], v_new)
    # Adjacent positions untouched.
    assert torch.equal(cache.layers[1].keys[..., :4, :], torch.zeros(1, 1, 4, 8))
    assert torch.equal(cache.layers[1].keys[..., 7:, :], torch.zeros(1, 1, 3, 8))
    # Sibling layers untouched.
    assert torch.equal(cache.layers[0].keys, torch.zeros(1, 1, 10, 8))


def test_kv_sharing_layer_writes_to_shared_layers_only():
    """Gemma path: layer >= sharing threshold, write goes to
    `shared_layers[src]`, NOT to `layers[li].keys`.
    """
    # 4 layers, layers 2-3 share from layer 1 (store_full_length_kv).
    model = _mk_model(n_layers=4, sharing_from=2, store_full_at=1)
    cache = _FakeCache(n_layers=4, seq=10, head_dim=8)
    # Populate `shared_layers[1]` as the source layer's full-length kv
    # would during a prefill. Use a distinct tensor (not aliased to
    # layers[1]) so we can assert writes landed where expected.
    shared_k = torch.zeros(1, 1, 10, 8)
    shared_v = torch.zeros(1, 1, 10, 8)
    cache.shared_layers = {1: (shared_k, shared_v)}

    k_new = torch.full((1, 1, 3, 8), 5.0)
    v_new = torch.full((1, 1, 3, 8), 6.0)

    # Write to a sharing layer (layer 3).
    write_kv_span(cache, model, 3, slice(0, 3), k_new, v_new)

    # shared_layers[1] got the update.
    assert torch.equal(shared_k[..., 0:3, :], k_new)
    assert torch.equal(shared_v[..., 0:3, :], v_new)
    # layers[3] was NOT written (sharing layers skip the layers[i]
    # path entirely — attention never reads it).
    assert torch.equal(cache.layers[3].keys, torch.zeros(1, 1, 10, 8))


def test_store_full_length_kv_writes_to_both_when_not_aliased():
    """Source layer (non-shared, store_full_length_kv=True): writes to
    `layers[li]` AND, if `shared_layers[li]` is a distinct tensor,
    mirrors the write there so sharing readers see it.
    """
    model = _mk_model(n_layers=4, sharing_from=2, store_full_at=1)
    cache = _FakeCache(n_layers=4, seq=10, head_dim=8)
    # Distinct (non-aliased) shared_layers entry for layer 1.
    shared_k = torch.zeros(1, 1, 10, 8)
    shared_v = torch.zeros(1, 1, 10, 8)
    cache.shared_layers = {1: (shared_k, shared_v)}

    k_new = torch.full((1, 1, 3, 8), 2.0)
    v_new = torch.full((1, 1, 3, 8), 3.0)
    write_kv_span(cache, model, 1, slice(5, 8), k_new, v_new)

    # layers[1] got it.
    assert torch.equal(cache.layers[1].keys[..., 5:8, :], k_new)
    # shared_layers[1] got it too (belt-and-suspenders for
    # sliding-window-cache cases where the two storages diverge).
    assert torch.equal(shared_k[..., 5:8, :], k_new)
    assert torch.equal(shared_v[..., 5:8, :], v_new)


def test_store_full_length_kv_single_write_when_aliased():
    """If `shared_layers[li]` is the same tensor as `layers[li].keys`
    (the common case: the prefill-assignment stores a reference),
    a single write through `layers[li]` already reflects in
    shared_layers. write_kv_span should NOT double-write because
    it checks `data_ptr()`.
    """
    model = _mk_model(n_layers=4, sharing_from=2, store_full_at=1)
    cache = _FakeCache(n_layers=4, seq=10, head_dim=8)
    # Alias shared_layers[1] to the same underlying tensor as layers[1].
    cache.shared_layers = {1: (cache.layers[1].keys, cache.layers[1].values)}

    k_new = torch.full((1, 1, 3, 8), 4.0)
    v_new = torch.full((1, 1, 3, 8), 5.0)
    write_kv_span(cache, model, 1, slice(2, 5), k_new, v_new)

    # Single write; the aliased shared_layers entry sees it via the
    # shared storage.
    assert torch.equal(cache.layers[1].keys[..., 2:5, :], k_new)
    assert torch.equal(cache.shared_layers[1][0][..., 2:5, :], k_new)


def test_unknown_backbone_falls_through_to_layers_entry():
    """If `attn_module` returns None (e.g. a custom backbone without
    the standard `.model.layers[i].self_attn` layout), write_kv_span
    must still write to `layers[li]` as a plain cache.
    """

    class NonStandardModel:
        pass

    model = NonStandardModel()  # no .model, no .layers
    cache = _FakeCache(n_layers=2, seq=10, head_dim=8)
    k_new = torch.full((1, 1, 2, 8), 1.0)
    v_new = torch.full((1, 1, 2, 8), 2.0)

    write_kv_span(cache, model, 0, slice(3, 5), k_new, v_new)
    assert torch.equal(cache.layers[0].keys[..., 3:5, :], k_new)
    assert torch.equal(cache.layers[0].values[..., 3:5, :], v_new)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
