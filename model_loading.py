"""Shared CLI args and loader for the reagent harnesses.

Both `measure_reuse_drift.py` and `measure_multi_splice.py` take the
same `--model/--dtype/--attn-impl/--device-map/--trust-remote-code`
quartet and load with `from_pretrained` the same way. Centralize it so
the dtype map lives in one place and CLI help text stays consistent.
"""

from __future__ import annotations

import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def add_model_args(parser: argparse.ArgumentParser) -> None:
    """Register the shared model-loading flags on `parser`."""
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--dtype",
        choices=list(_DTYPES),
        default="bfloat16",
    )
    parser.add_argument(
        "--attn-impl",
        default="sdpa",
        help="Use 'sdpa' for Llama / Gemma 4. Gemma 3 requires "
        "'eager' (its SDPA implementation in transformers "
        "5.5 produces wrong logits when a forward is split "
        "into prefill + per-token decode).",
    )
    parser.add_argument(
        "--device-map",
        default="balanced",
        help="Weight placement strategy passed to "
        "from_pretrained. 'balanced' spreads across all "
        "visible GPUs; use 'cuda:0' to pin a small model.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")


def load_tokenizer(args):
    print(f"[info] loading tokenizer {args.model}", flush=True)
    return AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=args.trust_remote_code
    )


def load_model(args):
    print(
        f"[info] loading model {args.model}  device_map={args.device_map}",
        flush=True,
    )
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=_DTYPES[args.dtype],
        device_map=args.device_map,
        attn_implementation=args.attn_impl,
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()
    print(f"[info] model loaded in {time.time() - t0:.1f}s", flush=True)
    return model
