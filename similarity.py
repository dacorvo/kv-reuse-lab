"""Sentence-embedding similarity shared by the reagent harnesses.

Thin wrapper around `sentence-transformers`. Runs the embedder on CPU
so it doesn't steal memory from the language model on the GPU, and
returns NaN for empty inputs (both harnesses compare against
potentially empty generations).
"""

from __future__ import annotations

from typing import Callable


def load_embedder_and_cos_sim(name: str) -> Callable[[str, str], float]:
    """Load the named sentence-transformers model and return a
    `cos_sim(a, b) -> float` closure. Printing is done here so callers
    don't each re-log the embedder's load message.
    """
    print(f"[info] loading embedder {name}", flush=True)
    from sentence_transformers import SentenceTransformer
    import numpy as np

    embedder = SentenceTransformer(name, device="cpu")

    def cos_sim(a: str, b: str) -> float:
        if not a.strip() or not b.strip():
            return float("nan")
        ea, eb = embedder.encode(
            [a, b], convert_to_numpy=True, normalize_embeddings=True
        )
        return float(np.dot(ea, eb))

    return cos_sim
