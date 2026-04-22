"""Small numeric helpers shared by the reagent harnesses.

Kept separate from `similarity.py` so this module has no heavy deps
(torch, sentence-transformers) and imports cheaply when tests only
need the stats functions.
"""

from __future__ import annotations

from typing import Sequence


def percentile(xs: Sequence[float], q: float) -> float:
    """Nearest-rank percentile (q in [0, 1]). Matches the ad-hoc
    implementation that lived in `measure_multi_splice.py`.
    """
    xs_sorted = sorted(xs)
    k = int(round(q * (len(xs_sorted) - 1)))
    return xs_sorted[k]
