#!/usr/bin/env bash
# Run measure_reuse_drift.py across the model panel.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${1:-${HERE}/results}"
mkdir -p "${OUT_DIR}"

MODELS="${MODELS:-meta-llama/Llama-3.2-1B-Instruct meta-llama/Llama-3.1-8B-Instruct google/gemma-4-E2B-it google/gemma-4-E4B-it google/gemma-4-26B-A4B-it google/gemma-4-31B-it}"
ATTN_IMPL="${ATTN_IMPL:-sdpa}"

for model in ${MODELS}; do
    tag="${model//\//_}"
    out="${OUT_DIR}/${tag}.json"
    log="${OUT_DIR}/${tag}.log"
    if [ -s "${out}" ] && [ "${FORCE:-0}" != "1" ]; then
        echo "=== ${model}  [skipping: ${out} already exists; set FORCE=1 to re-run]"
        continue
    fi
    echo "=== ${model}"
    uv run --script "${HERE}/measure_reuse_drift.py" \
        --model "${model}" \
        --drifts 0 50 100 200 500 1000 \
        --chunk-tokens 128 \
        --n-examples "${N_EXAMPLES:-20}" \
        --attn-impl "${ATTN_IMPL}" \
        --output "${out}" 2>&1 | tee "${log}" \
        | grep -E "^\[info\] ex|^\[info\] model" || true
done

echo
echo "=== Summary: mean KL(fresh||reused) ± stdev over examples"
uv run --no-project --python 3.10 python - <<'PY' "${OUT_DIR}"
import json, sys
from pathlib import Path
d = Path(sys.argv[1])
files = sorted(d.glob("*.json"))
if not files:
    print("no results"); sys.exit()
all_drifts = set()
rows = []
for f in files:
    data = json.loads(f.read_text())
    all_drifts.update(int(k) for k in data["per_drift"])
    rows.append(data)
drifts = sorted(all_drifts)
print(f"{'model':<42} " + " ".join(f"Δ={d:<13}" for d in drifts))
print("-" * (44 + 15*len(drifts)))
for r in rows:
    cells = []
    for d in drifts:
        e = r['per_drift'].get(str(d), {})
        mk, sk = e.get('mean_kl', float('nan')), e.get('stdev_kl', 0.0)
        cells.append(f"{mk:.2f}±{sk:.2f}")
    print(f"{r['model']:<42} " + " ".join(f"{c:<13}" for c in cells))
print(f"\ntop-1 agreement rate (N={rows[0].get('n_examples','?')} examples):")
for r in rows:
    cells = [f"{r['per_drift'].get(str(d),{}).get('agree_rate', float('nan')):.2f}"
             for d in drifts]
    print(f"{r['model']:<42} " + " ".join(f"{c:<13}" for c in cells))
print(f"\ntop-5 overlap rate:")
for r in rows:
    cells = [f"{r['per_drift'].get(str(d),{}).get('mean_top5_overlap', float('nan')):.2f}"
             for d in drifts]
    print(f"{r['model']:<42} " + " ".join(f"{c:<13}" for c in cells))
print(f"\nmean fresh entropy (nats) -- near 0 means trigger is near-deterministic "
      "and KL~0 may be trivial:")
for r in rows:
    cells = [f"{r['per_drift'].get(str(d),{}).get('mean_fresh_entropy', float('nan')):.2f}"
             for d in drifts]
    print(f"{r['model']:<42} " + " ".join(f"{c:<13}" for c in cells))
print(f"\nmean cosine similarity fresh-gen vs reused-gen (1.0 = identical meaning):")
for r in rows:
    cells = [f"{r['per_drift'].get(str(d),{}).get('mean_sim_fresh_reused', float('nan')):.2f}"
             for d in drifts]
    print(f"{r['model']:<42} " + " ".join(f"{c:<13}" for c in cells))
print(f"\nmean cosine similarity reused-gen vs dataset reference (quality):")
for r in rows:
    cells = [f"{r['per_drift'].get(str(d),{}).get('mean_sim_reused_reference', float('nan')):.2f}"
             for d in drifts]
    print(f"{r['model']:<42} " + " ".join(f"{c:<13}" for c in cells))
PY
