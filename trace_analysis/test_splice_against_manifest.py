#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx>=0.27", "pyarrow>=15", "huggingface_hub>=0.25"]
# ///
"""End-to-end splice-correctness measurement on real agentcap manifest
pairs through a patched llama-server.

Per pair, two server runs:

* **Spliced**: send donor (cache_prompt=true) then recipient
  (cache_prompt=true). The patched ``--cache-reuse`` fires; the
  recipient's first generated token comes from the model with the
  donor's K/V cells spliced into the recipient's prefill at the
  matched ranges, RoPE-rephased.

* **Cold**: send the recipient alone in a fresh server. No splice;
  the next token is the cold-prefill baseline.

The harness asks the server for top-N logprobs on the first generated
token (``logprobs: true, top_logprobs: N``) and reports per pair:

* top-1 agreement (does argmax match between spliced and cold).
* top-5 set Jaccard.
* approximate KL on the union of observed top-N tokens.

This is the splice-correctness measurement reagent has historically
done (KL on next-token distribution + top-1 agreement). We do *not*
decode tokens past the splice point — the original protocol cared
about distribution at the boundary, not downstream-generation drift.

Reasoning is disabled at server launch (``--reasoning off``) to match
what agentcap captured. Qwen3.5+/3.6 default to reasoning-on which
puts output into ``message.reasoning_content``; capture-time used
``--reasoning off`` so output goes to ``message.content`` directly.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import socket
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import httpx


REUSE_RE = re.compile(
    r"reusing chunk with size (\d+), shifting KV cache "
    r"\[(\d+), (\d+)\) -> \[(\d+), (\d+)\)"
)
SCHEDULED_RE = re.compile(
    r"scheduled splice: size (\d+), KV cache "
    r"\[(\d+), (\d+)\) -> \[(\d+), (\d+)\)"
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_health(port: int, timeout: float = 600.0) -> None:
    """Loading a 22 GB GGUF takes ~30 s on a warm cache, longer cold."""
    start = time.time()
    url = f"http://127.0.0.1:{port}/health"
    while time.time() - start < timeout:
        try:
            r = httpx.get(url, timeout=2.0)
            if r.status_code == 200:
                return
        except (httpx.HTTPError, OSError):
            pass
        time.sleep(2.0)
    raise RuntimeError(f"server did not become healthy on port {port}")


@contextmanager
def _server(args, port: int):
    """Launch a llama-server with --reasoning off and our patched
    --cache-reuse, yield (url, log_lines, log_lock). Tear down on exit."""
    cmd = [
        str(args.server_bin),
        "--model", str(args.gguf),
        "--port", str(port),
        "--host", "127.0.0.1",
        "--parallel", "1",
        "--ctx-size", str(args.ctx_size),
        "--cache-reuse", str(args.n_cache_reuse),
        "--kv-unified",
        "--jinja",
        "--no-warmup",
        "--no-cache-idle-slots",
        "--reasoning", "off",
        "--n-gpu-layers", "999",
        "-v",
    ]
    print(f"[server] starting on port {port} ...", flush=True)
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    log_lines: list[str] = []
    log_lock = threading.Lock()

    def _drain():
        for line in proc.stderr:
            with log_lock:
                log_lines.append(line)

    threading.Thread(target=_drain, daemon=True).start()

    try:
        _wait_health(port)
        url = f"http://127.0.0.1:{port}/v1/chat/completions"
        yield url, log_lines, log_lock
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def _fetch_pair_bodies(manifest_pair: dict) -> tuple[dict, dict]:
    """Stream the source parquet, find both rows by request_id, return
    their request bodies (messages + tools)."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    source = manifest_pair["source_parquet"]
    want = {manifest_pair["donor"]["request_id"],
            manifest_pair["recipient"]["request_id"]}
    bodies: dict[str, dict] = {}

    if source.startswith("hf://"):
        fs = HfFileSystem()
        bare = source[len("hf://"):].rstrip("/")
        info = fs.info(bare)
        files = [bare] if info.get("type") == "file" else sorted(
            fs.glob(bare + "/**/*.parquet"))
        opener = lambda p: fs.open(p, "rb")
    else:
        files = [Path(source)]
        opener = lambda p: open(p, "rb")

    for path in files:
        with opener(path) as fh:
            pf = pq.ParquetFile(fh)
            for batch in pf.iter_batches(
                batch_size=64,
                columns=["request_id", "request"],
            ):
                for row in batch.to_pylist():
                    if row["request_id"] not in want:
                        continue
                    req = row["request"]
                    if isinstance(req, str):
                        req = json.loads(req)
                    bodies[row["request_id"]] = req
                    if len(bodies) == 2:
                        return (
                            bodies[manifest_pair["donor"]["request_id"]],
                            bodies[manifest_pair["recipient"]["request_id"]],
                        )
    raise RuntimeError(
        f"could not find both request_ids in parquet; got {list(bodies)}")


def _body(req: dict, *, max_tokens: int, cache_prompt: bool,
          top_logprobs: int = 0) -> dict:
    out = {
        "messages": req.get("messages", []),
        "tools": req.get("tools"),
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "cache_prompt": cache_prompt,
    }
    if top_logprobs > 0:
        out["logprobs"] = True
        out["top_logprobs"] = top_logprobs
    return out


def _first_token_distribution(rr_json: dict, top_logprobs: int) -> tuple[int, dict[int, float]]:
    """Extract (top1_token_id, {token_id -> log_prob}) for the first
    generated token from a chat-completions response that asked for
    logprobs. The response carries top_logprobs entries per generated
    token; we want index 0 — the first sampled one."""
    choice = rr_json["choices"][0]
    lp = choice.get("logprobs") or {}
    content = lp.get("content") or []
    if not content:
        # Fall back to the verbose body if available; some llama-server
        # builds put the full sampler info there.
        verbose = choice.get("__verbose") or {}
        if "completion_probabilities" in verbose:
            entries = verbose["completion_probabilities"][0]
            top1 = entries.get("id")
            lps: dict[int, float] = {}
            for e in entries.get("top_probs") or entries.get("probs") or []:
                tid = e.get("id")
                lp_val = e.get("logprob") or (math.log(e["prob"]) if e.get("prob") else None)
                if tid is not None and lp_val is not None:
                    lps[tid] = float(lp_val)
            return top1, lps
        raise RuntimeError("response did not include first-token logprobs")
    first = content[0]
    top1 = first["id"]
    lps: dict[int, float] = {first["id"]: float(first["logprob"])}
    for e in first.get("top_logprobs") or []:
        lps[int(e["id"])] = float(e["logprob"])
    return top1, lps


def _compare_distributions(top_a: int, dist_a: dict[int, float],
                            top_b: int, dist_b: dict[int, float],
                            top_logprobs: int) -> dict:
    """Top-1 agreement, top-K overlap (Jaccard on the K most-probable),
    and an approximate KL on the union of observed token ids.
    Approximate because the tail (vocab \\ top-N) is not observed —
    we re-normalize over the union and compute KL between the
    re-normalized distributions, which understates true KL if the
    tails matter but is a useful first-order signal."""
    top_k = min(top_logprobs, len(dist_a), len(dist_b))
    sorted_a = sorted(dist_a.items(), key=lambda kv: -kv[1])[:top_k]
    sorted_b = sorted(dist_b.items(), key=lambda kv: -kv[1])[:top_k]
    set_a = {tid for tid, _ in sorted_a}
    set_b = {tid for tid, _ in sorted_b}
    overlap = len(set_a & set_b) / max(1, len(set_a | set_b))

    # Re-normalize each distribution over the union of observed ids.
    union = set(dist_a) | set(dist_b)
    LOG_NEG_INF = -50.0  # tiny mass for unobserved-on-this-side ids
    log_pa = {tid: dist_a.get(tid, LOG_NEG_INF) for tid in union}
    log_pb = {tid: dist_b.get(tid, LOG_NEG_INF) for tid in union}
    # Renormalize.
    z_a = math.log(sum(math.exp(v) for v in log_pa.values()))
    z_b = math.log(sum(math.exp(v) for v in log_pb.values()))
    kl = 0.0
    for tid in union:
        log_pa_n = log_pa[tid] - z_a
        log_pb_n = log_pb[tid] - z_b
        pa = math.exp(log_pa_n)
        kl += pa * (log_pa_n - log_pb_n)

    return {
        "top1_match": top_a == top_b,
        "top1_a": top_a,
        "top1_b": top_b,
        f"top{top_k}_jaccard": round(overlap, 4),
        "kl_approx_nats": round(max(0.0, kl), 6),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True,
                    type=Path, help="splice_candidates.jsonl from categorize_matches")
    ap.add_argument("--gguf", required=True, type=Path,
                    help="GGUF file matching the manifest's model")
    ap.add_argument("--top", type=int, default=3,
                    help="Number of manifest pairs to test (largest chunks first).")
    ap.add_argument("--n-cache-reuse", type=int, default=128)
    ap.add_argument("--ctx-size", type=int, default=65536)
    ap.add_argument("--top-logprobs", type=int, default=20,
                    help="Top-N logprobs to request on the first generated token.")
    ap.add_argument(
        "--server-bin",
        type=Path,
        default=Path("/home/ubuntu/llama.cpp/build/bin/llama-server"),
    )
    ap.add_argument("--output", type=Path, default=None,
                    help="Optional path to dump per-pair results as JSON.")
    args = ap.parse_args()

    if not args.gguf.exists():
        print(f"ERROR: gguf missing: {args.gguf}", file=sys.stderr)
        return 2
    if not args.server_bin.exists():
        print(f"ERROR: server bin missing: {args.server_bin}", file=sys.stderr)
        return 2

    pairs: list[dict] = []
    with args.manifest.open() as fh:
        for line in fh:
            pairs.append(json.loads(line))
    pairs.sort(key=lambda c: -c["chunk_n_tokens"])
    pairs = pairs[: args.top]
    print(f"[manifest] {len(pairs)} pair(s) selected (top by chunk size)",
          flush=True)

    results = []
    for i, pair in enumerate(pairs):
        print(f"\n[pair {i}] bucket={pair['bucket_id']} "
              f"chunk_n_tokens={pair['chunk_n_tokens']} "
              f"drift={pair['position_drift']:+}", flush=True)
        try:
            donor_body, recip_body = _fetch_pair_bodies(pair)
        except RuntimeError as e:
            print(f"  skip: {e}", flush=True)
            continue

        # ----- SPLICED run -----
        port = _free_port()
        with _server(args, port) as (url, log_lines, log_lock):
            print("  [spliced] sending donor (max_tokens=1, cache_prompt=true)...",
                  flush=True)
            rd = httpx.post(url, json=_body(donor_body, max_tokens=1, cache_prompt=True),
                            timeout=600.0)
            if rd.status_code != 200:
                print(f"  donor error: {rd.status_code} {rd.text[:300]}", flush=True)
                continue
            t = rd.json()["timings"]
            print(f"    donor: cache_n={t['cache_n']} prompt_n={t['prompt_n']} "
                  f"prompt_ms={t['prompt_ms']:.0f}", flush=True)
            time.sleep(0.5)
            with log_lock:
                fence = len(log_lines)

            print("  [spliced] sending recipient (max_tokens=1, logprobs)...",
                  flush=True)
            rr = httpx.post(url,
                            json=_body(recip_body, max_tokens=1, cache_prompt=True,
                                       top_logprobs=args.top_logprobs),
                            timeout=600.0)
            if rr.status_code != 200:
                print(f"  recipient error: {rr.status_code} {rr.text[:300]}", flush=True)
                continue
            rr_json = rr.json()
            with open(f"/tmp/spl_pair_{i}_resp.json", "w") as fh:
                json.dump(rr_json, fh, indent=2)
            t = rr_json["timings"]
            print(f"    recipient: cache_n={t['cache_n']} prompt_n={t['prompt_n']} "
                  f"prompt_ms={t['prompt_ms']:.0f}", flush=True)
            try:
                top1_spl, dist_spl = _first_token_distribution(rr_json, args.top_logprobs)
            except RuntimeError as e:
                print(f"  spliced first-token extraction failed: {e}", flush=True)
                continue
            time.sleep(0.5)
            with log_lock:
                recipient_lines = log_lines[fence:]
            scheduled = []
            applied = []
            for ln in recipient_lines:
                m = SCHEDULED_RE.search(ln)
                if m:
                    scheduled.append(int(m.group(1)))
                m = REUSE_RE.search(ln)
                if m:
                    size = int(m.group(1))
                    a, b, c, d = (int(m.group(j)) for j in range(2, 6))
                    applied.append({"size": size, "shift": c - a})
            print(f"    splice scheduled={scheduled}", flush=True)
            for h in applied:
                print(f"    splice applied: size={h['size']:5d} shift={h['shift']:+}",
                      flush=True)

        # ----- COLD run -----
        port = _free_port()
        with _server(args, port) as (url, _ll, _lock):
            print("  [cold] sending recipient (max_tokens=1, logprobs)...", flush=True)
            rr = httpx.post(url,
                            json=_body(recip_body, max_tokens=1, cache_prompt=True,
                                       top_logprobs=args.top_logprobs),
                            timeout=600.0)
            if rr.status_code != 200:
                print(f"  cold error: {rr.status_code} {rr.text[:300]}", flush=True)
                continue
            rr_json = rr.json()
            t = rr_json["timings"]
            print(f"    cold: cache_n={t['cache_n']} prompt_n={t['prompt_n']} "
                  f"prompt_ms={t['prompt_ms']:.0f}", flush=True)
            try:
                top1_cold, dist_cold = _first_token_distribution(rr_json, args.top_logprobs)
            except RuntimeError as e:
                print(f"  cold first-token extraction failed: {e}", flush=True)
                continue

        # ----- compare -----
        cmp = _compare_distributions(top1_spl, dist_spl, top1_cold, dist_cold,
                                      args.top_logprobs)
        print(f"  [compare] top1_match={cmp['top1_match']} "
              f"top1_spliced={cmp['top1_a']} top1_cold={cmp['top1_b']} "
              f"top{min(args.top_logprobs, len(dist_spl), len(dist_cold))}_jaccard="
              f"{cmp[f'top{min(args.top_logprobs, len(dist_spl), len(dist_cold))}_jaccard']} "
              f"kl_approx={cmp['kl_approx_nats']} nats", flush=True)

        results.append({
            "pair_idx": i,
            "bucket_id": pair["bucket_id"],
            "expected_chunk_n_tokens": pair["chunk_n_tokens"],
            "expected_drift": pair["position_drift"],
            "scheduled_splices": scheduled,
            "applied_splices": applied,
            **cmp,
        })

    print("\n=== summary ===")
    for r in results:
        sched = sum(r["scheduled_splices"])
        applied = sum(s["size"] for s in r["applied_splices"])
        print(f"pair {r['pair_idx']}: bucket={r['bucket_id']:<22} "
              f"applied={applied:>6} top1_match={r['top1_match']} "
              f"kl_approx={r['kl_approx_nats']:.4f}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2))
        print(f"[output] wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
