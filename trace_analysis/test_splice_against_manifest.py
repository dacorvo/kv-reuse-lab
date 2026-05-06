#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx>=0.27", "pyarrow>=15", "huggingface_hub>=0.25"]
# ///
"""End-to-end test of the patched ``--cache-reuse`` (symmetric) against
real agentcap manifest pairs.

For each pair in the manifest, fetch donor and recipient request bodies
from the source parquet, send them sequentially to a patched llama-server
running the matching GGUF, and parse the server's stderr to verify the
splice mechanism actually fires on real workload data.

Output is a per-pair summary: chunk size scheduled by the search loop,
chunk size actually applied (after [TAG_PROMPT_LOGITS] holdback and
last-token clipping), how many of those tokens were attributed to
``cache_n``, and the splice's KV-shift direction.

Usage:
    test_splice_against_manifest.py \\
        --manifest trace_analysis/results/.../*.splice_candidates.jsonl \\
        --gguf /path/to/model.gguf \\
        --top 5
"""
from __future__ import annotations

import argparse
import json
import re
import socket
import subprocess
import sys
import threading
import time
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True,
                    type=Path, help="splice_candidates.jsonl from categorize_matches")
    ap.add_argument("--gguf", required=True, type=Path,
                    help="GGUF file matching the manifest's model")
    ap.add_argument("--top", type=int, default=3,
                    help="Number of manifest pairs to test (largest chunks first).")
    ap.add_argument("--n-cache-reuse", type=int, default=128)
    ap.add_argument("--ctx-size", type=int, default=32768)
    ap.add_argument(
        "--server-bin",
        type=Path,
        default=Path("/home/ubuntu/llama.cpp/build/bin/llama-server"),
    )
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

    port = _free_port()
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
        "--n-gpu-layers", "999",
        "-v",
    ]
    print(f"[server] starting: {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
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
        print("[server] ready", flush=True)
        url = f"http://127.0.0.1:{port}/v1/chat/completions"

        results = []
        for i, pair in enumerate(pairs):
            print(f"\n[pair {i}] bucket={pair['bucket_id']} "
                  f"chunk_n_tokens={pair['chunk_n_tokens']} "
                  f"drift={pair['position_drift']:+}",
                  flush=True)
            try:
                donor_body, recip_body = _fetch_pair_bodies(pair)
            except RuntimeError as e:
                print(f"  skip: {e}", flush=True)
                continue

            def _strip(body: dict, max_tokens: int) -> dict:
                return {
                    "messages": body.get("messages", []),
                    "tools": body.get("tools"),
                    "max_tokens": max_tokens,
                    "temperature": 0.0,
                    "cache_prompt": True,
                }

            # Donor: prefill only (n=1 to flush prefill).
            print(f"  [donor] posting...", flush=True)
            rd = httpx.post(url, json=_strip(donor_body, 1), timeout=600.0)
            if rd.status_code != 200:
                print(f"  donor error: {rd.status_code} {rd.text[:500]}", flush=True)
                continue
            donor_timings = rd.json()["timings"]
            print(f"    cache_n={donor_timings['cache_n']} "
                  f"prompt_n={donor_timings['prompt_n']} "
                  f"prompt_ms={donor_timings['prompt_ms']:.0f}",
                  flush=True)

            # Mark the log fence for the recipient request.
            time.sleep(0.5)
            with log_lock:
                fence = len(log_lines)

            print(f"  [recipient] posting...", flush=True)
            rr = httpx.post(url, json=_strip(recip_body, 4), timeout=600.0)
            rr.raise_for_status()
            recip_timings = rr.json()["timings"]
            print(f"    cache_n={recip_timings['cache_n']} "
                  f"prompt_n={recip_timings['prompt_n']} "
                  f"prompt_ms={recip_timings['prompt_ms']:.0f}",
                  flush=True)
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

            print(f"  [splice] scheduled={scheduled}", flush=True)
            for h in applied:
                print(f"  [splice] applied: size={h['size']:5d} "
                      f"shift={h['shift']:+}", flush=True)

            results.append({
                "pair_idx": i,
                "bucket_id": pair["bucket_id"],
                "expected_chunk_n_tokens": pair["chunk_n_tokens"],
                "expected_drift": pair["position_drift"],
                "donor_cache_n": donor_timings["cache_n"],
                "donor_prompt_n": donor_timings["prompt_n"],
                "recipient_cache_n": recip_timings["cache_n"],
                "recipient_prompt_n": recip_timings["prompt_n"],
                "recipient_prompt_ms": recip_timings["prompt_ms"],
                "scheduled_splices": scheduled,
                "applied_splices": applied,
            })

        print("\n=== summary ===")
        for r in results:
            sched = sum(r["scheduled_splices"])
            applied = sum(s["size"] for s in r["applied_splices"])
            print(f"pair {r['pair_idx']}: bucket={r['bucket_id']:<22} "
                  f"expected_chunk={r['expected_chunk_n_tokens']:>6} "
                  f"scheduled={sched:>6} applied={applied:>6} "
                  f"cache_n={r['recipient_cache_n']:>6}")
        return 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


if __name__ == "__main__":
    sys.exit(main())
