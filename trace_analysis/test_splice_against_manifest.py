#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "httpx>=0.27",
#   "pyarrow>=15",
#   "huggingface_hub>=0.25",
#   "sentence-transformers>=3.0",
# ]
# ///
"""End-to-end splice-correctness measurement on real agentcap manifest
pairs through a patched llama-server.

Per pair, two server runs (separate processes — hybrid models leak
recurrent state across requests inside one server):

* **Spliced**: send donor (cache_prompt=true) then recipient
  (cache_prompt=true). The patched ``--cache-reuse`` fires; the
  recipient's prefill is built from the donor's K/V cells at the
  matched ranges, RoPE-rephased.

* **Cold**: send the recipient alone in a fresh server. No splice;
  cold-prefill baseline.

For each, the recipient generates ``--gen-tokens`` (default 64) tokens
greedy with logprobs on. Per pair:

* top-1 agreement at the first generated token.
* top-K Jaccard on the first-token top-N distributions.
* approximate KL on the union of observed top-N tokens (the API only
  exposes top-N logprobs, not full vocab — so this is renormalized
  over the union, an under-estimate of the true KL but useful as a
  first-order signal).
* sentence-embedding cosine similarity (``BAAI/bge-small-en-v1.5``)
  between the spliced and cold 64-token continuations.

Mirrors the metric set ``measure_multi_splice_b.py`` reported on the
transformers-based harness: ``kl``, ``agree``, ``sim_fresh_reused``,
plus splice coverage stats from the server log.

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
    if args.tensor_split:
        cmd += ["--tensor-split", args.tensor_split]
    if args.swa_full:
        cmd += ["--swa-full"]
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


def _strip_nulls(obj):
    """Drop ``None`` values recursively. The new agentcap parquet export
    carries explicit ``null`` for absent optional fields (e.g.
    ``messages[*].tool_call_id``, ``tools[*].function.parameters.
    properties.<param>``); llama-server's strict OpenAI parser rejects
    those with ``type must be string, but is null``."""
    if isinstance(obj, dict):
        return {k: _strip_nulls(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [_strip_nulls(v) for v in obj if v is not None]
    return obj


def _body(req: dict, *, max_tokens: int, cache_prompt: bool,
          top_logprobs: int = 0) -> dict:
    out = {
        "messages": _strip_nulls(req.get("messages", [])),
        "tools": _strip_nulls(req.get("tools")),
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "cache_prompt": cache_prompt,
        # Verbose response carries the *raw* generated text under
        # __verbose.content — before chat-format parsing strips
        # <think>/<tool_call>/etc. into separate fields. We need
        # that for the continuation-similarity metric since structured
        # tool calls leave message.content empty.
        "verbose": True,
    }
    if top_logprobs > 0:
        out["logprobs"] = True
        out["top_logprobs"] = top_logprobs
    return out


def _raw_continuation(rr_json: dict) -> str:
    """Return the unparsed model output. Tries three sources in order:
      1. ``__verbose.content`` — the raw string the model emitted, before
         peg-native chat-format splits it into ``content`` /
         ``reasoning_content`` / ``tool_calls``. Empty for some
         configurations.
      2. ``logprobs.content[*].token`` — concatenation of per-token text
         the sampler observed. Reliable since logprobs are requested.
      3. ``message.content`` — the parsed/extracted text, which is
         empty when the model emitted a structured tool call.
    The continuation-similarity metric needs whatever the model
    actually emitted, *before* chat-format parsing routes parts into
    structured fields, so the per-token reconstruction is the
    authoritative source when present."""
    choice = rr_json["choices"][0]
    verbose = choice.get("__verbose") or {}
    raw = verbose.get("content")
    if raw:
        return raw
    lp = (choice.get("logprobs") or {}).get("content") or []
    if lp:
        return "".join(e.get("token", "") for e in lp)
    return choice["message"].get("content") or ""


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
                    help="Top-N logprobs to request on each generated token.")
    ap.add_argument("--gen-tokens", type=int, default=64,
                    help="Tokens to greedy-decode after each prompt — first "
                    "token feeds the KL/top-1 metrics, the full continuation "
                    "feeds sim_fresh_reused.")
    ap.add_argument("--embedder", default="BAAI/bge-small-en-v1.5",
                    help="Sentence-transformer model for continuation similarity.")
    ap.add_argument(
        "--server-bin",
        type=Path,
        default=Path("/home/ubuntu/llama.cpp/build/bin/llama-server"),
    )
    ap.add_argument("--tensor-split", default=None,
                    help="llama-server --tensor-split argument (e.g. '1' for "
                    "single-GPU on a multi-GPU host). Default leaves it unset, "
                    "which uses pipeline parallelism across all visible GPUs.")
    ap.add_argument("--swa-full", action="store_true",
                    help="Pass --swa-full to llama-server. Required for SWA "
                    "models (Gemma-4 family) to allow K-shift / cache-reuse: "
                    "by default the SWA cache is sized smaller than the base "
                    "cache and the iswa size-mismatch assertion blocks shifts. "
                    "--swa-full equalizes both caches at the cost of the SWA "
                    "memory savings.")
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
    pairs.sort(key=lambda c: -c["chunk_n_chars"])
    pairs = pairs[: args.top]
    print(f"[manifest] {len(pairs)} pair(s) selected (top by chunk size)",
          flush=True)

    # Embedder for continuation similarity. CPU is fine — bge-small is
    # tiny and the model server has the GPUs.
    print(f"[embedder] loading {args.embedder} on CPU ...", flush=True)
    from sentence_transformers import SentenceTransformer
    import numpy as np
    embedder = SentenceTransformer(args.embedder, device="cpu")

    def _sim(a: str, b: str) -> float:
        if not a or not b:
            return float("nan")
        ea, eb = embedder.encode([a, b], convert_to_numpy=True, show_progress_bar=False)
        denom = float(np.linalg.norm(ea) * np.linalg.norm(eb))
        return float(np.dot(ea, eb) / denom) if denom > 0 else float("nan")

    results = []
    for i, pair in enumerate(pairs):
        print(f"\n[pair {i}] bucket={pair['bucket_id']} "
              f"chunk_n_chars={pair['chunk_n_chars']} "
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

            print(f"  [spliced] sending recipient (max_tokens={args.gen_tokens}, logprobs)...",
                  flush=True)
            rr = httpx.post(url,
                            json=_body(recip_body, max_tokens=args.gen_tokens,
                                       cache_prompt=True,
                                       top_logprobs=args.top_logprobs),
                            timeout=900.0)
            if rr.status_code != 200:
                print(f"  recipient error: {rr.status_code} {rr.text[:300]}", flush=True)
                continue
            rr_json = rr.json()
            t = rr_json["timings"]
            print(f"    recipient: cache_n={t['cache_n']} prompt_n={t['prompt_n']} "
                  f"prompt_ms={t['prompt_ms']:.0f} predicted_n={t.get('predicted_n')}",
                  flush=True)
            try:
                top1_spl, dist_spl = _first_token_distribution(rr_json, args.top_logprobs)
            except RuntimeError as e:
                print(f"  spliced first-token extraction failed: {e}", flush=True)
                continue
            text_spl = _raw_continuation(rr_json)
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
            print(f"  [cold] sending recipient (max_tokens={args.gen_tokens}, logprobs)...",
                  flush=True)
            rr = httpx.post(url,
                            json=_body(recip_body, max_tokens=args.gen_tokens,
                                       cache_prompt=True,
                                       top_logprobs=args.top_logprobs),
                            timeout=900.0)
            if rr.status_code != 200:
                print(f"  cold error: {rr.status_code} {rr.text[:300]}", flush=True)
                continue
            rr_json = rr.json()
            t = rr_json["timings"]
            print(f"    cold: cache_n={t['cache_n']} prompt_n={t['prompt_n']} "
                  f"prompt_ms={t['prompt_ms']:.0f} predicted_n={t.get('predicted_n')}",
                  flush=True)
            try:
                top1_cold, dist_cold = _first_token_distribution(rr_json, args.top_logprobs)
            except RuntimeError as e:
                print(f"  cold first-token extraction failed: {e}", flush=True)
                continue
            text_cold = _raw_continuation(rr_json)

        # ----- compare -----
        cmp = _compare_distributions(top1_spl, dist_spl, top1_cold, dist_cold,
                                      args.top_logprobs)
        sim = _sim(text_spl, text_cold)
        cmp["sim_fresh_reused"] = round(sim, 4) if sim == sim else None
        top_k = min(args.top_logprobs, len(dist_spl), len(dist_cold))
        print(f"  [compare] top1_match={cmp['top1_match']} "
              f"top1_spl={cmp['top1_a']} top1_cold={cmp['top1_b']} "
              f"top{top_k}_jaccard={cmp[f'top{top_k}_jaccard']} "
              f"kl_approx={cmp['kl_approx_nats']} nats "
              f"sim={cmp['sim_fresh_reused']}",
              flush=True)
        # Truncated continuations for inspection (also matches the
        # transformers harness's 256-char snippet convention).
        def _trunc(s: str, n: int = 256) -> str:
            s = s.replace("\n", "\\n")
            return s if len(s) <= n else s[:n] + "..."
        print(f"    spliced: {_trunc(text_spl)}", flush=True)
        print(f"    cold   : {_trunc(text_cold)}", flush=True)

        results.append({
            "pair_idx": i,
            "bucket_id": pair["bucket_id"],
            "expected_chunk_n_chars": pair["chunk_n_chars"],
            "expected_drift": pair["position_drift"],
            "scheduled_splices": scheduled,
            "applied_splices": applied,
            "spliced_text": text_spl[:512],
            "cold_text": text_cold[:512],
            **cmp,
        })

    print("\n=== summary ===")
    for r in results:
        applied = sum(s["size"] for s in r["applied_splices"])
        sim = r.get("sim_fresh_reused")
        sim_str = f"{sim:.3f}" if sim is not None else "n/a"
        print(f"pair {r['pair_idx']}: bucket={r['bucket_id']:<22} "
              f"applied={applied:>6} agree={int(r['top1_match'])} "
              f"kl={r['kl_approx_nats']:.4f} sim={sim_str}")
    if results:
        kls = [r["kl_approx_nats"] for r in results]
        agrees = [int(r["top1_match"]) for r in results]
        sims = [r.get("sim_fresh_reused") for r in results
                if r.get("sim_fresh_reused") is not None]
        sim_part = f" mean_sim={sum(sims)/len(sims):.4f}" if sims else ""
        print(f"  aggregate (N={len(results)}): "
              f"mean_kl={sum(kls)/len(kls):.4f} "
              f"agree_rate={sum(agrees)/len(agrees):.3f}{sim_part}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2))
        print(f"[output] wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
