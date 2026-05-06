#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx>=0.27"]
# ///
"""Smoke test for llama.cpp's --cache-reuse on a deliberately tailored
donor/recipient pair, using a tiny Llama-3.2-1B GGUF.

What it verifies (and what it does NOT)
---------------------------------------

llama.cpp's --cache-reuse loop in tools/server/server-context.cpp walks
``head_c`` (cache index) forward by 1 on every non-match while keeping
``head_p`` (recipient index) PINNED at the current matched-extent.
That means it can only find matches where ``cached[head_c:]`` starts
with ``input[head_p:]`` for some ``head_c`` past CP — i.e., the
recipient's content past CP must be a **contiguous suffix** of the
donor's content past CP. The KV shift it produces is therefore always
negative or zero (cached content moves to a *lower* position in the
recipient).

This test exercises that suffix case: donor has a long preamble +
SHARED chunk + tail; recipient has the SHARED chunk + same tail (no
preamble). After the CP scan, --cache-reuse should walk through the
preamble in cache, hit the SHARED chunk, find a long match against the
recipient's content, and splice with a negative shift equal to the
preamble length.

The test does NOT cover (and current --cache-reuse cannot handle) the
case our agent traces actually exhibit most: the SAME chunk recurring
at DIFFERENT positions in both donor and recipient, with divergent
content surrounding it on both sides. Verifying that case requires
extending the algorithm to also walk ``head_p`` — separate work.

Pass criteria
-------------

1. At least one ``reusing chunk with size N, shifting KV cache ...``
   line appears in server stderr for the recipient request.
2. The largest reused chunk's size is comfortably above the
   --cache-reuse threshold (≥ 30 tokens for a chunk we built to be
   ~50 tokens), guarding against accidental tiny-match hits.
3. KV shift is negative (cached chunk moved to a lower position in the
   recipient — confirms the suffix-match semantics).

Usage:
    uv run --script trace_analysis/test_cache_reuse_smoke.py
"""

from __future__ import annotations

import re
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx


SERVER_BIN = "/home/ubuntu/llama.cpp/build/bin/llama-server"
GGUF_PATH = (
    "/home/ubuntu/llama.cpp/models/llama-3.2-1b/"
    "Llama-3.2-1B-Instruct-Q4_K_M.gguf"
)

# Long enough to clear --cache-reuse 16 with margin. ~50 tokens.
SHARED_CHUNK = (
    "The Eiffel Tower is a wrought-iron lattice tower on the Champ de "
    "Mars in Paris, France. It is named after the engineer Gustave "
    "Eiffel, whose company designed and built the tower from 1887 to "
    "1889. Locally nicknamed 'La dame de fer', it was constructed as "
    "the centrepiece of the 1889 World's Fair."
)

# Preamble that exists ONLY in donor's cached prompt — its presence is
# what creates the position offset we want to verify gets RoPE-shifted.
# Trailing "\n\n" forces SHARED_CHUNK to start after a "\n\n" token in
# donor too. The recipient's user content starts directly with SHARED,
# preceded by the user-header template's own "\n\n". Same preceding
# token ⇒ same BPE state ⇒ same SHARED tokenization ⇒ a real match for
# --cache-reuse to find. Without this fence, donor's SHARED would start
# after ": " (different left-context than recipient's) and the tokens
# wouldn't agree at the first position, so the match-walker bails.
LONG_PREAMBLE = (
    "Before answering, please be aware that this is a test scenario "
    "with a deliberately verbose preamble that fills several dozen "
    "tokens of the donor's user message but is absent from the "
    "recipient's user message, so the shared content below ends up at "
    "a lower absolute position in the recipient than in the donor.\n\n"
)


REUSE_RE = re.compile(
    r"reusing chunk with size (\d+), shifting KV cache "
    r"\[(\d+), (\d+)\) -> \[(\d+), (\d+)\)"
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_health(port: int, timeout: float = 60.0) -> None:
    start = time.time()
    url = f"http://127.0.0.1:{port}/health"
    while time.time() - start < timeout:
        try:
            r = httpx.get(url, timeout=1.0)
            if r.status_code == 200:
                return
        except (httpx.HTTPError, ConnectionError, OSError):
            pass
        time.sleep(0.5)
    raise RuntimeError(f"server did not become healthy on port {port}")


def main() -> int:
    if not Path(SERVER_BIN).exists():
        print(f"ERROR: server binary missing: {SERVER_BIN}", file=sys.stderr)
        return 2
    if not Path(GGUF_PATH).exists():
        print(f"ERROR: gguf missing: {GGUF_PATH}", file=sys.stderr)
        return 2

    port = _free_port()
    cmd = [
        SERVER_BIN,
        "--model", GGUF_PATH,
        "--port", str(port),
        "--host", "127.0.0.1",
        # Single slot: deterministic — both requests land in the same
        # slot so donor's KV is the cache the recipient's request sees.
        "--parallel", "1",
        "--ctx-size", "4096",
        "--cache-reuse", "16",
        "--jinja",
        "--no-warmup",
        "--log-prefix",
        "--log-timestamps",
        "-v",  # turn on DBG so the "trying to reuse chunks" line is visible
    ]
    print(f"[smoke] starting: {' '.join(cmd)}", flush=True)
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
        url = f"http://127.0.0.1:{port}/v1/chat/completions"

        # Donor: preamble + SHARED. This populates the slot's cache
        # with a longer past-CP region than the recipient will have.
        donor = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": LONG_PREAMBLE + SHARED_CHUNK,
                },
            ],
            "max_tokens": 1,
            "temperature": 0.0,
            "cache_prompt": True,
        }
        # Recipient: just SHARED. Its content past CP is a suffix of the
        # donor's content past CP, so --cache-reuse should fire.
        recipient = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": SHARED_CHUNK},
            ],
            "max_tokens": 4,
            "temperature": 0.0,
            "cache_prompt": True,
        }

        rd = httpx.post(url, json=donor, timeout=120.0)
        rd.raise_for_status()
        time.sleep(0.3)

        with log_lock:
            fence = len(log_lines)

        rr = httpx.post(url, json=recipient, timeout=120.0)
        rr.raise_for_status()
        time.sleep(0.5)

        with log_lock:
            recipient_lines = log_lines[fence:]

        reuse_hits = []
        for ln in recipient_lines:
            m = REUSE_RE.search(ln)
            if m:
                size = int(m.group(1))
                a, b, c, d = (int(m.group(i)) for i in range(2, 6))
                reuse_hits.append(
                    {"size": size, "cache": [a, b], "recipient": [c, d]}
                )

        ok = True
        if not reuse_hits:
            print(
                "FAIL: no 'reusing chunk' log line after recipient request",
                file=sys.stderr,
            )
            print("--- last 30 stderr lines ---", file=sys.stderr)
            for ln in recipient_lines[-30:]:
                sys.stderr.write(ln)
            ok = False
        else:
            print(f"[smoke] recipient triggered {len(reuse_hits)} reuse(s):")
            for h in reuse_hits:
                shift = h["recipient"][0] - h["cache"][0]
                print(
                    f"  size={h['size']:4d}  "
                    f"cache {h['cache']} -> recipient {h['recipient']}  "
                    f"shift={shift:+d}"
                )
            largest = max(h["size"] for h in reuse_hits)
            largest_shift = next(
                h["recipient"][0] - h["cache"][0]
                for h in reuse_hits
                if h["size"] == largest
            )
            if largest < 30:
                print(
                    f"FAIL: largest reuse size {largest} below sanity "
                    f"floor 30 (the SHARED chunk should yield ≥ 30 "
                    f"matched tokens)",
                    file=sys.stderr,
                )
                ok = False
            if largest_shift >= 0:
                print(
                    f"FAIL: largest-reuse shift {largest_shift:+d} is "
                    f"non-negative; expected negative (cache → lower "
                    f"position in recipient because preamble is in "
                    f"donor only)",
                    file=sys.stderr,
                )
                ok = False
        if ok:
            print("[smoke] OK")
        return 0 if ok else 1
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


if __name__ == "__main__":
    sys.exit(main())
