"""Drift-mode implementations for the reagent harness.

A "drift mode" is a way of transforming a base Hermes agent trace into
a drifted variant whose tool-result chunk sits at a different absolute
position. The transformation must preserve the cached chunk's content
(so the reused-cache comparison is meaningful) and the final message
structure (the prompt still ends on a tool-result turn).

Four modes are provided:

* ``system-duplicate`` — append copies of the example's own system
  content until the prompt grows by ≥ Δ tokens. Semantically null:
  the expected next-token distribution should be unchanged. This is
  the original reagent protocol; used to isolate RoPE-phase error
  from context-conditioning error.
* ``system-instructions`` — append real instruction content from
  *other* Hermes examples' system prompts. Simulates realistic
  agent evolution (a new skill added, an AGENTS.md update).
* ``turn-insert`` — splice donor ``[user, assistant(tool_call),
  tool]`` triples between the original ``user`` turn and the
  original ``assistant(tool_call)`` turn. Simulates a side
  conversation that occurred between the user question and the
  model's decision to call the tool.
* ``prior-tool-exchange`` — splice donor triples between the
  ``system`` turn and the original ``user`` turn. Simulates a
  multi-turn session where earlier tool calls happened first.

Each public ``inflate_*`` function returns ``(drifted_msgs,
actual_delta_tokens)``. Δ targets are floors; achievable Δ depends
on the granularity of the mode (tokens for the *system-* modes,
whole donor triples for the *turn-*/*prior-* modes).
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

ROLE_MAP = {
    "system": "system",
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    # Tool responses are carried as user-role turns with
    # <tool_response>…</tool_response> markers in the content.
    # Gemma / Llama chat templates reject a distinct ``tool`` role.
    "tool": "user",
    "function": "user",
}

# Set of drift modes the dispatcher understands. Kept here so tests
# can enumerate without importing the measurement script.
DRIFT_MODES = (
    "system-duplicate",
    "system-instructions",
    "turn-insert",
    "prior-tool-exchange",
)


# ---------------------------------------------------------------------------
# Conversation extraction helpers
# ---------------------------------------------------------------------------


def hermes_to_messages(ex) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for t in ex.get("conversations") or ex.get("messages") or []:
        role = t.get("from") or t.get("role") or ""
        content = t.get("value") or t.get("content") or ""
        r = ROLE_MAP.get(role, role)
        if not r:
            continue
        out.append({"role": r, "content": content})
    return out


def reference_assistant_response(ex) -> str:
    """Return the text of the assistant turn that follows the tool turn
    in a Hermes example — the "gold" response we'd compare against.
    Returns an empty string if the example has no such turn.
    """
    convs = ex.get("conversations") or []
    saw_tool = False
    for t in convs:
        role = t.get("from") or t.get("role") or ""
        if saw_tool and role in ("gpt", "assistant"):
            return t.get("value") or t.get("content") or ""
        if role == "tool":
            saw_tool = True
    return ""


def prompt_msgs_through_tool(ex) -> List[Dict[str, str]]:
    """Return the Hermes conversation truncated up to and including the
    first tool-result turn. Drops the trailing assistant summary so the
    prompt ends where a real agent would — before decoding the next
    assistant turn.
    """
    convs = ex.get("conversations") or []
    truncated = []
    saw_tool = False
    for t in convs:
        truncated.append(t)
        if (t.get("from") or t.get("role")) == "tool":
            saw_tool = True
            break
    if not saw_tool:
        return []
    return hermes_to_messages({"conversations": truncated})


def extract_tool_triple(ex) -> List[Dict[str, str]] | None:
    """Extract a complete ``[user, assistant, tool]`` triple from a
    donor Hermes example. The triple is the first contiguous run of
    (user → assistant → tool) turns in the conversation. Returns
    ``None`` if no such triple exists.

    The triple is returned in post-role-mapping form (i.e. the tool
    turn becomes role=user with the tool_response content inline), so
    it can be spliced directly into a message list destined for a
    chat template.
    """
    msgs = hermes_to_messages(ex)
    tool_idx = _find_first_tool_index(ex)
    if tool_idx is None or tool_idx < 2:
        return None
    # Walk backwards from tool to find the preceding assistant and user.
    # In Hermes the layout is (user, assistant, tool, assistant, user, ...).
    # Grab the (tool_idx-2, tool_idx-1, tool_idx) triple.
    start = tool_idx - 2
    if msgs[start]["role"] != "user" or msgs[start + 1]["role"] != "assistant":
        return None
    return [dict(m) for m in msgs[start : tool_idx + 1]]


def _find_first_tool_index(ex) -> int | None:
    """Index into ``hermes_to_messages(ex)`` of the first tool-response
    turn (post-role-mapping, so role=user). Returns ``None`` if none.
    """
    convs = ex.get("conversations") or []
    out_i = 0
    for t in convs:
        role = t.get("from") or t.get("role") or ""
        if role == "tool":
            return out_i
        r = ROLE_MAP.get(role, role)
        if r:
            out_i += 1
    return None


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_hermes_examples(
    n_base: int,
    min_tokens_fn: Callable[[dict], bool],
    n_donors: int = 0,
    config: str = "func_calling",
) -> Tuple[List[dict], List[dict]]:
    """Stream the Hermes ``config`` subset and return
    ``(base, donors)``, each a list of examples that have a tool turn
    and pass ``min_tokens_fn``. Donors come from strictly later
    passing examples than the base set, so base and donor pools are
    disjoint.
    """
    from datasets import load_dataset

    ds = load_dataset(
        "NousResearch/hermes-function-calling-v1",
        config,
        split="train",
        streaming=True,
    )
    base: List[dict] = []
    donors: List[dict] = []
    for ex in ds:
        if len(base) >= n_base and len(donors) >= n_donors:
            break
        convs = ex.get("conversations") or []
        if not any((t.get("from") or t.get("role")) == "tool" for t in convs):
            continue
        try:
            if not min_tokens_fn(ex):
                continue
        except Exception:
            continue
        if len(base) < n_base:
            base.append(ex)
        else:
            donors.append(ex)
    if len(base) < n_base:
        raise RuntimeError(f"only {len(base)}/{n_base} base examples found")
    if len(donors) < n_donors:
        raise RuntimeError(f"only {len(donors)}/{n_donors} donor examples found")
    return base, donors


# ---------------------------------------------------------------------------
# Rendering / length measurement
# ---------------------------------------------------------------------------


def rendered_length(tokenizer, msgs, add_generation_prompt: bool = False) -> int:
    """Return the rendered-prompt token length. Works with both HF
    tokenizers (which return tensors) and the test-time
    :class:`DummyTokenizer` (which returns plain lists).
    """
    enc = tokenizer.apply_chat_template(
        msgs,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        return_dict=True,
    )
    ids = enc["input_ids"] if hasattr(enc, "__getitem__") else enc
    if hasattr(ids, "shape"):
        return int(ids.shape[-1])
    if ids and isinstance(ids[0], (list, tuple)):
        return len(ids[0])
    return len(ids)


# ---------------------------------------------------------------------------
# Inflate functions
# ---------------------------------------------------------------------------


def _binary_search_delta(
    build: Callable[[int], Tuple[List[Dict[str, str]], int]],
    target_delta: int,
    hi_guess: int,
) -> Tuple[List[Dict[str, str]], int]:
    """Given a monotone-ish ``build(k) -> (msgs, delta)`` helper, find
    the smallest k such that achieved delta ≥ ``target_delta``. If the
    search escalates ``hi`` beyond 1_000_000 without meeting the
    target, return the best found so far.
    """
    lo, hi = 1, max(hi_guess, 1)
    msgs_hi, d_hi = build(hi)
    while d_hi < target_delta:
        hi *= 2
        if hi > 1_000_000:
            return msgs_hi, d_hi
        msgs_hi, d_hi = build(hi)
    while lo < hi:
        mid = (lo + hi) // 2
        _, d_mid = build(mid)
        if d_mid < target_delta:
            lo = mid + 1
        else:
            hi = mid
    return build(lo)


def _set_system_content(msgs, new_content: str) -> List[Dict[str, str]]:
    """Return a new message list with the first system turn's content
    replaced by ``new_content``. Other turns are shallow-copied.
    Returns ``None``-equivalent (original list) if no system turn.
    """
    sys_idx = next((i for i, m in enumerate(msgs) if m["role"] == "system"), None)
    if sys_idx is None:
        return [dict(m) for m in msgs]
    out = [dict(m) for m in msgs]
    out[sys_idx] = {"role": "system", "content": new_content}
    return out


def inflate_system_duplicate(
    tokenizer, base_msgs, target_delta: int
) -> Tuple[List[Dict[str, str]], int]:
    """Append duplicates of the example's own system content to the
    system turn until the rendered length has grown by ≥ ``target_delta``.
    """
    T0 = rendered_length(tokenizer, base_msgs)
    if target_delta <= 0:
        return [dict(m) for m in base_msgs], 0
    sys_idx = next((i for i, m in enumerate(base_msgs) if m["role"] == "system"), None)
    if sys_idx is None:
        return [dict(m) for m in base_msgs], 0
    original = base_msgs[sys_idx]["content"]
    sys_tokens = tokenizer(original, add_special_tokens=False)["input_ids"]
    if not sys_tokens:
        return [dict(m) for m in base_msgs], 0

    def build(k: int) -> Tuple[List[Dict[str, str]], int]:
        repeats = (k + len(sys_tokens) - 1) // len(sys_tokens)
        repeated = sys_tokens * max(1, repeats)
        append_text = tokenizer.decode(repeated[:k])
        trial = _set_system_content(base_msgs, original + "\n" + append_text)
        return trial, rendered_length(tokenizer, trial) - T0

    return _binary_search_delta(
        build, target_delta, hi_guess=max(target_delta * 4, 4 * len(sys_tokens))
    )


def inflate_system_instructions(
    tokenizer, base_msgs, target_delta: int, donor_examples
) -> Tuple[List[Dict[str, str]], int]:
    """Append real instruction content pulled from donor examples'
    system prompts until the rendered length has grown by ≥ ``target_delta``.
    """
    T0 = rendered_length(tokenizer, base_msgs)
    if target_delta <= 0:
        return [dict(m) for m in base_msgs], 0
    sys_idx = next((i for i, m in enumerate(base_msgs) if m["role"] == "system"), None)
    if sys_idx is None or not donor_examples:
        return [dict(m) for m in base_msgs], 0

    original = base_msgs[sys_idx]["content"]
    # Concatenate donor system contents into a single token pool.
    pool: List[int] = []
    for donor in donor_examples:
        donor_msgs = hermes_to_messages(donor)
        donor_sys = next(
            (m["content"] for m in donor_msgs if m["role"] == "system"), None
        )
        if donor_sys:
            pool.extend(tokenizer(donor_sys, add_special_tokens=False)["input_ids"])
    if not pool:
        return [dict(m) for m in base_msgs], 0

    def build(k: int) -> Tuple[List[Dict[str, str]], int]:
        # Wrap around if the target k exceeds available donor material.
        repeats = (k + len(pool) - 1) // len(pool)
        repeated = pool * max(1, repeats)
        append_text = tokenizer.decode(repeated[:k])
        trial = _set_system_content(base_msgs, original + "\n" + append_text)
        return trial, rendered_length(tokenizer, trial) - T0

    return _binary_search_delta(
        build, target_delta, hi_guess=max(target_delta * 4, 4 * len(pool))
    )


def _collect_donor_triples(donors) -> List[List[Dict[str, str]]]:
    triples = []
    for donor in donors:
        t = extract_tool_triple(donor)
        if t is not None:
            triples.append(t)
    return triples


def _final_user_asst_indices(msgs) -> Tuple[int, int]:
    """Find the index of the last ``user`` turn followed by the last
    ``assistant`` turn before the final ``tool`` (user-role) turn that
    ends the prompt. Returns ``(user_idx, asst_idx)``. Raises
    ``ValueError`` if the expected structure is missing.
    """
    # Scan from the end to find: msgs[-1] should be the tool turn
    # (role=user), msgs[-2] the assistant tool-call, msgs[-3] the user
    # question. We return (tool-2 index, tool-1 index).
    if len(msgs) < 3:
        raise ValueError("message list too short to find user→asst→tool")
    n = len(msgs)
    if msgs[n - 1]["role"] != "user" or msgs[n - 2]["role"] != "assistant":
        raise ValueError("last two turns must be assistant→tool (user-role)")
    # Find the preceding user turn; it may not be exactly at n-3 if the
    # example has extra assistant/user banter, but for Hermes
    # ``func_calling`` it is.
    for i in range(n - 3, -1, -1):
        if msgs[i]["role"] == "user":
            return i, n - 2
    raise ValueError("could not find preceding user turn")


def _insert_triples_at(msgs, insert_at: int, triples) -> List[Dict[str, str]]:
    """Return a new message list with ``triples`` (a list of
    3-message donor triples) spliced at position ``insert_at``. Each
    triple is inlined in order. Does not mutate inputs.
    """
    out: List[Dict[str, str]] = [dict(m) for m in msgs[:insert_at]]
    for triple in triples:
        for m in triple:
            out.append(dict(m))
    for m in msgs[insert_at:]:
        out.append(dict(m))
    return out


def inflate_turn_insert(
    tokenizer, base_msgs, target_delta: int, donor_examples
) -> Tuple[List[Dict[str, str]], int]:
    """Splice whole donor ``[user, assistant, tool]`` triples between
    the original user turn and the original assistant tool-call turn.
    Granularity is one whole triple.
    """
    T0 = rendered_length(tokenizer, base_msgs)
    if target_delta <= 0:
        return [dict(m) for m in base_msgs], 0
    try:
        user_idx, _ = _final_user_asst_indices(base_msgs)
    except ValueError:
        return [dict(m) for m in base_msgs], 0
    triples = _collect_donor_triples(donor_examples or [])
    if not triples:
        return [dict(m) for m in base_msgs], 0

    def build(k: int) -> Tuple[List[Dict[str, str]], int]:
        # Wrap donors if k exceeds pool size.
        repeats = (k + len(triples) - 1) // len(triples)
        pool = triples * max(1, repeats)
        trial = _insert_triples_at(base_msgs, user_idx + 1, pool[:k])
        return trial, rendered_length(tokenizer, trial) - T0

    return _binary_search_delta(
        build, target_delta, hi_guess=max(1, target_delta // 100)
    )


def inflate_prior_tool_exchange(
    tokenizer, base_msgs, target_delta: int, donor_examples
) -> Tuple[List[Dict[str, str]], int]:
    """Splice whole donor triples between the original system turn and
    the original user turn. The cached chunk (last L tokens of the
    tool turn) remains in place at the end of the prompt.
    """
    T0 = rendered_length(tokenizer, base_msgs)
    if target_delta <= 0:
        return [dict(m) for m in base_msgs], 0
    # Insert right after the system turn, if any; otherwise at the start.
    sys_idx = next((i for i, m in enumerate(base_msgs) if m["role"] == "system"), None)
    insert_at = (sys_idx + 1) if sys_idx is not None else 0
    triples = _collect_donor_triples(donor_examples or [])
    if not triples:
        return [dict(m) for m in base_msgs], 0

    def build(k: int) -> Tuple[List[Dict[str, str]], int]:
        repeats = (k + len(triples) - 1) // len(triples)
        pool = triples * max(1, repeats)
        trial = _insert_triples_at(base_msgs, insert_at, pool[:k])
        return trial, rendered_length(tokenizer, trial) - T0

    return _binary_search_delta(
        build, target_delta, hi_guess=max(1, target_delta // 100)
    )


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def inflate(
    mode: str,
    tokenizer,
    base_msgs,
    target_delta: int,
    donor_examples=None,
) -> Tuple[List[Dict[str, str]], int]:
    """Dispatch to the inflate function for ``mode``. ``donor_examples``
    is required for every mode except ``system-duplicate``.
    """
    if mode == "system-duplicate":
        return inflate_system_duplicate(tokenizer, base_msgs, target_delta)
    if mode == "system-instructions":
        return inflate_system_instructions(
            tokenizer, base_msgs, target_delta, donor_examples
        )
    if mode == "turn-insert":
        return inflate_turn_insert(tokenizer, base_msgs, target_delta, donor_examples)
    if mode == "prior-tool-exchange":
        return inflate_prior_tool_exchange(
            tokenizer, base_msgs, target_delta, donor_examples
        )
    raise ValueError(f"unknown drift mode: {mode!r}; expected one of {DRIFT_MODES}")
