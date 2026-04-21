#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["pytest>=8.0"]
# ///
"""Unit tests for drift_modes.py.

Run with either
    uv run --script test_drift_modes.py
or, for the usual pytest CLI surface,
    uvx --with-editable . pytest -v test_drift_modes.py

Tests are split into four sections:
1. Extraction-helper tests (role mapping, truncation, donor-triple extraction).
2. Shared inflate invariants, parametrized across all four drift modes.
3. Mode-specific behaviours (one class per mode).
4. Dispatcher tests (`drift_modes.inflate`).
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import pytest

import drift_modes as dm


# ---------------------------------------------------------------------------
# Dummy tokenizer (whitespace-word, stable vocabulary). Implements just
# enough of the HF tokenizer interface for `drift_modes` to run without
# any model / HF dependency.
# ---------------------------------------------------------------------------


class DummyTokenizer:
    """Whitespace tokenizer with a growing word→id vocabulary.

    Supports:
      * ``__call__(text, add_special_tokens=False) -> {"input_ids": [ids]}``
      * ``decode(ids) -> text``
      * ``apply_chat_template(msgs, tokenize=True, add_generation_prompt,
        return_dict=True) -> {"input_ids": [ids]}``

    The chat template renders each message as ``"<role>: <content> <|sep|>"``
    so rendered length reflects both content and turn count.
    """

    def __init__(self) -> None:
        self._vocab: Dict[str, int] = {}
        self._ivocab: Dict[int, str] = {}

    def _id(self, tok: str) -> int:
        if tok not in self._vocab:
            i = len(self._vocab)
            self._vocab[tok] = i
            self._ivocab[i] = tok
        return self._vocab[tok]

    def __call__(self, text: str, add_special_tokens: bool = False):
        ids = [self._id(t) for t in text.split()]
        return {"input_ids": ids}

    def decode(self, ids: List[int]) -> str:
        return " ".join(self._ivocab[i] for i in ids)

    def apply_chat_template(
        self,
        msgs,
        tokenize: bool = True,
        add_generation_prompt: bool = False,
        return_dict: bool = False,
        return_tensors=None,
    ):
        parts: List[str] = []
        for m in msgs:
            parts.append(f"{m['role']}:")
            parts.append(m["content"])
            parts.append("<|sep|>")
        if add_generation_prompt:
            parts.append("assistant:")
        ids = self(" ".join(parts))["input_ids"]
        return {"input_ids": ids} if return_dict else ids


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def make_hermes_example(
    system: str = "SYS one two three four five",
    user: str = "USER question marker",
    asst: str = "ASST tool_call body body body",
    tool: str = "TOOL result aaa bbb ccc ddd eee fff",
    asst_reply: str = "ASST reply text",
) -> dict:
    """Hermes-shaped synthetic example with a complete
    ``[system, user, asst(tool_call), tool, asst(reply)]`` conversation.
    """
    return {
        "conversations": [
            {"from": "system", "value": system},
            {"from": "human", "value": user},
            {"from": "gpt", "value": asst},
            {"from": "tool", "value": tool},
            {"from": "gpt", "value": asst_reply},
        ]
    }


@pytest.fixture
def tok() -> DummyTokenizer:
    return DummyTokenizer()


@pytest.fixture
def base_msgs() -> List[Dict[str, str]]:
    return dm.prompt_msgs_through_tool(make_hermes_example())


@pytest.fixture
def donors() -> List[dict]:
    """Three donor examples with distinct system prompts and tool triples."""
    return [
        make_hermes_example(
            system="DONOR1_SYS beta gamma delta eps zeta eta theta iota",
            user="DUQ1 alpha",
            asst="DAC1 kappa lambda mu nu",
            tool="DTR1 xi omicron pi rho sigma tau",
        ),
        make_hermes_example(
            system="DONOR2_SYS aa bb cc dd ee ff gg",
            user="DUQ2 hh ii",
            asst="DAC2 jj kk ll",
            tool="DTR2 mm nn oo pp",
        ),
        make_hermes_example(
            system="DONOR3_SYS 1 2 3 4 5 6 7 8 9",
            user="DUQ3 a1",
            asst="DAC3 b2 c3",
            tool="DTR3 d4 e5 f6",
        ),
    ]


# Uniform adapter so parametrized tests can call each inflate function
# with the same signature (tok, base, target_delta, donors).
InflateCall = Callable[
    [DummyTokenizer, List[Dict[str, str]], int, List[dict]],
    Tuple[List[Dict[str, str]], int],
]

INFLATE_ADAPTERS: Dict[str, InflateCall] = {
    "system-duplicate": lambda tok, base, d, donors: dm.inflate_system_duplicate(
        tok, base, d
    ),
    "system-instructions": lambda tok, base, d, donors: dm.inflate_system_instructions(
        tok, base, d, donors
    ),
    "turn-insert": lambda tok, base, d, donors: dm.inflate_turn_insert(
        tok, base, d, donors
    ),
    "prior-tool-exchange": lambda tok, base, d, donors: dm.inflate_prior_tool_exchange(
        tok, base, d, donors
    ),
}

ALL_MODES = list(INFLATE_ADAPTERS.keys())


# ---------------------------------------------------------------------------
# 1. Extraction-helper tests
# ---------------------------------------------------------------------------


class TestHermesExtraction:
    @pytest.mark.parametrize(
        "role_in,role_out",
        [
            ("system", "system"),
            ("human", "user"),
            ("user", "user"),
            ("gpt", "assistant"),
            ("assistant", "assistant"),
            ("tool", "user"),
            ("function", "user"),
        ],
    )
    def test_role_mapping(self, role_in, role_out):
        ex = {"conversations": [{"from": role_in, "value": "x"}]}
        msgs = dm.hermes_to_messages(ex)
        assert msgs == [{"role": role_out, "content": "x"}]

    def test_hermes_to_messages_full_conversation(self):
        msgs = dm.hermes_to_messages(make_hermes_example())
        assert [m["role"] for m in msgs] == [
            "system",
            "user",
            "assistant",
            "user",
            "assistant",
        ]

    def test_prompt_msgs_through_tool_truncates_after_tool(self):
        msgs = dm.prompt_msgs_through_tool(make_hermes_example())
        assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]

    def test_prompt_msgs_through_tool_empty_when_no_tool(self):
        ex = {"conversations": [{"from": "human", "value": "hi"}]}
        assert dm.prompt_msgs_through_tool(ex) == []

    def test_reference_assistant_response(self):
        ex = make_hermes_example(asst_reply="THE REPLY")
        assert dm.reference_assistant_response(ex) == "THE REPLY"

    def test_reference_assistant_response_empty_when_missing(self):
        ex = {
            "conversations": [
                {"from": "human", "value": "q"},
                {"from": "gpt", "value": "call"},
                {"from": "tool", "value": "r"},
            ]
        }
        assert dm.reference_assistant_response(ex) == ""

    def test_extract_tool_triple(self):
        triple = dm.extract_tool_triple(
            make_hermes_example(user="Q", asst="CALL", tool="RESP")
        )
        assert triple is not None
        assert [m["role"] for m in triple] == ["user", "assistant", "user"]
        assert [m["content"] for m in triple] == ["Q", "CALL", "RESP"]

    def test_extract_tool_triple_none_when_no_tool(self):
        assert (
            dm.extract_tool_triple(
                {"conversations": [{"from": "human", "value": "hi"}]}
            )
            is None
        )


# ---------------------------------------------------------------------------
# 2. Shared inflate invariants, parametrized across all four modes
# ---------------------------------------------------------------------------


class TestInflateSharedInvariants:
    """Properties every inflate mode must satisfy."""

    @pytest.mark.parametrize("mode", ALL_MODES)
    def test_zero_delta_returns_zero_actual(self, mode, tok, base_msgs, donors):
        _, actual = INFLATE_ADAPTERS[mode](tok, base_msgs, 0, donors)
        assert actual == 0

    @pytest.mark.parametrize("mode", ALL_MODES)
    def test_returns_int_delta(self, mode, tok, base_msgs, donors):
        _, actual = INFLATE_ADAPTERS[mode](tok, base_msgs, 10, donors)
        assert isinstance(actual, int)
        assert actual >= 0

    @pytest.mark.parametrize("mode", ALL_MODES)
    def test_preserves_tool_content(self, mode, tok, base_msgs, donors):
        out, _ = INFLATE_ADAPTERS[mode](tok, base_msgs, 10, donors)
        # Final tool turn's content must be byte-identical to the input.
        assert out[-1]["content"] == base_msgs[-1]["content"]

    @pytest.mark.parametrize("mode", ALL_MODES)
    def test_preserves_tool_call_content(self, mode, tok, base_msgs, donors):
        out, _ = INFLATE_ADAPTERS[mode](tok, base_msgs, 10, donors)
        # Penultimate turn is the assistant tool_call; content unchanged.
        assert out[-2]["content"] == base_msgs[-2]["content"]

    @pytest.mark.parametrize("mode", ALL_MODES)
    def test_final_turn_structure(self, mode, tok, base_msgs, donors):
        out, _ = INFLATE_ADAPTERS[mode](tok, base_msgs, 10, donors)
        # Prompt must still end on assistant(tool_call) → user(tool_response).
        assert out[-2]["role"] == "assistant"
        assert out[-1]["role"] == "user"

    @pytest.mark.parametrize("mode", ALL_MODES)
    def test_does_not_mutate_input(self, mode, tok, base_msgs, donors):
        snapshot = [dict(m) for m in base_msgs]
        INFLATE_ADAPTERS[mode](tok, base_msgs, 10, donors)
        assert base_msgs == snapshot

    @pytest.mark.parametrize("mode", ALL_MODES)
    def test_hits_target_delta(self, mode, tok, base_msgs, donors):
        # With a tiny target and real donors, every mode should meet
        # or exceed the target (turn-granular modes will overshoot).
        _, actual = INFLATE_ADAPTERS[mode](tok, base_msgs, 5, donors)
        assert actual >= 5


# ---------------------------------------------------------------------------
# 3. Mode-specific behaviours
# ---------------------------------------------------------------------------


class TestSystemDuplicate:
    def test_appends_to_system_turn(self, tok, base_msgs):
        out, _ = dm.inflate_system_duplicate(tok, base_msgs, target_delta=5)
        assert out[0]["content"].startswith(base_msgs[0]["content"])
        assert len(out[0]["content"]) > len(base_msgs[0]["content"])

    def test_no_donors_needed(self, tok, base_msgs):
        # Signature does not accept donor_examples; runs standalone.
        out, actual = dm.inflate_system_duplicate(tok, base_msgs, 10)
        assert actual >= 10
        assert len(out) == len(base_msgs)  # turn count unchanged


class TestSystemInstructions:
    def test_pulls_from_donors(self, tok, base_msgs, donors):
        out, actual = dm.inflate_system_instructions(tok, base_msgs, 5, donors)
        # At least one donor token must appear in the inflated system turn.
        donor_words = {"beta", "gamma", "delta", "aa", "cc", "1", "2", "3"}
        system_words = set(out[0]["content"].split())
        assert donor_words & system_words
        assert actual >= 5

    def test_empty_donors_is_noop(self, tok, base_msgs):
        out, actual = dm.inflate_system_instructions(
            tok, base_msgs, target_delta=10, donor_examples=[]
        )
        assert actual == 0
        assert [m["content"] for m in out] == [m["content"] for m in base_msgs]

    def test_keeps_original_system_prefix(self, tok, base_msgs, donors):
        out, _ = dm.inflate_system_instructions(tok, base_msgs, 5, donors)
        assert out[0]["content"].startswith(base_msgs[0]["content"])


class TestTurnInsert:
    def test_inserts_after_original_user(self, tok, base_msgs, donors):
        out, _ = dm.inflate_turn_insert(tok, base_msgs, 10, donors)
        # base layout: [system, user=UQ, asst=ACALL, user=tool_response]
        # after insert: [system, user=UQ, <donor triples...>, asst=ACALL, user=tool_response]
        assert out[0]["role"] == "system"
        assert out[1]["role"] == "user"
        assert out[1]["content"] == base_msgs[1]["content"]
        # The message right after the original user is the first donor turn.
        donor_user_contents = {"DUQ1 alpha", "DUQ2 hh ii", "DUQ3 a1"}
        assert out[2]["content"] in donor_user_contents

    def test_does_not_modify_system_turn(self, tok, base_msgs, donors):
        out, _ = dm.inflate_turn_insert(tok, base_msgs, 10, donors)
        assert out[0]["content"] == base_msgs[0]["content"]

    def test_empty_donors_is_noop(self, tok, base_msgs):
        out, actual = dm.inflate_turn_insert(
            tok, base_msgs, target_delta=10, donor_examples=[]
        )
        assert actual == 0
        assert out == [dict(m) for m in base_msgs]


class TestPriorToolExchange:
    def test_inserts_between_system_and_user(self, tok, base_msgs, donors):
        out, _ = dm.inflate_prior_tool_exchange(tok, base_msgs, 10, donors)
        # Expected layout: [system, <donor triples...>, user=UQ, asst=ACALL, user=tool_response]
        assert out[0]["role"] == "system"
        # First non-system turn is a donor user, not the original.
        donor_user_contents = {"DUQ1 alpha", "DUQ2 hh ii", "DUQ3 a1"}
        assert out[1]["content"] in donor_user_contents
        assert out[1]["content"] != base_msgs[1]["content"]

    def test_original_user_appears_exactly_once(self, tok, base_msgs, donors):
        out, _ = dm.inflate_prior_tool_exchange(tok, base_msgs, 10, donors)
        marker = base_msgs[1]["content"]
        positions = [i for i, m in enumerate(out) if m["content"] == marker]
        assert len(positions) == 1
        assert positions[0] > 1  # after the donor triples

    def test_empty_donors_is_noop(self, tok, base_msgs):
        out, actual = dm.inflate_prior_tool_exchange(
            tok, base_msgs, target_delta=10, donor_examples=[]
        )
        assert actual == 0
        assert out == [dict(m) for m in base_msgs]


# ---------------------------------------------------------------------------
# 4. Dispatcher
# ---------------------------------------------------------------------------


class TestInflateDispatcher:
    @pytest.mark.parametrize("mode", ALL_MODES)
    def test_dispatches_to_registered_mode(self, mode, tok, base_msgs, donors):
        out, actual = dm.inflate(
            mode, tok, base_msgs, target_delta=3, donor_examples=donors
        )
        assert isinstance(out, list)
        assert isinstance(actual, int)

    def test_unknown_mode_raises(self, tok, base_msgs):
        with pytest.raises(ValueError):
            dm.inflate("nope", tok, base_msgs, 1, donor_examples=None)

    def test_drift_modes_registry_matches_adapters(self):
        assert set(dm.DRIFT_MODES) == set(ALL_MODES)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
