"""Tests for headroom.transforms.thinking_compactor.

The module ships its own ``_demo()`` self-check (kept for a quick manual run),
but had no pytest coverage until now. This file promotes those assertions to
real tests and adds the cases that motivated the ``bills_prior_thinking``
rewrite: a family-name short-circuit for known plain-text-reasoning providers
(deepseek/glm/qwen/minimax/kimi -- always billed, regardless of version
suffix), plus the version-number heuristic used as a fallback for everyone
else (dotted minor versions, date-like suffixes that must not be misread as
a version number).
"""

from __future__ import annotations

from typing import Any

import pytest

from headroom.transforms.thinking_compactor import (
    bills_prior_thinking,
    compact_reasoning_openai_chat,
    compact_thinking_to_text,
)


class FakeKompress:
    """Deterministic stand-in: always shrinks text to a fixed short summary."""

    def __init__(self) -> None:
        self.calls = 0

    def compress(self, text: str, allow_download: bool = True) -> Any:  # noqa: ARG002
        self.calls += 1
        return type("R", (), {"compressed": "short summary"})()


LONG = " ".join(["reasoning"] * 60)  # 60 words > default min_words=40


@pytest.mark.parametrize(
    ("model", "bills"),
    [
        # Anthropic hyphenated major-minor.
        ("claude-opus-4-6", True),
        ("claude-sonnet-4-6", True),
        ("claude-opus-4-8", True),
        ("claude-sonnet-5", True),
        ("claude-sonnet-4-5-20250929", False),
        ("claude-haiku-4-5-20251001", False),
        ("claude-3-5-sonnet-20241022", False),
        # Plain-text-reasoning families always bill, regardless of version suffix.
        ("glm-5.3", True),  # dotted minor version
        ("qwen3.7-max", True),  # fused family+version, dotted minor
        ("deepseek-v4-flash-0731", True),  # family match wins over the date-like suffix
        ("deepseek-v4.1-flash", True),  # family match wins over version parsing
        ("deepseek-v4-pro-0813", True),  # family match wins over the date-like suffix
        ("minimax-m2", True),
        ("kimi-k2.7", True),
        # Unrecognized third parties fall back to the Anthropic-style version heuristic.
        ("gpt-5.5", True),
        ("gpt-4.5", False),
    ],
)
def test_bills_prior_thinking(model: str, bills: bool) -> None:
    assert bills_prior_thinking(model) is bills


class TestCompactThinkingToText:
    def test_converts_old_thinking_to_marked_text(self) -> None:
        k = FakeKompress()
        msgs: list[dict[str, Any]] = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": LONG, "signature": "sig1"},
                    {"type": "tool_use", "id": "t1", "name": "calc", "input": {}},
                ],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}],
            },
            {  # last assistant turn: must stay verbatim
                "role": "assistant",
                "content": [{"type": "thinking", "thinking": LONG, "signature": "sig2"}],
            },
        ]

        out, stats = compact_thinking_to_text(msgs, kompress=k, keep_last_turns=1)

        old_content = out[1]["content"]
        assert old_content[0] == {"type": "text", "text": "[prior reasoning, compressed] short summary"}
        assert old_content[1]["type"] == "tool_use"
        assert out[3]["content"][0]["type"] == "thinking"
        assert stats["turns_compacted"] == 1
        assert stats["blocks"] == 1
        # Input must not be mutated in place.
        assert msgs[1]["content"][0]["type"] == "thinking"

    def test_determinism_hits_memo_cache(self) -> None:
        k = FakeKompress()
        msgs: list[dict[str, Any]] = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": [{"type": "thinking", "thinking": LONG, "signature": "sig1"}],
            },
        ]
        out1, _ = compact_thinking_to_text(msgs, kompress=k, keep_last_turns=0)
        calls_before = k.calls
        out2, _ = compact_thinking_to_text(msgs, kompress=k, keep_last_turns=0)
        assert out2[1]["content"][0] == out1[1]["content"][0]
        assert k.calls == calls_before

    def test_keep_last_turns_zero_compacts_everything(self) -> None:
        k = FakeKompress()
        msgs: list[dict[str, Any]] = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": [{"type": "thinking", "thinking": LONG, "signature": "sig1"}],
            },
            {"role": "user", "content": "next"},
            {
                "role": "assistant",
                "content": [{"type": "thinking", "thinking": LONG, "signature": "sig2"}],
            },
        ]
        out, stats = compact_thinking_to_text(msgs, kompress=k, keep_last_turns=0)
        assert out[3]["content"][0]["type"] == "text"
        assert stats["turns_compacted"] == 2

    def test_frozen_message_count_protects_cached_prefix(self) -> None:
        """A message still served from the provider's cache must not be flipped.

        Mirrors ``PrefixCacheTracker.get_frozen_message_count()``'s contract:
        messages at index < frozen_message_count are the real, cache_read_tokens
        -confirmed cached region. Compacting one there would bust an otherwise
        still-warm cache for no reason -- keep_last_turns alone can't protect it
        since it only looks at distance from the end, not what's actually cached.
        """
        k = FakeKompress()
        msgs: list[dict[str, Any]] = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": [{"type": "thinking", "thinking": LONG, "signature": "sig1"}],
            },
            {"role": "user", "content": "next"},
            {
                "role": "assistant",
                "content": [{"type": "thinking", "thinking": LONG, "signature": "sig2"}],
            },
        ]
        # keep_last_turns=0 would normally compact both turns; frozen_message_count=2
        # says index 0-1 are still cached, so only the turn past that boundary compacts.
        out, stats = compact_thinking_to_text(
            msgs, kompress=k, keep_last_turns=0, frozen_message_count=2
        )
        assert out[1]["content"][0]["type"] == "thinking"
        assert out[3]["content"][0]["type"] == "text"
        assert stats["turns_compacted"] == 1

    def test_cache_control_carried_to_emitted_text_block(self) -> None:
        k = FakeKompress()
        msgs: list[dict[str, Any]] = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": LONG,
                        "signature": "s",
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            },
            {"role": "user", "content": "next"},
        ]
        out, _ = compact_thinking_to_text(msgs, kompress=k, keep_last_turns=0)
        assert out[1]["content"][0].get("cache_control") == {"type": "ephemeral"}

    def test_short_thinking_block_left_untouched(self) -> None:
        k = FakeKompress()
        msgs: list[dict[str, Any]] = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": [{"type": "thinking", "thinking": "short", "signature": "s"}]},
            {"role": "user", "content": "next"},
        ]
        out, stats = compact_thinking_to_text(msgs, kompress=k, keep_last_turns=0)
        assert out[1]["content"][0]["type"] == "thinking"
        assert stats["blocks"] == 0

    def test_none_kompress_is_a_noop(self) -> None:
        msgs: list[dict[str, Any]] = [
            {"role": "assistant", "content": [{"type": "thinking", "thinking": LONG, "signature": "s"}]},
        ]
        out, stats = compact_thinking_to_text(msgs, kompress=None, keep_last_turns=0)
        assert out is msgs
        assert stats["turns_compacted"] == 0


class TestCompactReasoningOpenaiChat:
    def test_kimi_reasoning_content_compacted_and_glm_inline_think_compacted(self) -> None:
        k = FakeKompress()
        msgs: list[dict[str, Any]] = [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "kimi answer", "reasoning_content": LONG},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": f"<think>{LONG}</think> glm answer"},
            {"role": "user", "content": "q3"},
            {"role": "assistant", "content": "plain answer, no reasoning"},
        ]
        out, stats = compact_reasoning_openai_chat(msgs, kompress=k, keep_last_turns=1)
        assert out[1]["reasoning_content"] == "short summary"
        assert out[3]["content"] == "<think>short summary</think> glm answer"
        assert out[5]["content"] == "plain answer, no reasoning"
        assert stats["turns_compacted"] == 2
        assert stats["blocks"] == 2

    def test_keep_last_turns_protects_final_turn(self) -> None:
        k = FakeKompress()
        msgs: list[dict[str, Any]] = [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a", "reasoning_content": LONG},
        ]
        out, _ = compact_reasoning_openai_chat(msgs, kompress=k, keep_last_turns=1)
        assert out[1]["reasoning_content"] == LONG

    def test_drop_mode_removes_reasoning_without_a_compressor(self) -> None:
        msgs: list[dict[str, Any]] = [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "kimi answer", "reasoning_content": LONG},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": f"<think>{LONG}</think> glm answer"},
            {"role": "user", "content": "q3"},
            {"role": "assistant", "content": "plain answer, no reasoning"},
        ]
        out, stats = compact_reasoning_openai_chat(msgs, kompress=None, keep_last_turns=1, drop=True)
        assert out[1]["reasoning_content"] == ""
        assert out[3]["content"] == " glm answer"
        assert out[5]["content"] == "plain answer, no reasoning"
        assert stats["turns_compacted"] == 2
        assert stats["words_after"] == 0
