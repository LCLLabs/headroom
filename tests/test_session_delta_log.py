"""Tests for message-level session delta logging."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from headroom.proxy.session_delta_log import (
    build_message_delta,
    capture_session_delta_log,
    extract_forwarded_messages,
    reset_session_delta_state_for_tests,
    session_delta_log_enabled,
)


@pytest.fixture(autouse=True)
def _clean_delta_state(monkeypatch, tmp_path: Path):
    reset_session_delta_state_for_tests()
    monkeypatch.setenv("HEADROOM_SESSION_DELTA_LOG", "1")
    monkeypatch.setenv("HEADROOM_SESSION_DELTA_LOG_DIR", str(tmp_path / "sessions"))
    yield
    reset_session_delta_state_for_tests()


def test_enabled_gate(monkeypatch):
    monkeypatch.delenv("HEADROOM_SESSION_DELTA_LOG", raising=False)
    assert session_delta_log_enabled() is False
    monkeypatch.setenv("HEADROOM_SESSION_DELTA_LOG", "1")
    assert session_delta_log_enabled() is True


def test_extract_anthropic_and_responses_messages():
    assert extract_forwarded_messages(
        {"messages": [{"role": "user", "content": "hi"}]}, provider="anthropic"
    ) == [{"role": "user", "content": "hi"}]
    assert extract_forwarded_messages(
        {"input": [{"role": "user", "content": "hi"}]}, provider="openai"
    ) == [{"role": "user", "content": "hi"}]
    assert extract_forwarded_messages({"input": "plain"}, provider="openai") == [
        {"role": "user", "content": "plain"}
    ]


def test_build_snapshot_append_rewrite_unchanged():
    m1 = [{"role": "user", "content": "a"}]
    m2 = m1 + [{"role": "assistant", "content": "b"}]
    m3 = [
        {"role": "user", "content": "a-rewritten"},
        {"role": "assistant", "content": "b"},
        {"role": "user", "content": "c"},
    ]

    snap = build_message_delta(None, m1)
    assert snap["op"] == "snapshot"
    assert snap["messages"] == m1

    append = build_message_delta(m1, m2)
    assert append["op"] == "append"
    assert append["prefix_len"] == 1
    assert append["messages"] == [{"role": "assistant", "content": "b"}]
    assert append["rewrite_reasons"] == []

    rewrite = build_message_delta(m2, m3, rewrite_reasons=["explore_pruner", "compress"])
    assert rewrite["op"] == "rewrite"
    assert rewrite["from_index"] == 0
    assert rewrite["prefix_len"] == 0
    assert rewrite["messages"] == m3
    assert rewrite["rewrite_reasons"] == ["explore_pruner", "compress"]

    unchanged = build_message_delta(m3, m3, rewrite_reasons=["noop"])
    assert unchanged["op"] == "unchanged"
    assert unchanged["messages"] == []


def test_rewrite_last_message_only_keeps_prefix():
    prev = [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
    ]
    curr = [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b-compressed"},
    ]
    delta = build_message_delta(prev, curr, rewrite_reasons=["cache_mode_compress"])
    assert delta["op"] == "rewrite"
    assert delta["from_index"] == 1
    assert delta["prefix_len"] == 1
    assert delta["messages"] == [{"role": "assistant", "content": "b-compressed"}]
    assert delta["rewrite_reasons"] == ["cache_mode_compress"]


def test_capture_appends_jsonl_and_redacts(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("HEADROOM_SESSION_DELTA_LOG_DIR", str(tmp_path / "sessions"))
    sid = "sess-claude-1"

    path1 = capture_session_delta_log(
        provider="anthropic",
        body={
            "messages": [{"role": "user", "content": "hello", "api_key": "secret"}],
            "system": "sys",
        },
        session_id=sid,
        request_id="r1",
        rewrite_reasons=["initial"],
        tokens_saved=0,
    )
    assert path1 is not None
    assert path1.exists()

    path2 = capture_session_delta_log(
        provider="anthropic",
        body={
            "messages": [
                {"role": "user", "content": "hello", "api_key": "secret"},
                {"role": "assistant", "content": "world"},
            ],
        },
        session_id=sid,
        request_id="r2",
        transforms_applied=["lossless"],
        tokens_saved=10,
    )
    assert path2 == path1

    lines = path1.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    second = json.loads(lines[1])
    assert first["op"] == "snapshot"
    assert first["messages"][0]["api_key"] == "[REDACTED]"
    assert second["op"] == "append"
    assert second["messages"] == [{"role": "assistant", "content": "world"}]
    assert second["tokens_saved"] == 10


def test_capture_disabled_is_noop(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("HEADROOM_SESSION_DELTA_LOG", "0")
    monkeypatch.setenv("HEADROOM_SESSION_DELTA_LOG_DIR", str(tmp_path / "sessions"))
    assert (
        capture_session_delta_log(
            provider="openai",
            body={"input": [{"role": "user", "content": "x"}]},
            session_id="s",
            request_id="r",
        )
        is None
    )
    assert not (tmp_path / "sessions").exists() or not any((tmp_path / "sessions").iterdir())


def test_history_truncate_snapshots_with_reason():
    prev = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
    curr = [{"role": "user", "content": "fresh"}]
    delta = build_message_delta(prev, curr)
    assert delta["op"] == "snapshot"
    assert "history_truncated" in delta["rewrite_reasons"]
