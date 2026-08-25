"""Tests for opt-in ContextReducer before/after debug dumps."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from headroom.proxy.explore_pruner.debug_capture import (
    capture_explore_reducer_debug,
    explore_reducer_debug_enabled,
)
from headroom.proxy.explore_pruner.focus import EXPLORE_TOOL_NAME, PRUNED_CALL_IDS_KEY
from headroom.proxy.explore_pruner.protocol import clear_reducer_registry
from headroom.proxy.explore_pruner.service import ExploreToolService
from headroom.proxy.explore_pruner.types import ReduceInput, ReduceResult


class FakeTextReducer:
    name = "fake_text"

    def __init__(self, content: str = "REDUCED\nline2\n") -> None:
        self.content = content
        self.calls = 0

    async def reduce(self, inp: ReduceInput) -> ReduceResult | None:
        self.calls += 1
        return ReduceResult(content=self.content, metadata={"backend": "fake"})


@pytest.fixture(autouse=True)
def _clear_registry():
    clear_reducer_registry()
    yield
    clear_reducer_registry()


def test_explore_reducer_debug_enabled_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HEADROOM_EXPLORE_REDUCER_DEBUG", raising=False)
    assert explore_reducer_debug_enabled() is False
    monkeypatch.setenv("HEADROOM_EXPLORE_REDUCER_DEBUG", "1")
    assert explore_reducer_debug_enabled() is True


def test_capture_writes_content_files_with_real_newlines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HEADROOM_EXPLORE_REDUCER_DEBUG", "1")
    monkeypatch.setenv("HEADROOM_EXPLORE_REDUCER_DEBUG_DIR", str(tmp_path))

    before = "def foo():\n    return 1\n"
    after = "def foo():\n    return 2\n"
    paths = capture_explore_reducer_debug(
        reducer_name="fake_text",
        session_key="sess",
        call_id="c1",
        before=ReduceInput(content=before, query="Why?\nline2", config={"commands": ["sed"]}),
        after=ReduceResult(content=after, kept_frags=[1], metadata={"backend": "fake"}),
    )

    assert paths is not None
    meta_path, before_path, after_path = paths
    assert meta_path.exists()
    assert before_path.read_text(encoding="utf-8") == before
    assert after_path.read_text(encoding="utf-8") == after
    # Real newlines on disk, not escaped sequences in the text files.
    assert "\n" in before_path.read_text(encoding="utf-8")
    assert "\\n" not in before_path.read_bytes().decode("utf-8")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["reducer"] == "fake_text"
    assert meta["call_id"] == "c1"
    assert meta["session_key"] == "sess"
    assert meta["before"]["query"] == "Why?\nline2"
    assert meta["before"]["content_file"] == before_path.name
    assert meta["after"]["content_file"] == after_path.name
    assert meta["after"]["kept_frags"] == [1]
    assert "content" not in meta["before"]
    assert "content" not in meta["after"]


def test_capture_noop_when_disabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HEADROOM_EXPLORE_REDUCER_DEBUG", raising=False)
    monkeypatch.setenv("HEADROOM_EXPLORE_REDUCER_DEBUG_DIR", str(tmp_path))
    assert (
        capture_explore_reducer_debug(
            reducer_name="fake",
            session_key="s",
            call_id="c",
            before=ReduceInput(content="a\nb", query="q"),
            after=None,
        )
        is None
    )
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_prune_inbound_dumps_when_debug_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HEADROOM_EXPLORE_REDUCER_DEBUG", "1")
    monkeypatch.setenv("HEADROOM_EXPLORE_REDUCER_DEBUG_DIR", str(tmp_path))

    before_body = "\n".join(f"line_{i} = {i}" for i in range(200))
    after_body = "kept = 1\nkept = 2\n"
    reducer = FakeTextReducer(after_body)
    svc = ExploreToolService(reducer=reducer, min_chars_to_prune=100)
    svc.rewrite_outbound_items(
        [
            {
                "type": "function_call",
                "name": EXPLORE_TOOL_NAME,
                "call_id": "c1",
                "arguments": json.dumps(
                    {
                        "path": "/tmp/app.py",
                        "focus_question": "How does auth work?",
                        "start_line": 1,
                        "end_line": 50,
                    }
                ),
            }
        ],
        session_key="sess",
    )
    body = {"input": [{"type": "function_call_output", "call_id": "c1", "output": before_body}]}
    changed = await svc.prune_inbound(body, session_key="sess")
    assert changed is True
    assert "c1" in body[PRUNED_CALL_IDS_KEY]

    metas = list(tmp_path.glob("*_reduce.json"))
    befores = list(tmp_path.glob("*_before.txt"))
    afters = list(tmp_path.glob("*_after.txt"))
    assert len(metas) == 1
    assert len(befores) == 1
    assert len(afters) == 1
    assert afters[0].read_text(encoding="utf-8") == after_body
    assert "\n" in befores[0].read_text(encoding="utf-8")
