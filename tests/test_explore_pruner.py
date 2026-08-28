"""Unit tests for explore_tool + pluggable context reducers."""

from __future__ import annotations

import json
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from headroom.proxy.explore_pruner.focus import (
    EXPLORE_TOOL_INSTRUCTIONS,
    EXPLORE_TOOL_NAME,
    PRUNED_CALL_IDS_KEY,
    append_explore_source_code_tool,
    clamp_explore_window,
    has_python_file_scope,
    has_supported_source_file_scope,
    is_explore_source_code_call,
    rewrite_explore_call_to_exec,
    tool_output_looks_truncated,
)
from headroom.proxy.explore_pruner.protocol import (
    clear_reducer_registry,
    get_reducer,
    register_reducer,
)
from headroom.proxy.explore_pruner.service import ExploreToolService
from headroom.proxy.explore_pruner.store import ExplorePrunerRecord, ExplorePrunerStore
from headroom.proxy.explore_pruner.types import ReduceInput, ReduceResult
from headroom.proxy.models import ExplorePrunerConfig, ProxyConfig


class FakeTextReducer:
    """Reducer that only returns content (no kept_frags → no AST)."""

    name = "fake_text"

    def __init__(self, content: str = "REDUCED") -> None:
        self.content = content
        self.calls = 0

    async def reduce(self, inp: ReduceInput) -> ReduceResult | None:
        self.calls += 1
        return ReduceResult(content=self.content)


class FakeNoneReducer:
    name = "fake_none"

    async def reduce(self, inp: ReduceInput) -> ReduceResult | None:
        return None


class FakeKeptFragsReducer:
    name = "fake_kept"

    async def reduce(self, inp: ReduceInput) -> ReduceResult | None:
        return ReduceResult(
            content="(compressed 1 lines: omitted)\nx = 1\n",
            kept_frags=[2],
        )


def _mock_async_httpx(response: MagicMock) -> tuple[AsyncMock, AsyncMock]:
    """Return (AsyncClient context manager, client) with ``client.post`` mocked."""
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=response)
    mock_cm = AsyncMock()
    mock_cm.__aenter__.return_value = mock_client
    mock_cm.__aexit__.return_value = None
    return mock_cm, mock_client


def _long_py(n: int = 1200) -> str:
    return "\n".join(f"line_{i} = {i}" for i in range(n))


def _explore_call(
    call_id: str = "c1",
    path: str = "/tmp/app.py",
    start: int = 1,
    end: int = 50,
    focus: str = "How does auth work?",
) -> dict[str, Any]:
    return {
        "type": "function_call",
        "name": EXPLORE_TOOL_NAME,
        "call_id": call_id,
        "arguments": json.dumps(
            {
                "path": path,
                "focus_question": focus,
                "start_line": start,
                "end_line": end,
            }
        ),
    }


def _output(call_id: str, text: str) -> dict[str, Any]:
    return {"type": "function_call_output", "call_id": call_id, "output": text}


@pytest.fixture(autouse=True)
def _clear_registry():
    clear_reducer_registry()
    yield
    clear_reducer_registry()


def test_inject_tool_and_instructions_no_dup():
    svc = ExploreToolService(reducer=FakeTextReducer())
    body: dict[str, Any] = {"tools": [], "instructions": "hello"}
    svc.prepare_request(body)
    assert any(t.get("name") == EXPLORE_TOOL_NAME for t in body["tools"])
    assert EXPLORE_TOOL_INSTRUCTIONS in body["instructions"]
    assert "(compressed N lines: <brief summary>)" in body["instructions"]
    assert "(filtered N lines)" not in body["instructions"]
    tools_len = len(body["tools"])
    instr = body["instructions"]
    svc.prepare_request(body)
    assert len(body["tools"]) == tools_len
    assert body["instructions"].count(EXPLORE_TOOL_INSTRUCTIONS) == 1
    assert body["instructions"] == instr


def test_omission_marker_emits_coact_shape():
    from headroom.proxy.explore_pruner.ast_protect import _omission_marker

    assert _omission_marker(4) == "(compressed 4 lines: omitted)"
    assert _omission_marker(19, "    ") == "    (compressed 19 lines: omitted)"


def test_rewrite_legacy_filtered_markers_to_coact_shape():
    from headroom.proxy.explore_pruner.ast_protect import rewrite_legacy_filtered_markers

    text = "(filtered 12 lines)\ndef foo():\n(filtered 1 lines)\n"
    out = rewrite_legacy_filtered_markers(text)
    assert out == (
        "(compressed 12 lines: omitted)\ndef foo():\n(compressed 1 lines: omitted)\n"
    )
    # Already CoACT-shaped text is left alone.
    coact = "(compressed 4 lines: imports)\nx = 1\n"
    assert rewrite_legacy_filtered_markers(coact) == coact


@pytest.mark.asyncio
async def test_swe_pruner_fallback_rewrites_filtered_markers():
    from headroom.proxy.explore_pruner.reducers.swe_pruner import SwePrunerReducer

    reducer = SwePrunerReducer(
        api_base="http://127.0.0.1:9",
        ast_protect_enabled=True,
        rebuild_fallback="pruned",
    )
    raw = ReduceResult(
        content="(filtered 12 lines)\ndef is_separable(transform):\n    return True\n",
        kept_frags=[13],
    )
    rebuilt = MagicMock()
    rebuilt.text = None
    rebuilt.skip_reason = "rebuild_syntax"
    rebuilt.leaf_count = 0
    rebuilt.kept_count = 0
    with patch(
        "headroom.proxy.explore_pruner.reducers.swe_pruner.rebuild_python_from_pruned",
        return_value=rebuilt,
    ):
        out = reducer._apply_ast_rebuild(
            ReduceInput(
                content="x = 0\n",
                query="Why?",
                config={"commands": ["sed -n '1,40p' /tmp/app.py"]},
            ),
            raw,
        )
    assert out.metadata["ast_rebuild"] is False
    assert out.metadata["ast_skip_reason"] == "rebuild_syntax"
    assert "(filtered" not in out.content
    assert out.content.startswith("(compressed 12 lines: omitted)")


def test_omission_parser_accepts_coact_and_legacy_filtered():
    from headroom.proxy.explore_pruner.ast_protect import _parse_virtual_kept_lines

    coact = _parse_virtual_kept_lines(
        "(compressed 2 lines: imports)\n"
        "def foo():\n"
        "    return 1\n"
        "(compressed 3 lines: helpers)\n"
    )
    assert coact == [(3, "def foo():"), (4, "    return 1")]

    legacy = _parse_virtual_kept_lines(
        "(filtered 1 lines)\n"
        "x = 1\n"
    )
    assert legacy == [(2, "x = 1")]


def test_append_explore_tool_idempotent():
    tools = append_explore_source_code_tool([])
    tools2 = append_explore_source_code_tool(tools)
    assert tools2 is tools or tools2 == tools
    assert sum(1 for t in tools2 if t.get("name") == EXPLORE_TOOL_NAME) == 1


def test_explore_to_sed_rewrite_and_store():
    svc = ExploreToolService(reducer=FakeTextReducer(), explore_max_lines=400)
    item = _explore_call(end=900)
    out = svc.rewrite_outbound_items([item], session_key="s1")
    rewritten = out[0]
    assert rewritten["name"] == "exec_command"
    args = json.loads(rewritten["arguments"])
    assert "sed -n '1,400p'" in args["cmd"]
    assert "/tmp/app.py" in args["cmd"]
    rec = svc.store.get("s1", "c1")
    assert rec is not None
    assert rec.focus_question == "How does auth work?"
    assert rec.commands
    assert rec.explore_path == "/tmp/app.py"
    assert rec.explore_start_line == 1
    assert rec.explore_end_line == 900


def test_missing_line_args_no_rewrite():
    item = {
        "type": "function_call",
        "name": EXPLORE_TOOL_NAME,
        "call_id": "c2",
        "arguments": json.dumps({"path": "/tmp/a.py", "focus_question": "Why?"}),
    }
    rewritten, focus, _ = rewrite_explore_call_to_exec(item)
    assert is_explore_source_code_call(rewritten)
    assert focus is None


def test_clamp_explore_window():
    assert clamp_explore_window(10, 500, max_lines=100) == (10, 109)
    assert clamp_explore_window(5, 3, max_lines=50) == (5, 5)


@pytest.mark.asyncio
async def test_inbound_hits_store_and_calls_reducer():
    reducer = FakeTextReducer("PRUNED_BODY")
    svc = ExploreToolService(reducer=reducer, min_chars_to_prune=100)
    svc.rewrite_outbound_items([_explore_call()], session_key="sess")
    body = {
        "input": [_output("c1", _long_py(200))],
    }
    changed = await svc.prune_inbound(body, session_key="sess")
    assert changed.changed is True
    assert reducer.calls == 1
    assert body["input"][0]["output"] == "PRUNED_BODY"
    assert "c1" in body[PRUNED_CALL_IDS_KEY]
    assert isinstance(body[PRUNED_CALL_IDS_KEY], list)


@pytest.mark.asyncio
async def test_service_uses_reducer_content_as_is():
    """Orchestrator writes result.content verbatim; no AST in service layer."""
    reducer = FakeTextReducer("TEXT_ONLY")
    svc = ExploreToolService(reducer=reducer, min_chars_to_prune=50)
    svc.rewrite_outbound_items([_explore_call()], session_key="s")
    body = {"input": [_output("c1", _long_py(80))]}
    await svc.prune_inbound(body, session_key="s")
    assert body["input"][0]["output"] == "TEXT_ONLY"


@pytest.mark.asyncio
async def test_kept_frags_reducer_not_ast_processed_by_service():
    """Non-swe reducers that return kept_frags are not AST-rebuilt by ExploreToolService."""
    reducer = FakeKeptFragsReducer()
    svc = ExploreToolService(reducer=reducer, min_chars_to_prune=10)
    svc.rewrite_outbound_items([_explore_call()], session_key="s")
    body = {"input": [_output("c1", "x = 0\nx = 1\nx = 2\n" * 20)]}
    await svc.prune_inbound(body, session_key="s")
    assert body["input"][0]["output"] == "(compressed 1 lines: omitted)\nx = 1\n"


@pytest.mark.asyncio
async def test_swe_pruner_ast_rebuild():
    from headroom.proxy.explore_pruner.reducers.swe_pruner import SwePrunerReducer

    reducer = SwePrunerReducer(
        api_base="http://127.0.0.1:9",
        ast_protect_enabled=True,
    )
    raw = ReduceResult(
        content="(compressed 1 lines: omitted)\nx = 1\n",
        kept_frags=[2],
    )
    rebuilt = MagicMock()
    rebuilt.text = "REBUILT\n"
    rebuilt.leaf_count = 1
    rebuilt.kept_count = 1
    rebuilt.skip_reason = None
    with patch(
        "headroom.proxy.explore_pruner.reducers.swe_pruner.rebuild_python_from_pruned",
        return_value=rebuilt,
    ) as mock_ast:
        out = reducer._apply_ast_rebuild(
            ReduceInput(
                content="x = 0\nx = 1\n",
                query="Why?",
                config={"commands": ["sed -n '1,40p' /tmp/app.py"]},
            ),
            raw,
        )
        mock_ast.assert_called_once()
    assert out.content == "REBUILT\n"


def test_swe_pruner_skips_ast_rebuild_for_non_python():
    from headroom.proxy.explore_pruner.reducers.swe_pruner import SwePrunerReducer

    reducer = SwePrunerReducer(
        api_base="http://127.0.0.1:9",
        ast_protect_enabled=True,
    )
    raw = ReduceResult(
        content="(filtered 2 lines)\nfn main() {}\n",
        kept_frags=[3],
    )
    with patch(
        "headroom.proxy.explore_pruner.reducers.swe_pruner.rebuild_python_from_pruned"
    ) as mock_ast:
        out = reducer._apply_ast_rebuild(
            ReduceInput(
                content="mod a;\nmod b;\nfn main() {}\n",
                query="Why?",
                config={"commands": ["sed -n '1,40p' /tmp/main.rs"]},
            ),
            raw,
        )
        mock_ast.assert_not_called()
    assert out.metadata.get("ast_rebuild") is False
    assert out.metadata.get("ast_skip_reason") == "non_python"
    assert "(filtered" not in out.content
    assert out.content.startswith("(compressed 2 lines: omitted)")


def test_swe_pruner_skips_ast_rebuild_when_disabled():
    from headroom.proxy.explore_pruner.reducers.swe_pruner import SwePrunerReducer

    reducer = SwePrunerReducer(
        api_base="http://127.0.0.1:9",
        ast_protect_enabled=False,
    )
    raw = ReduceResult(
        content="(filtered 1 lines)\nx = 1\n",
        kept_frags=[2],
    )
    with patch(
        "headroom.proxy.explore_pruner.reducers.swe_pruner.rebuild_python_from_pruned"
    ) as mock_ast:
        out = reducer._apply_ast_rebuild(
            ReduceInput(
                content="x = 0\nx = 1\n",
                query="Why?",
                config={"commands": ["sed -n '1,40p' /tmp/app.py"]},
            ),
            raw,
        )
        mock_ast.assert_not_called()
    assert out.metadata.get("ast_rebuild") is False
    assert out.metadata.get("ast_skip_reason") == "disabled"
    assert "(filtered" not in out.content
    assert out.content.startswith("(compressed 1 lines: omitted)")


@pytest.mark.asyncio
async def test_swe_pruner_reduce_applies_ast_after_http():
    from unittest.mock import AsyncMock

    from headroom.proxy.explore_pruner.reducers.swe_pruner import SwePrunerReducer

    reducer = SwePrunerReducer(
        api_base="http://127.0.0.1:9",
        ast_protect_enabled=True,
    )

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "pruned_code": "(compressed 1 lines: omitted)\nx = 1\n",
        "kept_frags": [2],
    }
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    mock_cm = AsyncMock()
    mock_cm.__aenter__.return_value = mock_client
    mock_cm.__aexit__.return_value = None

    rebuilt = MagicMock()
    rebuilt.text = "FROM_AST\n"
    rebuilt.leaf_count = 1
    rebuilt.kept_count = 1
    rebuilt.skip_reason = None
    with (
        patch(
            "headroom.proxy.explore_pruner.reducers.swe_pruner.httpx.AsyncClient",
            return_value=mock_cm,
        ),
        patch(
            "headroom.proxy.explore_pruner.reducers.swe_pruner.rebuild_python_from_pruned",
            return_value=rebuilt,
        ),
    ):
        out = await reducer.reduce(
            ReduceInput(
                content="x = 0\nx = 1\n",
                query="Why?",
                config={"commands": ["sed -n '1,40p' /tmp/app.py"]},
            )
        )
    assert out is not None
    assert out.content == "FROM_AST\n"


def test_parse_coact_response_code_type():
    from headroom.proxy.explore_pruner.types import parse_coact_response

    result = parse_coact_response(
        {
            "pruned_code": "def validate():\n    return x\n",
            "origin_token_cnt": 511,
            "left_token_cnt": 78,
            "model_input_token_cnt": 1368,
            "kept_frags": [20, 21],
            "compression_type": "code",
            "error_msg": None,
        }
    )
    assert result is not None
    assert result.content.startswith("def validate")
    assert result.kept_frags == [20, 21]
    assert result.metadata["backend"] == "coact"
    assert result.metadata["compression_type"] == "code"
    assert result.metadata["origin_token_cnt"] == 511
    assert result.metadata["left_token_cnt"] == 78
    assert result.metadata["model_input_token_cnt"] == 1368


def test_parse_coact_response_plain_and_unchanged():
    from headroom.proxy.explore_pruner.types import parse_coact_response

    plain = parse_coact_response(
        {
            "pruned_code": "3 tests passed; no failure traceback was present.",
            "compression_type": "plain",
        }
    )
    assert plain is not None
    assert plain.content.startswith("3 tests passed")
    assert plain.metadata["compression_type"] == "plain"

    unchanged = parse_coact_response(
        {
            "pruned_code": "def foo(): pass\n",
            "compression_type": "unchanged",
        }
    )
    assert unchanged is not None
    assert unchanged.content == "def foo(): pass\n"
    assert unchanged.metadata["compression_type"] == "unchanged"


def test_parse_coact_response_rejects_invalid_and_empty():
    from headroom.proxy.explore_pruner.types import parse_coact_response

    assert (
        parse_coact_response(
            {"pruned_code": "same as original", "compression_type": "invalid"}
        )
        is None
    )
    assert parse_coact_response({"pruned_code": "", "compression_type": "code"}) is None
    assert parse_coact_response({"compression_type": "code"}) is None


@pytest.mark.asyncio
async def test_coact_payload_maps_query_code_goal_and_tool_call():
    from headroom.proxy.explore_pruner.reducers.coact import CoactReducer

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "pruned_code": "kept",
        "compression_type": "code",
        "error_msg": None,
        "kept_frags": [1],
    }
    mock_cm, mock_client = _mock_async_httpx(mock_response)
    reducer = CoactReducer(api_base="http://127.0.0.1:8002")
    with patch(
        "headroom.proxy.explore_pruner.reducers.coact.httpx.AsyncClient",
        return_value=mock_cm,
    ):
        await reducer.reduce(
            ReduceInput(
                content="def foo(): pass",
                query="Where is foo?",
                config={
                    "commands": ["sed -n '1,80p' src/utils.py"],
                    "goal": "Fix empty string validation",
                },
            )
        )
    mock_client.post.assert_called_once()
    payload = mock_client.post.call_args.kwargs["json"]
    assert payload["query"] == "Where is foo?"
    assert payload["code"] == "def foo(): pass"
    assert payload["goal"] == "Fix empty string validation"
    assert payload["tool_call"] == "sed -n '1,80p' src/utils.py"
    assert "threshold" not in payload


@pytest.mark.asyncio
async def test_coact_reduce_returns_pruned_without_ast():
    from headroom.proxy.explore_pruner.reducers.coact import CoactReducer

    pruned = "(compressed 19 lines: imports)\ndef validate():\n    return x\n"
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "pruned_code": pruned,
        "kept_frags": [20, 21, 22],
        "compression_type": "code",
        "error_msg": None,
        "origin_token_cnt": 511,
        "left_token_cnt": 78,
    }
    mock_cm, _client = _mock_async_httpx(mock_response)
    reducer = CoactReducer(api_base="http://127.0.0.1:8002")
    with (
        patch(
            "headroom.proxy.explore_pruner.reducers.coact.httpx.AsyncClient",
            return_value=mock_cm,
        ),
        patch(
            "headroom.proxy.explore_pruner.ast_protect.rebuild_python_from_pruned",
        ) as mock_ast,
    ):
        out = await reducer.reduce(
            ReduceInput(content="x = 0\nx = 1\n", query="Why?", config={"commands": []})
        )
    mock_ast.assert_not_called()
    assert out is not None
    assert out.content == pruned
    assert out.kept_frags == [20, 21, 22]
    assert out.metadata["compression_type"] == "code"


@pytest.mark.asyncio
async def test_coact_reduce_fail_open_on_error_msg_invalid_and_http():
    from headroom.proxy.explore_pruner.reducers.coact import CoactReducer

    reducer = CoactReducer(api_base="http://127.0.0.1:8002")
    inp = ReduceInput(content="def foo(): pass", query="Where is foo?")

    error_resp = MagicMock()
    error_resp.status_code = 200
    error_resp.json.return_value = {
        "pruned_code": "def foo(): pass",
        "error_msg": "backend failed",
        "compression_type": "invalid",
    }
    mock_cm, _ = _mock_async_httpx(error_resp)
    with patch(
        "headroom.proxy.explore_pruner.reducers.coact.httpx.AsyncClient",
        return_value=mock_cm,
    ):
        assert await reducer.reduce(inp) is None

    invalid_resp = MagicMock()
    invalid_resp.status_code = 200
    invalid_resp.json.return_value = {
        "pruned_code": "def foo(): pass",
        "error_msg": None,
        "compression_type": "invalid",
    }
    mock_cm, _ = _mock_async_httpx(invalid_resp)
    with patch(
        "headroom.proxy.explore_pruner.reducers.coact.httpx.AsyncClient",
        return_value=mock_cm,
    ):
        assert await reducer.reduce(inp) is None

    http_resp = MagicMock()
    http_resp.status_code = 500
    http_resp.content = b"boom"
    mock_cm, _ = _mock_async_httpx(http_resp)
    with patch(
        "headroom.proxy.explore_pruner.reducers.coact.httpx.AsyncClient",
        return_value=mock_cm,
    ):
        assert await reducer.reduce(inp) is None


@pytest.mark.asyncio
async def test_fail_open_paths():
    # truncation
    svc = ExploreToolService(reducer=FakeTextReducer(), min_chars_to_prune=10)
    svc.rewrite_outbound_items([_explore_call()], session_key="s")
    truncated = "a" * 100 + "\nWarning: truncated output\n"
    body = {"input": [_output("c1", truncated)]}
    assert (await svc.prune_inbound(body, session_key="s")).changed is False

    # unsupported file
    svc2 = ExploreToolService(reducer=FakeTextReducer(), min_chars_to_prune=10)
    svc2.rewrite_outbound_items(
        [_explore_call(path="/tmp/readme.md")], session_key="s2"
    )
    body2 = {"input": [_output("c1", _long_py(80))]}
    assert (await svc2.prune_inbound(body2, session_key="s2")).changed is False

    # short text
    svc3 = ExploreToolService(reducer=FakeTextReducer(), min_chars_to_prune=5000)
    svc3.rewrite_outbound_items([_explore_call()], session_key="s3")
    body3 = {"input": [_output("c1", "short")]}
    assert (await svc3.prune_inbound(body3, session_key="s3")).changed is False

    # reducer None
    svc4 = ExploreToolService(
        reducer=FakeNoneReducer(), min_chars_to_prune=10, fail_open=True
    )
    svc4.rewrite_outbound_items([_explore_call()], session_key="s4")
    body4 = {"input": [_output("c1", _long_py(80))]}
    assert (await svc4.prune_inbound(body4, session_key="s4")).changed is False


@pytest.mark.parametrize(
    "path",
    [
        "/tmp/app.py",
        "/tmp/app.js",
        "/tmp/app.ts",
        "/tmp/app.tsx",
        "/tmp/app.go",
        "/tmp/app.rs",
        "/tmp/App.java",
        "/tmp/app.c",
        "/tmp/app.h",
        "/tmp/app.cc",
        "/tmp/app.cpp",
        "/tmp/app.cxx",
        "/tmp/app.hpp",
        "/tmp/app.hxx",
        "/tmp/app.hh",
        "/tmp/App.JS",
        "/tmp/mod.CPP",
    ],
)
def test_supported_source_file_scope(path: str):
    cmd = f"sed -n '1,80p' {path}"
    assert has_supported_source_file_scope([cmd]) is True


@pytest.mark.parametrize(
    "path",
    ["/tmp/readme.md", "/tmp/data.json", "/tmp/notes.txt", "/tmp/app.pyi"],
)
def test_unsupported_source_file_scope(path: str):
    cmd = f"sed -n '1,80p' {path}"
    assert has_supported_source_file_scope([cmd]) is not True
    assert has_python_file_scope([cmd]) is not True


def test_python_file_scope_only_py():
    assert has_python_file_scope(["sed -n '1,80p' /tmp/app.py"]) is True
    assert has_python_file_scope(["sed -n '1,80p' /tmp/app.js"]) is False
    assert has_python_file_scope(["sed"]) is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path",
    [
        "/tmp/app.js",
        "/tmp/app.ts",
        "/tmp/app.tsx",
        "/tmp/app.go",
        "/tmp/app.rs",
        "/tmp/App.java",
        "/tmp/app.c",
        "/tmp/app.cpp",
    ],
)
async def test_inbound_prunes_supported_non_python_languages(path: str):
    reducer = FakeTextReducer("PRUNED_BODY")
    svc = ExploreToolService(reducer=reducer, min_chars_to_prune=10)
    svc.rewrite_outbound_items([_explore_call(path=path)], session_key="s")
    body = {"input": [_output("c1", _long_py(80))]}
    result = await svc.prune_inbound(body, session_key="s")
    assert result.changed is True
    assert reducer.calls == 1
    assert body["input"][0]["output"] == "PRUNED_BODY"


@pytest.mark.asyncio
async def test_pruned_call_ids_skip_content_router_candidates():
    """Simulate ContentRouter skip using the same key the service stamps."""
    svc = ExploreToolService(reducer=FakeTextReducer("X"), min_chars_to_prune=10)
    svc.rewrite_outbound_items([_explore_call()], session_key="s")
    body = {"input": [_output("c1", _long_py(80))]}
    await svc.prune_inbound(body, session_key="s")
    pruned = body[PRUNED_CALL_IDS_KEY]
    assert "c1" in pruned
    # Candidate extraction would skip these ids (wired in openai handler).
    items = body["input"]
    candidates = [
        i
        for i, item in enumerate(items)
        if isinstance(item, dict)
        and item.get("type") == "function_call_output"
        and item.get("call_id") not in pruned
    ]
    assert candidates == []


@pytest.mark.asyncio
async def test_cached_pruned_output_skips_reducer():
    reducer = FakeTextReducer("ONCE")
    svc = ExploreToolService(reducer=reducer, min_chars_to_prune=10)
    svc.rewrite_outbound_items([_explore_call()], session_key="s")
    body = {"input": [_output("c1", _long_py(80))]}
    await svc.prune_inbound(body, session_key="s")
    assert reducer.calls == 1
    body2 = {"input": [_output("c1", _long_py(80))]}
    await svc.prune_inbound(body2, session_key="s")
    assert reducer.calls == 1
    assert body2["input"][0]["output"] == "ONCE"


@pytest.mark.asyncio
async def test_inbound_restores_explore_tool_name_for_upstream():
    """Client history stores exec_command; upstream must see explore_source_code."""
    reducer = FakeTextReducer("PRUNED")
    svc = ExploreToolService(reducer=reducer, min_chars_to_prune=10)
    rewritten = svc.rewrite_outbound_items([_explore_call()], session_key="s")
    assert rewritten[0]["name"] == "exec_command"
    body = {
        "input": [
            rewritten[0],
            _output("c1", _long_py(80)),
        ]
    }
    changed = await svc.prune_inbound(body, session_key="s")
    assert changed.changed is True
    call = body["input"][0]
    assert call["name"] == EXPLORE_TOOL_NAME
    assert call["call_id"] == "c1"
    args = json.loads(call["arguments"])
    assert args["path"] == "/tmp/app.py"
    assert args["focus_question"] == "How does auth work?"
    assert args["start_line"] == 1
    assert args["end_line"] == 50
    assert "cmd" not in args
    assert body["input"][1]["output"] == "PRUNED"


@pytest.mark.asyncio
async def test_inbound_does_not_restore_native_sed():
    svc = ExploreToolService(reducer=FakeTextReducer(), min_chars_to_prune=10)
    native = {
        "type": "function_call",
        "name": "exec_command",
        "call_id": "native-sed",
        "arguments": json.dumps({"cmd": "sed -n '1,80p' /tmp/app.py"}),
    }
    body = {"input": [native, _output("native-sed", _long_py(80))]}
    await svc.prune_inbound(body, session_key="s")
    assert body["input"][0]["name"] == "exec_command"
    assert json.loads(body["input"][0]["arguments"])["cmd"].startswith("sed -n")


@pytest.mark.asyncio
async def test_later_turn_restores_explore_name_from_cache():
    reducer = FakeTextReducer("ONCE")
    svc = ExploreToolService(reducer=reducer, min_chars_to_prune=10)
    rewritten = svc.rewrite_outbound_items([_explore_call()], session_key="s")
    await svc.prune_inbound(
        {"input": [rewritten[0], _output("c1", _long_py(80))]},
        session_key="s",
    )
    replay = {
        "input": [
            dict(rewritten[0]),
            _output("c1", _long_py(80)),
        ]
    }
    await svc.prune_inbound(replay, session_key="s")
    assert reducer.calls == 1
    assert replay["input"][0]["name"] == EXPLORE_TOOL_NAME
    assert replay["input"][1]["output"] == "ONCE"


@pytest.mark.asyncio
async def test_inbound_restores_explore_name_even_when_output_not_pruned():
    svc = ExploreToolService(reducer=FakeTextReducer(), min_chars_to_prune=5000)
    rewritten = svc.rewrite_outbound_items([_explore_call()], session_key="s")
    body = {"input": [rewritten[0], _output("c1", "short")]}
    changed = await svc.prune_inbound(body, session_key="s")
    assert changed.changed is True
    assert body["input"][0]["name"] == EXPLORE_TOOL_NAME
    assert body["input"][1]["output"] == "short"


def test_tool_output_truncation_markers():
    assert tool_output_looks_truncated("tokens truncated")
    assert not tool_output_looks_truncated("normal output")


def test_proxy_config_explore_defaults():
    cfg = ProxyConfig()
    assert isinstance(cfg.explore_pruner, ExplorePrunerConfig)
    assert cfg.explore_pruner.enabled is False
    assert cfg.explore_pruner.reducer == "swe_pruner"


def test_proxy_config_dict_round_trip():
    cfg = ProxyConfig(explore_pruner={"enabled": True, "reducer": "fake_text"})  # type: ignore[arg-type]
    assert isinstance(cfg.explore_pruner, ExplorePrunerConfig)
    assert cfg.explore_pruner.enabled is True
    assert cfg.explore_pruner.reducer == "fake_text"


def test_register_and_get_reducer():
    r = FakeTextReducer()
    register_reducer(r)
    assert get_reducer("fake_text") is r


def test_store_ttl_preserves_pruned_on_upsert():
    store = ExplorePrunerStore(ttl_seconds=60)
    now = time.time()
    store.upsert(
        "s",
        "c",
        ExplorePrunerRecord(
            commands=["sed"],
            focus_question="q?",
            created_at=now,
            pruned_output="cached",
            explore_path="/tmp/app.py",
            explore_start_line=1,
            explore_end_line=50,
        ),
    )
    store.upsert(
        "s",
        "c",
        ExplorePrunerRecord(
            commands=["sed"],
            focus_question="q?",
            created_at=now + 1,
            pruned_output=None,
        ),
    )
    rec = store.get("s", "c")
    assert rec.pruned_output == "cached"
    assert rec.explore_path == "/tmp/app.py"
    assert rec.explore_start_line == 1
    assert rec.explore_end_line == 50


def test_factory_disabled_returns_none():
    from headroom.proxy.explore_pruner.factory import build_explore_tool_service

    assert build_explore_tool_service(ExplorePrunerConfig(enabled=False)) is None


def test_factory_ast_protect_env_switch(monkeypatch):
    from headroom.proxy.explore_pruner.factory import (
        build_explore_tool_service,
        explore_pruner_config_from_env,
    )
    from headroom.proxy.explore_pruner.reducers.swe_pruner import SwePrunerReducer

    monkeypatch.setenv("HEADROOM_EXPLORE_AST_PROTECT", "0")
    cfg = explore_pruner_config_from_env(ExplorePrunerConfig(enabled=True))
    assert cfg.ast_protect_enabled is False

    monkeypatch.setattr(
        "headroom.proxy.explore_pruner.factory.init_tree_sitter",
        lambda: True,
    )
    svc = build_explore_tool_service(
        ExplorePrunerConfig(
            enabled=True,
            api_base="http://127.0.0.1:9",
            ast_protect_enabled=False,
        )
    )
    assert svc is not None
    reducer = get_reducer("swe_pruner")
    assert isinstance(reducer, SwePrunerReducer)
    assert reducer._ast_protect_enabled is False


def test_factory_builds_swe_pruner(monkeypatch):
    from headroom.proxy.explore_pruner.factory import build_explore_tool_service
    from headroom.proxy.explore_pruner.reducers.swe_pruner import SwePrunerReducer

    monkeypatch.setattr(
        "headroom.proxy.explore_pruner.factory.init_tree_sitter",
        lambda: False,
    )
    svc = build_explore_tool_service(
        ExplorePrunerConfig(
            enabled=True,
            api_base="http://127.0.0.1:9",
            ast_protect_enabled=True,
        )
    )
    assert svc is not None
    assert svc.reducer_name == "swe_pruner"
    reducer = get_reducer("swe_pruner")
    assert isinstance(reducer, SwePrunerReducer)
    assert reducer._ast_protect_enabled is False


def test_factory_builds_coact_without_tree_sitter(monkeypatch):
    from headroom.proxy.explore_pruner.factory import build_explore_tool_service
    from headroom.proxy.explore_pruner.reducers.coact import CoactReducer

    def _tree_sitter_must_not_run() -> bool:
        raise AssertionError("tree-sitter should not be used for coact")

    monkeypatch.setattr(
        "headroom.proxy.explore_pruner.factory.init_tree_sitter",
        _tree_sitter_must_not_run,
    )
    svc = build_explore_tool_service(
        ExplorePrunerConfig(
            enabled=True,
            reducer="coact",
            api_base="http://127.0.0.1:9",
            timeout_seconds=90.0,
            ast_protect_enabled=True,
        )
    )
    assert svc is not None
    assert svc.reducer_name == "coact"
    reducer = get_reducer("coact")
    assert isinstance(reducer, CoactReducer)
    assert reducer._url == "http://127.0.0.1:9/prune"
    assert reducer._timeout == 90.0


def test_factory_coact_defaults_port_8002_and_120s_timeout():
    from headroom.proxy.explore_pruner.factory import build_explore_tool_service
    from headroom.proxy.explore_pruner.reducers.coact import CoactReducer

    svc = build_explore_tool_service(ExplorePrunerConfig(enabled=True, reducer="coact"))
    assert svc is not None
    reducer = get_reducer("coact")
    assert isinstance(reducer, CoactReducer)
    assert reducer._url == "http://127.0.0.1:8002/prune"
    assert reducer._timeout == 120.0


def test_stream_rewriter_renames_added_and_rewrites_done():
    """HTTP/WS stream events must never expose explore_source_code to the client."""
    from headroom.proxy.explore_pruner.stream_rewrite import ExploreStreamRewriter

    svc = ExploreToolService(reducer=FakeTextReducer(), explore_max_lines=400)
    rewriter = ExploreStreamRewriter(svc, session_key="sess")
    item_id = "item-explore-1"
    added = rewriter.rewrite_event(
        {
            "type": "response.output_item.added",
            "output_index": 1,
            "item": {
                "type": "function_call",
                "id": item_id,
                "status": "in_progress",
                "name": EXPLORE_TOOL_NAME,
                "call_id": "c1",
                "arguments": "",
            },
        }
    )
    assert added is not None
    assert added["item"]["name"] == "exec_command"

    # Explore-shaped arg deltas must be dropped after added rename.
    assert (
        rewriter.rewrite_event(
            {
                "type": "response.function_call_arguments.delta",
                "output_index": 1,
                "item_id": item_id,
                "delta": '{"path"',
            }
        )
        is None
    )

    args_done = rewriter.rewrite_event(
        {
            "type": "response.function_call_arguments.done",
            "output_index": 1,
            "item_id": item_id,
            "arguments": json.dumps(
                {
                    "path": "/tmp/app.py",
                    "focus_question": "How does auth work?",
                    "start_line": 1,
                    "end_line": 50,
                }
            ),
        }
    )
    assert args_done is not None
    done_args = json.loads(args_done["arguments"])
    assert "sed -n" in done_args["cmd"]
    assert "focus_question" not in done_args

    done = rewriter.rewrite_event(
        {
            "type": "response.output_item.done",
            "output_index": 1,
            "item": {
                "type": "function_call",
                "id": item_id,
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
            },
        }
    )
    assert done is not None
    assert done["item"]["name"] == "exec_command"
    assert "sed -n" in json.loads(done["item"]["arguments"])["cmd"]
    assert svc.store.get("sess", "c1") is not None


def test_stream_rewriter_rewrites_completed_output():
    from headroom.proxy.explore_pruner.stream_rewrite import ExploreStreamRewriter

    svc = ExploreToolService(reducer=FakeTextReducer())
    rewriter = ExploreStreamRewriter(svc, session_key="sess")
    completed = rewriter.rewrite_event(
        {
            "type": "response.completed",
            "response": {
                "id": "resp_1",
                "output": [_explore_call()],
            },
        }
    )
    assert completed is not None
    out = completed["response"]["output"][0]
    assert out["name"] == "exec_command"
    assert "sed -n" in json.loads(out["arguments"])["cmd"]


def test_rewrite_sse_event_bytes_drops_explore_deltas():
    from headroom.proxy.explore_pruner.stream_rewrite import (
        ExploreStreamRewriter,
        rewrite_sse_event_bytes,
    )

    svc = ExploreToolService(reducer=FakeTextReducer())
    rewriter = ExploreStreamRewriter(svc, session_key="sess")
    added = (
        b"event: response.output_item.added\n"
        b'data: {"type":"response.output_item.added","item":'
        b'{"type":"function_call","id":"i1","name":"explore_source_code",'
        b'"call_id":"c1","arguments":""}}\n\n'
    )
    added_out = rewrite_sse_event_bytes(added, rewriter)
    assert added_out is not None
    assert b"explore_source_code" not in added_out
    assert b"exec_command" in added_out

    delta = (
        b"event: response.function_call_arguments.delta\n"
        b'data: {"type":"response.function_call_arguments.delta",'
        b'"item_id":"i1","delta":"{\\"path\\""}\n\n'
    )
    assert rewrite_sse_event_bytes(delta, rewriter) is None


@pytest.mark.asyncio
async def test_http_sse_stream_rewrites_explore_before_client():
    """Codex HTTP streaming must not forward explore_source_code to the client."""
    from unittest.mock import AsyncMock, MagicMock

    import httpx

    from headroom.proxy.server import HeadroomProxy

    explore_args = json.dumps(
        {
            "path": "/tmp/app.py",
            "focus_question": "How does auth work?",
            "start_line": 10,
            "end_line": 40,
        }
    )
    sse = (
        b"event: response.output_item.added\n"
        b'data: {"type":"response.output_item.added","output_index":1,'
        b'"item":{"type":"function_call","id":"i1","status":"in_progress",'
        b'"name":"explore_source_code","call_id":"c1","arguments":""}}\n\n'
        b"event: response.output_item.done\n"
        b'data: {"type":"response.output_item.done","output_index":1,'
        b'"item":{"type":"function_call","id":"i1","name":"explore_source_code",'
        b'"call_id":"c1","arguments":'
        + json.dumps(explore_args).encode()
        + b"}}\n\n"
        b"event: response.completed\n"
        b'data: {"type":"response.completed","response":{"id":"resp_1","output":['
        b'{"type":"function_call","id":"i1","name":"explore_source_code",'
        b'"call_id":"c1","arguments":'
        + json.dumps(explore_args).encode()
        + b"}]}}\n\n"
    )

    proxy = object.__new__(HeadroomProxy)
    proxy.http_client = MagicMock(spec=httpx.AsyncClient)
    proxy._config = MagicMock()
    proxy._config.memory_enabled = False
    proxy._config.ccr_inject_tool = False
    proxy._config.retry_enabled = False
    proxy._config.retry_max_attempts = 1
    proxy._config.retry_base_delay_ms = 0
    proxy._config.retry_max_delay_ms = 0
    proxy.config = proxy._config
    proxy.memory_handler = None
    proxy.memory_manager = None
    proxy._parse_sse_usage_from_buffer = MagicMock(return_value=None)
    proxy._finalize_stream_response = AsyncMock(return_value=None)
    proxy.explore_tool_service = ExploreToolService(
        reducer=FakeTextReducer(), explore_max_lines=400
    )

    mock_response = AsyncMock()
    mock_response.headers = httpx.Headers({"content-type": "text/event-stream"})
    mock_response.status_code = 200

    async def aiter_bytes():
        # Split mid-event to prove we wait for SSE boundaries.
        yield sse[:40]
        yield sse[40:]

    mock_response.aiter_bytes = aiter_bytes
    mock_response.aclose = AsyncMock()
    proxy.http_client.build_request = MagicMock(return_value=MagicMock())
    proxy.http_client.send = AsyncMock(return_value=mock_response)

    result = await proxy._stream_response(
        url="https://api.openai.com/v1/responses",
        headers={"authorization": "Bearer test", "session-id": "sess-1"},
        body={"model": "gpt-test", "stream": True, "input": []},
        provider="openai",
        model="gpt-test",
        request_id="req-explore-sse",
        original_tokens=10,
        optimized_tokens=10,
        tokens_saved=0,
        transforms_applied=[],
        tags={},
        optimization_latency=0.0,
        session_key="sess-1",
    )
    body = b"".join([chunk async for chunk in result.body_iterator])
    assert b"explore_source_code" not in body
    assert b"exec_command" in body
    assert b"sed -n" in body
    rec = proxy.explore_tool_service.store.get("sess-1", "c1")
    assert rec is not None
    assert rec.focus_question == "How does auth work?"


@pytest.mark.asyncio
async def test_explore_prepare_and_prune_returns_savings_report():
    from headroom.proxy.handlers.openai import _explore_prepare_and_prune

    original = _long_py(80)
    pruned = "SHORT"
    svc = ExploreToolService(reducer=FakeTextReducer(pruned), min_chars_to_prune=10)
    svc.rewrite_outbound_items([_explore_call()], session_key="s")
    body: dict[str, Any] = {"input": [_output("c1", original)], "tools": []}
    report = await _explore_prepare_and_prune(
        svc,
        body,
        session_key="s",
        request_id="r1",
        count_text=_count_chars,
    )
    assert report.changed is True
    assert report.tokens_saved == len(original) - len(pruned)
    assert EXPLORE_TOOL_NAME in [t.get("name") for t in body["tools"]]


def _count_chars(text: str) -> int:
    return len(text)


@pytest.mark.asyncio
async def test_prune_inbound_reports_token_savings_with_same_counter():
    from headroom.proxy.explore_pruner.types import PruneSavings

    original = _long_py(200)
    pruned = "PRUNED_BODY"
    reducer = FakeTextReducer(pruned)
    svc = ExploreToolService(reducer=reducer, min_chars_to_prune=100)
    svc.rewrite_outbound_items([_explore_call()], session_key="sess")
    body = {"input": [_output("c1", original)]}

    report = await svc.prune_inbound(body, session_key="sess", count_text=_count_chars)

    assert isinstance(report, PruneSavings)
    assert report.changed is True
    assert report.items == 1
    assert report.tokens_before == len(original)
    assert report.tokens_after == len(pruned)
    assert report.tokens_saved == len(original) - len(pruned)
    assert report.tokens_saved > 0


@pytest.mark.asyncio
async def test_prune_inbound_cache_hit_still_counts_savings():
    original = _long_py(80)
    pruned = "ONCE"
    reducer = FakeTextReducer(pruned)
    svc = ExploreToolService(reducer=reducer, min_chars_to_prune=10)
    svc.rewrite_outbound_items([_explore_call()], session_key="s")
    await svc.prune_inbound(
        {"input": [_output("c1", original)]},
        session_key="s",
        count_text=_count_chars,
    )
    report = await svc.prune_inbound(
        {"input": [_output("c1", original)]},
        session_key="s",
        count_text=_count_chars,
    )
    assert reducer.calls == 1
    assert report.changed is True
    assert report.tokens_saved == len(original) - len(pruned)
    assert report.items == 1


@pytest.mark.asyncio
async def test_prune_inbound_skip_paths_report_zero_savings():
    svc = ExploreToolService(reducer=FakeTextReducer(), min_chars_to_prune=10)
    svc.rewrite_outbound_items([_explore_call()], session_key="s")
    truncated = "a" * 100 + "\nWarning: truncated output\n"
    report = await svc.prune_inbound(
        {"input": [_output("c1", truncated)]},
        session_key="s",
        count_text=_count_chars,
    )
    assert report.changed is False
    assert report.tokens_saved == 0
    assert report.items == 0


@pytest.mark.asyncio
async def test_prune_inbound_inflation_clamps_saved_to_zero():
    original = _long_py(80)
    inflated = original + "\n" + original
    reducer = FakeTextReducer(inflated)
    svc = ExploreToolService(reducer=reducer, min_chars_to_prune=10)
    svc.rewrite_outbound_items([_explore_call()], session_key="s")
    report = await svc.prune_inbound(
        {"input": [_output("c1", original)]},
        session_key="s",
        count_text=_count_chars,
    )
    assert report.changed is True
    assert report.tokens_before == len(original)
    assert report.tokens_after == len(inflated)
    assert report.tokens_saved == 0


def test_apply_explore_prune_savings_folds_into_headline_and_attribution():
    from headroom.proxy.explore_pruner.savings import apply_explore_prune_savings
    from headroom.proxy.explore_pruner.types import PruneSavings
    from headroom.proxy.prometheus_metrics import PrometheusMetrics
    from headroom.proxy.savings_attribution import from_tags
    from headroom.proxy.tool_schema_savings_policy import headline_tokens_saved

    report = PruneSavings(
        tokens_before=100,
        tokens_after=40,
        tokens_saved=60,
        items=1,
        changed=True,
    )
    metrics = PrometheusMetrics()
    tags: dict[str, object] = {}
    transforms: list[str] = ["router:json"]
    tokens_saved, attempted, original = apply_explore_prune_savings(
        report,
        tokens_saved=10,
        attempted_input_tokens=20,
        original_tokens=30,
        transforms=transforms,
        tags=tags,
        metrics=metrics,
    )
    assert tokens_saved == 70
    assert attempted == 120
    assert original == 130
    assert "explore_pruner" in transforms
    assert metrics.tokens_saved_by_strategy["explore_pruner"] == 60
    attributed = from_tags(tags)
    assert attributed[0]["source"] == "explore_pruner"
    assert attributed[0]["tokens"] == 60
    assert headline_tokens_saved(tokens_saved, tags) == tokens_saved


def test_apply_explore_prune_savings_noops_when_nothing_pruned():
    from headroom.proxy.explore_pruner.savings import apply_explore_prune_savings
    from headroom.proxy.explore_pruner.types import PruneSavings
    from headroom.proxy.prometheus_metrics import PrometheusMetrics

    metrics = PrometheusMetrics()
    tags: dict[str, object] = {}
    transforms: list[str] = []
    tokens_saved, attempted, original = apply_explore_prune_savings(
        PruneSavings(),
        tokens_saved=5,
        attempted_input_tokens=8,
        original_tokens=9,
        transforms=transforms,
        tags=tags,
        metrics=metrics,
    )
    assert (tokens_saved, attempted, original) == (5, 8, 9)
    assert transforms == []
    assert metrics.tokens_saved_by_strategy == {}


def test_ws_prune_attribution_accumulates_then_clears_without_mutating_session_tags():
    """WS sessions reuse one tag dict; prune ledgers must not land on it."""
    from headroom.proxy.explore_pruner.savings import (
        apply_explore_prune_savings,
        attach_explore_prune_tags,
        merge_explore_prune_attribution,
    )
    from headroom.proxy.explore_pruner.types import PruneSavings
    from headroom.proxy.savings_attribution import SAVINGS_ATTRIBUTION_TAG, from_tags

    session_tags = {"client": "codex"}
    pending: dict[str, object] = {}

    turn1: dict[str, object] = {}
    apply_explore_prune_savings(
        PruneSavings(tokens_before=10, tokens_after=4, tokens_saved=6, changed=True),
        tokens_saved=0,
        attempted_input_tokens=0,
        tags=turn1,
    )
    merge_explore_prune_attribution(pending, turn1)

    turn2: dict[str, object] = {}
    apply_explore_prune_savings(
        PruneSavings(tokens_before=20, tokens_after=13, tokens_saved=7, changed=True),
        tokens_saved=0,
        attempted_input_tokens=0,
        tags=turn2,
    )
    merge_explore_prune_attribution(pending, turn2)

    outcome_tags = attach_explore_prune_tags(session_tags, pending)
    assert SAVINGS_ATTRIBUTION_TAG not in session_tags
    attributed = [row for row in from_tags(outcome_tags) if row.get("source") == "explore_pruner"]
    assert [row["tokens"] for row in attributed] == [6, 7]

    pending.clear()
    later = attach_explore_prune_tags(session_tags, pending)
    assert from_tags(later) == []

