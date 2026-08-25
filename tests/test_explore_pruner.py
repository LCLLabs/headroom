"""Unit tests for explore_tool + pluggable context reducers."""

from __future__ import annotations

import json
import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from headroom.proxy.explore_pruner.focus import (
    EXPLORE_TOOL_INSTRUCTIONS,
    EXPLORE_TOOL_NAME,
    PRUNED_CALL_IDS_KEY,
    append_explore_source_code_tool,
    clamp_explore_window,
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
            content="(filtered 1 lines)\nx = 1\n",
            kept_frags=[2],
        )


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
    tools_len = len(body["tools"])
    instr = body["instructions"]
    svc.prepare_request(body)
    assert len(body["tools"]) == tools_len
    assert body["instructions"].count(EXPLORE_TOOL_INSTRUCTIONS) == 1
    assert body["instructions"] == instr


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
    assert changed is True
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
    assert body["input"][0]["output"] == "(filtered 1 lines)\nx = 1\n"


@pytest.mark.asyncio
async def test_swe_pruner_ast_rebuild():
    from headroom.proxy.explore_pruner.reducers.swe_pruner import SwePrunerReducer

    reducer = SwePrunerReducer(
        api_base="http://127.0.0.1:9",
        ast_protect_enabled=True,
    )
    raw = ReduceResult(
        content="(filtered 1 lines)\nx = 1\n",
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
            ReduceInput(content="x = 0\nx = 1\n", query="Why?", config={"commands": ["sed"]}),
            raw,
        )
        mock_ast.assert_called_once()
    assert out.content == "REBUILT\n"


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
        "pruned_code": "(filtered 1 lines)\nx = 1\n",
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
            ReduceInput(content="x = 0\nx = 1\n", query="Why?", config={"commands": []})
        )
    assert out is not None
    assert out.content == "FROM_AST\n"


@pytest.mark.asyncio
async def test_fail_open_paths():
    # truncation
    svc = ExploreToolService(reducer=FakeTextReducer(), min_chars_to_prune=10)
    svc.rewrite_outbound_items([_explore_call()], session_key="s")
    truncated = "a" * 100 + "\nWarning: truncated output\n"
    body = {"input": [_output("c1", truncated)]}
    assert await svc.prune_inbound(body, session_key="s") is False

    # non-py scope
    svc2 = ExploreToolService(reducer=FakeTextReducer(), min_chars_to_prune=10)
    svc2.rewrite_outbound_items(
        [_explore_call(path="/tmp/readme.md")], session_key="s2"
    )
    body2 = {"input": [_output("c1", _long_py(80))]}
    assert await svc2.prune_inbound(body2, session_key="s2") is False

    # short text
    svc3 = ExploreToolService(reducer=FakeTextReducer(), min_chars_to_prune=5000)
    svc3.rewrite_outbound_items([_explore_call()], session_key="s3")
    body3 = {"input": [_output("c1", "short")]}
    assert await svc3.prune_inbound(body3, session_key="s3") is False

    # reducer None
    svc4 = ExploreToolService(
        reducer=FakeNoneReducer(), min_chars_to_prune=10, fail_open=True
    )
    svc4.rewrite_outbound_items([_explore_call()], session_key="s4")
    body4 = {"input": [_output("c1", _long_py(80))]}
    assert await svc4.prune_inbound(body4, session_key="s4") is False


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
    assert store.get("s", "c").pruned_output == "cached"


def test_factory_disabled_returns_none():
    from headroom.proxy.explore_pruner.factory import build_explore_tool_service

    assert build_explore_tool_service(ExplorePrunerConfig(enabled=False)) is None


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


