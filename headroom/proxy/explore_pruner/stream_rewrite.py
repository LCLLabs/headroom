"""Rewrite explore_source_code SSE/WS events before they reach the client.

Codex (and other harnesses) only know exec_command. Headroom injects
explore_source_code for the model, then must rewrite outbound calls to
``exec_command(sed -n ...)`` so the client can execute them. Non-stream
JSON already did this; HTTP SSE / WS must apply the same rewrite at
event boundaries or the harness returns ``unsupported call``.
"""

from __future__ import annotations

import json
from typing import Any

from headroom.proxy.explore_pruner.focus import (
    EXPLORE_TOOL_NAME,
    FUNCTION_CALL_TYPE,
    is_explore_source_code_call,
)


def resolve_explore_session_key(
    *,
    headers: Any,
    body: dict[str, Any] | None,
    request_id: str,
) -> str:
    """Stable session key for explore prune store lookups."""
    for name in ("x-headroom-session-id", "session-id"):
        try:
            value = headers.get(name) if headers is not None else None
        except Exception:
            value = None
        if isinstance(value, str) and value.strip():
            return value.strip()
    if isinstance(body, dict):
        prev = body.get("previous_response_id")
        if isinstance(prev, str) and prev.strip():
            return prev.strip()
    return request_id


class ExploreStreamRewriter:
    """Stateful rewriter for Responses API stream events.

    Tracks explore calls from ``output_item.added`` so argument deltas can be
    suppressed and replaced once the full explore args are known.
    """

    def __init__(self, service: Any, *, session_key: str) -> None:
        self._service = service
        self._session_key = session_key
        self._pending_explore_item_ids: set[str] = set()

    def set_session_key(self, session_key: str) -> None:
        self._session_key = session_key

    def rewrite_event(self, event: dict[str, Any]) -> dict[str, Any] | None:
        """Rewrite one Responses stream event.

        Returns the event to forward (possibly modified), or ``None`` to drop
        it (explore argument deltas that would corrupt a rewritten exec call).
        """
        if not isinstance(event, dict):
            return event

        event_type = event.get("type")
        if event_type == "response.output_item.added":
            return self._rewrite_item_added(event)
        if event_type == "response.output_item.done":
            return self._rewrite_item_done(event)
        if event_type == "response.function_call_arguments.delta":
            return self._rewrite_args_delta(event)
        if event_type == "response.function_call_arguments.done":
            return self._rewrite_args_done(event)
        if event_type == "response.completed":
            return self._rewrite_completed(event)
        return event

    def _rewrite_item_added(self, event: dict[str, Any]) -> dict[str, Any]:
        item = event.get("item")
        if not isinstance(item, dict) or not is_explore_source_code_call(item):
            return event
        item_id = item.get("id")
        if isinstance(item_id, str) and item_id:
            self._pending_explore_item_ids.add(item_id)
        # Args are usually empty on added; only rename so the client never sees
        # an unknown tool. Full sed rewrite happens on done.
        return {**event, "item": {**item, "name": "exec_command"}}

    def _rewrite_item_done(self, event: dict[str, Any]) -> dict[str, Any]:
        item = event.get("item")
        if not isinstance(item, dict):
            return event
        item_id = item.get("id")
        if isinstance(item_id, str):
            self._pending_explore_item_ids.discard(item_id)

        if is_explore_source_code_call(item):
            source_item = item
        elif self._looks_like_explore_args(item.get("arguments")):
            source_item = {**item, "name": EXPLORE_TOOL_NAME, "type": FUNCTION_CALL_TYPE}
        else:
            return event

        rewritten_list = self._service.rewrite_outbound_items(
            [source_item], session_key=self._session_key
        )
        if (
            not isinstance(rewritten_list, list)
            or not rewritten_list
            or rewritten_list[0] is source_item
        ):
            if is_explore_source_code_call(item):
                return {**event, "item": {**item, "name": "exec_command"}}
            return event
        return {**event, "item": rewritten_list[0]}

    def _rewrite_args_delta(self, event: dict[str, Any]) -> dict[str, Any] | None:
        item_id = event.get("item_id")
        if isinstance(item_id, str) and item_id in self._pending_explore_item_ids:
            return None
        return event

    def _rewrite_args_done(self, event: dict[str, Any]) -> dict[str, Any] | None:
        item_id = event.get("item_id")
        if not (isinstance(item_id, str) and item_id in self._pending_explore_item_ids):
            return event

        raw_args = event.get("arguments")
        fake_item = {
            "type": FUNCTION_CALL_TYPE,
            "name": EXPLORE_TOOL_NAME,
            "call_id": event.get("call_id") or f"explore-{item_id}",
            "id": item_id,
            "arguments": raw_args if isinstance(raw_args, str) else json.dumps(raw_args or {}),
        }
        rewritten_list = self._service.rewrite_outbound_items(
            [fake_item], session_key=self._session_key
        )
        if (
            not isinstance(rewritten_list, list)
            or not rewritten_list
            or is_explore_source_code_call(rewritten_list[0])
        ):
            return event
        return {**event, "arguments": rewritten_list[0].get("arguments")}

    def _rewrite_completed(self, event: dict[str, Any]) -> dict[str, Any]:
        response = event.get("response")
        if not isinstance(response, dict):
            return event
        output = response.get("output")
        if not isinstance(output, list):
            return event
        rewritten = self._service.rewrite_outbound_items(
            output, session_key=self._session_key
        )
        if rewritten is output:
            return event
        return {**event, "response": {**response, "output": rewritten}}

    @staticmethod
    def _looks_like_explore_args(raw: object) -> bool:
        if isinstance(raw, dict):
            args = raw
        elif isinstance(raw, str) and raw.strip():
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                return False
            if not isinstance(parsed, dict):
                return False
            args = parsed
        else:
            return False
        return (
            "path" in args
            and "focus_question" in args
            and ("start_line" in args or "end_line" in args)
        )


def rewrite_explore_sse_data_line(
    data: str,
    rewriter: ExploreStreamRewriter,
) -> str | None:
    """Rewrite a single SSE ``data:`` payload; return None to drop the event."""
    if data == "[DONE]":
        return data
    try:
        event = json.loads(data)
    except json.JSONDecodeError:
        return data
    if not isinstance(event, dict):
        return data
    rewritten = rewriter.rewrite_event(event)
    if rewritten is None:
        return None
    if rewritten is event:
        return data
    return json.dumps(rewritten, ensure_ascii=False)


def pop_complete_sse_event(buf: bytearray) -> bytes | None:
    """Remove and return the next complete SSE event from ``buf``, or None."""
    crlf = buf.find(b"\r\n\r\n")
    lf = buf.find(b"\n\n")
    if crlf < 0 and lf < 0:
        return None
    if crlf >= 0 and (lf < 0 or crlf < lf):
        end = crlf + 4
    else:
        end = lf + 2
    event = bytes(buf[:end])
    del buf[:end]
    return event


def rewrite_sse_event_bytes(
    raw: bytes,
    rewriter: ExploreStreamRewriter,
) -> bytes | None:
    """Rewrite one complete SSE event (including trailing blank line).

    Returns ``None`` to drop the event. Unparseable bytes pass through.
    """
    if not raw:
        return raw

    terminator = b"\n\n"
    body = raw
    if raw.endswith(b"\r\n\r\n"):
        terminator = b"\r\n\r\n"
        body = raw[: -len(terminator)]
    elif raw.endswith(b"\n\n"):
        body = raw[:-2]

    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError:
        return raw

    lines = text.split("\n")
    data_parts: list[str] = []
    for line in lines:
        stripped = line[:-1] if line.endswith("\r") else line
        if stripped.startswith("data:"):
            data_parts.append(stripped[5:].lstrip())
    if not data_parts:
        return raw

    new_data = rewrite_explore_sse_data_line("\n".join(data_parts), rewriter)
    if new_data is None:
        return None
    if new_data == "\n".join(data_parts):
        return raw

    rebuilt: list[str] = []
    data_emitted = False
    for line in lines:
        stripped = line[:-1] if line.endswith("\r") else line
        if stripped.startswith("data:"):
            if not data_emitted:
                rebuilt.append(f"data: {new_data}")
                data_emitted = True
        else:
            rebuilt.append(line)
    return ("\n".join(rebuilt).encode("utf-8")) + terminator
