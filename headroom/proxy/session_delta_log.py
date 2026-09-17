"""Opt-in append-only per-session delta log of post-processed upstream messages.

Records message-level deltas (not full transcript re-dumps) after Headroom /
ContextReducer processing, for Claude Code (Anthropic) and Codex (Responses).
Never raises into the request path.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

from headroom import paths as _paths
from headroom.proxy.wire_debug_format_policy import safe_wire_debug_name
from headroom.proxy.wire_debug_redaction_policy import redact_for_wire_debug

logger = logging.getLogger("headroom.proxy")

_ENV = "HEADROOM_SESSION_DELTA_LOG"
_DIR_ENV = "HEADROOM_SESSION_DELTA_LOG_DIR"
SCHEMA_VERSION = 1

_TRUE = ("1", "true", "yes", "on")

# Last successfully logged forwarded message list per session (process-local).
_last_logged: dict[str, list[dict[str, Any]]] = {}
_last_logged_lock = threading.Lock()

fcntl: Any | None
try:
    import fcntl as _fcntl

    fcntl = _fcntl
    _HAS_FCNTL = True
except ImportError:  # pragma: no cover - Windows
    fcntl = None
    _HAS_FCNTL = False


def session_delta_log_enabled() -> bool:
    """Return whether opt-in session delta logging is enabled."""
    return os.environ.get(_ENV, "").strip().lower() in _TRUE


def _log_dir() -> Path:
    explicit = os.environ.get(_DIR_ENV, "").strip()
    if explicit:
        return Path(explicit).expanduser()
    return _paths.session_delta_log_dir()


def _stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _as_message_list(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [item if isinstance(item, dict) else {"value": item} for item in value]
    if isinstance(value, str) and value:
        return [{"role": "user", "content": value}]
    if isinstance(value, dict):
        return [value]
    return []


def extract_forwarded_messages(body: Any, *, provider: str) -> list[dict[str, Any]]:
    """Normalize the post-processed conversation sequence for message-level deltas.

    Anthropic / chat: ``body["messages"]``.
    OpenAI Responses / Codex: ``body["input"]`` items (string coerced to one user msg).
    """
    if not isinstance(body, dict):
        return []
    provider_l = (provider or "").lower()
    if provider_l in ("openai", "codex", "responses"):
        if "input" in body:
            return _as_message_list(body.get("input"))
        # Some WS frames nest under ``response``.
        nested = body.get("response")
        if isinstance(nested, dict) and "input" in nested:
            return _as_message_list(nested.get("input"))
    messages = body.get("messages")
    if messages is not None:
        return _as_message_list(messages)
    if "input" in body:
        return _as_message_list(body.get("input"))
    return []


def common_prefix_len(previous: list[dict[str, Any]], current: list[dict[str, Any]]) -> int:
    """Return how many leading messages are deep-equal."""
    n = 0
    limit = min(len(previous), len(current))
    while n < limit and previous[n] == current[n]:
        n += 1
    return n


def build_message_delta(
    previous: list[dict[str, Any]] | None,
    current: list[dict[str, Any]],
    *,
    rewrite_reasons: list[str] | None = None,
) -> dict[str, Any]:
    """Classify a message-level delta against the last logged forwarded list.

    Ops:
      - ``snapshot``: first turn or truncated history
      - ``append``: previous is an exact prefix; only new trailing messages
      - ``unchanged``: identical sequence (meta-only)
      - ``rewrite``: divergence inside/at prefix; includes ``rewrite_reasons``
    """
    reasons = [str(r) for r in (rewrite_reasons or []) if r]
    content_hash = _stable_hash(current)

    if previous is None:
        return {
            "op": "snapshot",
            "prefix_len": 0,
            "prefix_hash": None,
            "content_hash": content_hash,
            "messages": redact_for_wire_debug(current),
            "rewrite_reasons": reasons,
        }

    if previous == current:
        return {
            "op": "unchanged",
            "prefix_len": len(current),
            "prefix_hash": _stable_hash(current),
            "content_hash": content_hash,
            "messages": [],
            "rewrite_reasons": reasons,
        }

    prefix_len = common_prefix_len(previous, current)

    # History shrank relative to last log → full snapshot (session reset / compact).
    if len(current) < len(previous) and prefix_len < len(current):
        return {
            "op": "snapshot",
            "prefix_len": 0,
            "prefix_hash": None,
            "content_hash": content_hash,
            "messages": redact_for_wire_debug(current),
            "rewrite_reasons": reasons or ["history_truncated"],
        }
    if len(current) < len(previous):
        return {
            "op": "snapshot",
            "prefix_len": 0,
            "prefix_hash": None,
            "content_hash": content_hash,
            "messages": redact_for_wire_debug(current),
            "rewrite_reasons": reasons or ["history_truncated"],
        }

    # Pure append of new messages.
    if prefix_len == len(previous) and len(current) > len(previous):
        return {
            "op": "append",
            "prefix_len": prefix_len,
            "prefix_hash": _stable_hash(previous),
            "content_hash": content_hash,
            "messages": redact_for_wire_debug(current[prefix_len:]),
            "rewrite_reasons": [],
        }

    # Divergence: rewrite from first differing index (message-level replacement).
    return {
        "op": "rewrite",
        "prefix_len": prefix_len,
        "prefix_hash": _stable_hash(previous[:prefix_len]) if prefix_len else None,
        "from_index": prefix_len,
        "content_hash": content_hash,
        "messages": redact_for_wire_debug(current[prefix_len:]),
        "rewrite_reasons": reasons or ["message_divergence"],
    }


def _session_file(session_id: str) -> Path:
    safe = safe_wire_debug_name(session_id or "unknown_session")
    return _log_dir() / f"{safe}.jsonl"


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
    with open(path, "a", encoding="utf-8") as handle:
        if _HAS_FCNTL and fcntl is not None:
            fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            handle.write(line)
            handle.flush()
        finally:
            if _HAS_FCNTL and fcntl is not None:
                fcntl.flock(handle, fcntl.LOCK_UN)


def resolve_session_delta_id(
    *,
    headers: Any = None,
    body: dict[str, Any] | None = None,
    session_id: str | None = None,
    request_id: str | None = None,
) -> str:
    """Pick a stable session key for the delta log file."""
    if session_id and str(session_id).strip():
        return str(session_id).strip()
    if headers is not None:
        for name in ("x-headroom-session-id", "session-id"):
            try:
                value = headers.get(name)
            except Exception:
                value = None
            if isinstance(value, str) and value.strip():
                return value.strip()
    if isinstance(body, dict):
        prev = body.get("previous_response_id")
        if isinstance(prev, str) and prev.strip():
            return prev.strip()
    if request_id and str(request_id).strip():
        return str(request_id).strip()
    return "unknown_session"


def capture_session_delta_log(
    *,
    provider: str,
    body: Any,
    session_id: str | None = None,
    request_id: str | None = None,
    transport: str = "http",
    model: str | None = None,
    rewrite_reasons: list[str] | tuple[str, ...] | None = None,
    transforms_applied: list[str] | tuple[str, ...] | None = None,
    mutation_reasons: list[str] | tuple[str, ...] | None = None,
    tokens_saved: int | None = None,
    metadata: dict[str, Any] | None = None,
    headers: Any = None,
) -> Path | None:
    """Append one message-level delta line for the session. Returns path or None."""
    if not session_delta_log_enabled():
        return None

    try:
        sid = resolve_session_delta_id(
            headers=headers, body=body if isinstance(body, dict) else None,
            session_id=session_id, request_id=request_id,
        )
        current = extract_forwarded_messages(body, provider=provider)
        reasons: list[str] = []
        for group in (rewrite_reasons, transforms_applied, mutation_reasons):
            if not group:
                continue
            for item in group:
                text = str(item).strip()
                if text and text not in reasons:
                    reasons.append(text)

        with _last_logged_lock:
            previous = _last_logged.get(sid)
            # Copy so callers cannot mutate stored state.
            prev_copy = [dict(m) for m in previous] if previous is not None else None
            delta = build_message_delta(prev_copy, current, rewrite_reasons=reasons)
            _last_logged[sid] = [dict(m) for m in current]

        extras: dict[str, Any] = {}
        if isinstance(body, dict):
            if body.get("system") is not None:
                extras["system_hash"] = _stable_hash(redact_for_wire_debug(body.get("system")))
            if body.get("instructions") is not None:
                extras["instructions_hash"] = _stable_hash(
                    redact_for_wire_debug(body.get("instructions"))
                )
            if body.get("tools") is not None:
                tools = body.get("tools")
                extras["tools_count"] = len(tools) if isinstance(tools, list) else 1
                extras["tools_hash"] = _stable_hash(redact_for_wire_debug(tools))

        record = {
            "v": SCHEMA_VERSION,
            "ts_ns": time.time_ns(),
            "session_id": sid,
            "provider": provider,
            "request_id": request_id,
            "transport": transport,
            "model": model,
            "op": delta["op"],
            "prefix_len": delta.get("prefix_len"),
            "prefix_hash": delta.get("prefix_hash"),
            "content_hash": delta.get("content_hash"),
            "from_index": delta.get("from_index"),
            "messages": delta.get("messages") or [],
            "rewrite_reasons": delta.get("rewrite_reasons") or [],
            "tokens_saved": tokens_saved,
            "extras": extras,
            "metadata": redact_for_wire_debug(metadata or {}),
        }

        path = _session_file(sid)
        _append_jsonl(path, record)
        logger.info(
            "event=session_delta_log path=%s session_id=%s op=%s request_id=%s "
            "prefix_len=%s messages=%d rewrite_reasons=%s",
            path,
            sid,
            record["op"],
            request_id or "",
            record.get("prefix_len"),
            len(record["messages"]),
            ",".join(record["rewrite_reasons"]) if record["rewrite_reasons"] else "",
        )
        return path
    except Exception as exc:  # pragma: no cover - must never break traffic
        logger.warning("event=session_delta_log_failed error=%s", exc)
        return None


def reset_session_delta_state_for_tests() -> None:
    """Clear in-memory last-logged state (tests only)."""
    with _last_logged_lock:
        _last_logged.clear()
