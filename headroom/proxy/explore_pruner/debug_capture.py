"""Opt-in disk capture of ContextReducer inputs/outputs (local debugging).

Mirrors Codex wire-debug: env-gated, file-based, never breaks the request path.

Large ``content`` bodies are written as sibling ``.txt`` files so editors show
real newlines instead of JSON ``\\n`` escapes. Metadata stays in a compact JSON
sidecar that references those files.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from headroom import paths as _paths
from headroom.proxy.explore_pruner.types import ReduceInput, ReduceResult
from headroom.proxy.wire_debug_format_policy import safe_wire_debug_name
from headroom.proxy.wire_debug_redaction_policy import redact_for_wire_debug

logger = logging.getLogger(__name__)

_DEBUG_ENV = "HEADROOM_EXPLORE_REDUCER_DEBUG"
_DEBUG_DIR_ENV = "HEADROOM_EXPLORE_REDUCER_DEBUG_DIR"


def explore_reducer_debug_enabled() -> bool:
    """Return whether opt-in explore-reducer capture is enabled."""
    return os.environ.get(_DEBUG_ENV, "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _debug_dir() -> Path:
    explicit = os.environ.get(_DEBUG_DIR_ENV, "").strip()
    if explicit:
        return Path(explicit).expanduser()
    return _paths.explore_reducer_debug_dir()


def capture_explore_reducer_debug(
    *,
    reducer_name: str,
    session_key: str,
    call_id: str,
    before: ReduceInput,
    after: ReduceResult | None,
) -> tuple[Path, Path, Path] | None:
    """Write before/after reducer snapshots; content goes to ``.txt`` with real newlines.

    Returns ``(meta_json, before_txt, after_txt)`` or ``None`` when disabled / on error.
    """
    if not explore_reducer_debug_enabled():
        return None

    try:
        out_dir = _debug_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        ts_ns = time.time_ns()
        safe_call = safe_wire_debug_name(call_id or "no_call")
        stem = f"{ts_ns}_{safe_call}"
        meta_path = out_dir / f"{stem}_reduce.json"
        before_path = out_dir / f"{stem}_before.txt"
        after_path = out_dir / f"{stem}_after.txt"

        before_path.write_text(before.content, encoding="utf-8")
        after_text = after.content if after is not None else ""
        after_path.write_text(after_text, encoding="utf-8")

        after_meta: dict[str, Any] | None
        if after is None:
            after_meta = None
        else:
            after_meta = {
                "content_file": after_path.name,
                "chars": len(after.content),
                "kept_frags": list(after.kept_frags),
                "token_scores": list(after.token_scores),
                "metadata": redact_for_wire_debug(after.metadata or {}),
            }

        payload = {
            "event": "explore_reducer_debug",
            "timestamp_ns": ts_ns,
            "reducer": reducer_name,
            "session_key": session_key,
            "call_id": call_id,
            "before": {
                "content_file": before_path.name,
                "chars": len(before.content),
                "query": before.query,
                "config": redact_for_wire_debug(before.config or {}),
            },
            "after": after_meta,
        }
        meta_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        logger.info(
            "event=explore_reducer_debug_capture path=%s call_id=%s reducer=%s "
            "chars_before=%d chars_after=%s",
            meta_path,
            call_id,
            reducer_name,
            len(before.content),
            len(after.content) if after is not None else "none",
        )
        return meta_path, before_path, after_path
    except Exception as exc:  # pragma: no cover - debug path must never break traffic
        logger.warning("event=explore_reducer_debug_capture_failed error=%s", exc)
        return None
