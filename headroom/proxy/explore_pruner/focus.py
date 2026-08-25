"""Explore-tool schema, instructions, and outbound sed rewrite helpers."""

from __future__ import annotations

import json
import shlex
from pathlib import PurePath
from typing import Literal

FocusSource = Literal["arg", "none"]

FOCUS_ARG_KEY = "focus_question"
FUNCTION_CALL_TYPE = "function_call"
FUNCTION_OUTPUT_TYPE = "function_call_output"

EXPLORE_TOOL_NAME = "explore_source_code"
EXPLORE_PATH_KEY = "path"
EXPLORE_START_KEY = "start_line"
EXPLORE_END_KEY = "end_line"
DEFAULT_EXPLORE_MAX_LINES = 400

TRUNCATED_OUTPUT_MARKERS = (
    "warning: truncated output",
    "tokens truncated",
    "truncated output (original token count",
)

EXPLORE_OUTPUT_FORMAT_HINT = (
    "Omitted regions appear as `(compressed N lines: <brief summary>)` where N "
    "is the number of original source lines dropped; surrounding code structure "
    "is kept."
)

EXPLORE_TOOL_INSTRUCTIONS = (
    "When you need to understand a source file, prefer shell rg/grep first to "
    "locate relevant line numbers, then call explore_source_code with path, "
    "focus_question, and a line window (start_line + end_line, 1-based). "
    "Do not cat whole files; use explore windows for file reads. Shell is fine "
    "for search and other non-read commands.\n\n"
    "Window size: keep end_line - start_line + 1 within about 400 lines "
    "(gateway may clamp longer windows). Put line numbers only in start_line/"
    "end_line, never in focus_question.\n\n"
    "What to write in focus_question: a complete self-contained interrogative "
    "question (ends with '?') that states what you are trying to understand or "
    "debug about this window. It must be a real question, not a statement or "
    "imperative. Do NOT list symbols to keep, function names to preserve, or "
    "line ranges — the pruner infers relevance from the question alone. "
    "Keywords or vague phrases are not enough.\n\n"
    "If the tool output shows truncation warnings (e.g. 'truncated output' or "
    "'tokens truncated'), call explore_source_code again with a different "
    "window — do not assume the file was complete.\n\n"
    "Output format: " + EXPLORE_OUTPUT_FORMAT_HINT + "\n\n"
    "Good examples:\n"
    "- Why does auth fail when a token is expired?\n"
    "- How is pagination wired from the request into page-size handling?\n"
    "- Where is the scene graph built, and how are nodes attached after a "
    "null-pointer in the renderer?\n\n"
    "Bad examples:\n"
    "- load_raw function (not a question; no intent)\n"
    "- Keep the token validation and expiry-check logic (statement / keep list; "
    "not a question)\n"
    "- lines 50-100 of data_loader.py (put lines in start_line/end_line, not "
    "focus_question)\n"
    "- fix the bug (not a question; no reading intent)"
)

EXPLORE_SOURCE_CODE_TOOL: dict = {
    "type": "function",
    "name": EXPLORE_TOOL_NAME,
    "description": (
        "Read a line window of a source file; return parts matching "
        "focus_question. Prefer rg/grep to find lines, then explore with "
        "start_line/end_line. See injected instructions."
    ),
    "strict": True,
    "parameters": {
        "type": "object",
        "properties": {
            EXPLORE_PATH_KEY: {
                "type": "string",
                "description": "File path to read.",
            },
            FOCUS_ARG_KEY: {
                "type": "string",
                "description": (
                    "Required. A complete interrogative question (ends with '?') "
                    "about what you want to understand in this window. Not a "
                    "statement or keep-list. No paths, line numbers, or symbol "
                    "names to preserve. See injected instructions for examples."
                ),
            },
            EXPLORE_START_KEY: {
                "type": "integer",
                "minimum": 1,
                "description": "1-based start line of the window to read.",
            },
            EXPLORE_END_KEY: {
                "type": "integer",
                "minimum": 1,
                "description": (
                    "1-based end line of the window (inclusive). Prefer spans "
                    "within ~400 lines; longer windows may be clamped."
                ),
            },
        },
        "required": [
            EXPLORE_PATH_KEY,
            FOCUS_ARG_KEY,
            EXPLORE_START_KEY,
            EXPLORE_END_KEY,
        ],
        "additionalProperties": False,
    },
}

# Internal body key recording call_ids already pruned this request.
PRUNED_CALL_IDS_KEY = "_headroom_explore_pruned_call_ids"


def is_explore_source_code_call(item: dict) -> bool:
    """True for function_call named explore_source_code."""
    return (
        item.get("type") == FUNCTION_CALL_TYPE
        and str(item.get("name") or "") == EXPLORE_TOOL_NAME
    )


def truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars]


def _parse_call_arguments(item: dict) -> dict | None:
    raw = item.get("arguments")
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, dict):
            return parsed
    return None


def shell_commands(item: dict) -> list[str]:
    """Extract command strings from function_call(exec_command)."""
    if item.get("type") != FUNCTION_CALL_TYPE:
        return []
    raw = item.get("arguments")
    if isinstance(raw, dict):
        cmd = raw.get("cmd") or raw.get("command")
        return [str(cmd)] if cmd else []
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return [raw]
        if isinstance(parsed, dict):
            cmd = parsed.get("cmd") or parsed.get("command")
            return [str(cmd)] if cmd else [raw]
        return [raw]
    return []


def _looks_like_explicit_file_path(token: str) -> bool:
    value = token.strip()
    if not value or value.startswith("-"):
        return False
    if value in {"|", "||", "&&", ";", ">", ">>", "<", "2>", "&>", "1>"}:
        return False
    if value.endswith("/"):
        return False
    if "://" in value:
        return False
    path = PurePath(value)
    if path.suffix:
        return True
    return "/" in value


def extract_scoped_file_paths(commands: list[str]) -> list[str]:
    paths: list[str] = []
    for command in commands:
        if not isinstance(command, str) or not command.strip():
            continue
        try:
            tokens = shlex.split(command)
        except ValueError:
            continue
        for token in tokens[1:]:
            if _looks_like_explicit_file_path(token):
                paths.append(token)
    return paths


def has_python_file_scope(commands: list[str]) -> bool | None:
    """Return True when scope includes .py, False for known non-.py, else None."""
    paths = extract_scoped_file_paths(commands)
    if not paths:
        return None
    return any(path.endswith(".py") for path in paths)


def append_explore_source_code_tool(tools: list) -> list:
    """Append explore_source_code tool schema if not already present."""
    if not tools:
        return [dict(EXPLORE_SOURCE_CODE_TOOL)]
    for tool in tools:
        if isinstance(tool, dict) and tool.get("name") == EXPLORE_TOOL_NAME:
            return tools
    return [*tools, dict(EXPLORE_SOURCE_CODE_TOOL)]


def _parse_explore_line(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip():
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def clamp_explore_window(
    start: int, end: int, *, max_lines: int = DEFAULT_EXPLORE_MAX_LINES
) -> tuple[int, int]:
    """Normalize start/end and clamp window length to max_lines."""
    start = max(1, start)
    if end < start:
        end = start
    if max_lines < 1:
        max_lines = 1
    if end - start + 1 > max_lines:
        end = start + max_lines - 1
    return start, end


def tool_output_looks_truncated(text: str) -> bool:
    lowered = (text or "").lower()
    return any(marker in lowered for marker in TRUNCATED_OUTPUT_MARKERS)


def parse_explore_source_fields(
    item: dict,
) -> tuple[str, int, int, str | None] | None:
    """Return unclamped (path, start, end, focus) from an explore_source_code call."""
    if not is_explore_source_code_call(item):
        return None
    args = _parse_call_arguments(item)
    if not args:
        return None
    path = args.get(EXPLORE_PATH_KEY)
    if not isinstance(path, str) or not path.strip():
        return None
    start = _parse_explore_line(args.get(EXPLORE_START_KEY))
    end = _parse_explore_line(args.get(EXPLORE_END_KEY))
    if start is None or end is None:
        return None
    focus: str | None = None
    raw_focus = args.get(FOCUS_ARG_KEY)
    if isinstance(raw_focus, str) and raw_focus.strip():
        focus = raw_focus.strip()
    return path.strip(), start, end, focus


def restore_explore_function_call(
    item: dict,
    *,
    path: str,
    focus_question: str | None,
    start_line: int,
    end_line: int,
) -> dict:
    """Rewrite a client-facing exec_command item back to explore_source_code."""
    payload = {
        EXPLORE_PATH_KEY: path,
        FOCUS_ARG_KEY: focus_question or "",
        EXPLORE_START_KEY: start_line,
        EXPLORE_END_KEY: end_line,
    }
    raw = item.get("arguments")
    arguments: dict | str
    if isinstance(raw, dict):
        arguments = payload
    else:
        arguments = json.dumps(payload, ensure_ascii=False)
    return {
        **item,
        "type": FUNCTION_CALL_TYPE,
        "name": EXPLORE_TOOL_NAME,
        "arguments": arguments,
    }


def rewrite_explore_call_to_exec(
    item: dict,
    max_chars: int = 300,
    *,
    max_lines: int = DEFAULT_EXPLORE_MAX_LINES,
) -> tuple[dict, str | None, FocusSource]:
    """Rewrite explore_source_code to exec_command sed window; return focus.

    Returns the original item unchanged (and focus=None) when args are invalid.
    """
    if not is_explore_source_code_call(item):
        return item, None, "none"

    args = _parse_call_arguments(item)
    if not args:
        return item, None, "none"

    path = args.get(EXPLORE_PATH_KEY)
    if not isinstance(path, str) or not path.strip():
        return item, None, "none"

    start = _parse_explore_line(args.get(EXPLORE_START_KEY))
    end = _parse_explore_line(args.get(EXPLORE_END_KEY))
    if start is None or end is None:
        return item, None, "none"

    start, end = clamp_explore_window(start, end, max_lines=max_lines)

    focus: str | None = None
    source: FocusSource = "none"
    raw_focus = args.get(FOCUS_ARG_KEY)
    if isinstance(raw_focus, str) and raw_focus.strip():
        focus = truncate(raw_focus.strip(), max_chars)
        source = "arg"

    cmd = f"sed -n '{start},{end}p' {shlex.quote(path.strip())}"
    new_args = json.dumps({"cmd": cmd}, ensure_ascii=False)
    rewritten = {
        **item,
        "name": "exec_command",
        "arguments": new_args,
    }
    return rewritten, focus, source


def aggregate_output_text(item: dict) -> str:
    """Join tool output text (function_call_output string or chunk list)."""
    raw = item.get("output")
    if isinstance(raw, str):
        return raw
    if not isinstance(raw, list):
        return ""
    parts: list[str] = []
    for chunk in raw:
        if isinstance(chunk, str):
            parts.append(chunk)
            continue
        if not isinstance(chunk, dict):
            continue
        for key in ("stdout", "stderr", "text"):
            val = chunk.get(key)
            if isinstance(val, str) and val:
                parts.append(val)
    return "".join(parts)


def rewrite_pruned_output(item: dict, pruned: str) -> dict:
    """Replace tool output text with pruned content; keep call_id / outcomes."""
    new_item = dict(item)
    output = item.get("output")

    if isinstance(output, str) or item.get("type") == FUNCTION_OUTPUT_TYPE:
        new_item["output"] = pruned
        return new_item

    if not isinstance(output, list) or not output:
        new_item["output"] = [{"type": "stdout", "stdout": pruned, "text": pruned}]
        return new_item

    new_chunks: list = []
    wrote = False
    for chunk in output:
        if not isinstance(chunk, dict):
            new_chunks.append(chunk)
            continue
        new_chunk = dict(chunk)
        if not wrote and (
            "stdout" in new_chunk or new_chunk.get("type") in (None, "stdout", "text")
        ):
            if "stdout" in new_chunk or new_chunk.get("type") == "stdout":
                new_chunk["stdout"] = pruned
            if "text" in new_chunk:
                new_chunk["text"] = pruned
            if "stderr" in new_chunk:
                new_chunk["stderr"] = ""
            wrote = True
        else:
            if "stdout" in new_chunk:
                new_chunk["stdout"] = ""
            if "stderr" in new_chunk and "outcome" not in new_chunk:
                new_chunk["stderr"] = ""
            if "text" in new_chunk and "outcome" not in new_chunk:
                new_chunk["text"] = ""
        new_chunks.append(new_chunk)

    if not wrote:
        new_chunks.insert(0, {"type": "stdout", "stdout": pruned, "text": pruned})

    new_item["output"] = new_chunks
    return new_item


def normalize_responses_input(raw: object) -> list:
    if isinstance(raw, list):
        return list(raw)
    if isinstance(raw, str):
        return [{"type": "message", "role": "user", "content": raw}]
    return []
