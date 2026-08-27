"""Explore-tool orchestration: inject, rewrite outbound, prune inbound."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

from headroom.proxy.explore_pruner.ast_protect import strip_shell_line_numbers
from headroom.proxy.explore_pruner.focus import (
    EXPLORE_TOOL_INSTRUCTIONS,
    FUNCTION_CALL_TYPE,
    FUNCTION_OUTPUT_TYPE,
    PRUNED_CALL_IDS_KEY,
    aggregate_output_text,
    append_explore_source_code_tool,
    has_python_file_scope,
    is_explore_source_code_call,
    normalize_responses_input,
    parse_explore_source_fields,
    restore_explore_function_call,
    rewrite_explore_call_to_exec,
    rewrite_pruned_output,
    shell_commands,
    tool_output_looks_truncated,
)
from headroom.proxy.explore_pruner.protocol import ContextReducer
from headroom.proxy.explore_pruner.store import ExplorePrunerRecord, ExplorePrunerStore
from headroom.proxy.explore_pruner.types import PruneSavings, ReduceInput

logger = logging.getLogger(__name__)


def _safe_count(count_text: Callable[[str], int] | None, text: str) -> int:
    """Count tokens with the caller-supplied ruler; never raise."""
    if count_text is None:
        return 0
    try:
        return max(int(count_text(text)), 0)
    except Exception:
        return 0


class ExploreToolService:
    """Responses-path explore_tool orchestration (no CCR).

    Reducers return final ``ReduceResult.content``; backend-specific post-processing
    (e.g. swe-pruner AST rebuild) stays inside the reducer implementation.
    """

    def __init__(
        self,
        reducer: ContextReducer,
        store: ExplorePrunerStore | None = None,
        *,
        min_chars_to_prune: int = 1000,
        focus_max_chars: int = 500,
        explore_max_lines: int = 400,
        instructions_enabled: bool = True,
        fail_open: bool = True,
    ) -> None:
        self._reducer = reducer
        self._store = store or ExplorePrunerStore()
        self._min_chars = min_chars_to_prune
        self._focus_max = focus_max_chars
        self._max_lines = explore_max_lines
        self._instructions_enabled = instructions_enabled
        self._fail_open = fail_open

    @property
    def store(self) -> ExplorePrunerStore:
        return self._store

    @property
    def reducer_name(self) -> str:
        return self._reducer.name

    def prepare_request(self, body: dict[str, Any]) -> None:
        """Inject explore_source_code tool and fixed instructions."""
        tools = body.get("tools")
        if isinstance(tools, list):
            body["tools"] = append_explore_source_code_tool(tools)
        elif tools is None:
            body["tools"] = append_explore_source_code_tool([])

        if not self._instructions_enabled:
            return
        existing = body.get("instructions") or ""
        if not isinstance(existing, str):
            existing = str(existing)
        if EXPLORE_TOOL_INSTRUCTIONS in existing:
            return
        if existing.strip():
            body["instructions"] = f"{existing.rstrip()}\n\n{EXPLORE_TOOL_INSTRUCTIONS}"
        else:
            body["instructions"] = EXPLORE_TOOL_INSTRUCTIONS

    def rewrite_outbound_items(
        self,
        output_items: list[Any],
        *,
        session_key: str,
    ) -> list[Any]:
        """Rewrite explore_source_code → exec_command(sed); register store."""
        new_items: list[Any] = []
        changed = False
        for item in output_items:
            if not isinstance(item, dict) or not is_explore_source_code_call(item):
                new_items.append(item)
                continue

            rewritten, focus, _source = rewrite_explore_call_to_exec(
                item,
                self._focus_max,
                max_lines=self._max_lines,
            )
            # Invalid args: leave original item untouched.
            if is_explore_source_code_call(rewritten):
                logger.info(
                    "explore_pruner outbound skip call_id=%s reason=invalid_args",
                    item.get("call_id"),
                )
                new_items.append(rewritten)
                continue

            changed = True
            call_id = rewritten.get("call_id") or item.get("call_id")
            cmds = shell_commands(rewritten)
            logger.info(
                "explore_pruner outbound rewrite call_id=%s cmd=%r focus=%r",
                call_id,
                cmds[0] if cmds else "",
                focus,
            )
            if call_id:
                fields = parse_explore_source_fields(item)
                path, start, end = (None, None, None)
                if fields is not None:
                    path, start, end, _orig_focus = fields
                self._store.upsert(
                    session_key,
                    str(call_id),
                    ExplorePrunerRecord(
                        commands=cmds,
                        focus_question=focus,
                        created_at=time.time(),
                        explore_path=path,
                        explore_start_line=start,
                        explore_end_line=end,
                    ),
                )
            new_items.append(rewritten)
        return new_items if changed else output_items

    async def prune_inbound(
        self,
        body: dict[str, Any],
        *,
        session_key: str,
        count_text: Callable[[str], int] | None = None,
    ) -> PruneSavings:
        """Prune function_call_output items and restore explore calls for upstream.

        Client transcripts store rewritten ``exec_command`` items. This rewrites
        those back to ``explore_source_code`` (using the outbound store) so the
        model sees a consistent tool name, then prunes matching outputs.

        ``count_text`` is the same tokenizer ContentRouter uses. When omitted,
        the rewrite still runs but token totals stay zero.

        Returns a :class:`PruneSavings` report (``changed`` is True when the
        body input was modified).
        """
        started = time.perf_counter()
        items = normalize_responses_input(body.get("input"))
        if not items:
            return PruneSavings(elapsed_ms=(time.perf_counter() - started) * 1000.0)

        changed = False
        tokens_before = 0
        tokens_after = 0
        tokens_saved = 0
        pruned_item_count = 0

        def _note_rewrite(original_text: str, rewritten_text: str) -> None:
            nonlocal tokens_before, tokens_after, tokens_saved, pruned_item_count
            before = _safe_count(count_text, original_text)
            after = _safe_count(count_text, rewritten_text)
            tokens_before += before
            tokens_after += after
            tokens_saved += max(0, before - after)
            pruned_item_count += 1

        working: list[Any] = []
        for item in items:
            restored = self._restore_explore_call_item(item, session_key)
            if restored is not item:
                changed = True
                working.append(restored)
            else:
                working.append(item)
        items = working

        pruned_ids: set[str] = set()
        existing = body.get(PRUNED_CALL_IDS_KEY)
        if isinstance(existing, (set, list, tuple, frozenset)):
            pruned_ids.update(str(x) for x in existing)

        new_items: list[Any] = []
        for item in items:
            if not isinstance(item, dict) or item.get("type") != FUNCTION_OUTPUT_TYPE:
                new_items.append(item)
                continue

            call_id = item.get("call_id")
            if not call_id:
                new_items.append(item)
                continue
            call_id_s = str(call_id)

            record = self._store.get(session_key, call_id_s)
            if record is not None and record.pruned_output:
                original_text = aggregate_output_text(item)
                _note_rewrite(original_text, record.pruned_output)
                new_items.append(rewrite_pruned_output(item, record.pruned_output))
                pruned_ids.add(call_id_s)
                changed = True
                continue

            if record is None or not record.focus_question:
                # Same-request fallback: explore call may still be in input.
                focus, commands = self._focus_from_input_items(items, call_id_s)
                if not focus:
                    new_items.append(item)
                    continue
            else:
                focus = record.focus_question
                commands = record.commands

            text = aggregate_output_text(item)
            if len(text) < self._min_chars:
                new_items.append(item)
                continue
            if tool_output_looks_truncated(text):
                logger.info(
                    "explore_pruner skip call_id=%s reason=truncated_output",
                    call_id_s,
                )
                new_items.append(item)
                continue
            if has_python_file_scope(commands) is not True:
                logger.debug(
                    "explore_pruner skip call_id=%s reason=no_python_file",
                    call_id_s,
                )
                new_items.append(item)
                continue

            prepared = text
            stripped = strip_shell_line_numbers(text)
            if stripped.strip():
                prepared = stripped

            reduce_input = ReduceInput(
                content=prepared,
                query=focus,
                config={"commands": commands},
            )
            result = await self._reducer.reduce(reduce_input)
            from headroom.proxy.explore_pruner.debug_capture import (
                capture_explore_reducer_debug,
            )

            capture_explore_reducer_debug(
                reducer_name=self._reducer.name,
                session_key=session_key,
                call_id=call_id_s,
                before=reduce_input,
                after=result,
            )
            if result is None:
                if self._fail_open:
                    logger.warning(
                        "explore_pruner fail_open call_id=%s chars=%d",
                        call_id_s,
                        len(text),
                    )
                    new_items.append(item)
                    continue
                raise RuntimeError(f"explore reducer failed for call_id={call_id_s}")

            pruned = result.content

            logger.info(
                "explore_pruner applied call_id=%s reducer=%s chars_before=%d chars_after=%d",
                call_id_s,
                self._reducer.name,
                len(text),
                len(pruned),
            )

            self._store.upsert(
                session_key,
                call_id_s,
                ExplorePrunerRecord(
                    commands=commands,
                    focus_question=focus,
                    created_at=time.time(),
                    pruned_output=pruned,
                ),
            )
            _note_rewrite(text, pruned)
            new_items.append(rewrite_pruned_output(item, pruned))
            pruned_ids.add(call_id_s)
            changed = True

        if changed:
            body["input"] = new_items
            if pruned_ids:
                # Use a list so later json.dumps(body) in compression stays valid.
                body[PRUNED_CALL_IDS_KEY] = sorted(pruned_ids)
        return PruneSavings(
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            tokens_saved=tokens_saved,
            items=pruned_item_count,
            changed=changed,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )

    def _restore_explore_call_item(self, item: Any, session_key: str) -> Any:
        """Map client exec_command history back to explore_source_code for upstream."""
        if not isinstance(item, dict) or item.get("type") != FUNCTION_CALL_TYPE:
            return item
        if is_explore_source_code_call(item):
            return item
        if str(item.get("name") or "") != "exec_command":
            return item
        call_id = item.get("call_id")
        if not call_id:
            return item
        record = self._store.get(session_key, str(call_id))
        if (
            record is None
            or not record.explore_path
            or record.explore_start_line is None
            or record.explore_end_line is None
            or not record.focus_question
        ):
            return item
        logger.debug(
            "explore_pruner inbound restore call_id=%s path=%r",
            call_id,
            record.explore_path,
        )
        return restore_explore_function_call(
            item,
            path=record.explore_path,
            focus_question=record.focus_question,
            start_line=record.explore_start_line,
            end_line=record.explore_end_line,
        )

    def _focus_from_input_items(
        self, items: list[Any], call_id: str
    ) -> tuple[str | None, list[str]]:
        """Recover focus from a co-located explore/exec call in the same input."""
        for item in items:
            if not isinstance(item, dict):
                continue
            if str(item.get("call_id") or "") != call_id:
                continue
            if is_explore_source_code_call(item):
                rewritten, focus, _ = rewrite_explore_call_to_exec(
                    item, self._focus_max, max_lines=self._max_lines
                )
                if focus:
                    return focus, shell_commands(rewritten)
            cmds = shell_commands(item)
            # Already rewritten exec_command: focus only available from store.
            if cmds:
                return None, cmds
        return None, []
