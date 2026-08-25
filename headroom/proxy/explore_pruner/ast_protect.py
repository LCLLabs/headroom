"""AST round-up rebuild after pruner compression.

The external pruner is a relevance oracle: its ``pruned_code`` is never spliced
back as Python. Original tree-sitter AST is the only structure source; kept
fragments are rounded up to complete leaf statements, then the tree is re-emitted.

Parser is loaded at gateway startup when code_ast_protect.enabled is true.
Load failure aborts the process; when disabled, this module is skipped.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from headroom.proxy.explore_pruner.pruner_types import PrunerResult
from headroom.proxy.explore_pruner.score_retention import ScoreRetentionSettings

logger = logging.getLogger(__name__)

# CoACT-style omission markers, plus legacy swe-pruner ``(filtered N lines)``.
_OMISSION_MARKER_RE = re.compile(
    r"\(\s*(?:filtered|compressed)\s+\d+\s+lines?(?:\s*:\s*[^)]*)?\s*\)",
    re.IGNORECASE,
)
_OMISSION_COUNT_RE = re.compile(
    r"\(\s*(?:filtered|compressed)\s+(\d+)\s+lines?(?:\s*:\s*[^)]*)?\s*\)",
    re.IGNORECASE,
)
_OMISSION_MARKER = "(compressed {n} lines: omitted)"
# Back-compat aliases used by older call sites / tests.
_FILTERED_RE = _OMISSION_MARKER_RE
_FILTERED_COUNT_RE = _OMISSION_COUNT_RE
_FILTERED_MARKER = _OMISSION_MARKER
# cat -n / nl / GitHub: consume the number + separator, keep the original indent
_LINE_NUM_PREFIX = re.compile(r"^(\s*\d+)(?:\t+|\|(?:\s)?|:(?:\s)?| )")
_FILE_BANNER = re.compile(r"^==>\s+.*\s+<==\s*$")

_IMPORT_TYPES = frozenset(
    {
        "import_statement",
        "import_from_statement",
        "future_import_statement",
    }
)

# Block/module children we will keep/drop. Identifiers inside ERROR nodes are ignored.
_LEAF_TYPES = _IMPORT_TYPES | frozenset(
    {
        "assignment",
        "augmented_assignment",
        "expression_statement",
        "return_statement",
        "pass_statement",
        "break_statement",
        "continue_statement",
        "raise_statement",
        "assert_statement",
        "delete_statement",
        "global_statement",
        "nonlocal_statement",
        "type_alias_statement",
        "string",
        "call",
        "yield",
        "await",
    }
)

_COMPOUND_STMT = frozenset(
    {
        "if_statement",
        "try_statement",
        "for_statement",
        "while_statement",
        "with_statement",
        "match_statement",
        "function_definition",
        "class_definition",
        "decorated_definition",
    }
)

_CLAUSE = frozenset(
    {
        "elif_clause",
        "else_clause",
        "except_clause",
        "except_group_clause",
        "finally_clause",
        "case_clause",
    }
)

_CHAIN_CLAUSES: dict[str, tuple[str, ...]] = {
    "if_statement": ("elif_clause", "else_clause"),
    "try_statement": (
        "except_clause",
        "except_group_clause",
        "else_clause",
        "finally_clause",
    ),
    "for_statement": ("else_clause",),
    "while_statement": ("else_clause",),
    "with_statement": (),
}

# Process-level Python parser (set by init_tree_sitter at startup).
_parser: Any | None = None


@dataclass
class CodeAstProtectSettings:
    """Settings for AST round-up rebuild after pruning."""

    enabled: bool = False
    log_pruned_text: bool = False
    preserve_imports: bool = True
    filtered_markers: bool = True
    # On rebuild miss: "original" (pre-prune prepared) | "pruned" (pruner text)
    rebuild_fallback: str = "pruned"
    function_block_mode: str = "off"
    function_block_max_lines: int = 80
    function_block_max_count: int = 5
    score_retention: ScoreRetentionSettings = field(
        default_factory=ScoreRetentionSettings
    )


@dataclass
class RebuildResult:
    """Round-up rebuild outcome. ``text`` is None when skipped."""

    text: str | None
    leaf_count: int = 0
    kept_count: int = 0
    skip_reason: str | None = None


@dataclass
class _Leaf:
    start: int
    end: int
    type: str


def init_tree_sitter() -> bool:
    """Load and probe the Python tree-sitter parser at startup.

    Returns True and sets module _parser on success; False otherwise.
    Safe to call more than once (idempotent when already initialized).
    """
    global _parser
    if _parser is not None:
        return True
    try:
        from tree_sitter import Language, Parser

        parser = Parser()
        # Prefer the bundled tree-sitter-python grammar (no GitHub download).
        # Fall back to language-pack, which fetches parsers on first use.
        try:
            import tree_sitter_python as tsp

            parser.language = Language(tsp.language())
        except Exception:
            from tree_sitter_language_pack import get_language

            parser.language = get_language("python")
        tree = parser.parse(b"def _probe():\n    return 1\n")
        root = tree.root_node
        ok = (
            root is not None
            and root.type == "module"
            and root.child_count > 0
            and not root.has_error
        )
        if not ok:
            logger.warning("tree-sitter probe parse failed")
            return False
        _parser = parser
        logger.info("tree-sitter python parser loaded")
        return True
    except Exception:
        logger.warning(
            "tree-sitter unavailable (install with: uv sync --extra code-ast; needs tree-sitter-python)",
            exc_info=True,
        )
        _parser = None
        return False


def reset_tree_sitter_for_tests() -> None:
    """Clear module parser (tests only)."""
    global _parser
    _parser = None


def get_parser() -> Any:
    """Return the startup-loaded parser; raise if init_tree_sitter was not called."""
    if _parser is None:
        raise RuntimeError(
            "tree-sitter parser not initialized; call init_tree_sitter() at startup"
        )
    return _parser


def strip_shell_line_numbers(text: str) -> str:
    """Remove listing prefixes (``cat -n`` / ``nl`` / ``==> file <==``) when present."""
    raw_lines = text.splitlines(keepends=True)
    bodies: list[tuple[str, str]] = []
    for line in raw_lines:
        if line.endswith("\r\n"):
            body, nl = line[:-2], "\r\n"
        elif line.endswith("\n"):
            body, nl = line[:-1], "\n"
        else:
            body, nl = line, ""
        bodies.append((body, nl))

    nonempty = [b for b, _nl in bodies if b.strip() and not _FILE_BANNER.match(b)]
    numbered = [_LINE_NUM_PREFIX.match(b) for b in nonempty]
    looks_numbered = (
        len(nonempty) >= 3
        and sum(1 for m in numbered if m) >= max(3, int(0.8 * len(nonempty)))
        and _line_numbers_mostly_consecutive(
            [int(m.group(1)) for m in numbered if m]
        )
    )

    out: list[str] = []
    for body, nl in bodies:
        if _FILE_BANNER.match(body):
            continue
        if looks_numbered:
            matched = _LINE_NUM_PREFIX.match(body)
            if matched:
                body = body[matched.end() :]
        out.append(body + nl)
    return "".join(out)


def _line_numbers_mostly_consecutive(nums: list[int]) -> bool:
    if len(nums) < 2:
        return False
    step_one = sum(1 for a, b in zip(nums, nums[1:]) if b - a == 1)
    return step_one / (len(nums) - 1) >= 0.8


def rebuild_python_from_pruned(
    original: str,
    pruned: str,
    settings: CodeAstProtectSettings | None = None,
    *,
    focus_question: str | None = None,
    commands: list[str] | None = None,
    focus_strategy_name: str | None = None,
    pruner_result: PrunerResult | None = None,
) -> RebuildResult:
    """Round kept regions up to complete original statements and re-emit.

    Primary keep oracle: ``pruner_result.kept_frags`` (1-based line numbers).
    Legacy: fragment-align ``pruned`` when kept_frags is empty/missing.
    Score retention is not used for keep selection.

    Original may contain tree-sitter ERROR nodes (truncated ``head`` output,
    leftover banners). Only well-typed statement leaves are kept.

    On skip, ``text`` is None and ``skip_reason`` is one of: empty, parse_error,
    no_leaves, no_matches, empty_rebuild, rebuild_error, rebuild_syntax.
    """
    settings = settings or CodeAstProtectSettings()
    kept_frags = (
        list(pruner_result.kept_frags)
        if pruner_result is not None and pruner_result.kept_frags
        else []
    )
    has_frags = bool(kept_frags)
    if not original or not original.strip():
        return RebuildResult(text=None, skip_reason="empty")
    if not has_frags and (not pruned or not pruned.strip()):
        return RebuildResult(text=None, skip_reason="empty")

    parsed = _parse_module(original, allow_error=True)
    if parsed is None:
        return RebuildResult(text=None, skip_reason="parse_error")
    root, code_bytes = parsed

    leaves = _collect_leaves(root)
    if not leaves:
        return RebuildResult(text=None, skip_reason="no_leaves")

    if has_frags:
        kept = kept_from_line_frags(leaves, code_bytes, kept_frags)
    else:
        kept = _align_leaves(code_bytes, pruned or "", leaves)

    if not kept:
        return RebuildResult(
            text=None,
            skip_reason="no_matches",
            leaf_count=len(leaves),
        )

    kept = _maybe_expand_to_functions(
        root,
        code_bytes,
        kept,
        settings,
        focus_question=focus_question,
        commands=commands or [],
        focus_strategy_name=focus_strategy_name,
        pruner_result=pruner_result,
    )

    if settings.preserve_imports:
        for leaf in leaves:
            if leaf.type in _IMPORT_TYPES:
                kept.add((leaf.start, leaf.end))

    ctx = _EmitCtx(code_bytes=code_bytes, kept=kept, markers=False)
    text = ctx.emit_stmt(root).rstrip() + "\n"
    if not text.strip():
        return RebuildResult(
            text=None,
            skip_reason="empty_rebuild",
            leaf_count=len(leaves),
            kept_count=len(kept),
        )

    check = _parse_module(text, allow_error=False)
    if check is None:
        return RebuildResult(
            text=None,
            skip_reason="rebuild_error",
            leaf_count=len(leaves),
            kept_count=len(kept),
        )
    try:
        compile(text, "<ast_rebuild>", "exec")
    except SyntaxError:
        return RebuildResult(
            text=None,
            skip_reason="rebuild_syntax",
            leaf_count=len(leaves),
            kept_count=len(kept),
        )

    if settings.filtered_markers:
        marked = _EmitCtx(
            code_bytes=code_bytes, kept=kept, markers=True
        ).emit_stmt(root).rstrip() + "\n"
        if marked.strip():
            text = marked

    return RebuildResult(text=text, leaf_count=len(leaves), kept_count=len(kept))


def _line_byte_ranges(code_bytes: bytes) -> list[tuple[int, int]]:
    """Byte spans for each line; index ``line_num - 1`` for 1-based line numbers.

    Matches swe-pruner ``str.splitlines()`` line count (no keepends content).
    """
    ranges: list[tuple[int, int]] = []
    offset = 0
    for chunk in code_bytes.splitlines(keepends=True):
        content_len = len(chunk.rstrip(b"\r\n"))
        ranges.append((offset, offset + content_len))
        offset += len(chunk)
    return ranges


def kept_from_line_frags(
    leaves: list[_Leaf],
    code_bytes: bytes,
    kept_frags: list[int],
) -> set[tuple[int, int]]:
    """Keep any leaf whose byte span overlaps a kept line's byte span."""
    ranges = _line_byte_ranges(code_bytes)
    line_spans: list[tuple[int, int]] = []
    for ln in kept_frags:
        if isinstance(ln, int) and 1 <= ln <= len(ranges):
            line_spans.append(ranges[ln - 1])
    if not line_spans:
        return set()

    kept: set[tuple[int, int]] = set()
    for leaf in leaves:
        for start, end in line_spans:
            if leaf.start < end and leaf.end > start:
                kept.add((leaf.start, leaf.end))
                break
    return kept


def _count_error_nodes(node: Any) -> int:
    count = 0
    if node.type == "ERROR" or getattr(node, "is_missing", False):
        count += 1
    for child in node.children:
        count += _count_error_nodes(child)
    return count


def _parse_module(
    code: str, *, allow_error: bool = False
) -> tuple[Any, bytes] | None:
    try:
        code_bytes = code.encode("utf-8")
    except UnicodeEncodeError:
        return None
    try:
        tree = get_parser().parse(code_bytes)
        root = tree.root_node
    except Exception:
        logger.debug("shell_prune ast_rebuild parse failed", exc_info=True)
        return None
    if root is None or root.type != "module":
        return None
    if not allow_error and (root.has_error or _count_error_nodes(root) > 0):
        return None
    return root, code_bytes


def _named_children(node: Any) -> list[Any]:
    return [c for c in node.children if getattr(c, "is_named", True)]


def _find_body_child(node: Any) -> Any | None:
    for child in node.children:
        if child.type == "block":
            return child
    return None


def _is_error(node: Any) -> bool:
    return node.type == "ERROR" or getattr(node, "is_missing", False)


def _collect_leaves(node: Any) -> list[_Leaf]:
    """Keep/drop units: named statement-like children that are not compounds."""
    out: list[_Leaf] = []
    if _is_error(node):
        return out
    ntype = node.type

    if ntype in ("module", "block"):
        for child in _named_children(node):
            if child.type == "comment" or _is_error(child):
                continue
            if child.type in _COMPOUND_STMT or child.type in _CLAUSE:
                out.extend(_collect_leaves(child))
            elif child.type in _LEAF_TYPES:
                out.append(_Leaf(child.start_byte, child.end_byte, child.type))
        return out

    if ntype == "decorated_definition":
        for child in node.children:
            if child.type in ("function_definition", "class_definition"):
                out.extend(_collect_leaves(child))
        return out

    if ntype in ("function_definition", "class_definition"):
        body = _find_body_child(node)
        if body is not None:
            out.extend(_collect_leaves(body))
        return out

    if ntype == "match_statement":
        body = _find_body_child(node)
        if body is not None:
            out.extend(_collect_leaves(body))
        return out

    if ntype in _COMPOUND_STMT or ntype in _CLAUSE:
        for child in node.children:
            if child.type == "block":
                out.extend(_collect_leaves(child))
            elif child.type in _CLAUSE or child.type == "cases":
                out.extend(_collect_leaves(child))
        return out

    if ntype == "cases":
        for child in node.children:
            if child.type == "case_clause":
                out.extend(_collect_leaves(child))
        return out

    for child in node.children:
        out.extend(_collect_leaves(child))
    return out


def _split_fragments(pruned: str) -> list[str]:
    parts = _OMISSION_MARKER_RE.split(pruned)
    return [p for p in parts if p.strip()]


def _lines_equivalent(original: str, pruned: str) -> bool:
    return original.rstrip() == pruned.rstrip() or original.strip() == pruned.strip()


def _original_line_texts(code_bytes: bytes) -> list[str]:
    return code_bytes.decode("utf-8").splitlines()


def _line_byte_span(code_bytes: bytes, line_no: int) -> tuple[int, int] | None:
    """Map 1-based line number to [start, end) bytes without trailing newline."""
    if line_no < 1:
        return None
    offset = 0
    for i, chunk in enumerate(code_bytes.splitlines(keepends=True), start=1):
        content_len = len(chunk.rstrip(b"\r\n"))
        if i == line_no:
            return offset, offset + content_len
        offset += len(chunk)
    return None


def _parse_virtual_kept_lines(pruned: str) -> list[tuple[int, str]]:
    """Walk pruner output; map each kept line to its original 1-based line number.

    ``(compressed N lines: …)`` / legacy ``(filtered N lines)`` advances the
    counter by N without emitting a kept line.
    """
    orig_line = 1
    kept: list[tuple[int, str]] = []
    for raw_line in pruned.splitlines():
        stripped = raw_line.strip()
        marker = _OMISSION_COUNT_RE.fullmatch(stripped) if stripped else None
        if marker:
            orig_line += int(marker.group(1))
            continue
        kept.append((orig_line, raw_line))
        orig_line += 1
    return kept


def _virtual_timeline_anchored(
    code_bytes: bytes,
    pruned: str,
    specs: list[tuple[int, str]],
) -> bool:
    """Virtual line numbers apply only when anchored at file start."""
    if not specs:
        return False
    first_physical = pruned.lstrip().splitlines()[0].strip() if pruned.strip() else ""
    if first_physical and _OMISSION_COUNT_RE.fullmatch(first_physical):
        return True
    first_line_no, first_text = specs[0]
    if first_line_no != 1:
        return False
    lines = _original_line_texts(code_bytes)
    if not lines:
        return False
    return _lines_equivalent(lines[0], first_text)


def _align_leaves_by_virtual_lines(
    code_bytes: bytes,
    pruned: str,
    leaves: list[_Leaf],
) -> set[tuple[int, int]]:
    """Align pruned lines via ``(compressed|filtered N lines…)`` virtual timeline."""
    if not _OMISSION_MARKER_RE.search(pruned):
        return set()

    specs = _parse_virtual_kept_lines(pruned)
    if not specs or not _virtual_timeline_anchored(code_bytes, pruned, specs):
        return set()

    orig_lines = _original_line_texts(code_bytes)
    matched_spans: list[tuple[int, int]] = []
    for line_no, pruned_line in specs:
        if line_no < 1 or line_no > len(orig_lines):
            continue
        if not _lines_equivalent(orig_lines[line_no - 1], pruned_line):
            continue
        span = _line_byte_span(code_bytes, line_no)
        if span is not None:
            matched_spans.append(span)

    if not matched_spans:
        return set()

    kept: set[tuple[int, int]] = set()
    for leaf in leaves:
        for start, end in matched_spans:
            if leaf.start < end and leaf.end > start:
                kept.add((leaf.start, leaf.end))
    return kept


def _find_bytes(haystack: bytes, needle: bytes, cursor: int) -> int:
    if not needle:
        return -1
    return haystack.find(needle, cursor)


def _locate_fragment(
    haystack: bytes, frag: str, cursor: int
) -> list[tuple[int, int]]:
    """Return match spans in original bytes; empty if this fragment cannot align."""
    stripped_nl = frag.strip("\n")
    candidates = [stripped_nl]
    stripped_all = frag.strip()
    if stripped_all != stripped_nl:
        candidates.append(stripped_all)

    for text in candidates:
        needle = text.encode("utf-8")
        idx = _find_bytes(haystack, needle, cursor)
        if idx >= 0:
            return [(idx, idx + len(needle))]

    spans: list[tuple[int, int]] = []
    pos = cursor
    for line in frag.splitlines():
        raw = line.rstrip()
        stripped = line.strip()
        if not stripped:
            continue
        idx = _find_bytes(haystack, raw.encode("utf-8"), pos)
        length = len(raw.encode("utf-8"))
        if idx < 0 and stripped != raw:
            idx = _find_bytes(haystack, stripped.encode("utf-8"), pos)
            length = len(stripped.encode("utf-8"))
        if idx < 0:
            continue
        spans.append((idx, idx + length))
        pos = idx + length
    return spans


def _align_leaves_by_fragments(
    code_bytes: bytes,
    pruned: str,
    leaves: list[_Leaf],
) -> set[tuple[int, int]]:
    """Monotonic overlap of pruned text fragments onto original leaf byte ranges."""
    kept: set[tuple[int, int]] = set()
    cursor = 0
    for frag in _split_fragments(pruned):
        spans = _locate_fragment(code_bytes, frag, cursor)
        if not spans:
            continue
        for start, end in spans:
            for leaf in leaves:
                if leaf.start < end and leaf.end > start:
                    kept.add((leaf.start, leaf.end))
            cursor = max(cursor, end)
    return kept


def _align_leaves(
    code_bytes: bytes,
    pruned: str,
    leaves: list[_Leaf],
) -> set[tuple[int, int]]:
    """Combine fragment substring alignment with virtual line-number alignment."""
    kept = _align_leaves_by_fragments(code_bytes, pruned, leaves)
    kept |= _align_leaves_by_virtual_lines(code_bytes, pruned, leaves)
    return kept


def _slice(code_bytes: bytes, start: int, end: int) -> str:
    return code_bytes[start:end].decode("utf-8")


def _line_indent(code_bytes: bytes, start: int) -> str:
    line_start = code_bytes.rfind(b"\n", 0, start) + 1
    return _slice(code_bytes, line_start, start)


def _colon_header(code_bytes: bytes, node: Any) -> str:
    """Text from node start through ``:``, dropping indent of the first body line."""
    body = _find_body_child(node)
    if body is None:
        return _slice(code_bytes, node.start_byte, node.end_byte)
    raw = _slice(code_bytes, node.start_byte, body.start_byte)
    if "\n" in raw:
        return raw.rstrip(" \t")
    return raw


def _body_indent(code_bytes: bytes, node: Any, body: Any | None) -> str:
    if body is not None:
        for child in _named_children(body):
            return _line_indent(code_bytes, child.start_byte)
    return _line_indent(code_bytes, node.start_byte) + "    "


def _span_line_count(code_bytes: bytes, start: int, end: int) -> int:
    if end <= start:
        return 0
    return len(_slice(code_bytes, start, end).splitlines())


def _iter_nodes(node: Any):
    yield node
    for child in node.children:
        yield from _iter_nodes(child)


def _function_nodes(root: Any) -> list[Any]:
    return [
        node
        for node in _iter_nodes(root)
        if node.type in ("function_definition", "decorated_definition")
    ]


def _function_span(node: Any) -> tuple[int, int] | None:
    if node.type == "function_definition":
        return (node.start_byte, node.end_byte)
    if node.type != "decorated_definition":
        return None
    for child in node.children:
        if child.type == "function_definition":
            return (node.start_byte, node.end_byte)
    return None


def _function_body_line_count(code_bytes: bytes, node: Any) -> int:
    span = _function_span(node)
    if span is None:
        return 0
    return _span_line_count(code_bytes, span[0], span[1])


def _is_understanding_focus(focus_question: str | None) -> bool:
    if not focus_question:
        return False
    text = focus_question.lower()
    understanding_terms = (
        "debug",
        "why",
        "how",
        "validation",
        "control flow",
        "data flow",
        "call flow",
        "locate where",
        "logic",
        "behavior",
        "fails",
        "failure",
        "returned",
        "built",
        "called",
    )
    constant_terms = (
        "constant",
        "default",
        "env",
        "config",
        "import",
        "line",
        "assignment",
        "string",
        "value",
    )
    if any(term in text for term in constant_terms):
        return False
    return any(term in text for term in understanding_terms)


def _function_block_mode_active(
    settings: CodeAstProtectSettings,
    *,
    focus_question: str | None,
    commands: list[str],
    focus_strategy_name: str | None,
) -> bool:
    """True when focus heuristics request whole-function expansion."""
    if settings.function_block_mode == "off":
        return False
    if settings.function_block_mode == "always":
        return True
    if settings.function_block_mode != "scenario":
        return False
    if focus_strategy_name != "explore_tool":
        return False
    return _is_understanding_focus(focus_question)


def _function_score_wants_whole(
    node: Any,
    settings: CodeAstProtectSettings,
    pruner_result: PrunerResult | None,
    code_bytes: bytes = b"",
) -> bool:
    """True when aligned token scores justify expanding this function wholly."""
    sr = settings.score_retention
    if not sr.enabled or pruner_result is None or not pruner_result.token_spans:
        return False
    from headroom.proxy.explore_pruner.score_retention import (
        aggregate_span_score,
        body_coverage,
        header_ultra_keep_whole_function,
        max_window_score,
    )

    spans = pruner_result.token_spans
    if header_ultra_keep_whole_function(node, code_bytes, spans, sr):
        return True

    body = _find_body_child(node)
    if body is None:
        return False
    body_score = aggregate_span_score(spans, body.start_byte, body.end_byte)
    window = max_window_score(
        spans, body.start_byte, body.end_byte, sr.window_size
    )
    body_score = max(body_score, window)
    coverage = body_coverage(spans, body.start_byte, body.end_byte, sr.t_mid)
    return body_score >= sr.t_high and coverage >= sr.body_coverage_min


def _matching_function_node(
    leaf_span: tuple[int, int], functions: list[Any]
) -> Any | None:
    start, end = leaf_span
    best = None
    for node in functions:
        span = _function_span(node)
        if span is None:
            continue
        if span[0] <= start and span[1] >= end:
            if best is None:
                best = node
                continue
            best_span = _function_span(best)
            if best_span is not None and (span[1] - span[0]) < (
                best_span[1] - best_span[0]
            ):
                best = node
    return best


def _maybe_expand_to_functions(
    root: Any,
    code_bytes: bytes,
    kept: set[tuple[int, int]],
    settings: CodeAstProtectSettings,
    *,
    focus_question: str | None,
    commands: list[str],
    focus_strategy_name: str | None,
    pruner_result: PrunerResult | None = None,
) -> set[tuple[int, int]]:
    focus_active = _function_block_mode_active(
        settings,
        focus_question=focus_question,
        commands=commands,
        focus_strategy_name=focus_strategy_name,
    )
    score_active = (
        settings.score_retention.enabled
        and pruner_result is not None
        and bool(pruner_result.token_spans)
    )
    if not focus_active and not score_active:
        return kept

    functions = _function_nodes(root)
    if not functions:
        return kept

    expanded: set[tuple[int, int]] = set(kept)
    expanded_count = 0
    for leaf_span in sorted(kept):
        node = _matching_function_node(leaf_span, functions)
        if node is None:
            continue
        function_span = _function_span(node)
        if function_span is None or function_span in expanded:
            continue
        if _function_body_line_count(code_bytes, node) > settings.function_block_max_lines:
            continue
        if expanded_count >= settings.function_block_max_count:
            continue

        fn_node = node
        if node.type == "decorated_definition":
            for child in node.children:
                if child.type == "function_definition":
                    fn_node = child
                    break

        wants_whole = focus_active or _function_score_wants_whole(
            fn_node, settings, pruner_result, code_bytes=code_bytes
        )
        if not wants_whole:
            continue
        expanded.add(function_span)
        expanded_count += 1
    return expanded


def _omission_marker(n: int, indent: str = "") -> str:
    """Emit a CoACT-compatible omission placeholder (generic summary)."""
    return f"{indent}{_OMISSION_MARKER.format(n=n)}"


def _filtered_marker(n: int, indent: str = "") -> str:
    """Backward-compatible alias for :func:`_omission_marker`."""
    return _omission_marker(n, indent)


def _subtree_kept(node: Any, kept: set[tuple[int, int]]) -> bool:
    key = (node.start_byte, node.end_byte)
    if key in kept:
        return True
    # Header-only / partial spans stored as byte ranges inside this node.
    for start, end in kept:
        if node.start_byte <= start and end <= node.end_byte:
            return True
    for child in node.children:
        if _subtree_kept(child, kept):
            return True
    return False


class _EmitCtx:
    """Walk the original tree and emit original slices for kept leaves."""

    def __init__(
        self,
        code_bytes: bytes,
        kept: set[tuple[int, int]],
        markers: bool = False,
    ) -> None:
        self.code_bytes = code_bytes
        self.kept = kept
        self.markers = markers

    def emit_stmt(self, node: Any) -> str:
        ntype = node.type
        if _is_error(node):
            return ""
        if ntype == "module":
            return self._emit_module(node)
        if ntype == "decorated_definition":
            return self._emit_decorated(node)
        if ntype in ("function_definition", "class_definition"):
            return self._emit_def(node)
        if ntype == "match_statement":
            return self._emit_match(node)
        if ntype in _CHAIN_CLAUSES:
            return self._emit_chain(node)
        if ntype == "comment":
            return ""
        if (node.start_byte, node.end_byte) in self.kept:
            return _slice(self.code_bytes, node.start_byte, node.end_byte)
        return ""

    def _flush_gap(
        self, parts: list[str], gap: list[int | None], indent: str
    ) -> None:
        start, end = gap[0], gap[1]
        gap[0] = gap[1] = None
        if not self.markers or start is None or end is None:
            return
        n = _span_line_count(self.code_bytes, start, end)
        if n > 0:
            parts.append(_omission_marker(n, indent))

    def _note_gap(self, gap: list[int | None], node: Any) -> None:
        if not self.markers:
            return
        if gap[0] is None:
            gap[0] = node.start_byte
        gap[1] = node.end_byte

    def _emit_children(
        self,
        children: list[Any],
        *,
        indent_of,
        rstrip_chunk: bool = False,
        trailing_indent: str = "",
    ) -> str:
        parts: list[str] = []
        gap: list[int | None] = [None, None]
        last_indent = trailing_indent
        for child in children:
            if child.type == "comment" or _is_error(child):
                self._note_gap(gap, child)
                continue
            chunk = self.emit_stmt(child)
            indent = indent_of(child)
            if not chunk:
                self._note_gap(gap, child)
                continue
            self._flush_gap(parts, gap, indent)
            parts.append(indent + (chunk.rstrip() if rstrip_chunk else chunk))
            last_indent = indent
        self._flush_gap(parts, gap, last_indent)
        return "\n".join(parts)

    def _emit_module(self, node: Any) -> str:
        return self._emit_children(
            _named_children(node),
            indent_of=lambda _c: "",
            rstrip_chunk=True,
        )

    def _emit_decorated(self, node: Any) -> str:
        if (node.start_byte, node.end_byte) in self.kept:
            return _slice(self.code_bytes, node.start_byte, node.end_byte)
        inner = None
        for child in node.children:
            if child.type in ("function_definition", "class_definition"):
                inner = child
                break
        if inner is None:
            return ""
        body = self._emit_def(inner)
        if not body:
            return ""
        prefix = _slice(self.code_bytes, node.start_byte, inner.start_byte)
        return prefix + body

    def _emit_def(self, node: Any) -> str:
        if not _subtree_kept(node, self.kept):
            return ""
        if (node.start_byte, node.end_byte) in self.kept:
            return _slice(self.code_bytes, node.start_byte, node.end_byte)
        body = _find_body_child(node)
        header = _colon_header(self.code_bytes, node)
        inner = self._emit_block(body) if body is not None else ""
        if not inner.strip():
            inner = _body_indent(self.code_bytes, node, body) + "pass"
        return header + inner

    def _emit_block(self, body: Any | None) -> str:
        if body is None:
            return ""
        children = _named_children(body)
        trailing = (
            _line_indent(self.code_bytes, children[0].start_byte) if children else ""
        )
        return self._emit_children(
            children,
            indent_of=lambda c: _line_indent(self.code_bytes, c.start_byte),
            trailing_indent=trailing,
        )

    def _emit_chain(self, node: Any) -> str:
        """if/try/for/while/with: drop empty constructs; keep leading header if a later clause lives."""
        if not _subtree_kept(node, self.kept):
            return ""
        body = _find_body_child(node)
        header = _colon_header(self.code_bytes, node)
        primary = self._emit_block(body)
        clause_parts: list[str] = []
        allowed = _CHAIN_CLAUSES.get(node.type, ())
        gap: list[int | None] = [None, None]
        has_kept_clause = False
        for child in node.children:
            if child.type not in allowed:
                continue
            if not _subtree_kept(child, self.kept):
                self._note_gap(gap, child)
                continue
            has_kept_clause = True
            indent = _line_indent(self.code_bytes, child.start_byte)
            marker_parts: list[str] = []
            self._flush_gap(marker_parts, gap, indent)
            if marker_parts:
                clause_parts.append("\n" + marker_parts[0])
            clause_parts.append(self._emit_clause(child))
        if self.markers and has_kept_clause:
            trail: list[str] = []
            self._flush_gap(
                trail, gap, _line_indent(self.code_bytes, node.start_byte)
            )
            if trail:
                clause_parts.append("\n" + trail[0])
        if not primary.strip() and not has_kept_clause:
            return ""
        # Bare `try` (no except/finally) is invalid; emit the body statements instead.
        if node.type == "try_statement" and not has_kept_clause:
            unwrapped = self._unwrap_block(
                body, _line_indent(self.code_bytes, node.start_byte)
            )
            if self.markers:
                trail: list[str] = []
                self._flush_gap(
                    trail, gap, _line_indent(self.code_bytes, node.start_byte)
                )
                if trail:
                    unwrapped = unwrapped + "\n" + trail[0]
            return unwrapped
        if not primary.strip():
            primary = _body_indent(self.code_bytes, node, body) + "pass"
        return header + primary + "".join(clause_parts)

    def _unwrap_block(self, body: Any | None, continuation_indent: str) -> str:
        """Emit block children as a sibling chunk (parent supplies first-line indent)."""
        if body is None:
            return ""
        children = _named_children(body)
        if not children:
            return ""
        first = True

        def indent_of(child: Any) -> str:
            nonlocal first
            if first:
                first = False
                return ""
            return continuation_indent

        return self._emit_children(children, indent_of=indent_of)

    def _emit_clause(self, node: Any) -> str:
        indent = _line_indent(self.code_bytes, node.start_byte)
        header = _colon_header(self.code_bytes, node)
        body = _find_body_child(node)
        inner = self._emit_block(body)
        if not inner.strip():
            inner = _body_indent(self.code_bytes, node, body) + "pass"
        return "\n" + indent + header + inner

    def _emit_match(self, node: Any) -> str:
        if not _subtree_kept(node, self.kept):
            return ""
        body = _find_body_child(node)
        cases = (
            [c for c in _named_children(body) if c.type == "case_clause"]
            if body is not None
            else []
        )
        kept_any = [c for c in cases if _subtree_kept(c, self.kept)]
        if not kept_any:
            return ""
        header = _colon_header(self.code_bytes, node)
        parts: list[str] = []
        gap: list[int | None] = [None, None]
        for case in cases:
            if not _subtree_kept(case, self.kept):
                self._note_gap(gap, case)
                continue
            indent = _line_indent(self.code_bytes, case.start_byte)
            marker_parts: list[str] = []
            self._flush_gap(marker_parts, gap, indent)
            if marker_parts:
                parts.append("\n" + marker_parts[0])
            parts.append(self._emit_clause(case))
        trail: list[str] = []
        self._flush_gap(
            trail, gap, _line_indent(self.code_bytes, node.start_byte)
        )
        if trail:
            parts.append("\n" + trail[0])
        return header + "".join(parts)
