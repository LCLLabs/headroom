"""Pure-data types for explore context reduction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ReduceInput:
    """Input to a :class:`~headroom.proxy.explore_pruner.protocol.ContextReducer`."""

    content: str
    query: str
    config: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReduceResult:
    """Output from a context reducer.

    ``content`` is the final text written back to tool output. Backend-specific
    post-processing (e.g. swe-pruner AST rebuild) must finish before returning;
    the explore orchestrator only reads ``content``.
    """

    content: str
    kept_frags: list[int] = field(default_factory=list)
    token_scores: list[tuple[str, float]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TokenScore:
    """One entry from swe-pruner ``token_scores``: ``(token_string, score)``."""

    token: str
    score: float


def parse_token_scores(raw: Any) -> list[TokenScore]:
    """Parse ``List[Tuple[str, float]]`` (JSON list-of-pairs) into TokenScore."""
    if not isinstance(raw, list):
        return []
    out: list[TokenScore] = []
    for item in raw:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            token, score = item[0], item[1]
            if isinstance(token, str) and isinstance(score, (int, float)):
                out.append(TokenScore(token=token, score=float(score)))
                continue
        if isinstance(item, dict):
            token = item.get("token")
            score = item.get("score")
            if isinstance(token, str) and isinstance(score, (int, float)):
                out.append(TokenScore(token=token, score=float(score)))
    return out


def parse_kept_frags(raw: Any) -> list[int]:
    """Parse 1-based kept line numbers; skip non-positive / non-int entries."""
    if not isinstance(raw, list):
        return []
    out: list[int] = []
    for item in raw:
        if isinstance(item, bool):
            continue
        if isinstance(item, int) and item >= 1:
            out.append(item)
        elif isinstance(item, float) and item == int(item) and int(item) >= 1:
            out.append(int(item))
    return out


def parse_swe_pruner_response(data: dict[str, Any]) -> ReduceResult | None:
    """Build a ReduceResult from a successful swe-pruner JSON body."""
    pruned = data.get("pruned_code")
    if not isinstance(pruned, str) or not pruned:
        return None

    score_raw = data.get("score")
    score = float(score_raw) if isinstance(score_raw, (int, float)) else None
    origin = data.get("origin_token_cnt")
    left = data.get("left_token_cnt")
    token_scores = parse_token_scores(data.get("token_scores"))
    return ReduceResult(
        content=pruned,
        kept_frags=parse_kept_frags(data.get("kept_frags")),
        token_scores=[(t.token, t.score) for t in token_scores],
        metadata={
            "backend": "swe_pruner",
            "score": score,
            "origin_token_cnt": int(origin) if isinstance(origin, int) else None,
            "left_token_cnt": int(left) if isinstance(left, int) else None,
        },
    )
