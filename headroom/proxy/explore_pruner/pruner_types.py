"""Compatibility shim: PrunerResult shape expected by ast_protect."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from headroom.proxy.explore_pruner.types import ReduceResult, TokenScore, parse_token_scores


@dataclass
class TokenSpanScore:
    token: str
    score: float
    start: int | None = None
    end: int | None = None

    @property
    def aligned(self) -> bool:
        return self.start is not None and self.end is not None


@dataclass
class PrunerResult:
    """Full pruner response used by AST rebuild (kept_frags primary)."""

    pruned_code: str
    score: float | None = None
    origin_token_cnt: int | None = None
    left_token_cnt: int | None = None
    token_scores: list[TokenScore] = field(default_factory=list)
    kept_frags: list[int] = field(default_factory=list)
    token_spans: list[TokenSpanScore] = field(default_factory=list)


def reduce_result_to_pruner_result(result: ReduceResult) -> PrunerResult:
    """Adapt a ReduceResult into the AST rebuild oracle shape."""
    return PrunerResult(
        pruned_code=result.content,
        score=result.metadata.get("score") if isinstance(result.metadata, dict) else None,
        origin_token_cnt=result.metadata.get("origin_token_cnt")
        if isinstance(result.metadata, dict)
        else None,
        left_token_cnt=result.metadata.get("left_token_cnt")
        if isinstance(result.metadata, dict)
        else None,
        token_scores=[
            TokenScore(token=t, score=s) for t, s in result.token_scores if isinstance(t, str)
        ],
        kept_frags=list(result.kept_frags),
    )


def parse_pruner_response(data: dict[str, Any]) -> PrunerResult | None:
    from headroom.proxy.explore_pruner.types import parse_kept_frags, parse_swe_pruner_response

    reduced = parse_swe_pruner_response(data)
    if reduced is None:
        return None
    return reduce_result_to_pruner_result(reduced)
