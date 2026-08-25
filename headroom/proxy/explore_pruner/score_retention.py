"""Score-retention settings stub (full logic deferred; kept_frags is primary).

``CodeAstProtectSettings`` references this type. With ``enabled=False`` (default)
function-score expansion is a no-op.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ScoreRetentionSettings:
    """Controls score-aware keep/drop decisions after pruning (opt-in later)."""

    enabled: bool = False
    t_high: float = 0.75
    t_mid: float = 0.5
    t_ultra: float = 0.95
    header_ultra_fn_max_lines: int = 20
    header_ultra_class_member_max_lines: int = 40
    body_coverage_min: float = 0.12
    header_keeps_docstring: bool = True
    member_level_enabled: bool = True
    merge_with_fragment_align: bool = True
    log_scores: bool = False
    gray_zone_refinement: bool = False
    max_unaligned_ratio: float = 0.15
    window_size: int = 16


def aggregate_span_score(*_args, **_kwargs) -> float:
    return 0.0


def body_coverage(*_args, **_kwargs) -> float:
    return 0.0


def header_ultra_keep_whole_function(*_args, **_kwargs) -> bool:
    return False


def max_window_score(*_args, **_kwargs) -> float:
    return 0.0
