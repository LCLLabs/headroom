"""Fold explore-pruner token savings into the proxy accounting funnel."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from typing import Any

from headroom.proxy.explore_pruner.types import PruneSavings
from headroom.proxy.savings_attribution import SAVINGS_ATTRIBUTION_TAG, record_savings

TRANSFORM_NAME = "explore_pruner"


def merge_explore_prune_attribution(
    dest: MutableMapping[str, Any],
    src: Mapping[str, Any] | None,
) -> None:
    """Append prune ledger rows from ``src`` onto ``dest``.

    WebSocket sessions reuse one tag dict across turns. Recording prune
    savings onto that dict would replay every prior ``explore_pruner`` row
    into ``by_source`` on later outcomes. Callers record onto a short-lived
    dict and merge it once per outcome.
    """
    if not src:
        return
    ledger = src.get(SAVINGS_ATTRIBUTION_TAG)
    if not isinstance(ledger, list) or not ledger:
        return
    current = dest.get(SAVINGS_ATTRIBUTION_TAG)
    if isinstance(current, list):
        dest[SAVINGS_ATTRIBUTION_TAG] = [*current, *ledger]
    else:
        dest[SAVINGS_ATTRIBUTION_TAG] = list(ledger)


def attach_explore_prune_tags(
    base_tags: Mapping[str, Any] | None,
    prune_tags: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Copy ``base_tags`` and overlay pending prune attribution."""
    out: dict[str, Any] = dict(base_tags) if base_tags else {}
    merge_explore_prune_attribution(out, prune_tags)
    return out


def apply_explore_prune_savings(
    report: PruneSavings,
    *,
    tokens_saved: int,
    attempted_input_tokens: int,
    original_tokens: int = 0,
    transforms: list[str] | None = None,
    tags: MutableMapping[str, Any] | None = None,
    metrics: Any | None = None,
) -> tuple[int, int, int]:
    """Add prune deltas to handler accumulators and attribute the source.

    Compaction-shaped: the delta is folded into ``tokens_saved`` (the headline)
    and ``attempted_input_tokens`` (the active-savings denominator). Attribution
    via ``record_savings`` / ``record_compression`` is explanatory and is NOT
    added to the headline again.

    Returns ``(tokens_saved, attempted_input_tokens, original_tokens)``.
    """
    before = max(int(report.tokens_before), 0)
    saved = max(int(report.tokens_saved), 0)
    after = max(int(report.tokens_after), 0)
    if before <= 0 and saved <= 0:
        return tokens_saved, attempted_input_tokens, original_tokens

    tokens_saved += saved
    attempted_input_tokens += before
    original_tokens += before

    if saved <= 0:
        return tokens_saved, attempted_input_tokens, original_tokens

    if transforms is not None and TRANSFORM_NAME not in transforms:
        transforms.append(TRANSFORM_NAME)
    if tags is not None:
        record_savings(tags, TRANSFORM_NAME, tokens=saved)
    if metrics is not None:
        record_compression = getattr(metrics, "record_compression", None)
        if callable(record_compression):
            record_compression(
                TRANSFORM_NAME,
                original_tokens=before,
                compressed_tokens=after,
            )
        record_unit = getattr(metrics, "record_codex_ws_unit", None)
        if callable(record_unit):
            record_unit(
                strategy=TRANSFORM_NAME,
                reason_category="applied",
                elapsed_ms=max(float(report.elapsed_ms), 0.0),
                text_bytes=0,
                tokens_before=before,
                tokens_after=after,
                tokens_saved=saved,
                modified=True,
                content_type="code",
            )
    return tokens_saved, attempted_input_tokens, original_tokens
