"""Category-shift telemetry for compression requests.

Tracks how often content type changes between the pre- and post-compression
message bodies, and produces sample records for a dedicated JSONL log.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable

from headroom.compression.detector import ContentType, get_detector


_DETECTOR = None


def _get_detector():
    global _DETECTOR
    if _DETECTOR is None:
        _DETECTOR = get_detector(prefer_magika=True)
    return _DETECTOR


def _walk_texts(value: Any) -> Iterable[str]:
    if value is None:
        return
    if isinstance(value, str):
        yield value
        return
    if isinstance(value, list):
        for item in value:
            yield from _walk_texts(item)
        return
    if isinstance(value, dict):
        content = value.get("content")
        if content is not None:
            yield from _walk_texts(content)
        parts = value.get("parts")
        if parts is not None:
            yield from _walk_texts(parts)
        text = value.get("text")
        if isinstance(text, str):
            yield text
        elif text is not None and not isinstance(text, (dict, list)):
            yield str(text)


def _detect_content_type(text: str) -> str:
    result = _get_detector().detect(text)
    return result.content_type.value


def _family_from_transforms(transforms_applied: Iterable[str] | None) -> str:
    labels = " ".join(str(item).lower() for item in (transforms_applied or []))
    if "coact" in labels:
        return "coact"
    if "swe_pruner" in labels or "explore_pruner" in labels or "pruner" in labels:
        return "pruner"
    if "router:" in labels or "pipeline:" in labels:
        return "native"
    return "native"


@dataclass
class CategoryShiftObservation:
    family: str
    request_changed: bool
    unit_count: int
    changed_units: int
    before_counts: dict[str, int] = field(default_factory=dict)
    after_counts: dict[str, int] = field(default_factory=dict)
    transitions: dict[str, int] = field(default_factory=dict)
    request_messages: list[dict[str, Any]] | None = None
    compressed_messages: list[dict[str, Any]] | None = None

    @property
    def change_rate(self) -> float:
        return round((self.changed_units / self.unit_count) * 100.0, 2) if self.unit_count else 0.0


@dataclass
class CategoryShiftStats:
    requests_analyzed: int = 0
    requests_changed: int = 0
    units_compared: int = 0
    units_changed: int = 0
    before_counts: Counter[str] = field(default_factory=Counter)
    after_counts: Counter[str] = field(default_factory=Counter)
    transitions: Counter[str] = field(default_factory=Counter)

    def record(self, obs: CategoryShiftObservation) -> None:
        self.requests_analyzed += 1
        self.units_compared += obs.unit_count
        self.units_changed += obs.changed_units
        if obs.request_changed:
            self.requests_changed += 1
        self.before_counts.update(obs.before_counts)
        self.after_counts.update(obs.after_counts)
        self.transitions.update(obs.transitions)

    def snapshot(self) -> dict[str, Any]:
        return {
            "requests_analyzed": self.requests_analyzed,
            "requests_changed": self.requests_changed,
            "request_change_rate": round((self.requests_changed / self.requests_analyzed) * 100.0, 2)
            if self.requests_analyzed
            else 0.0,
            "units_compared": self.units_compared,
            "units_changed": self.units_changed,
            "unit_change_rate": round((self.units_changed / self.units_compared) * 100.0, 2)
            if self.units_compared
            else 0.0,
            "before_counts": dict(self.before_counts),
            "after_counts": dict(self.after_counts),
            "transitions": dict(self.transitions),
        }


@dataclass
class CategoryShiftCollector:
    total: CategoryShiftStats = field(default_factory=CategoryShiftStats)
    by_family: dict[str, CategoryShiftStats] = field(default_factory=dict)

    def record(self, obs: CategoryShiftObservation) -> None:
        self.total.record(obs)
        self.by_family.setdefault(obs.family, CategoryShiftStats()).record(obs)

    def snapshot(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "total": self.total.snapshot(),
            "by_family": {family: stats.snapshot() for family, stats in sorted(self.by_family.items())},
        }


def analyze_category_shift(
    request_messages: list[dict[str, Any]] | None,
    compressed_messages: list[dict[str, Any]] | None,
    *,
    transforms_applied: Iterable[str] | None = None,
) -> CategoryShiftObservation | None:
    if not request_messages or not compressed_messages:
        return None

    before_texts = list(_walk_texts(request_messages))
    after_texts = list(_walk_texts(compressed_messages))
    if not before_texts and not after_texts:
        return None

    before_types = [_detect_content_type(text) for text in before_texts]
    after_types = [_detect_content_type(text) for text in after_texts]

    unit_count = max(len(before_types), len(after_types))
    changed_units = 0
    before_counts: Counter[str] = Counter()
    after_counts: Counter[str] = Counter()
    transitions: Counter[str] = Counter()

    for idx in range(unit_count):
        before_type = before_types[idx] if idx < len(before_types) else ContentType.UNKNOWN.value
        after_type = after_types[idx] if idx < len(after_types) else ContentType.UNKNOWN.value
        before_counts[before_type] += 1
        after_counts[after_type] += 1
        if before_type != after_type:
            changed_units += 1
            transitions[f"{before_type}->{after_type}"] += 1

    return CategoryShiftObservation(
        family=_family_from_transforms(transforms_applied),
        request_changed=changed_units > 0,
        unit_count=unit_count,
        changed_units=changed_units,
        before_counts=dict(before_counts),
        after_counts=dict(after_counts),
        transitions=dict(transitions),
        request_messages=request_messages,
        compressed_messages=compressed_messages,
    )


def serialize_category_shift_record(
    *,
    request_id: str,
    timestamp: str,
    provider: str,
    model: str,
    observation: CategoryShiftObservation,
) -> dict[str, Any]:
    return {
        "request_id": request_id,
        "timestamp": timestamp,
        "provider": provider,
        "model": model,
        "family": observation.family,
        "request_changed": observation.request_changed,
        "unit_count": observation.unit_count,
        "changed_units": observation.changed_units,
        "change_rate": observation.change_rate,
        "before_counts": observation.before_counts,
        "after_counts": observation.after_counts,
        "transitions": observation.transitions,
        "request_messages": observation.request_messages,
        "compressed_messages": observation.compressed_messages,
    }


def json_line(record: dict[str, Any]) -> str:
    return json.dumps(record, ensure_ascii=False)
