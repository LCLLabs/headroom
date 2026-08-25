"""In-memory TTL store for explore prune metadata keyed by session + call_id."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from threading import Lock

logger = logging.getLogger(__name__)


@dataclass
class ExplorePrunerRecord:
    commands: list[str]
    focus_question: str | None
    created_at: float
    pruned_output: str | None = None


class ExplorePrunerStore:
    """Process-local store. Multi-worker deployments need a shared backend later."""

    def __init__(self, ttl_seconds: int = 900, max_entries: int = 10_000) -> None:
        self._ttl = ttl_seconds
        self._max_entries = max_entries
        self._entries: dict[str, ExplorePrunerRecord] = {}
        self._lock = Lock()

    @staticmethod
    def _key(session_key: str, call_id: str) -> str:
        return f"explore_prune:{session_key}:{call_id}"

    def upsert(self, session_key: str, call_id: str, record: ExplorePrunerRecord) -> None:
        key = self._key(session_key, call_id)
        with self._lock:
            self._purge_locked()
            existing = self._entries.get(key)
            # Outbound re-register often omits pruned_output; keep cache.
            if (
                existing is not None
                and record.pruned_output is None
                and existing.pruned_output is not None
            ):
                record.pruned_output = existing.pruned_output
            self._entries[key] = record
            if len(self._entries) > self._max_entries:
                overflow = len(self._entries) - self._max_entries
                oldest = sorted(self._entries.items(), key=lambda kv: kv[1].created_at)
                for drop_key, _ in oldest[:overflow]:
                    del self._entries[drop_key]
        logger.debug(
            "explore_prune store upsert call_id=%s focus=%r pruned=%s",
            call_id,
            record.focus_question,
            record.pruned_output is not None,
        )

    def get(self, session_key: str, call_id: str) -> ExplorePrunerRecord | None:
        key = self._key(session_key, call_id)
        with self._lock:
            self._purge_locked()
            return self._entries.get(key)

    def _purge_locked(self) -> None:
        now = time.time()
        expired = [k for k, v in self._entries.items() if now - v.created_at > self._ttl]
        for k in expired:
            del self._entries[k]
