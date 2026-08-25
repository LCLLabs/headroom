"""Pluggable context-reducer protocol and in-process registry."""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

from headroom.proxy.explore_pruner.types import ReduceInput, ReduceResult

logger = logging.getLogger(__name__)

_REGISTRY: dict[str, ContextReducer] = {}


@runtime_checkable
class ContextReducer(Protocol):
    """Async pure-data context reducer (no AST knowledge required)."""

    name: str

    async def reduce(self, inp: ReduceInput) -> ReduceResult | None:
        """Return a reduced result, or ``None`` on failure (caller fail-opens)."""
        ...


def register_reducer(reducer: ContextReducer) -> None:
    """Register a reducer by ``name`` (last write wins)."""
    _REGISTRY[reducer.name] = reducer
    logger.debug("explore_pruner registered reducer=%s", reducer.name)


def get_reducer(name: str) -> ContextReducer | None:
    """Return a registered reducer, or ``None`` if unknown."""
    return _REGISTRY.get(name)


def clear_reducer_registry() -> None:
    """Clear the registry (tests only)."""
    _REGISTRY.clear()


def registered_reducer_names() -> list[str]:
    """Return registered reducer names (sorted)."""
    return sorted(_REGISTRY)
