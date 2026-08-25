"""Build ExploreToolService from ProxyConfig / ExplorePrunerConfig."""

from __future__ import annotations

import logging
import os

from headroom.proxy.explore_pruner.ast_protect import CodeAstProtectSettings, init_tree_sitter
from headroom.proxy.explore_pruner.protocol import get_reducer, register_reducer
from headroom.proxy.explore_pruner.reducers.coact import (
    COACT_DEFAULT_API_BASE,
    COACT_DEFAULT_TIMEOUT_SECONDS,
    COACT_NAME,
    CoactReducer,
)
from headroom.proxy.explore_pruner.reducers.swe_pruner import SWE_PRUNER_NAME, SwePrunerReducer
from headroom.proxy.explore_pruner.service import ExploreToolService
from headroom.proxy.explore_pruner.store import ExplorePrunerStore
from headroom.proxy.models import ExplorePrunerConfig

logger = logging.getLogger(__name__)


def explore_pruner_config_from_env(
    base: ExplorePrunerConfig | None = None,
) -> ExplorePrunerConfig:
    """Overlay HEADROOM_* env vars onto an ExplorePrunerConfig."""
    cfg = ExplorePrunerConfig(**{**vars(base)}) if base is not None else ExplorePrunerConfig()

    enabled_raw = os.environ.get("HEADROOM_EXPLORE_PRUNER_ENABLED")
    if enabled_raw is not None:
        cfg.enabled = enabled_raw.strip().lower() in ("1", "true", "yes", "on")

    reducer = os.environ.get("HEADROOM_EXPLORE_REDUCER")
    if reducer:
        cfg.reducer = reducer.strip()

    api_base = os.environ.get("HEADROOM_PRUNER_API_BASE")
    if api_base:
        cfg.api_base = api_base.strip()

    api_key = os.environ.get("HEADROOM_PRUNER_API_KEY")
    if api_key is not None:
        cfg.api_key = api_key.strip() or None

    timeout = os.environ.get("HEADROOM_PRUNER_TIMEOUT_SECONDS")
    if timeout:
        try:
            cfg.timeout_seconds = float(timeout)
        except ValueError:
            pass

    threshold = os.environ.get("HEADROOM_PRUNER_THRESHOLD")
    if threshold:
        try:
            cfg.threshold = float(threshold)
        except ValueError:
            pass

    min_chars = os.environ.get("HEADROOM_EXPLORE_MIN_CHARS")
    if min_chars:
        try:
            cfg.min_chars_to_prune = int(min_chars)
        except ValueError:
            pass

    max_lines = os.environ.get("HEADROOM_EXPLORE_MAX_LINES")
    if max_lines:
        try:
            cfg.explore_max_lines = int(max_lines)
        except ValueError:
            pass

    ast_raw = os.environ.get("HEADROOM_EXPLORE_AST_PROTECT")
    if ast_raw is not None:
        cfg.ast_protect_enabled = ast_raw.strip().lower() in ("1", "true", "yes", "on")

    return cfg


def _build_swe_pruner_reducer(cfg: ExplorePrunerConfig) -> SwePrunerReducer:
    """Construct swe_pruner with optional AST (tree-sitter warmed when enabled)."""
    ast_ok = True
    if cfg.ast_protect_enabled:
        ast_ok = init_tree_sitter()
        if not ast_ok:
            logger.warning(
                "explore_pruner: tree-sitter unavailable; swe_pruner AST disabled (fail-open)"
            )

    ast_enabled = bool(cfg.ast_protect_enabled and ast_ok)
    return SwePrunerReducer(
        api_base=cfg.api_base,
        api_key=cfg.api_key,
        timeout_seconds=cfg.timeout_seconds,
        threshold=cfg.threshold,
        always_keep_first_frags=cfg.always_keep_first_frags,
        chunk_overlap_tokens=cfg.chunk_overlap_tokens,
        ast_protect_enabled=ast_enabled,
        rebuild_fallback=cfg.rebuild_fallback,
        ast_settings=CodeAstProtectSettings(
            enabled=ast_enabled,
            rebuild_fallback=cfg.rebuild_fallback,
            preserve_imports=True,
            filtered_markers=True,
        ),
    )


def _build_coact_reducer(cfg: ExplorePrunerConfig) -> CoactReducer:
    """Construct CoACT reducer. No AST. Default port 8002 / 120s when swe defaults remain."""
    defaults = ExplorePrunerConfig()
    api_base = cfg.api_base
    if api_base == defaults.api_base:
        api_base = COACT_DEFAULT_API_BASE
    timeout = cfg.timeout_seconds
    if timeout == defaults.timeout_seconds:
        timeout = COACT_DEFAULT_TIMEOUT_SECONDS
    return CoactReducer(
        api_base=api_base,
        api_key=cfg.api_key,
        timeout_seconds=timeout,
    )


def build_explore_tool_service(cfg: ExplorePrunerConfig) -> ExploreToolService | None:
    """Construct and register an ExploreToolService, or None when disabled."""
    if not cfg.enabled:
        return None

    reducer_name = (cfg.reducer or SWE_PRUNER_NAME).strip() or SWE_PRUNER_NAME
    reducer = get_reducer(reducer_name)
    if reducer is None:
        if reducer_name == SWE_PRUNER_NAME:
            reducer = _build_swe_pruner_reducer(cfg)
        elif reducer_name == COACT_NAME:
            reducer = _build_coact_reducer(cfg)
        else:
            logger.error(
                "explore_pruner unknown reducer=%r; feature disabled",
                reducer_name,
            )
            return None
        register_reducer(reducer)

    store = ExplorePrunerStore(
        ttl_seconds=cfg.store_ttl_seconds,
        max_entries=cfg.store_max_entries,
    )
    return ExploreToolService(
        reducer=reducer,
        store=store,
        min_chars_to_prune=cfg.min_chars_to_prune,
        focus_max_chars=cfg.focus_max_chars,
        explore_max_lines=cfg.explore_max_lines,
        instructions_enabled=cfg.instructions_enabled,
        fail_open=cfg.fail_open,
    )
