"""Explore-tool + pluggable context reducers for OpenAI Responses."""

from __future__ import annotations

from headroom.proxy.explore_pruner.ast_protect import (
    CodeAstProtectSettings,
    init_tree_sitter,
    rebuild_python_from_pruned,
    strip_shell_line_numbers,
)
from headroom.proxy.explore_pruner.factory import (
    build_explore_tool_service,
    explore_pruner_config_from_env,
)
from headroom.proxy.explore_pruner.focus import (
    EXPLORE_TOOL_INSTRUCTIONS,
    EXPLORE_TOOL_NAME,
    PRUNED_CALL_IDS_KEY,
    append_explore_source_code_tool,
    is_explore_source_code_call,
    rewrite_explore_call_to_exec,
)
from headroom.proxy.explore_pruner.protocol import (
    ContextReducer,
    clear_reducer_registry,
    get_reducer,
    register_reducer,
    registered_reducer_names,
)
from headroom.proxy.explore_pruner.reducers import SWE_PRUNER_NAME, SwePrunerReducer
from headroom.proxy.explore_pruner.service import ExploreToolService
from headroom.proxy.explore_pruner.store import ExplorePrunerRecord, ExplorePrunerStore
from headroom.proxy.explore_pruner.types import ReduceInput, ReduceResult

__all__ = [
    "CodeAstProtectSettings",
    "ContextReducer",
    "EXPLORE_TOOL_INSTRUCTIONS",
    "EXPLORE_TOOL_NAME",
    "ExplorePrunerRecord",
    "ExplorePrunerStore",
    "ExploreToolService",
    "PRUNED_CALL_IDS_KEY",
    "ReduceInput",
    "ReduceResult",
    "SWE_PRUNER_NAME",
    "SwePrunerReducer",
    "append_explore_source_code_tool",
    "build_explore_tool_service",
    "clear_reducer_registry",
    "explore_pruner_config_from_env",
    "get_reducer",
    "init_tree_sitter",
    "is_explore_source_code_call",
    "rebuild_python_from_pruned",
    "register_reducer",
    "registered_reducer_names",
    "rewrite_explore_call_to_exec",
    "strip_shell_line_numbers",
]
