"""swe-pruner HTTP backend as a ContextReducer (includes optional AST rebuild)."""

from __future__ import annotations

import logging
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx

from headroom.proxy.explore_pruner.ast_protect import (
    CodeAstProtectSettings,
    rebuild_python_from_pruned,
)
from headroom.proxy.explore_pruner.pruner_types import reduce_result_to_pruner_result
from headroom.proxy.explore_pruner.types import ReduceInput, ReduceResult, parse_swe_pruner_response

logger = logging.getLogger(__name__)

_BLOCKED_HOSTS = frozenset(
    {
        "169.254.169.254",
        "metadata.google.internal",
        "metadata.goog",
    }
)

SWE_PRUNER_NAME = "swe_pruner"


class SwePrunerReducer:
    """POST {query, code, ...} → final pruned text (optional AST round-up)."""

    name = SWE_PRUNER_NAME

    def __init__(
        self,
        api_base: str,
        path: str = "/prune",
        api_key: str | None = None,
        timeout_seconds: float = 60.0,
        threshold: float | None = None,
        always_keep_first_frags: bool | None = None,
        chunk_overlap_tokens: int | None = None,
        *,
        ast_protect_enabled: bool = True,
        rebuild_fallback: str = "pruned",
        ast_settings: CodeAstProtectSettings | None = None,
    ) -> None:
        self._url = urljoin(api_base.rstrip("/") + "/", path.lstrip("/"))
        self._api_key = api_key
        self._timeout = timeout_seconds
        self._threshold = threshold
        self._always_keep_first_frags = always_keep_first_frags
        self._chunk_overlap_tokens = chunk_overlap_tokens
        self._ast_protect_enabled = ast_protect_enabled
        if ast_settings is None:
            ast_settings = CodeAstProtectSettings(
                enabled=ast_protect_enabled,
                rebuild_fallback=rebuild_fallback,
                preserve_imports=True,
                filtered_markers=True,
            )
        self._ast_settings = ast_settings
        self._validate_url()

    def _validate_url(self) -> None:
        parsed = urlparse(self._url)
        host = (parsed.hostname or "").lower()
        if host in _BLOCKED_HOSTS:
            raise ValueError(f"pruner api_base host blocked: {host}")

    def _build_payload(self, code: str, query: str) -> dict[str, Any]:
        payload: dict[str, Any] = {"query": query, "code": code}
        if self._threshold is not None:
            payload["threshold"] = self._threshold
        if self._always_keep_first_frags is not None:
            payload["always_keep_first_frags"] = self._always_keep_first_frags
        if self._chunk_overlap_tokens is not None:
            payload["chunk_overlap_tokens"] = self._chunk_overlap_tokens
        return payload

    def _apply_ast_rebuild(
        self,
        inp: ReduceInput,
        raw: ReduceResult,
    ) -> ReduceResult:
        """Round up kept_frags to complete statements; return final text-only result."""
        if not self._ast_protect_enabled or not raw.kept_frags:
            return ReduceResult(content=raw.content, metadata=dict(raw.metadata))

        commands_raw = inp.config.get("commands")
        commands = commands_raw if isinstance(commands_raw, list) else []
        pruner_result = reduce_result_to_pruner_result(raw)
        rebuilt = rebuild_python_from_pruned(
            inp.content,
            raw.content,
            self._ast_settings,
            focus_question=inp.query,
            commands=commands,
            focus_strategy_name="explore_tool",
            pruner_result=pruner_result,
        )
        if rebuilt.text is not None:
            logger.info(
                "swe_pruner ast_rebuild leaves=%d kept=%d chars_before=%d chars_after=%d",
                rebuilt.leaf_count,
                rebuilt.kept_count,
                len(inp.content),
                len(rebuilt.text),
            )
            return ReduceResult(
                content=rebuilt.text,
                metadata={**raw.metadata, "ast_rebuild": True},
            )

        use_pruned = self._ast_settings.rebuild_fallback == "pruned"
        fallback = raw.content if use_pruned else inp.content
        logger.info(
            "swe_pruner ast_rebuild fallback_%s reason=%s",
            "pruned" if use_pruned else "original",
            rebuilt.skip_reason,
        )
        return ReduceResult(
            content=fallback,
            metadata={**raw.metadata, "ast_rebuild": False, "ast_skip_reason": rebuilt.skip_reason},
        )

    async def reduce(self, inp: ReduceInput) -> ReduceResult | None:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        payload = self._build_payload(inp.content, inp.query)
        for key in ("threshold", "always_keep_first_frags", "chunk_overlap_tokens"):
            if key in inp.config and inp.config[key] is not None:
                payload[key] = inp.config[key]

        try:
            logger.info(
                "explore_pruner swe_pruner_call query=%r chars=%d url=%s",
                inp.query,
                len(inp.content),
                self._url,
            )
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(self._url, json=payload, headers=headers)
            if resp.status_code >= 400:
                logger.warning(
                    "swe_pruner HTTP %s url=%s body_len=%d",
                    resp.status_code,
                    self._url,
                    len(resp.content),
                )
                return None
            data = resp.json()
            err = data.get("error_msg")
            if err:
                logger.warning("swe_pruner error_msg=%s", err)
                return None
            raw = parse_swe_pruner_response(data)
            if raw is None:
                logger.warning("swe_pruner empty/invalid pruned_code")
                return None
            logger.info(
                "swe_pruner ok query=%r kept_frags=%d token_scores=%d",
                inp.query,
                len(raw.kept_frags),
                len(raw.token_scores),
            )
            return self._apply_ast_rebuild(inp, raw)
        except Exception:
            logger.exception("swe_pruner request failed url=%s", self._url)
            return None
