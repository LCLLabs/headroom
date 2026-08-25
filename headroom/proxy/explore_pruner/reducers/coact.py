"""CoACT HTTP backend as a ContextReducer (semantic prune; no AST rebuild)."""

from __future__ import annotations

import logging
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx

from headroom.proxy.explore_pruner.types import ReduceInput, ReduceResult, parse_coact_response

logger = logging.getLogger(__name__)

_BLOCKED_HOSTS = frozenset(
    {
        "169.254.169.254",
        "metadata.google.internal",
        "metadata.goog",
    }
)

COACT_NAME = "coact"
COACT_DEFAULT_API_BASE = "http://127.0.0.1:8002"
COACT_DEFAULT_TIMEOUT_SECONDS = 120.0


class CoactReducer:
    """POST {query, code, goal?, tool_call?} → pruned text from CoACT."""

    name = COACT_NAME

    def __init__(
        self,
        api_base: str,
        path: str = "/prune",
        api_key: str | None = None,
        timeout_seconds: float = COACT_DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self._url = urljoin(api_base.rstrip("/") + "/", path.lstrip("/"))
        self._api_key = api_key
        self._timeout = timeout_seconds
        self._validate_url()

    def _validate_url(self) -> None:
        parsed = urlparse(self._url)
        host = (parsed.hostname or "").lower()
        if host in _BLOCKED_HOSTS:
            raise ValueError(f"pruner api_base host blocked: {host}")

    def _build_payload(self, inp: ReduceInput) -> dict[str, Any]:
        payload: dict[str, Any] = {"query": inp.query, "code": inp.content}
        goal = inp.config.get("goal")
        if isinstance(goal, str) and goal.strip():
            payload["goal"] = goal.strip()
        commands = inp.config.get("commands")
        if isinstance(commands, list) and commands:
            payload["tool_call"] = str(commands[0])
        else:
            tool_call = inp.config.get("tool_call")
            if isinstance(tool_call, str) and tool_call.strip():
                payload["tool_call"] = tool_call.strip()
        return payload

    async def reduce(self, inp: ReduceInput) -> ReduceResult | None:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        payload = self._build_payload(inp)
        try:
            logger.info(
                "explore_pruner coact_call query=%r chars=%d url=%s",
                inp.query,
                len(inp.content),
                self._url,
            )
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(self._url, json=payload, headers=headers)
            if resp.status_code >= 400:
                logger.warning(
                    "coact HTTP %s url=%s body_len=%d",
                    resp.status_code,
                    self._url,
                    len(resp.content),
                )
                return None
            data = resp.json()
            if not isinstance(data, dict):
                logger.warning("coact non-object JSON response")
                return None
            err = data.get("error_msg")
            if err:
                logger.warning("coact error_msg=%s", err)
                return None
            raw = parse_coact_response(data)
            if raw is None:
                logger.warning("coact empty/invalid pruned_code")
                return None
            logger.info(
                "coact ok query=%r type=%s kept_frags=%d",
                inp.query,
                raw.metadata.get("compression_type"),
                len(raw.kept_frags),
            )
            return raw
        except Exception:
            logger.exception("coact request failed url=%s", self._url)
            return None
