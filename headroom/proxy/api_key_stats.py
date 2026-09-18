"""Privacy-safe API-key attribution for proxy compression statistics.

Only a truncated SHA-256 fingerprint is carried beyond request ingress.  The
raw credential must never enter request logs, metrics state, or dashboard
payloads.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

API_KEY_ID_TAG = "_headroom_api_key_id"
API_KEY_ID_PREFIX = "key_"
API_KEY_DIGEST_LENGTH = 16

# Prefer provider-native key headers when more than one credential is present.
# Authorization remains the OpenAI/Codex path and is therefore still covered.
_CREDENTIAL_HEADERS = (
    "x-api-key",
    "api-key",
    "x-goog-api-key",
    "authorization",
)


def _credential_value(header_name: str, value: Any) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    if header_name == "authorization":
        parts = text.split(None, 1)
        if parts[0].lower() in {"bearer", "basic"}:
            if len(parts) != 2:
                return None
            text = parts[1].strip()
    return text or None


def fingerprint_api_key(headers: Mapping[str, Any] | Any) -> str | None:
    """Return a stable, non-reversible identifier for an inbound credential.

    Header matching is case-insensitive.  The returned value contains no raw
    key prefix or suffix, so it is safe to persist and display.  ``None`` means
    the request did not carry a supported API-key credential.
    """

    try:
        normalized = {str(name).lower(): value for name, value in headers.items()}
    except (AttributeError, TypeError):
        return None

    for header_name in _CREDENTIAL_HEADERS:
        credential = _credential_value(header_name, normalized.get(header_name))
        if credential is None:
            continue
        digest = hashlib.sha256(credential.encode("utf-8")).hexdigest()
        return API_KEY_ID_PREFIX + digest[:API_KEY_DIGEST_LENGTH]
    return None


def api_key_id_from_tags(tags: Mapping[str, Any] | None) -> str | None:
    """Read the internal key identifier carried through ``RequestOutcome``."""

    value = (tags or {}).get(API_KEY_ID_TAG)
    if not isinstance(value, str) or not value.startswith(API_KEY_ID_PREFIX):
        return None
    digest = value[len(API_KEY_ID_PREFIX) :]
    if len(digest) != API_KEY_DIGEST_LENGTH:
        return None
    try:
        int(digest, 16)
    except ValueError:
        return None
    return value
