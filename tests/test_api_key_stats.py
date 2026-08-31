"""Privacy and protocol coverage for per-API-key attribution."""

from __future__ import annotations

from headroom.proxy.api_key_stats import API_KEY_ID_TAG, fingerprint_api_key
from headroom.proxy.helpers import extract_tags
from headroom.proxy.savings_attribution import public_tags


def test_fingerprint_is_stable_across_supported_key_headers_without_leaking_secret() -> None:
    secret = "company-key-super-secret"
    expected = fingerprint_api_key({"Authorization": f"Bearer {secret}"})

    assert expected is not None
    assert expected.startswith("key_")
    assert len(expected) == 20
    assert secret not in expected
    assert fingerprint_api_key({"x-api-key": secret}) == expected
    assert fingerprint_api_key({"API-Key": secret}) == expected
    assert fingerprint_api_key({"x-goog-api-key": secret}) == expected


def test_extract_tags_carries_only_internal_fingerprint_and_public_tags_strip_it() -> None:
    secret = "sk-ant-never-persist-this"
    tags = extract_tags(
        {
            "Authorization": f"Bearer {secret}",
            "X-Headroom-Project": "storage",
        }
    )

    assert tags["project"] == "storage"
    assert tags[API_KEY_ID_TAG].startswith("key_")
    assert secret not in repr(tags)
    assert public_tags(tags) == {"project": "storage"}


def test_missing_or_empty_credentials_are_not_attributed() -> None:
    assert fingerprint_api_key({}) is None
    assert fingerprint_api_key({"Authorization": "Bearer "}) is None
    assert API_KEY_ID_TAG not in extract_tags({"User-Agent": "codex-cli"})
