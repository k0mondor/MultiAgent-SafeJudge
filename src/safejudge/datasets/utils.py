"""Validation helpers for untrusted dataset metadata."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from safejudge.core.errors import AdapterError


def require_text(record: Mapping[str, Any], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise AdapterError(f"required text field {key!r} is missing or empty")
    return value.strip()


def optional_text(record: Mapping[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def id_fragment(value: object) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9._-]+", "-", str(value).strip()).strip("-")
    if not normalized:
        raise AdapterError(f"cannot create stable id from {value!r}")
    return normalized

