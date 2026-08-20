"""Explicit adapter registry; importing a plugin cannot silently replace a built-in."""

from __future__ import annotations

from collections.abc import Callable

from safejudge.core.errors import AdapterError
from safejudge.datasets.adapters import (
    JailBreakV28KAdapter,
    MMSafetyBenchAdapter,
    MOSSBenchAdapter,
    OmniSafetyBenchAdapter,
)
from safejudge.datasets.base import DatasetAdapter

AdapterFactory = Callable[[], DatasetAdapter]

_BUILT_INS: dict[str, AdapterFactory] = {
    JailBreakV28KAdapter.name: JailBreakV28KAdapter,
    MMSafetyBenchAdapter.name: MMSafetyBenchAdapter,
    MOSSBenchAdapter.name: MOSSBenchAdapter,
    OmniSafetyBenchAdapter.name: OmniSafetyBenchAdapter,
}


def list_adapters() -> tuple[str, ...]:
    return tuple(sorted(_BUILT_INS))


def create_adapter(name: str) -> DatasetAdapter:
    try:
        return _BUILT_INS[name]()
    except KeyError as error:
        choices = ", ".join(list_adapters())
        raise AdapterError(f"unknown dataset adapter {name!r}; choose one of: {choices}") from error

