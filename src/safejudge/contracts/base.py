"""Shared validation policy for persisted contracts."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ContractModel(BaseModel):
    """Strict and immutable base for versioned cross-layer data."""

    model_config = ConfigDict(extra="forbid", frozen=True)

