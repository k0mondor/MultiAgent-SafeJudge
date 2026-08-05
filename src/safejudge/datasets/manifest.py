"""Reproducibility manifest emitted beside each canonical JSONL file."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field

from safejudge.contracts.base import ContractModel
from safejudge.core.time import utc_now


class FileDigest(ContractModel):
    name: str
    sha256: str


class DatasetManifest(ContractModel):
    manifest_version: Literal["1.0"] = "1.0"
    canonical_schema_version: Literal["1.0"] = "1.0"
    adapter: str
    source: FileDigest
    output: FileDigest
    sample_count: int = Field(ge=0)
    dataset_names: tuple[str, ...]
    modality_counts: dict[str, int]
    request_intent_counts: dict[str, int]
    created_at: datetime = Field(default_factory=utc_now)

