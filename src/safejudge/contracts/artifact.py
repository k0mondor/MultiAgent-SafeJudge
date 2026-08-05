"""References to immutable raw artifacts stored outside workflow state."""

from __future__ import annotations

from pydantic import Field

from safejudge.contracts.base import ContractModel
from safejudge.contracts.dataset import NonEmptyString, Sha256


class ArtifactRef(ContractModel):
    uri: NonEmptyString
    sha256: Sha256
    content_type: NonEmptyString
    size_bytes: int = Field(ge=0)
