"""Shared contracts for files persisted beside reproducible batch artifacts."""

from __future__ import annotations

from safejudge.contracts.base import ContractModel
from safejudge.contracts.dataset import Sha256


class FileDigest(ContractModel):
    name: str
    sha256: Sha256
