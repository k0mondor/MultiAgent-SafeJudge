"""Framework-independent dataset adapter interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from safejudge.contracts.dataset import CanonicalMultimodalSample, DatasetSplit


@dataclass(frozen=True, slots=True)
class AdapterContext:
    media_root: Path
    dataset_version: str = "main"
    split: DatasetSplit = DatasetSplit.TEST
    subset: str | None = None
    verify_media: bool = True
    hash_media: bool = False
    variants: tuple[str, ...] = ()


class DatasetAdapter(ABC):
    """Convert one official dataset format without downloading it."""

    name: str

    @abstractmethod
    def convert_file(
        self,
        source_file: Path,
        context: AdapterContext,
    ) -> Iterator[CanonicalMultimodalSample]:
        """Yield canonical samples or raise a classified adapter error."""
