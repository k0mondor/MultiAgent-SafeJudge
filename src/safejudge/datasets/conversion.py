"""Canonical JSONL conversion with validation, deduplication, and manifests."""

from __future__ import annotations

import os
import tempfile
from collections import Counter
from dataclasses import dataclass
from itertools import islice
from pathlib import Path

from safejudge.contracts.dataset import MediaPart
from safejudge.contracts.files import FileDigest
from safejudge.core.errors import AdapterError
from safejudge.core.files import sha256_file
from safejudge.datasets.base import AdapterContext
from safejudge.datasets.manifest import DatasetManifest
from safejudge.datasets.registry import create_adapter


def file_sha256(path: Path) -> str:
    return sha256_file(path)


@dataclass(frozen=True, slots=True)
class ConversionResult:
    output_path: Path
    manifest_path: Path
    manifest: DatasetManifest


def convert_dataset(
    *,
    adapter_name: str,
    source_file: Path,
    output_path: Path,
    context: AdapterContext,
    overwrite: bool = False,
    limit: int | None = None,
) -> ConversionResult:
    source_file = source_file.resolve()
    output_path = output_path.resolve()
    manifest_path = output_path.with_suffix(f"{output_path.suffix}.manifest.json")

    if not source_file.is_file():
        raise AdapterError(f"source metadata file does not exist: {source_file}")
    if limit is not None and limit < 1:
        raise AdapterError("conversion limit must be at least 1")
    if not overwrite and (output_path.exists() or manifest_path.exists()):
        raise AdapterError(
            f"output or manifest already exists: {output_path}; pass --overwrite to replace it"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    adapter = create_adapter(adapter_name)
    sample_ids: set[str] = set()
    dataset_names: set[str] = set()
    modality_counts: Counter[str] = Counter()
    intent_counts: Counter[str] = Counter()
    sample_count = 0

    temporary_output: Path | None = None
    temporary_manifest: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            dir=output_path.parent,
            delete=False,
        ) as stream:
            temporary_output = Path(stream.name)
            samples = adapter.convert_file(source_file, context)
            for sample in islice(samples, limit):
                if sample.sample_id in sample_ids:
                    raise AdapterError(f"duplicate canonical sample_id: {sample.sample_id}")
                sample_ids.add(sample.sample_id)
                dataset_names.add(sample.source.dataset_name)
                intent_counts[sample.request_context.intent.value] += 1
                for part in sample.parts:
                    if isinstance(part, MediaPart):
                        modality_counts[part.media.media_type.value] += 1
                stream.write(sample.model_dump_json())
                stream.write("\n")
                sample_count += 1

        output_digest = file_sha256(temporary_output)
        manifest = DatasetManifest(
            adapter=adapter_name,
            source=FileDigest(name=source_file.name, sha256=file_sha256(source_file)),
            output=FileDigest(name=output_path.name, sha256=output_digest),
            sample_count=sample_count,
            dataset_names=tuple(sorted(dataset_names)),
            modality_counts=dict(sorted(modality_counts.items())),
            request_intent_counts=dict(sorted(intent_counts.items())),
        )
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{manifest_path.name}.",
            suffix=".tmp",
            dir=manifest_path.parent,
            delete=False,
        ) as stream:
            temporary_manifest = Path(stream.name)
            stream.write(manifest.model_dump_json(indent=2))
            stream.write("\n")

        os.replace(temporary_output, output_path)
        temporary_output = None
        os.replace(temporary_manifest, manifest_path)
        temporary_manifest = None
        return ConversionResult(
            output_path=output_path,
            manifest_path=manifest_path,
            manifest=manifest,
        )
    finally:
        for temporary_path in (temporary_output, temporary_manifest):
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
