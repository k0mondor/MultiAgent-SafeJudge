"""Adapter for MOSSBench oversensitivity metadata (JSON, JSONL, or CSV)."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from contextlib import suppress
from pathlib import Path
from typing import Any

from safejudge.contracts.dataset import (
    CanonicalMultimodalSample,
    MediaPart,
    MediaType,
    RequestContext,
    RequestIntent,
    SourceRecord,
    TextPart,
)
from safejudge.core.errors import AdapterError
from safejudge.datasets.base import AdapterContext, DatasetAdapter
from safejudge.datasets.media import build_media_ref
from safejudge.datasets.readers import iter_mapping_records
from safejudge.datasets.utils import id_fragment, optional_text, require_text

_METADATA_KEYS = ("over", "human", "child", "syn", "ocr", "harm")


def _image_path(record: Mapping[str, Any]) -> str:
    value = record.get("image")
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, Mapping):
        for key in ("path", "file_name", "filename"):
            nested = value.get(key)
            if isinstance(nested, str) and nested.strip():
                return nested.strip()
    for key in ("file_name", "image_path", "path"):
        candidate = record.get(key)
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    filename = getattr(value, "filename", None)
    if isinstance(filename, str) and filename.strip():
        return filename.strip()
    raise AdapterError(
        "MOSSBench image path is missing; export the Hugging Face image column "
        "with file_name/path metadata instead of an in-memory image only"
    )


def _metadata(record: Mapping[str, Any]) -> dict[str, Any]:
    nested = record.get("metadata")
    if isinstance(nested, Mapping):
        return {str(key): value for key, value in nested.items()}

    metadata: dict[str, Any] = {}
    for key in _METADATA_KEYS:
        value = record.get(key, record.get(f"metadata_{key}"))
        if value in (None, ""):
            continue
        if isinstance(value, str):
            with suppress(json.JSONDecodeError):
                value = json.loads(value)
        metadata[key] = value
    return metadata


class MOSSBenchAdapter(DatasetAdapter):
    name = "mossbench"

    def convert_file(
        self,
        source_file: Path,
        context: AdapterContext,
    ) -> Iterator[CanonicalMultimodalSample]:
        subset = context.subset or source_file.parent.name
        for row_number, record in enumerate(iter_mapping_records(source_file), start=1):
            problem_id = record.get("pid", record.get("id", row_number))
            question = require_text(record, "question")
            metadata = _metadata(record)
            oversensitivity_type = metadata.get("over")
            harm_type = metadata.get("harm")
            media = build_media_ref(
                _image_path(record),
                MediaType.IMAGE,
                media_root=context.media_root,
                verify=context.verify_media,
                hash_media=context.hash_media,
            )
            yield CanonicalMultimodalSample(
                sample_id=f"mossbench:{id_fragment(subset)}:{id_fragment(problem_id)}",
                parts=(MediaPart(media=media), TextPart(text=question)),
                source=SourceRecord(
                    dataset_name="MOSSBench",
                    dataset_version=context.dataset_version,
                    split=context.split,
                    original_id=str(problem_id),
                    original_labels={**metadata, "subset": subset},
                    license="CC-BY-SA-4.0",
                    source_url="https://github.com/xirui-li/MOSSBench",
                ),
                request_context=RequestContext(
                    intent=RequestIntent.BENIGN,
                    risk_category=f"harm:{harm_type}" if harm_type is not None else None,
                    attack_type=str(oversensitivity_type) if oversensitivity_type else None,
                ),
                metadata={
                    "short_description": optional_text(
                        record, "short description", "short_description"
                    ),
                    "intended_use": "test-only; training prohibited by dataset terms",
                    "subset": subset,
                },
            )
