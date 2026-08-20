"""Adapter for the official JailBreakV-28K CSV manifest."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
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


def _optional_bool(record: Mapping[str, Any], key: str) -> bool | None:
    value = optional_text(record, key)
    if value is None:
        return None
    normalized = value.casefold()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise AdapterError(f"JailBreakV-28K field {key!r} is not a boolean: {value!r}")


class JailBreakV28KAdapter(DatasetAdapter):
    """Convert JailBreakV-28K text-image jailbreak pairs without inference."""

    name = "jailbreakv-28k"

    def convert_file(
        self,
        source_file: Path,
        context: AdapterContext,
    ) -> Iterator[CanonicalMultimodalSample]:
        if source_file.suffix.casefold() != ".csv":
            raise AdapterError("JailBreakV-28K adapter requires the official CSV manifest")

        subset = context.subset or source_file.stem
        for row_number, record in enumerate(iter_mapping_records(source_file), start=1):
            original_id = require_text(record, "id")
            jailbreak_query = require_text(record, "jailbreak_query")
            underlying_query = require_text(record, "redteam_query")
            attack_format = require_text(record, "format")
            policy = require_text(record, "policy")
            source_name = require_text(record, "from")
            media = build_media_ref(
                require_text(record, "image_path"),
                MediaType.IMAGE,
                media_root=context.media_root,
                verify=context.verify_media,
                hash_media=context.hash_media,
            )
            yield CanonicalMultimodalSample(
                sample_id=(
                    f"jailbreakv-28k:{id_fragment(subset)}:{id_fragment(original_id)}"
                ),
                parts=(MediaPart(media=media), TextPart(text=jailbreak_query)),
                source=SourceRecord(
                    dataset_name="JailBreakV-28K",
                    dataset_version=context.dataset_version,
                    split=context.split,
                    original_id=original_id,
                    original_labels={
                        "policy": policy,
                        "attack_format": attack_format,
                        "source": source_name,
                        "subset": subset,
                    },
                    source_url="https://huggingface.co/datasets/EddyLuo/JailBreakV_28K",
                ),
                request_context=RequestContext(
                    intent=RequestIntent.HARMFUL,
                    risk_category=policy,
                    attack_type=attack_format,
                ),
                metadata={
                    "underlying_harmful_query": underlying_query,
                    "source": source_name,
                    "selected_mini": _optional_bool(record, "selected_mini"),
                    "transfer_from_llm": _optional_bool(record, "transfer_from_llm"),
                    "subset": subset,
                    "row_number": row_number,
                    "intended_use": "research safety evaluation only",
                },
            )
