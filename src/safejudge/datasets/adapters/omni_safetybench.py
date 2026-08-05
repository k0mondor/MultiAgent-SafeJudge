"""Adapter for one dual-modal Omni-SafetyBench JSONL subset at a time."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

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

_MEDIA_FIELDS = {
    "image_path": MediaType.IMAGE,
    "audio_path": MediaType.AUDIO,
    "video_path": MediaType.VIDEO,
}


class OmniSafetyBenchAdapter(DatasetAdapter):
    name = "omni-safetybench"

    def convert_file(
        self,
        source_file: Path,
        context: AdapterContext,
    ) -> Iterator[CanonicalMultimodalSample]:
        subset = context.subset or f"{source_file.parent.name}-{source_file.stem}"
        for row_number, record in enumerate(iter_mapping_records(source_file), start=1):
            present_media = [
                (field, media_type)
                for field, media_type in _MEDIA_FIELDS.items()
                if isinstance(record.get(field), str) and str(record[field]).strip()
            ]
            if len(present_media) != 1:
                raise AdapterError(
                    "Omni-SafetyBench phase-one adapter requires exactly one of "
                    f"image_path/audio_path/video_path at row {row_number}; "
                    f"got {len(present_media)}"
                )
            media_field, media_type = present_media[0]
            seed_id = record.get("id", row_number)
            question = require_text(record, "text")
            harmful_category = require_text(record, "harmful_category")
            media = build_media_ref(
                require_text(record, media_field),
                media_type,
                media_root=context.media_root,
                verify=context.verify_media,
                hash_media=context.hash_media,
            )
            yield CanonicalMultimodalSample(
                sample_id=(
                    f"omni-safetybench:{id_fragment(subset)}:{id_fragment(seed_id)}"
                ),
                parts=(MediaPart(media=media), TextPart(text=question)),
                source=SourceRecord(
                    dataset_name="Omni-SafetyBench",
                    dataset_version=context.dataset_version,
                    split=context.split,
                    original_id=str(seed_id),
                    original_labels={
                        "harmful_category": harmful_category,
                        "subset": subset,
                    },
                    license="CC-BY-NC-4.0",
                    source_url="https://huggingface.co/datasets/Leyiii/Omni-SafetyBench",
                ),
                request_context=RequestContext(
                    intent=RequestIntent.HARMFUL,
                    risk_category=harmful_category,
                    attack_type=f"modality_transfer:{media_type.value}-text",
                ),
                metadata={
                    "parallel_seed_id": str(seed_id),
                    "modality_combination": f"{media_type.value}-text",
                    "corresponding_text": optional_text(record, "corresponding_text"),
                    "key_phrase": optional_text(record, "key_phrase"),
                    "subset": subset,
                },
            )
