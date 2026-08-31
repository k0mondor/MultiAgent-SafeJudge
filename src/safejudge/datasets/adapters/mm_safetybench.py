"""Adapter for the official MM-SafetyBench processed question files."""

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
from safejudge.datasets.readers import read_json_object
from safejudge.datasets.utils import id_fragment, optional_text, require_text

_VARIANTS = {
    "sd": ("SD", "Rephrased Question(SD)"),
    "sd_typo": ("SD_TYPO", "Rephrased Question"),
    "typo": ("TYPO", "Rephrased Question"),
}


class MMSafetyBenchAdapter(DatasetAdapter):
    name = "mm-safetybench"

    def convert_file(
        self,
        source_file: Path,
        context: AdapterContext,
    ) -> Iterator[CanonicalMultimodalSample]:
        scenario = source_file.stem
        variants = context.variants or tuple(_VARIANTS)
        unsupported = sorted(set(variants) - _VARIANTS.keys())
        if unsupported:
            raise AdapterError(f"unsupported MM-SafetyBench variants: {unsupported}")

        for question_id, value in read_json_object(source_file).items():
            if not isinstance(value, Mapping):
                raise AdapterError(f"question {question_id!r} in {source_file} is not an object")
            record: Mapping[str, Any] = value
            for variant in variants:
                image_directory, question_field = _VARIANTS[variant]
                question = require_text(record, question_field)
                raw_media_uri = f"{scenario}/{image_directory}/{question_id}.jpg"
                media = build_media_ref(
                    raw_media_uri,
                    MediaType.IMAGE,
                    media_root=context.media_root,
                    verify=context.verify_media,
                    hash_media=context.hash_media,
                )

                source_id = id_fragment(question_id)
                yield CanonicalMultimodalSample(
                    sample_id=f"mm-safetybench:{id_fragment(scenario)}:{source_id}:{variant}",
                    parts=(MediaPart(media=media), TextPart(text=question)),
                    source=SourceRecord(
                        dataset_name="MM-SafetyBench",
                        dataset_version=context.dataset_version,
                        split=context.split,
                        original_id=f"{question_id}:{variant}",
                        original_labels={
                            "scenario": scenario,
                            "variant": variant,
                        },
                        license="CC-BY-NC-4.0",
                        source_url="https://github.com/isXinLiu/MM-SafetyBench",
                    ),
                    request_context=RequestContext(
                        intent=RequestIntent.HARMFUL,
                        risk_category=scenario,
                        attack_type="query_relevant_image",
                    ),
                    metadata={
                        "variant": variant,
                        "original_question": optional_text(record, "Question"),
                        "changed_question": optional_text(record, "Changed Question"),
                        "key_phrase": optional_text(record, "Key Phrase"),
                        "phrase_type": optional_text(record, "Phrase Type"),
                    },
                )
