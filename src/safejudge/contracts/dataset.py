"""Canonical input records independent of any benchmark or model provider."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, JsonValue, StringConstraints, model_validator

from safejudge.contracts.base import ContractModel

SCHEMA_VERSION: Literal["1.0"] = "1.0"
NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[a-fA-F0-9]{64}$")]


class DatasetSplit(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    UNKNOWN = "unknown"


class MediaType(StrEnum):
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"


class RequestIntent(StrEnum):
    """Request-side context; this is never the target response score."""

    BENIGN = "benign"
    HARMFUL = "harmful"
    AMBIGUOUS = "ambiguous"
    UNKNOWN = "unknown"


class MediaRef(ContractModel):
    media_type: MediaType
    uri: NonEmptyString
    mime_type: NonEmptyString
    sha256: Sha256 | None = None
    size_bytes: int | None = Field(default=None, ge=0)
    duration_ms: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def duration_only_for_temporal_media(self) -> MediaRef:
        if self.media_type is MediaType.IMAGE and self.duration_ms is not None:
            raise ValueError("duration_ms is only valid for audio or video")
        return self


class TextPart(ContractModel):
    kind: Literal["text"] = "text"
    text: NonEmptyString


class MediaPart(ContractModel):
    kind: Literal["media"] = "media"
    media: MediaRef


PromptPart = Annotated[TextPart | MediaPart, Field(discriminator="kind")]


class SourceRecord(ContractModel):
    dataset_name: NonEmptyString
    dataset_version: NonEmptyString
    split: DatasetSplit = DatasetSplit.UNKNOWN
    original_id: NonEmptyString
    original_labels: dict[str, JsonValue] = Field(default_factory=dict)
    license: str | None = None
    source_url: str | None = None


class RequestContext(ContractModel):
    """Optional source-side labels used to interpret a target response."""

    intent: RequestIntent = RequestIntent.UNKNOWN
    risk_category: str | None = None
    attack_type: str | None = None


class CanonicalMultimodalSample(ContractModel):
    """A benchmark request before any target model has answered it."""

    schema_version: Literal["1.0"] = SCHEMA_VERSION
    sample_id: NonEmptyString
    parts: tuple[PromptPart, ...] = Field(min_length=1)
    source: SourceRecord
    request_context: RequestContext = Field(default_factory=RequestContext)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
