"""Versioned contracts for blind and benchmark-assisted request grounding."""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from safejudge.contracts.base import ContractModel
from safejudge.contracts.dataset import MediaType


class GroundingMode(StrEnum):
    BLIND = "blind"
    BENCHMARK_ASSISTED = "benchmark_assisted"


class GroundingStatus(StrEnum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    UNAVAILABLE = "unavailable"


class ObservationModality(StrEnum):
    IMAGE_OCR = "image_ocr"
    IMAGE_VLM = "image_vlm"
    AUDIO_ASR = "audio_asr"
    AUDIO_MODEL = "audio_model"
    VIDEO_OCR = "video_ocr"
    VIDEO_ASR = "video_asr"
    VIDEO_VLM = "video_vlm"
    BENCHMARK_LABEL = "benchmark_label"


class RawGroundingObservation(ContractModel):
    modality: ObservationModality
    text: str = Field(min_length=1, max_length=2_000)
    confidence: float = Field(ge=0, le=1)
    media_type: MediaType | None = None
    media_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    page_or_frame: int | None = Field(default=None, ge=0)
    bbox: tuple[float, float, float, float] | None = None
    start_ms: int | None = Field(default=None, ge=0)
    end_ms: int | None = Field(default=None, ge=0)
    tool_id: str = Field(min_length=1)
    tool_version: str = Field(min_length=1)
    call_id: str | None = None
    provider_response_id: str | None = None

    @model_validator(mode="after")
    def location_is_valid(self) -> RawGroundingObservation:
        if self.bbox is not None and any(value < 0 or value > 1 for value in self.bbox):
            raise ValueError("grounding bbox values must be normalized to 0..1")
        if self.end_ms is not None and self.start_ms is None:
            raise ValueError("grounding end_ms requires start_ms")
        if (
            self.start_ms is not None
            and self.end_ms is not None
            and self.end_ms <= self.start_ms
        ):
            raise ValueError("grounding end_ms must be greater than start_ms")
        return self


class GroundingObservation(RawGroundingObservation):
    evidence_id: str = Field(pattern=r"^G[0-9]{3}$")
    observation_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def observation_hash_matches(self) -> GroundingObservation:
        payload = self.model_dump(
            mode="json", exclude={"evidence_id", "observation_sha256"}
        )
        if self.observation_sha256 != _canonical_hash(payload):
            raise ValueError("grounding observation hash does not match its content")
        return self

    @classmethod
    def from_raw(
        cls,
        raw: RawGroundingObservation,
        *,
        evidence_id: str,
    ) -> GroundingObservation:
        payload = raw.model_dump(mode="json")
        return cls(
            **payload,
            evidence_id=evidence_id,
            observation_sha256=_canonical_hash(payload),
        )


class GroundingArtifact(ContractModel):
    schema_version: Literal["1.0"] = "1.0"
    artifact_id: str = Field(pattern=r"^grounding_[a-f0-9]{16}$")
    sample_id: str = Field(min_length=1)
    mode: GroundingMode
    status: GroundingStatus
    pipeline_id: str = Field(min_length=1)
    pipeline_version: str = Field(min_length=1)
    pipeline_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    observations: tuple[GroundingObservation, ...] = ()
    error_codes: tuple[str, ...] = ()
    artifact_hash: str = Field(pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def artifact_is_consistent(self) -> GroundingArtifact:
        payload = self.model_dump(mode="json", exclude={"artifact_id", "artifact_hash"})
        expected = _canonical_hash(payload)
        if self.artifact_hash != expected:
            raise ValueError("grounding artifact hash does not match its content")
        if self.artifact_id != f"grounding_{expected[:16]}":
            raise ValueError("grounding artifact_id does not match artifact_hash")
        ids = [item.evidence_id for item in self.observations]
        if len(ids) != len(set(ids)):
            raise ValueError("grounding evidence IDs must be unique")
        if self.status is GroundingStatus.COMPLETE and self.error_codes:
            raise ValueError("complete grounding cannot contain error codes")
        return self

    @classmethod
    def create(
        cls,
        *,
        sample_id: str,
        mode: GroundingMode,
        status: GroundingStatus,
        pipeline_id: str,
        pipeline_version: str,
        pipeline_hash: str,
        observations: tuple[GroundingObservation, ...] = (),
        error_codes: tuple[str, ...] = (),
    ) -> GroundingArtifact:
        payload = {
            "schema_version": "1.0",
            "sample_id": sample_id,
            "mode": mode,
            "status": status,
            "pipeline_id": pipeline_id,
            "pipeline_version": pipeline_version,
            "pipeline_hash": pipeline_hash,
            "observations": observations,
            "error_codes": error_codes,
        }
        encoded = {
            "schema_version": "1.0",
            "sample_id": sample_id,
            "mode": mode.value,
            "status": status.value,
            "pipeline_id": pipeline_id,
            "pipeline_version": pipeline_version,
            "pipeline_hash": pipeline_hash,
            "observations": [item.model_dump(mode="json") for item in observations],
            "error_codes": list(error_codes),
        }
        artifact_hash = _canonical_hash(encoded)
        return cls(
            **payload,
            artifact_id=f"grounding_{artifact_hash[:16]}",
            artifact_hash=artifact_hash,
        )


def _canonical_hash(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
