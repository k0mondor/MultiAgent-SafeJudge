"""Provider-neutral grounding pipeline with explicit blind/assisted behavior."""

from __future__ import annotations

import asyncio
import hashlib
import json
from typing import Protocol

from safejudge.contracts.dataset import (
    CanonicalMultimodalSample,
    MediaPart,
    MediaRef,
    MediaType,
)
from safejudge.contracts.model import InvocationContext
from safejudge.grounding.contracts import (
    GroundingArtifact,
    GroundingMode,
    GroundingObservation,
    GroundingStatus,
    ObservationModality,
    RawGroundingObservation,
)


class GroundingTool(Protocol):
    @property
    def tool_id(self) -> str: ...

    @property
    def tool_version(self) -> str: ...

    @property
    def supported_media_types(self) -> frozenset[MediaType]: ...

    async def observe(
        self,
        *,
        sample_id: str,
        media: MediaRef,
        context: InvocationContext,
    ) -> tuple[RawGroundingObservation, ...]: ...


class GroundingPipeline:
    def __init__(
        self,
        *,
        mode: GroundingMode,
        tools: tuple[GroundingTool, ...] = (),
        pipeline_id: str = "safejudge-request-grounding",
        pipeline_version: str = "1.0",
    ) -> None:
        self.mode = mode
        self.tools = tools
        self.pipeline_id = pipeline_id
        self.pipeline_version = pipeline_version

    @property
    def fingerprint(self) -> str:
        payload = {
            "pipeline_id": self.pipeline_id,
            "pipeline_version": self.pipeline_version,
            "mode": self.mode.value,
            "tools": [
                {
                    "tool_id": tool.tool_id,
                    "tool_version": tool.tool_version,
                    "media_types": sorted(item.value for item in tool.supported_media_types),
                }
                for tool in self.tools
            ],
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    async def run(
        self,
        sample: CanonicalMultimodalSample,
        *,
        context: InvocationContext,
    ) -> GroundingArtifact:
        if self.mode is GroundingMode.BENCHMARK_ASSISTED:
            return self._benchmark_artifact(sample)
        media = tuple(part.media for part in sample.parts if isinstance(part, MediaPart))
        if not media:
            return GroundingArtifact.create(
                sample_id=sample.sample_id,
                mode=self.mode,
                status=GroundingStatus.COMPLETE,
                pipeline_id=self.pipeline_id,
                pipeline_version=self.pipeline_version,
                pipeline_hash=self.fingerprint,
            )
        jobs: list[tuple[MediaRef, GroundingTool]] = []
        missing_media_types: set[MediaType] = set()
        for item in media:
            matching = tuple(
                tool for tool in self.tools if item.media_type in tool.supported_media_types
            )
            if not matching:
                missing_media_types.add(item.media_type)
            jobs.extend((item, tool) for tool in matching)
        results = await asyncio.gather(
            *(
                tool.observe(sample_id=sample.sample_id, media=item, context=context)
                for item, tool in jobs
            ),
            return_exceptions=True,
        )
        raw: list[RawGroundingObservation] = []
        error_codes = {f"NO_TOOL_FOR_{item.value.upper()}" for item in missing_media_types}
        for (media_item, _tool), result in zip(jobs, results, strict=True):
            if isinstance(result, BaseException):
                error_codes.add(f"GROUNDING_TOOL_ERROR_{type(result).__name__.upper()}")
            else:
                raw.extend(result)
                if not result:
                    identity = (media_item.sha256 or media_item.media_type.value)[:12].upper()
                    error_codes.add(f"NO_OBSERVATION_FOR_{identity}")
        observations = tuple(
            GroundingObservation.from_raw(item, evidence_id=f"G{index:03d}")
            for index, item in enumerate(raw)
        )
        if not observations:
            status = GroundingStatus.UNAVAILABLE
            if not error_codes:
                error_codes.add("NO_GROUNDING_OBSERVATIONS")
        elif error_codes:
            status = GroundingStatus.PARTIAL
        else:
            status = GroundingStatus.COMPLETE
        return GroundingArtifact.create(
            sample_id=sample.sample_id,
            mode=self.mode,
            status=status,
            pipeline_id=self.pipeline_id,
            pipeline_version=self.pipeline_version,
            pipeline_hash=self.fingerprint,
            observations=observations,
            error_codes=tuple(sorted(error_codes)),
        )

    def _benchmark_artifact(
        self,
        sample: CanonicalMultimodalSample,
    ) -> GroundingArtifact:
        context = sample.request_context
        text = json.dumps(
            {
                "declared_intent": context.intent.value,
                "declared_risk_category": context.risk_category,
                "declared_attack_type": context.attack_type,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        raw = RawGroundingObservation(
            modality=ObservationModality.BENCHMARK_LABEL,
            text=text,
            confidence=1,
            tool_id="benchmark-adapter-context",
            tool_version="1.0",
        )
        observation = GroundingObservation.from_raw(raw, evidence_id="G000")
        return GroundingArtifact.create(
            sample_id=sample.sample_id,
            mode=self.mode,
            status=GroundingStatus.COMPLETE,
            pipeline_id=self.pipeline_id,
            pipeline_version=self.pipeline_version,
            pipeline_hash=self.fingerprint,
            observations=(observation,),
        )
