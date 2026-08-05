"""Conversion and execution boundary for immutable target-model responses."""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import JsonValue

from safejudge.contracts.dataset import (
    CanonicalMultimodalSample,
    MediaPart,
    TextPart,
)
from safejudge.contracts.evaluation import TargetModelRef, TargetResponse
from safejudge.contracts.model import (
    InvocationContext,
    ModelMediaPart,
    ModelRequest,
    ModelRole,
    ModelTextPart,
)
from safejudge.core.errors import UnsupportedModalityError
from safejudge.models.invocation import ModelInvoker


class TargetRunner:
    def __init__(self, invoker: ModelInvoker) -> None:
        self.invoker = invoker

    async def run(
        self,
        sample: CanonicalMultimodalSample,
        *,
        context: InvocationContext,
        parameters: Mapping[str, JsonValue] | None = None,
    ) -> TargetResponse:
        request = self.build_request(sample, parameters=parameters)
        capabilities = self.invoker.provider.capabilities
        if not capabilities.supports(request.modalities):
            actual = "+".join(sorted(item.value for item in request.modalities))
            supported = [
                "+".join(sorted(item.value for item in combination.modalities))
                for combination in capabilities.input_combinations
            ]
            raise UnsupportedModalityError(
                f"model {self.invoker.provider.model_id!r} does not support {actual}; "
                f"supported combinations: {supported}"
            )

        result = await self.invoker.invoke(request, context=context)
        response = result.response
        return TargetResponse(
            response_id=f"target_{result.call_id.removeprefix('call_')}",
            sample_id=sample.sample_id,
            call_id=result.call_id,
            provider_response_id=response.response_id,
            request_hash=response.request_hash,
            model=TargetModelRef(
                provider=response.provider,
                model=response.model,
                revision=response.model_version,
            ),
            text=response.answer,
            finish_reason=response.finish_reason,
            token_usage=response.token_usage,
            latency_ms=response.latency_ms,
            cost=response.cost,
            billed_cost_usd=result.billed_usd,
            cache_hit=result.cache_hit,
            raw_artifact=response.raw_artifact,
            raw_artifact_uri=(
                response.raw_artifact.uri if response.raw_artifact is not None else None
            ),
        )

    @staticmethod
    def build_request(
        sample: CanonicalMultimodalSample,
        *,
        parameters: Mapping[str, JsonValue] | None = None,
    ) -> ModelRequest:
        parts: list[ModelTextPart | ModelMediaPart] = []
        for part in sample.parts:
            if isinstance(part, TextPart):
                parts.append(ModelTextPart(text=part.text))
            elif isinstance(part, MediaPart):
                parts.append(ModelMediaPart(media=part.media))
        return ModelRequest(
            request_id=f"target:{sample.sample_id}",
            role=ModelRole.TARGET,
            parts=tuple(parts),
            parameters=dict(parameters or {}),
        )
