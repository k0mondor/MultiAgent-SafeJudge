"""Conversion and execution boundary for immutable target-model responses."""

from __future__ import annotations

import hashlib
import json
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
from safejudge.core.errors import ProviderError, ProviderErrorKind, UnsupportedModalityError
from safejudge.models.invocation import ModelInvoker
from safejudge.models.profiles import ModelProfile

_ADAPTIVE_RETRY_KINDS = {
    ProviderErrorKind.EMPTY_RESPONSE,
    ProviderErrorKind.REASONING_ONLY,
    ProviderErrorKind.TRUNCATED_RESPONSE,
}


class TargetRunner:
    def __init__(self, invoker: ModelInvoker, *, profile: ModelProfile) -> None:
        self.invoker = invoker
        self.profile = profile
        if not profile.supports_role(ModelRole.TARGET):
            raise ValueError("target profile does not support the target role")
        if profile.provider != invoker.provider.provider_name:
            raise ValueError("target profile provider does not match ModelInvoker provider")
        if profile.model_id != invoker.provider.model_id:
            raise ValueError("target profile model_id does not match ModelInvoker model")

    async def run(
        self,
        sample: CanonicalMultimodalSample,
        *,
        context: InvocationContext,
    ) -> TargetResponse:
        resolved = self.profile.default_parameters
        parameter_attempts = [resolved]
        parameter_attempts.extend(
            {**resolved, **retry_parameters}
            for retry_parameters in self.profile.retry_parameters
        )
        request = self.build_request(sample, parameters=resolved)
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

        for attempt, attempt_parameters in enumerate(parameter_attempts):
            request = self.build_request(sample, parameters=attempt_parameters)
            if attempt:
                request = request.model_copy(
                    update={"request_id": f"target:{sample.sample_id}:adaptive:{attempt}"}
                )
            try:
                result = await self.invoker.invoke(request, context=context)
                break
            except ProviderError as error:
                if error.kind not in _ADAPTIVE_RETRY_KINDS or attempt + 1 >= len(
                    parameter_attempts
                ):
                    raise
        else:
            raise AssertionError("unreachable target adaptive retry state")
        response = result.response
        return TargetResponse(
            response_id=_stable_target_response_id(
                sample_id=sample.sample_id,
                request_hash=response.request_hash,
                provider=response.provider,
                model=response.model,
                model_version=response.model_version,
                answer=response.answer,
                finish_reason=response.finish_reason,
            ),
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
        parameters: Mapping[str, JsonValue],
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
            parameters=dict(parameters),
        )


def _stable_target_response_id(
    *,
    sample_id: str,
    request_hash: str,
    provider: str,
    model: str,
    model_version: str | None,
    answer: str,
    finish_reason: str | None,
) -> str:
    """Identify the immutable semantic response, not the local invocation event."""

    payload = {
        "schema": "safejudge-target-response-id-v1",
        "sample_id": sample_id,
        "request_hash": request_hash,
        "provider": provider,
        "model": model,
        "model_version": model_version,
        "answer": answer,
        "finish_reason": finish_reason,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"target_{hashlib.sha256(encoded).hexdigest()}"
