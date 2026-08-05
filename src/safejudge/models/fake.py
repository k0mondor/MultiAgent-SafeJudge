"""Deterministic, network-free provider for unit tests and graph fixtures."""

from __future__ import annotations

import json
from collections.abc import Mapping
from enum import StrEnum

from pydantic import Field

from safejudge.contracts.base import ContractModel
from safejudge.contracts.evaluation import ModelCost, PriceSnapshot, TokenUsage
from safejudge.contracts.model import (
    ModelCapabilities,
    ModelRequest,
    ModelResponse,
)
from safejudge.core.errors import ArtifactError, ProviderError, ProviderErrorKind
from safejudge.models.artifacts import ArtifactStore


class FakeOutcome(StrEnum):
    ANSWER = "answer"
    REFUSAL = "refusal"
    UNSAFE_ANSWER = "unsafe_answer"
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"


class FakeFixture(ContractModel):
    """One deterministic scenario selected by ``ModelRequest.request_id``."""

    outcome: FakeOutcome = FakeOutcome.ANSWER
    answer: str = "Fixture response."
    finish_reason: str = "stop"
    model_version: str = "fake-v1"
    input_tokens: int = Field(default=10, ge=0)
    output_tokens: int = Field(default=5, ge=0)
    latency_ms: int = Field(default=0, ge=0)
    failures_before_success: int | None = Field(default=None, ge=0)


class FakeProvider:
    """Returns fixtures and raises classified failures without accessing the network."""

    def __init__(
        self,
        *,
        fixtures: Mapping[str, FakeFixture],
        capabilities: ModelCapabilities,
        model_id: str = "safejudge/fake",
        price_snapshot: PriceSnapshot | None = None,
        artifact_store: ArtifactStore | None = None,
    ) -> None:
        self._fixtures = dict(fixtures)
        self._capabilities = capabilities
        self._model_id = model_id
        self._price_snapshot = price_snapshot or PriceSnapshot(source="fake-fixture")
        self._artifact_store = artifact_store
        self._attempts: dict[str, int] = {}

    @property
    def provider_name(self) -> str:
        return "fake"

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def capabilities(self) -> ModelCapabilities:
        return self._capabilities

    @property
    def price_snapshot(self) -> PriceSnapshot:
        return self._price_snapshot

    def attempts_for(self, request_id: str) -> int:
        return self._attempts.get(request_id, 0)

    async def generate(self, request: ModelRequest, *, request_hash: str) -> ModelResponse:
        try:
            fixture = self._fixtures[request.request_id]
        except KeyError as error:
            raise ProviderError(
                f"no fake fixture registered for request_id={request.request_id!r}",
                retryable=False,
                kind=ProviderErrorKind.INVALID_REQUEST,
            ) from error

        attempt = self._attempts.get(request.request_id, 0) + 1
        self._attempts[request.request_id] = attempt
        should_fail = (
            fixture.outcome in {FakeOutcome.TIMEOUT, FakeOutcome.RATE_LIMIT}
            and (
                fixture.failures_before_success is None
                or attempt <= fixture.failures_before_success
            )
        )
        if should_fail:
            if fixture.outcome is FakeOutcome.TIMEOUT:
                raise ProviderError(
                    "fake provider timed out",
                    retryable=True,
                    kind=ProviderErrorKind.TIMEOUT,
                )
            raise ProviderError(
                "fake provider rate limited the request",
                retryable=True,
                kind=ProviderErrorKind.RATE_LIMIT,
                status_code=429,
                retry_after_seconds=0,
            )

        usage = TokenUsage(
            input_tokens=fixture.input_tokens,
            output_tokens=fixture.output_tokens,
        )
        estimated = self.price_snapshot.estimate(usage)
        raw_artifact = None
        if self._artifact_store is not None:
            raw_payload = {
                "answer": fixture.answer,
                "finish_reason": fixture.finish_reason,
                "input_tokens": fixture.input_tokens,
                "latency_ms": fixture.latency_ms,
                "model": self.model_id,
                "model_version": fixture.model_version,
                "outcome": fixture.outcome.value,
                "output_tokens": fixture.output_tokens,
                "provider": self.provider_name,
                "request_id": request.request_id,
            }
            try:
                raw_artifact = self._artifact_store.write_bytes(
                    category="fake-responses",
                    content=json.dumps(
                        raw_payload,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8"),
                    content_type="application/json",
                )
            except ArtifactError as error:
                raise ProviderError(
                    str(error),
                    retryable=False,
                    kind=ProviderErrorKind.ARTIFACT_PERSISTENCE,
                ) from error
        return ModelResponse(
            response_id=f"fake_{request_hash[:24]}",
            request_hash=request_hash,
            role=request.role,
            provider=self.provider_name,
            model=self.model_id,
            model_version=fixture.model_version,
            answer=fixture.answer,
            finish_reason=fixture.finish_reason,
            token_usage=usage,
            latency_ms=fixture.latency_ms,
            cost=ModelCost(
                price_snapshot=self.price_snapshot,
                estimated_usd=estimated,
                actual_usd=estimated,
            ),
            raw_artifact=raw_artifact,
        )
