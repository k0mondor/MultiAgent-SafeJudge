"""Contracts created after a target model receives a canonical sample."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import IntEnum
from typing import Literal

from pydantic import Field

from safejudge.contracts.artifact import ArtifactRef
from safejudge.contracts.base import ContractModel
from safejudge.contracts.dataset import NonEmptyString
from safejudge.core.time import utc_now


class ResponseComplianceLevel(IntEnum):
    """Safety-Rubric compliance of the target response, not request risk."""

    FULLY_COMPLIANT = 0
    PARTIALLY_COMPLIANT = 1
    SEVERELY_NONCOMPLIANT = 2


class TargetModelRef(ContractModel):
    role: Literal["target"] = "target"
    provider: NonEmptyString
    model: NonEmptyString
    revision: str | None = None


class TokenUsage(ContractModel):
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class PriceSnapshot(ContractModel):
    """Prices frozen at call time, expressed per one million tokens."""

    currency: Literal["USD"] = "USD"
    input_per_million_tokens: Decimal = Field(default=Decimal("0"), ge=0)
    output_per_million_tokens: Decimal = Field(default=Decimal("0"), ge=0)
    source: NonEmptyString = "configuration"

    def estimate(self, usage: TokenUsage) -> Decimal:
        million = Decimal(1_000_000)
        return (
            Decimal(usage.input_tokens) * self.input_per_million_tokens
            + Decimal(usage.output_tokens) * self.output_per_million_tokens
        ) / million


class ModelCost(ContractModel):
    price_snapshot: PriceSnapshot
    estimated_usd: Decimal = Field(ge=0)
    actual_usd: Decimal | None = Field(default=None, ge=0)


class TargetResponse(ContractModel):
    """The immutable answer that judge agents will evaluate."""

    schema_version: Literal["1.1"] = "1.1"
    response_id: NonEmptyString
    sample_id: NonEmptyString
    call_id: NonEmptyString | None = None
    provider_response_id: NonEmptyString | None = None
    request_hash: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    model: TargetModelRef
    text: NonEmptyString
    finish_reason: str | None = None
    token_usage: TokenUsage | None = None
    latency_ms: int | None = Field(default=None, ge=0)
    cost: ModelCost | None = None
    billed_cost_usd: Decimal = Field(default=Decimal("0"), ge=0)
    cache_hit: bool = False
    created_at: datetime = Field(default_factory=utc_now)
    raw_artifact: ArtifactRef | None = None
    raw_artifact_uri: str | None = None
