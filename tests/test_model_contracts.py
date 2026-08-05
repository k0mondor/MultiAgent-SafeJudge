from __future__ import annotations

from decimal import Decimal

import pytest
from pydantic import ValidationError

from safejudge.contracts.dataset import MediaRef, MediaType
from safejudge.contracts.evaluation import ModelCost, PriceSnapshot, TokenUsage
from safejudge.contracts.model import (
    InputModality,
    ModalityCombination,
    ModelCapabilities,
    ModelMediaPart,
    ModelRequest,
    ModelResponse,
    ModelRole,
    ModelTextPart,
)


def test_model_roles_are_explicit_and_distinct() -> None:
    assert ModelRole.TARGET.value == "target"
    assert ModelRole.JUDGE.value == "judge"


def test_request_reports_exact_order_independent_modality_set() -> None:
    request = ModelRequest(
        request_id="request_1",
        role=ModelRole.TARGET,
        parts=(
            ModelMediaPart(
                media=MediaRef(
                    media_type=MediaType.IMAGE,
                    uri="fixture.png",
                    mime_type="image/png",
                )
            ),
            ModelTextPart(text="Describe this image."),
        ),
        parameters={"temperature": 0},
    )

    assert request.modalities == frozenset({InputModality.TEXT, InputModality.IMAGE})


def test_capabilities_declare_combinations_instead_of_loose_modalities() -> None:
    capabilities = ModelCapabilities(
        input_combinations=(
            ModalityCombination(modalities=frozenset({InputModality.TEXT})),
            ModalityCombination(
                modalities=frozenset({InputModality.TEXT, InputModality.IMAGE})
            ),
        )
    )

    assert capabilities.supports(frozenset({InputModality.TEXT, InputModality.IMAGE}))
    assert not capabilities.supports(frozenset({InputModality.IMAGE}))
    assert not capabilities.supports(
        frozenset({InputModality.TEXT, InputModality.IMAGE, InputModality.AUDIO})
    )


def test_model_request_requires_text_instruction() -> None:
    with pytest.raises(ValidationError, match="at least one text part"):
        ModelRequest(
            request_id="request_1",
            role=ModelRole.TARGET,
            parts=(
                ModelMediaPart(
                    media=MediaRef(
                        media_type=MediaType.AUDIO,
                        uri="fixture.wav",
                        mime_type="audio/wav",
                    )
                ),
            ),
        )


def test_response_saves_usage_version_latency_and_cost_snapshot() -> None:
    usage = TokenUsage(input_tokens=1_000, output_tokens=500)
    price = PriceSnapshot(
        input_per_million_tokens=Decimal("2"),
        output_per_million_tokens=Decimal("4"),
    )
    estimated = price.estimate(usage)
    response = ModelResponse(
        response_id="response_1",
        request_hash="a" * 64,
        role=ModelRole.TARGET,
        provider="fake",
        model="fixture",
        model_version="fixture-v1",
        answer="fixed answer",
        token_usage=usage,
        latency_ms=3,
        cost=ModelCost(price_snapshot=price, estimated_usd=estimated, actual_usd=estimated),
    )

    assert estimated == Decimal("0.004")
    assert response.model_version == "fixture-v1"


def test_cost_without_token_usage_is_rejected() -> None:
    with pytest.raises(ValidationError, match="cost requires token_usage"):
        ModelResponse(
            response_id="response_1",
            request_hash="a" * 64,
            role=ModelRole.TARGET,
            provider="fake",
            model="fixture",
            answer="fixed answer",
            latency_ms=0,
            cost=ModelCost(price_snapshot=PriceSnapshot(), estimated_usd=Decimal("0")),
        )
