from __future__ import annotations

import asyncio

import pytest

from safejudge.contracts.model import (
    InputModality,
    ModalityCombination,
    ModelCapabilities,
    ModelRequest,
    ModelRole,
    ModelTextPart,
)
from safejudge.core.errors import ProviderError, ProviderErrorKind
from safejudge.models.fake import FakeFixture, FakeOutcome, FakeProvider

CAPABILITIES = ModelCapabilities(
    input_combinations=(
        ModalityCombination(modalities=frozenset({InputModality.TEXT})),
    )
)


def request(request_id: str) -> ModelRequest:
    return ModelRequest(
        request_id=request_id,
        role=ModelRole.TARGET,
        parts=(ModelTextPart(text="fixture prompt"),),
    )


def test_fake_provider_returns_fixed_refusal_and_unsafe_answers() -> None:
    provider = FakeProvider(
        capabilities=CAPABILITIES,
        fixtures={
            "refusal": FakeFixture(
                outcome=FakeOutcome.REFUSAL,
                answer="I cannot assist with that.",
            ),
            "unsafe": FakeFixture(
                outcome=FakeOutcome.UNSAFE_ANSWER,
                answer="Unsafe fixture instructions.",
            ),
        },
    )

    refusal = asyncio.run(provider.generate(request("refusal"), request_hash="a" * 64))
    unsafe = asyncio.run(provider.generate(request("unsafe"), request_hash="b" * 64))

    assert refusal.answer == "I cannot assist with that."
    assert unsafe.answer == "Unsafe fixture instructions."
    assert refusal.provider == "fake"


@pytest.mark.parametrize(
    ("outcome", "kind"),
    [
        (FakeOutcome.TIMEOUT, ProviderErrorKind.TIMEOUT),
        (FakeOutcome.RATE_LIMIT, ProviderErrorKind.RATE_LIMIT),
    ],
)
def test_fake_provider_simulates_retryable_failures(
    outcome: FakeOutcome,
    kind: ProviderErrorKind,
) -> None:
    provider = FakeProvider(
        capabilities=CAPABILITIES,
        fixtures={"failure": FakeFixture(outcome=outcome)},
    )

    with pytest.raises(ProviderError) as caught:
        asyncio.run(provider.generate(request("failure"), request_hash="a" * 64))

    assert caught.value.retryable is True
    assert caught.value.kind is kind


def test_fake_provider_can_recover_after_configured_transient_failures() -> None:
    provider = FakeProvider(
        capabilities=CAPABILITIES,
        fixtures={
            "transient": FakeFixture(
                outcome=FakeOutcome.RATE_LIMIT,
                failures_before_success=1,
                answer="eventual fixture answer",
            )
        },
    )

    with pytest.raises(ProviderError):
        asyncio.run(provider.generate(request("transient"), request_hash="a" * 64))
    response = asyncio.run(provider.generate(request("transient"), request_hash="a" * 64))

    assert response.answer == "eventual fixture answer"
    assert provider.attempts_for("transient") == 2


def test_missing_fake_fixture_is_non_retryable() -> None:
    provider = FakeProvider(capabilities=CAPABILITIES, fixtures={})

    with pytest.raises(ProviderError) as caught:
        asyncio.run(provider.generate(request("missing"), request_hash="a" * 64))

    assert caught.value.retryable is False
    assert caught.value.kind is ProviderErrorKind.INVALID_REQUEST
