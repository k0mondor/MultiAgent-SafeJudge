from __future__ import annotations

import asyncio
from decimal import Decimal
from pathlib import Path

import pytest

from safejudge.contracts.dataset import (
    CanonicalMultimodalSample,
    MediaPart,
    MediaRef,
    MediaType,
    SourceRecord,
    TextPart,
)
from safejudge.contracts.evaluation import PriceSnapshot
from safejudge.contracts.model import (
    InputModality,
    InvocationContext,
    ModalityCombination,
    ModelCapabilities,
)
from safejudge.core.errors import (
    BudgetExceededError,
    ProviderError,
    UnsupportedModalityError,
)
from safejudge.models.cache import SQLiteModelStore
from safejudge.models.fake import FakeFixture, FakeOutcome, FakeProvider
from safejudge.models.invocation import InvocationPolicy, ModelInvoker
from safejudge.models.target import TargetRunner

ALL_CAPABILITIES = ModelCapabilities(
    input_combinations=(
        ModalityCombination(modalities=frozenset({InputModality.TEXT, InputModality.IMAGE})),
        ModalityCombination(modalities=frozenset({InputModality.TEXT, InputModality.AUDIO})),
        ModalityCombination(modalities=frozenset({InputModality.TEXT, InputModality.VIDEO})),
    )
)
CONTEXT = InvocationContext(experiment_id="experiment_test", run_id="run_test")


def sample(sample_id: str, media_type: MediaType) -> CanonicalMultimodalSample:
    extension = {MediaType.IMAGE: "png", MediaType.AUDIO: "wav", MediaType.VIDEO: "mp4"}[
        media_type
    ]
    return CanonicalMultimodalSample(
        sample_id=sample_id,
        parts=(
            MediaPart(
                media=MediaRef(
                    media_type=media_type,
                    uri=f"media/{sample_id}.{extension}",
                    mime_type=f"{media_type.value}/{extension}",
                )
            ),
            TextPart(text="Evaluate this media."),
        ),
        source=SourceRecord(
            dataset_name="fixture",
            dataset_version="1",
            original_id=sample_id,
        ),
    )


def runner(
    tmp_path: Path,
    *,
    fixtures: dict[str, FakeFixture],
    capabilities: ModelCapabilities = ALL_CAPABILITIES,
    policy: InvocationPolicy | None = None,
) -> tuple[TargetRunner, FakeProvider, SQLiteModelStore]:
    provider = FakeProvider(
        fixtures=fixtures,
        capabilities=capabilities,
        price_snapshot=PriceSnapshot(
            input_per_million_tokens=Decimal("10"),
            output_per_million_tokens=Decimal("20"),
        ),
    )
    store = SQLiteModelStore(tmp_path / "model-calls.sqlite3")
    invoker = ModelInvoker(provider=provider, store=store, policy=policy)
    return TargetRunner(invoker), provider, store


@pytest.mark.parametrize("media_type", list(MediaType))
def test_target_runner_handles_image_audio_and_video_text_offline(
    tmp_path: Path,
    media_type: MediaType,
) -> None:
    item = sample(f"{media_type.value}-sample", media_type)
    target_runner, _, _ = runner(
        tmp_path,
        fixtures={
            f"target:{item.sample_id}": FakeFixture(
                answer=f"{media_type.value} fixture answer",
                model_version="fake-2026-08",
            )
        },
    )

    response = asyncio.run(
        target_runner.run(item, context=CONTEXT, parameters={"temperature": 0})
    )

    assert response.sample_id == item.sample_id
    assert response.model.role == "target"
    assert response.model.revision == "fake-2026-08"
    assert response.text == f"{media_type.value} fixture answer"
    assert response.request_hash is not None
    assert response.provider_response_id is not None
    assert response.cost is not None


def test_target_runner_fails_before_provider_for_unsupported_combination(
    tmp_path: Path,
) -> None:
    item = sample("audio-sample", MediaType.AUDIO)
    image_only = ModelCapabilities(
        input_combinations=(
            ModalityCombination(
                modalities=frozenset({InputModality.TEXT, InputModality.IMAGE})
            ),
        )
    )
    target_runner, provider, _ = runner(
        tmp_path,
        fixtures={f"target:{item.sample_id}": FakeFixture()},
        capabilities=image_only,
    )

    with pytest.raises(UnsupportedModalityError, match=r"audio\+text"):
        asyncio.run(target_runner.run(item, context=CONTEXT))

    assert provider.attempts_for(f"target:{item.sample_id}") == 0


def test_target_runner_retries_transient_but_not_permanent_error(tmp_path: Path) -> None:
    transient = sample("transient", MediaType.IMAGE)
    permanent = sample("permanent", MediaType.IMAGE)
    target_runner, provider, store = runner(
        tmp_path,
        fixtures={
            "target:transient": FakeFixture(
                outcome=FakeOutcome.RATE_LIMIT,
                failures_before_success=1,
                answer="recovered",
            ),
        },
        policy=InvocationPolicy(max_retries=1),
    )

    recovered = asyncio.run(target_runner.run(transient, context=CONTEXT))
    assert recovered.text == "recovered"
    assert provider.attempts_for("target:transient") == 2

    with pytest.raises(ProviderError) as caught:
        asyncio.run(target_runner.run(permanent, context=CONTEXT))
    assert caught.value.retryable is False
    assert store.call_count() == 2


def test_target_runner_cache_hit_preserves_answer_for_new_judge_runs(tmp_path: Path) -> None:
    item = sample("cached", MediaType.VIDEO)
    target_runner, provider, _ = runner(
        tmp_path,
        fixtures={"target:cached": FakeFixture(answer="immutable target answer")},
    )

    first = asyncio.run(target_runner.run(item, context=CONTEXT))
    second = asyncio.run(target_runner.run(item, context=CONTEXT))

    assert first.cache_hit is False
    assert second.cache_hit is True
    assert second.text == first.text
    assert second.provider_response_id == first.provider_response_id
    assert second.billed_cost_usd == 0
    assert provider.attempts_for("target:cached") == 1


def test_target_runner_budget_cap_blocks_before_provider(tmp_path: Path) -> None:
    item = sample("over-budget", MediaType.IMAGE)
    target_runner, provider, _ = runner(
        tmp_path,
        fixtures={"target:over-budget": FakeFixture()},
        policy=InvocationPolicy(
            max_total_cost_usd=Decimal("0.01"),
            uncached_call_reservation_usd=Decimal("0.02"),
        ),
    )

    with pytest.raises(BudgetExceededError, match="projected cost"):
        asyncio.run(target_runner.run(item, context=CONTEXT))

    assert provider.attempts_for("target:over-budget") == 0
