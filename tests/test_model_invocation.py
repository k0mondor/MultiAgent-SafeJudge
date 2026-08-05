from __future__ import annotations

import asyncio
import sqlite3
from decimal import Decimal
from pathlib import Path

import pytest

from safejudge.contracts.artifact import ArtifactRef
from safejudge.contracts.evaluation import PriceSnapshot, TokenUsage
from safejudge.contracts.model import (
    InputModality,
    InvocationContext,
    ModalityCombination,
    ModelCapabilities,
    ModelRequest,
    ModelResponse,
    ModelRole,
    ModelTextPart,
)
from safejudge.core.errors import BudgetExceededError, ProviderError, ProviderErrorKind
from safejudge.models.cache import SQLiteModelStore, stable_request_hash
from safejudge.models.fake import FakeFixture, FakeOutcome, FakeProvider
from safejudge.models.invocation import InvocationPolicy, InvocationResult, ModelInvoker

CAPABILITIES = ModelCapabilities(
    input_combinations=(
        ModalityCombination(modalities=frozenset({InputModality.TEXT})),
    )
)
CONTEXT = InvocationContext(experiment_id="experiment_test", run_id="run_test")


def request(request_id: str, *, temperature: float = 0) -> ModelRequest:
    return ModelRequest(
        request_id=request_id,
        role=ModelRole.TARGET,
        parts=(ModelTextPart(text="same semantic prompt"),),
        parameters={"temperature": temperature},
    )


def provider(fixtures: dict[str, FakeFixture]) -> FakeProvider:
    return FakeProvider(
        capabilities=CAPABILITIES,
        fixtures=fixtures,
        price_snapshot=PriceSnapshot(
            input_per_million_tokens=Decimal("100000"),
            output_per_million_tokens=Decimal("100000"),
        ),
    )


def test_stable_hash_ignores_request_id_but_includes_parameters_and_model() -> None:
    first = request("run_a")
    second = request("run_b")

    assert stable_request_hash(provider="fake", model="m1", request=first) == stable_request_hash(
        provider="fake", model="m1", request=second
    )
    assert stable_request_hash(provider="fake", model="m1", request=first) != stable_request_hash(
        provider="fake", model="m1", request=request("run_c", temperature=0.5)
    )
    assert stable_request_hash(provider="fake", model="m1", request=first) != stable_request_hash(
        provider="fake", model="m2", request=first
    )


def test_repeated_request_hits_persistent_cache_without_second_provider_call(
    tmp_path: Path,
) -> None:
    fake = provider(
        {
            "first": FakeFixture(answer="cached answer", input_tokens=4, output_tokens=1),
            "second": FakeFixture(answer="should not be called"),
        }
    )
    store = SQLiteModelStore(tmp_path / "calls.sqlite3")
    invoker = ModelInvoker(provider=fake, store=store)

    first = asyncio.run(invoker.invoke(request("first"), context=CONTEXT))
    second = asyncio.run(invoker.invoke(request("second"), context=CONTEXT))

    assert first.cache_hit is False
    assert second.cache_hit is True
    assert second.response.answer == "cached answer"
    assert fake.attempts_for("second") == 0
    assert first.billed_usd == Decimal("0.5")
    assert second.billed_usd == 0
    assert store.total_billed_usd(context=CONTEXT, role=ModelRole.TARGET) == Decimal("0.5")
    assert store.call_count(cache_hit=True) == 1


def test_target_and_judge_roles_have_different_cache_keys() -> None:
    target = request("target")
    judge = target.model_copy(update={"role": ModelRole.JUDGE})

    assert stable_request_hash(provider="fake", model="m", request=target) != stable_request_hash(
        provider="fake", model="m", request=judge
    )


def test_invoker_retries_retryable_fake_error(tmp_path: Path) -> None:
    fake = provider(
        {
            "retry": FakeFixture(
                outcome=FakeOutcome.TIMEOUT,
                failures_before_success=2,
                answer="recovered",
            )
        }
    )
    database = tmp_path / "calls.sqlite3"
    store = SQLiteModelStore(database)
    invoker = ModelInvoker(
        provider=fake,
        store=store,
        policy=InvocationPolicy(max_retries=2),
    )

    result = asyncio.run(
        invoker.invoke(request("retry", temperature=0.25), context=CONTEXT)
    )

    assert result.attempts == 3
    assert result.response.answer == "recovered"
    assert fake.attempts_for("retry") == 3
    with sqlite3.connect(database) as connection:
        attempt_statuses = connection.execute(
            """
            SELECT attempt_number, status FROM model_attempts
            ORDER BY attempt_number
            """
        ).fetchall()
    assert attempt_statuses == [(1, "failed"), (2, "failed"), (3, "success")]


def test_budget_blocks_uncached_call_but_allows_cache_hit(tmp_path: Path) -> None:
    fake = provider({"cached": FakeFixture(answer="cached"), "blocked": FakeFixture()})
    store = SQLiteModelStore(tmp_path / "calls.sqlite3")
    warm = ModelInvoker(provider=fake, store=store)
    asyncio.run(warm.invoke(request("cached"), context=CONTEXT))
    guarded = ModelInvoker(
        provider=fake,
        store=store,
        policy=InvocationPolicy(
            max_total_cost_usd=Decimal("1.6"),
            uncached_call_reservation_usd=Decimal("0.2"),
        ),
    )

    cached = asyncio.run(guarded.invoke(request("another-id"), context=CONTEXT))
    assert cached.cache_hit is True

    with pytest.raises(BudgetExceededError, match="blocked"):
        asyncio.run(
            guarded.invoke(request("blocked", temperature=0.7), context=CONTEXT)
        )
    assert fake.attempts_for("blocked") == 0


class TrackingProvider:
    provider_name = "tracking"
    model_id = "tracking/model"
    capabilities = CAPABILITIES
    price_snapshot = PriceSnapshot(source="tracking")

    def __init__(self) -> None:
        self.calls = 0
        self.active = 0
        self.max_active = 0

    async def generate(self, request: ModelRequest, *, request_hash: str) -> ModelResponse:
        self.calls += 1
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(0.02)
        finally:
            self.active -= 1
        return ModelResponse(
            response_id=f"tracking_{self.calls}",
            request_hash=request_hash,
            role=request.role,
            provider=self.provider_name,
            model=self.model_id,
            answer=f"answer for {request.request_id}",
            token_usage=TokenUsage(input_tokens=1, output_tokens=1),
            latency_ms=20,
        )


def test_different_cache_keys_run_concurrently(tmp_path: Path) -> None:
    tracking = TrackingProvider()
    invoker = ModelInvoker(
        provider=tracking,
        store=SQLiteModelStore(tmp_path / "calls.sqlite3"),
        policy=InvocationPolicy(max_concurrency=3),
    )

    async def run_all() -> None:
        await asyncio.gather(
            invoker.invoke(request("a", temperature=0.1), context=CONTEXT),
            invoker.invoke(request("b", temperature=0.2), context=CONTEXT),
            invoker.invoke(request("c", temperature=0.3), context=CONTEXT),
        )

    asyncio.run(run_all())

    assert tracking.calls == 3
    assert tracking.max_active == 3


def test_identical_cache_keys_use_singleflight(tmp_path: Path) -> None:
    tracking = TrackingProvider()
    invoker = ModelInvoker(
        provider=tracking,
        store=SQLiteModelStore(tmp_path / "calls.sqlite3"),
        policy=InvocationPolicy(max_concurrency=3),
    )

    async def run_both() -> tuple[InvocationResult, InvocationResult]:
        first, second = await asyncio.gather(
            invoker.invoke(request("same-a"), context=CONTEXT),
            invoker.invoke(request("same-b"), context=CONTEXT),
        )
        return first, second

    first, second = asyncio.run(run_both())

    assert tracking.calls == 1
    assert {first.cache_hit, second.cache_hit} == {False, True}


def test_budget_is_scoped_by_experiment_run_and_role(tmp_path: Path) -> None:
    context_a = InvocationContext(experiment_id="experiment_a", run_id="run_1")
    context_b = InvocationContext(experiment_id="experiment_b", run_id="run_1")
    fake = provider(
        {
            "first": FakeFixture(answer="first"),
            "second": FakeFixture(answer="second"),
            "blocked": FakeFixture(answer="must not run"),
        }
    )
    store = SQLiteModelStore(tmp_path / "calls.sqlite3")
    invoker = ModelInvoker(
        provider=fake,
        store=store,
        policy=InvocationPolicy(
            max_total_cost_usd=Decimal("1.6"),
            uncached_call_reservation_usd=Decimal("0.2"),
        ),
    )

    asyncio.run(invoker.invoke(request("first"), context=context_a))
    asyncio.run(
        invoker.invoke(request("second", temperature=0.8), context=context_b)
    )
    with pytest.raises(BudgetExceededError):
        asyncio.run(
            invoker.invoke(request("blocked", temperature=0.7), context=context_a)
        )

    assert fake.attempts_for("second") == 1
    assert fake.attempts_for("blocked") == 0
    assert store.total_billed_usd(
        context=context_a,
        role=ModelRole.TARGET,
    ) == Decimal("1.5")
    assert store.total_billed_usd(
        context=context_b,
        role=ModelRole.TARGET,
    ) == Decimal("1.5")


def test_concurrent_budget_reservations_prevent_overspend(tmp_path: Path) -> None:
    tracking = TrackingProvider()
    invoker = ModelInvoker(
        provider=tracking,
        store=SQLiteModelStore(tmp_path / "calls.sqlite3"),
        policy=InvocationPolicy(
            max_concurrency=2,
            max_total_cost_usd=Decimal("1.0"),
            uncached_call_reservation_usd=Decimal("0.6"),
        ),
    )

    async def compete() -> tuple[
        InvocationResult | BaseException,
        InvocationResult | BaseException,
    ]:
        return await asyncio.gather(
            invoker.invoke(request("budget-a", temperature=0.1), context=CONTEXT),
            invoker.invoke(request("budget-b", temperature=0.2), context=CONTEXT),
            return_exceptions=True,
        )

    results = asyncio.run(compete())

    assert tracking.calls == 1
    assert sum(isinstance(item, BudgetExceededError) for item in results) == 1


class FailingArtifactProvider(TrackingProvider):
    async def generate(self, request: ModelRequest, *, request_hash: str) -> ModelResponse:
        del request, request_hash
        self.calls += 1
        raise ProviderError(
            "invalid fixture response",
            retryable=False,
            kind=ProviderErrorKind.MALFORMED_RESPONSE,
            raw_artifact=ArtifactRef(
                uri="memory://provider/raw-error",
                sha256="a" * 64,
                content_type="application/json",
                size_bytes=17,
            ),
        )


def test_non_retryable_failure_records_raw_artifact_once(tmp_path: Path) -> None:
    database = tmp_path / "calls.sqlite3"
    failing = FailingArtifactProvider()
    invoker = ModelInvoker(
        provider=failing,
        store=SQLiteModelStore(database),
        policy=InvocationPolicy(max_retries=5),
    )

    with pytest.raises(ProviderError):
        asyncio.run(invoker.invoke(request("failure"), context=CONTEXT))

    with sqlite3.connect(database) as connection:
        row = connection.execute(
            """
            SELECT attempts, raw_artifact_uri, raw_artifact_sha256
            FROM model_calls WHERE status = 'failed'
            """
        ).fetchone()
        [attempt_row] = connection.execute(
            """
            SELECT attempt_number, raw_artifact_uri, raw_artifact_sha256
            FROM model_attempts WHERE status = 'failed'
            """
        ).fetchall()
    assert failing.calls == 1
    assert row == (1, "memory://provider/raw-error", "a" * 64)
    assert attempt_row == (1, "memory://provider/raw-error", "a" * 64)
