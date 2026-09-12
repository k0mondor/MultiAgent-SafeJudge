from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from safejudge.contracts.dataset import (
    CanonicalMultimodalSample,
    DatasetSplit,
    SourceRecord,
    TextPart,
)
from safejudge.contracts.evaluation import PriceSnapshot
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
from safejudge.core.errors import CacheMissError
from safejudge.models.cache import SQLiteModelStore
from safejudge.models.invocation import InvocationPolicy, ModelInvoker
from safejudge.models.profiles import ModelProfile
from safejudge.models.target import TargetRunner


class _CountingProvider:
    provider_name = "local-openai"
    model_id = "replay-test-model"
    capabilities = ModelCapabilities(
        input_combinations=(
            ModalityCombination(modalities=frozenset({InputModality.TEXT})),
        )
    )
    price_snapshot = PriceSnapshot(source="test")

    def __init__(self) -> None:
        self.call_count = 0

    async def generate(
        self,
        request: ModelRequest,
        *,
        request_hash: str,
    ) -> ModelResponse:
        self.call_count += 1
        return ModelResponse(
            response_id=f"provider-response-{self.call_count}",
            request_hash=request_hash,
            role=request.role,
            provider=self.provider_name,
            model=self.model_id,
            model_version="test-v1",
            answer="A stable answer.",
            finish_reason="stop",
            latency_ms=1,
        )


class ReplayCacheTests(unittest.IsolatedAsyncioTestCase):
    async def test_cached_target_keeps_semantic_response_id(self) -> None:
        provider = _CountingProvider()
        profile = ModelProfile(
            profile_id="replay-test-target",
            profile_version="1",
            provider="local-openai",
            model_id=provider.model_id,
            roles=frozenset({ModelRole.TARGET}),
        )
        sample = CanonicalMultimodalSample(
            sample_id="sample-1",
            parts=(TextPart(text="Test request"),),
            source=SourceRecord(
                dataset_name="replay-test",
                dataset_version="1",
                split=DatasetSplit.TEST,
                original_id="source-1",
            ),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SQLiteModelStore(Path(temp_dir) / "calls.sqlite3")
            runner = TargetRunner(ModelInvoker(provider=provider, store=store), profile=profile)
            first = await runner.run(
                sample,
                context=InvocationContext(experiment_id="test", run_id="first"),
            )
            second = await runner.run(
                sample,
                context=InvocationContext(experiment_id="test", run_id="second"),
            )

        self.assertEqual(provider.call_count, 1)
        self.assertFalse(first.cache_hit)
        self.assertTrue(second.cache_hit)
        self.assertNotEqual(first.call_id, second.call_id)
        self.assertEqual(first.response_id, second.response_id)

    async def test_cache_only_miss_never_calls_provider(self) -> None:
        provider = _CountingProvider()
        request = ModelRequest(
            request_id="judge:missing",
            role=ModelRole.JUDGE,
            parts=(ModelTextPart(text="Judge this"),),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            invoker = ModelInvoker(
                provider=provider,
                store=SQLiteModelStore(Path(temp_dir) / "calls.sqlite3"),
                policy=InvocationPolicy(cache_only=True),
            )
            with self.assertRaises(CacheMissError):
                await invoker.invoke(
                    request,
                    context=InvocationContext(experiment_id="test", run_id="replay"),
                )

        self.assertEqual(provider.call_count, 0)

    async def test_cache_only_hit_reuses_seeded_response(self) -> None:
        provider = _CountingProvider()
        request = ModelRequest(
            request_id="judge:cached",
            role=ModelRole.JUDGE,
            parts=(ModelTextPart(text="Judge this"),),
        )
        context = InvocationContext(experiment_id="test", run_id="seed")
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SQLiteModelStore(Path(temp_dir) / "calls.sqlite3")
            await ModelInvoker(provider=provider, store=store).invoke(request, context=context)
            replay = await ModelInvoker(
                provider=provider,
                store=store,
                policy=InvocationPolicy(cache_only=True),
            ).invoke(
                request,
                context=InvocationContext(experiment_id="test", run_id="replay"),
            )

        self.assertTrue(replay.cache_hit)
        self.assertEqual(provider.call_count, 1)

