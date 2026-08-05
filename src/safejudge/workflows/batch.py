"""Recoverable JSONL batch runner for frozen target responses and M3 judges."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import tempfile
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Literal, cast

from langchain_core.runnables import RunnableConfig
from pydantic import Field, JsonValue, ValidationError

from safejudge.contracts.base import ContractModel
from safejudge.contracts.dataset import CanonicalMultimodalSample, RequestIntent, Sha256
from safejudge.contracts.evaluation import TargetResponse
from safejudge.contracts.judging import EvaluationResult
from safejudge.contracts.model import (
    InputModality,
    InvocationContext,
    ModalityCombination,
    ModelCapabilities,
)
from safejudge.core.errors import ConfigurationError, ContractValidationError
from safejudge.core.time import utc_now
from safejudge.datasets.readers import iter_mapping_records
from safejudge.models.artifacts import FileArtifactStore
from safejudge.models.base import ModelProvider
from safejudge.models.cache import SQLiteModelStore
from safejudge.models.fake import FakeFixture, FakeProvider
from safejudge.models.invocation import InvocationPolicy, ModelInvoker
from safejudge.workflows.checkpoint import sqlite_checkpointer
from safejudge.workflows.graph import (
    EvaluationContext,
    EvaluationInput,
    build_evaluation_graph,
    create_evaluation_identity,
)
from safejudge.workflows.judge import JudgeRunner
from safejudge.workflows.ledger import SQLiteNodeLedger

_TEXT_CAPABILITIES = ModelCapabilities(
    input_combinations=(ModalityCombination(modalities=frozenset({InputModality.TEXT})),)
)


class EvaluationFileDigest(ContractModel):
    name: str
    sha256: Sha256


class EvaluationBatchManifest(ContractModel):
    manifest_version: Literal["1.0"] = "1.0"
    evaluation_result_schema_version: Literal["1.2"] = "1.2"
    samples_input: EvaluationFileDigest
    targets_input: EvaluationFileDigest
    output: EvaluationFileDigest
    judge_provider: str
    judge_model: str
    experiment_id: str
    run_id: str
    sample_count: int = Field(ge=0)
    arbitration_count: int = Field(ge=0)
    cache_hit_count: int = Field(ge=0)
    cache_miss_count: int = Field(ge=0)
    billed_cost_usd: Decimal = Field(ge=0)
    compliance_level_counts: dict[str, int]
    created_at: datetime = Field(default_factory=utc_now)


@dataclass(frozen=True, slots=True)
class EvaluationBatchResult:
    output_path: Path
    manifest_path: Path
    store_path: Path
    checkpoint_path: Path
    node_ledger_path: Path
    manifest: EvaluationBatchManifest


async def run_fake_evaluation_batch(
    *,
    samples_path: Path,
    target_responses_path: Path,
    output_path: Path,
    store_path: Path,
    checkpoint_path: Path,
    node_ledger_path: Path,
    artifact_root: Path,
    context: InvocationContext,
    max_concurrency: int = 3,
    max_sample_concurrency: int = 1,
    limit: int | None = None,
    overwrite: bool = False,
) -> EvaluationBatchResult:
    samples = _limited(_read_samples(samples_path.resolve()), limit)
    targets = _read_targets(target_responses_path.resolve())
    fixtures: dict[str, FakeFixture] = {}
    for sample in samples:
        target = _target_for(sample, targets)
        quote = target.text[: min(120, len(target.text))]
        intent = sample.request_context.intent
        fixtures[f"judge:intent:{sample.sample_id}"] = FakeFixture(
            answer=json.dumps(
                {
                    "request_intent": intent.value,
                    "risk_category": sample.request_context.risk_category,
                    "request_summary": f"Offline fixture for {sample.sample_id}.",
                    "response_summary": "Offline frozen target response.",
                    "expanded_task_context": "Apply the configured rubric to frozen data.",
                    "confidence": 0.9,
                }
            )
        )
        fixtures[f"judge:compliance:{sample.sample_id}"] = FakeFixture(
            answer=_fake_verdict("compliant", "OFFLINE_FIXTURE", quote)
        )
        fixtures[f"judge:harm_enablement:{sample.sample_id}"] = FakeFixture(
            answer=_fake_enablement(quote)
        )
        oversensitivity_label = (
            "appropriate" if intent is RequestIntent.BENIGN else "not_applicable"
        )
        fixtures[f"judge:oversensitivity:{sample.sample_id}"] = FakeFixture(
            answer=_fake_verdict(
                oversensitivity_label,
                "OFFLINE_FIXTURE",
                quote,
            )
        )
    provider = FakeProvider(
        fixtures=fixtures,
        capabilities=_TEXT_CAPABILITIES,
        model_id="safejudge/fake-m3-judge",
        artifact_store=FileArtifactStore(artifact_root.resolve()),
    )
    return await run_evaluation_batch(
        samples_path=samples_path,
        target_responses_path=target_responses_path,
        output_path=output_path,
        store_path=store_path,
        checkpoint_path=checkpoint_path,
        node_ledger_path=node_ledger_path,
        context=context,
        provider=provider,
        max_concurrency=max_concurrency,
        max_sample_concurrency=max_sample_concurrency,
        limit=limit,
        overwrite=overwrite,
    )


async def run_evaluation_batch(
    *,
    samples_path: Path,
    target_responses_path: Path,
    output_path: Path,
    store_path: Path,
    checkpoint_path: Path,
    node_ledger_path: Path,
    context: InvocationContext,
    provider: ModelProvider,
    max_concurrency: int = 3,
    max_sample_concurrency: int = 1,
    max_retries: int = 1,
    parameters: Mapping[str, JsonValue] | None = None,
    limit: int | None = None,
    overwrite: bool = False,
) -> EvaluationBatchResult:
    samples_path = samples_path.resolve()
    target_responses_path = target_responses_path.resolve()
    output_path = output_path.resolve()
    store_path = store_path.resolve()
    checkpoint_path = checkpoint_path.resolve()
    node_ledger_path = node_ledger_path.resolve()
    manifest_path = output_path.with_suffix(f"{output_path.suffix}.manifest.json")
    _validate_paths(
        samples_path=samples_path,
        target_responses_path=target_responses_path,
        output_path=output_path,
        manifest_path=manifest_path,
        overwrite=overwrite,
    )
    if max_concurrency < 1 or max_sample_concurrency < 1:
        raise ConfigurationError("concurrency values must be at least 1")
    if max_retries < 0:
        raise ConfigurationError("max retries must be non-negative")

    samples = _limited(_read_samples(samples_path), limit)
    targets = _read_targets(target_responses_path)
    pairs = tuple((sample, _target_for(sample, targets)) for sample in samples)
    store = SQLiteModelStore(store_path)
    judge_runner = JudgeRunner(
        ModelInvoker(
            provider=provider,
            store=store,
            policy=InvocationPolicy(
                max_concurrency=max_concurrency,
                max_retries=max_retries,
                retry_backoff_seconds=0.5,
            ),
        )
    )
    graph_context = EvaluationContext(
        invocation=context,
        judge_runner=judge_runner,
        judge_parameters=dict(parameters or {"temperature": 0, "max_tokens": 512}),
        node_ledger=SQLiteNodeLedger(node_ledger_path),
    )
    sample_slots = asyncio.Semaphore(max_sample_concurrency)

    async with sqlite_checkpointer(checkpoint_path) as saver:
        graph = build_evaluation_graph(checkpointer=saver)

        async def evaluate_one(
            sample: CanonicalMultimodalSample,
            target: TargetResponse,
        ) -> EvaluationResult:
            _, spec = create_evaluation_identity(
                sample=sample,
                target_response=target,
                context=graph_context,
            )
            config: RunnableConfig = {
                "configurable": {"thread_id": f"{context.run_id}:{spec.evaluation_key}"}
            }
            async with sample_slots:
                snapshot = await graph.aget_state(config)
                existing = snapshot.values.get("result") if snapshot.values else None
                if existing is not None:
                    return EvaluationResult.model_validate(existing)
                graph_input: EvaluationInput | None
                graph_input = (
                    None
                    if snapshot.next
                    else EvaluationInput(
                        sample=sample,
                        target_response=target,
                    )
                )
                output = await graph.ainvoke(
                    graph_input,
                    config=config,
                    context=graph_context,
                )
                return EvaluationResult.model_validate(output["result"])

        results = await asyncio.gather(*(evaluate_one(sample, target) for sample, target in pairs))

    output_content = "".join(f"{item.model_dump_json()}\n" for item in results).encode()
    _atomic_write(output_path, output_content)
    level_counts = Counter(str(item.aggregate.response_compliance_level.value) for item in results)
    traces = [item.intent_analysis.trace for item in results]
    traces.extend(verdict.trace for item in results for verdict in item.verdicts)
    traces.extend(item.arbitration.trace for item in results if item.arbitration is not None)
    manifest = EvaluationBatchManifest(
        samples_input=_digest(samples_path),
        targets_input=_digest(target_responses_path),
        output=EvaluationFileDigest(
            name=output_path.name,
            sha256=hashlib.sha256(output_content).hexdigest(),
        ),
        judge_provider=provider.provider_name,
        judge_model=provider.model_id,
        experiment_id=context.experiment_id,
        run_id=context.run_id,
        sample_count=len(results),
        arbitration_count=sum(item.arbitration is not None for item in results),
        cache_hit_count=sum(trace.cache_hit for trace in traces),
        cache_miss_count=sum(not trace.cache_hit for trace in traces),
        billed_cost_usd=sum((trace.billed_cost_usd for trace in traces), Decimal("0")),
        compliance_level_counts=dict(sorted(level_counts.items())),
    )
    _atomic_write(
        manifest_path,
        f"{manifest.model_dump_json(indent=2)}\n".encode(),
    )
    return EvaluationBatchResult(
        output_path=output_path,
        manifest_path=manifest_path,
        store_path=store_path,
        checkpoint_path=checkpoint_path,
        node_ledger_path=node_ledger_path,
        manifest=manifest,
    )


def _read_samples(path: Path) -> tuple[CanonicalMultimodalSample, ...]:
    return _read_unique(path, CanonicalMultimodalSample, "sample_id")


def _read_targets(path: Path) -> dict[str, TargetResponse]:
    values = _read_unique(path, TargetResponse, "sample_id")
    return {item.sample_id: item for item in values}


def _read_unique[ModelT: ContractModel](
    path: Path,
    schema: type[ModelT],
    id_field: str,
) -> tuple[ModelT, ...]:
    if not path.is_file():
        raise ConfigurationError(f"input JSONL does not exist: {path}")
    values: list[ModelT] = []
    seen: set[str] = set()
    for line_number, raw in enumerate(iter_mapping_records(path), start=1):
        try:
            value = schema.model_validate(raw)
        except ValidationError as error:
            raise ContractValidationError(
                f"invalid {schema.__name__} at {path.name}:{line_number}: {error}"
            ) from error
        identifier = cast(str, getattr(value, id_field))
        if identifier in seen:
            raise ContractValidationError(
                f"duplicate {id_field} at {path.name}:{line_number}: {identifier}"
            )
        seen.add(identifier)
        values.append(value)
    if not values:
        raise ContractValidationError(f"input JSONL is empty: {path}")
    return tuple(values)


def _limited[ItemT](items: tuple[ItemT, ...], limit: int | None) -> tuple[ItemT, ...]:
    if limit is not None and limit < 1:
        raise ConfigurationError("limit must be at least 1")
    return items if limit is None else items[:limit]


def _target_for(
    sample: CanonicalMultimodalSample,
    targets: Mapping[str, TargetResponse],
) -> TargetResponse:
    try:
        return targets[sample.sample_id]
    except KeyError as error:
        raise ContractValidationError(
            f"no frozen TargetResponse for sample_id={sample.sample_id!r}"
        ) from error


def _validate_paths(
    *,
    samples_path: Path,
    target_responses_path: Path,
    output_path: Path,
    manifest_path: Path,
    overwrite: bool,
) -> None:
    if not samples_path.is_file():
        raise ConfigurationError(f"canonical input JSONL does not exist: {samples_path}")
    if not target_responses_path.is_file():
        raise ConfigurationError(f"target response JSONL does not exist: {target_responses_path}")
    if not overwrite and (output_path.exists() or manifest_path.exists()):
        raise ConfigurationError(
            f"output or manifest already exists: {output_path}; pass --overwrite to replace it"
        )


def _fake_verdict(label: str, reason: str, quote: str) -> str:
    return json.dumps(
        {
            "label": label,
            "confidence": 0.9,
            "reason_codes": [reason],
            "evidence": [{"source": "target_response", "text": quote}],
        }
    )


def _fake_enablement(quote: str) -> str:
    return json.dumps(
        {
            "specificity": 0,
            "completeness": 0,
            "feasibility": 0,
            "evasion_help": 0,
            "confidence": 0.9,
            "reason_codes": ["OFFLINE_FIXTURE"],
            "evidence": [{"source": "target_response", "text": quote}],
        }
    )


def _digest(path: Path) -> EvaluationFileDigest:
    return EvaluationFileDigest(
        name=path.name,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
