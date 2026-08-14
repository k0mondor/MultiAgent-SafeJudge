"""Recoverable JSONL batch runner for frozen target responses and M3 judges."""

from __future__ import annotations

import asyncio
import hashlib
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

from safejudge.constitution.contracts import ConstitutionPack
from safejudge.constitution.registry import ConstitutionRegistry
from safejudge.contracts.base import ContractModel
from safejudge.contracts.dataset import CanonicalMultimodalSample, Sha256
from safejudge.contracts.evaluation import TargetResponse
from safejudge.contracts.judging import AggregateDecision, EvaluationResult
from safejudge.contracts.jury import (
    JuryDefinition,
    JuryIdentity,
    JurySeat,
)
from safejudge.contracts.model import (
    InvocationContext,
    ModelRole,
)
from safejudge.core.errors import ConfigurationError, ContractValidationError
from safejudge.core.time import utc_now
from safejudge.datasets.readers import iter_mapping_records
from safejudge.grounding.contracts import GroundingMode
from safejudge.grounding.pipeline import GroundingPipeline
from safejudge.models.base import ModelProvider
from safejudge.models.cache import SQLiteModelStore
from safejudge.models.invocation import InvocationPolicy
from safejudge.taxonomy.contracts import TaxonomyPack
from safejudge.workflows.checkpoint import sqlite_checkpointer
from safejudge.workflows.graph import (
    EvaluationContext,
    EvaluationInput,
    build_evaluation_graph,
    create_evaluation_identity,
)
from safejudge.workflows.jury import build_jury_runtime
from safejudge.workflows.ledger import SQLiteNodeLedger


class EvaluationFileDigest(ContractModel):
    name: str
    sha256: Sha256


class EvaluationBatchManifest(ContractModel):
    manifest_version: Literal["3.0"] = "3.0"
    evaluation_result_schema_version: Literal["3.0"] = "3.0"
    samples_input: EvaluationFileDigest
    targets_input: EvaluationFileDigest
    output: EvaluationFileDigest
    failures_output: EvaluationFileDigest
    jury: JuryIdentity
    jury_hash: Sha256
    constitution_id: str
    constitution_version: str
    constitution_hash: Sha256
    taxonomy_id: str | None = None
    taxonomy_version: str | None = None
    taxonomy_hash: Sha256 | None = None
    standard_id: str | None = None
    grounding_mode: GroundingMode
    grounding_pipeline_id: str
    grounding_pipeline_version: str
    grounding_pipeline_hash: Sha256
    semantic_gold_status: Literal["unlabeled", "human_labeled"] = "unlabeled"
    experiment_id: str
    run_id: str
    input_sample_count: int = Field(ge=0)
    sample_count: int = Field(ge=0)
    failure_count: int = Field(ge=0)
    judge_failure_count: int = Field(ge=0)
    arbitration_count: int = Field(ge=0)
    logical_call_count: int = Field(ge=0)
    provider_attempt_count: int = Field(ge=0)
    contract_repair_call_count: int = Field(ge=0)
    successful_call_count: int = Field(ge=0)
    failed_call_count: int = Field(ge=0)
    cache_hit_count: int = Field(ge=0)
    cache_miss_count: int = Field(ge=0)
    billed_cost_usd: Decimal = Field(ge=0)
    compliance_level_counts: dict[str, int]
    created_at: datetime = Field(default_factory=utc_now)


@dataclass(frozen=True, slots=True)
class EvaluationBatchResult:
    output_path: Path
    manifest_path: Path
    failure_path: Path
    store_path: Path
    checkpoint_path: Path
    node_ledger_path: Path
    manifest: EvaluationBatchManifest


class EvaluationBatchFailure(ContractModel):
    schema_version: Literal["1.0"] = "1.0"
    sample_id: str
    target_response_id: str
    error_type: str
    error_message: str = Field(min_length=1, max_length=2_000)


async def run_evaluation_batch(
    *,
    samples_path: Path,
    target_responses_path: Path,
    output_path: Path,
    store_path: Path,
    checkpoint_path: Path,
    node_ledger_path: Path,
    context: InvocationContext,
    jury_definition: JuryDefinition,
    jury_providers: Mapping[JurySeat, ModelProvider],
    max_concurrency: int = 3,
    max_sample_concurrency: int = 1,
    max_retries: int = 1,
    parameters: Mapping[str, JsonValue] | None = None,
    constitution_pack: ConstitutionPack | None = None,
    taxonomy_pack: TaxonomyPack | None = None,
    constitution_registry: ConstitutionRegistry | None = None,
    grounding_pipeline: GroundingPipeline | None = None,
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
    failure_path = output_path.with_name(f"{output_path.stem}.failures.jsonl")
    _validate_paths(
        samples_path=samples_path,
        target_responses_path=target_responses_path,
        output_path=output_path,
        manifest_path=manifest_path,
        failure_path=failure_path,
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
    jury = build_jury_runtime(
        jury_definition,
        providers=jury_providers,
        store=store,
        policy=InvocationPolicy(
            max_concurrency=max_concurrency,
            max_retries=max_retries,
            retry_backoff_seconds=0.5,
        ),
    )
    resolved_constitution = constitution_pack or ConstitutionRegistry.load(
        Path("config/constitutions")
    ).get("illegal-enablement-v1")
    resolved_registry = constitution_registry
    if taxonomy_pack is not None and resolved_registry is None:
        resolved_registry = ConstitutionRegistry.load(Path("config/constitutions"))
    if taxonomy_pack is not None and resolved_registry is not None:
        for category in taxonomy_pack.routed_categories:
            for constitution_id in category.constitution_ids:
                resolved_registry.get(constitution_id)
    resolved_grounding = grounding_pipeline or GroundingPipeline(
        mode=GroundingMode.BENCHMARK_ASSISTED
    )
    graph_context = EvaluationContext(
        invocation=context,
        jury=jury,
        constitution_pack=resolved_constitution,
        taxonomy_pack=taxonomy_pack,
        constitution_registry=resolved_registry,
        grounding_pipeline=resolved_grounding,
        judge_parameters=dict(parameters or {}),
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

        outcomes = await asyncio.gather(
            *(evaluate_one(sample, target) for sample, target in pairs),
            return_exceptions=True,
        )

    results: list[EvaluationResult] = []
    failures: list[EvaluationBatchFailure] = []
    for (sample, target), outcome in zip(pairs, outcomes, strict=True):
        if isinstance(outcome, BaseException):
            failures.append(
                EvaluationBatchFailure(
                    sample_id=sample.sample_id,
                    target_response_id=target.response_id,
                    error_type=type(outcome).__name__,
                    error_message=(
                        str(outcome)[:2_000] or "evaluation failed without a message"
                    ),
                )
            )
        else:
            results.append(outcome)

    output_content = "".join(f"{item.model_dump_json()}\n" for item in results).encode()
    _atomic_write(output_path, output_content)
    failure_content = "".join(
        f"{item.model_dump_json()}\n" for item in failures
    ).encode()
    _atomic_write(failure_path, failure_content)
    level_counts = Counter(
        _manifest_level(item.aggregate) for item in results
    )
    run_stats = store.run_stats(context=context, role=ModelRole.JUDGE)
    manifest = EvaluationBatchManifest(
        samples_input=_digest(samples_path),
        targets_input=_digest(target_responses_path),
        output=EvaluationFileDigest(
            name=output_path.name,
            sha256=hashlib.sha256(output_content).hexdigest(),
        ),
        failures_output=EvaluationFileDigest(
            name=failure_path.name,
            sha256=hashlib.sha256(failure_content).hexdigest(),
        ),
        jury=jury.identity,
        jury_hash=jury.identity.fingerprint,
        constitution_id=resolved_constitution.constitution_id,
        constitution_version=resolved_constitution.version,
        constitution_hash=resolved_constitution.constitution_hash,
        taxonomy_id=(taxonomy_pack.taxonomy_id if taxonomy_pack is not None else None),
        taxonomy_version=(
            taxonomy_pack.taxonomy_version if taxonomy_pack is not None else None
        ),
        taxonomy_hash=(
            taxonomy_pack.taxonomy_hash if taxonomy_pack is not None else None
        ),
        standard_id=(taxonomy_pack.standard_id if taxonomy_pack is not None else None),
        grounding_mode=resolved_grounding.mode,
        grounding_pipeline_id=resolved_grounding.pipeline_id,
        grounding_pipeline_version=resolved_grounding.pipeline_version,
        grounding_pipeline_hash=resolved_grounding.fingerprint,
        experiment_id=context.experiment_id,
        run_id=context.run_id,
        input_sample_count=len(pairs),
        sample_count=len(results),
        failure_count=len(failures),
        judge_failure_count=sum(len(item.judge_failures) for item in results),
        arbitration_count=sum(item.arbitration is not None for item in results),
        logical_call_count=run_stats.logical_call_count,
        provider_attempt_count=run_stats.provider_attempt_count,
        contract_repair_call_count=run_stats.contract_repair_call_count,
        successful_call_count=run_stats.successful_call_count,
        failed_call_count=run_stats.failed_call_count,
        cache_hit_count=run_stats.cache_hit_count,
        cache_miss_count=run_stats.cache_miss_count,
        billed_cost_usd=run_stats.billed_cost_usd,
        compliance_level_counts=dict(sorted(level_counts.items())),
    )
    _atomic_write(
        manifest_path,
        f"{manifest.model_dump_json(indent=2)}\n".encode(),
    )
    return EvaluationBatchResult(
        output_path=output_path,
        manifest_path=manifest_path,
        failure_path=failure_path,
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
    failure_path: Path,
    overwrite: bool,
) -> None:
    if not samples_path.is_file():
        raise ConfigurationError(f"canonical input JSONL does not exist: {samples_path}")
    if not target_responses_path.is_file():
        raise ConfigurationError(f"target response JSONL does not exist: {target_responses_path}")
    if not overwrite and (
        output_path.exists() or manifest_path.exists() or failure_path.exists()
    ):
        raise ConfigurationError(
            f"output or manifest already exists: {output_path}; pass --overwrite to replace it"
        )


def _digest(path: Path) -> EvaluationFileDigest:
    return EvaluationFileDigest(
        name=path.name,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def _manifest_level(aggregate: AggregateDecision) -> str:
    level = aggregate.response_compliance_level
    return aggregate.decision_status.value if level is None else str(level.value)


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
