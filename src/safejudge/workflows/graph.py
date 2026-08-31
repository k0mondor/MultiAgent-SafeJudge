"""Typed LangGraph implementation of the M3 main/sub-judge workflow."""

from __future__ import annotations

import hashlib
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from decimal import Decimal
from typing import Annotated, Any, Literal

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from langgraph.types import Send
from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from safejudge.aggregation import ShiftedProductPolicy, load_default_aggregation_policy
from safejudge.constitution.compiler import (
    CompiledConstitution,
    compile_constitution,
    validate_triggered_rule_ids,
)
from safejudge.constitution.contracts import ConstitutionPack
from safejudge.constitution.registry import ConstitutionRegistry
from safejudge.constitution.router import (
    CategoryConstitutionBinding,
    ConstitutionRoute,
    ConstitutionRouteAction,
    MultiCategoryRoute,
    route_categories,
    route_constitution,
)
from safejudge.contracts.base import ContractModel
from safejudge.contracts.dataset import CanonicalMultimodalSample, RequestIntent
from safejudge.contracts.evaluation import ResponseComplianceLevel, TargetResponse
from safejudge.contracts.judging import (
    AggregateDecision,
    AmbiguityKind,
    ArbitrationDecision,
    ArbitrationExecutionFailure,
    ArbitrationResolutionStatus,
    CategoryAnalysis,
    CategoryEvaluationResult,
    DecisionStatus,
    EnablementScores,
    EvaluationResult,
    EvaluationSpec,
    GuardrailExecutionFailure,
    GuardrailVerdict,
    IntentAnalysis,
    JudgeAxis,
    JudgeExecutionFailure,
    JudgeFailureCode,
    JudgeLabel,
    JudgeVerdict,
    RequestSnapshot,
    ScopeStatus,
)
from safejudge.contracts.model import InvocationContext
from safejudge.core.errors import (
    BudgetExceededError,
    ConfigurationError,
    ContractValidationError,
    JudgeContractError,
    ProviderError,
    SafeJudgeError,
    UnsupportedModalityError,
)
from safejudge.grounding.contracts import (
    GroundingArtifact,
    GroundingMode,
    GroundingStatus,
)
from safejudge.grounding.pipeline import GroundingPipeline
from safejudge.models.target import TargetRunner
from safejudge.taxonomy.contracts import TaxonomyPack
from safejudge.workflows.jury import JuryRuntime
from safejudge.workflows.ledger import NodeLedger, NodeRunSpan
from safejudge.workflows.prompts import (
    PROMPT_BUNDLE_VERSION,
    RUBRIC_VERSION,
    prompt_bundle_hash,
    rubric_hash,
)


def _merge_verdicts(
    left: tuple[JudgeVerdict, ...] | list[JudgeVerdict],
    right: tuple[JudgeVerdict, ...] | list[JudgeVerdict],
) -> tuple[JudgeVerdict, ...]:
    """Checkpoint serializers may restore pending tuple writes as lists."""

    return (*left, *right)


def _merge_failures(
    left: tuple[JudgeExecutionFailure, ...] | list[JudgeExecutionFailure],
    right: tuple[JudgeExecutionFailure, ...] | list[JudgeExecutionFailure],
) -> tuple[JudgeExecutionFailure, ...]:
    return (*left, *right)


def _merge_guardrail_verdicts(
    left: tuple[GuardrailVerdict, ...] | list[GuardrailVerdict],
    right: tuple[GuardrailVerdict, ...] | list[GuardrailVerdict],
) -> tuple[GuardrailVerdict, ...]:
    return (*left, *right)


def _merge_guardrail_failures(
    left: tuple[GuardrailExecutionFailure, ...] | list[GuardrailExecutionFailure],
    right: tuple[GuardrailExecutionFailure, ...] | list[GuardrailExecutionFailure],
) -> tuple[GuardrailExecutionFailure, ...]:
    return (*left, *right)


class EvaluationInput(ContractModel):
    """One canonical sample, optionally paired with an already frozen target answer."""

    sample: CanonicalMultimodalSample
    target_response: TargetResponse | None = None


class EvaluationOutput(ContractModel):
    result: EvaluationResult


class EvaluationState(BaseModel):
    """Checkpointed graph state. Run dependencies live in EvaluationContext."""

    model_config = ConfigDict(extra="forbid")

    sample: CanonicalMultimodalSample
    target_response: TargetResponse | None = None
    request_snapshot: RequestSnapshot | None = None
    evaluation_spec: EvaluationSpec | None = None
    grounding_artifact: GroundingArtifact | None = None
    intent_analysis: IntentAnalysis | None = None
    category_analysis: CategoryAnalysis | None = None
    constitution_route: ConstitutionRoute | None = None
    category_route: MultiCategoryRoute | None = None
    verdicts: Annotated[tuple[JudgeVerdict, ...], _merge_verdicts] = ()
    judge_failures: Annotated[tuple[JudgeExecutionFailure, ...], _merge_failures] = ()
    guardrail_verdicts: Annotated[tuple[GuardrailVerdict, ...], _merge_guardrail_verdicts] = ()
    guardrail_failures: Annotated[
        tuple[GuardrailExecutionFailure, ...], _merge_guardrail_failures
    ] = ()
    category_results: tuple[CategoryEvaluationResult, ...] = ()
    routed_axes: tuple[JudgeAxis, ...] = ()
    aggregate: AggregateDecision | None = None
    arbitration: ArbitrationDecision | None = None
    result: EvaluationResult | None = None


class EvaluationContext(BaseModel):
    """Immutable, non-checkpointed dependencies and trusted run configuration."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        arbitrary_types_allowed=True,
    )

    invocation: InvocationContext
    jury: JuryRuntime
    constitution_pack: ConstitutionPack
    taxonomy_pack: TaxonomyPack | None = None
    constitution_registry: ConstitutionRegistry | None = None
    grounding_pipeline: GroundingPipeline = Field(
        default_factory=lambda: GroundingPipeline(mode=GroundingMode.BENCHMARK_ASSISTED)
    )
    target_runner: TargetRunner | None = None
    judge_parameters: dict[str, JsonValue] = Field(default_factory=dict)
    rubric_version: str = RUBRIC_VERSION
    aggregation_policy: ShiftedProductPolicy = Field(
        default_factory=load_default_aggregation_policy
    )
    node_ledger: NodeLedger | None = None

    @model_validator(mode="after")
    def taxonomy_dependencies_are_paired(self) -> EvaluationContext:
        if (self.taxonomy_pack is None) != (self.constitution_registry is None):
            raise ValueError("taxonomy_pack and constitution_registry must be provided together")
        return self


class JudgeTaskInput(ContractModel):
    sample_id: str
    evaluation_key: str
    request_snapshot: RequestSnapshot
    target_response: TargetResponse
    grounding_artifact: GroundingArtifact
    intent_analysis: IntentAnalysis
    constitution_scenarios: tuple[str, ...]
    category_id: str | None = None
    category_origin: Literal["request", "response_added"] = "request"
    category_name: str | None = None
    parent_id: str | None = None
    parent_name: str | None = None
    standard_clause: str | None = None
    operational_definition: str | None = None
    inclusion_anchors: tuple[str, ...] = ()
    exclusion_anchors: tuple[str, ...] = ()
    constitution_id: str | None = None


class JudgeTaskState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sample_id: str
    evaluation_key: str
    request_snapshot: RequestSnapshot
    target_response: TargetResponse
    grounding_artifact: GroundingArtifact
    intent_analysis: IntentAnalysis
    constitution_scenarios: tuple[str, ...]
    category_id: str | None = None
    category_origin: Literal["request", "response_added"] = "request"
    category_name: str | None = None
    parent_id: str | None = None
    parent_name: str | None = None
    standard_clause: str | None = None
    operational_definition: str | None = None
    inclusion_anchors: tuple[str, ...] = ()
    exclusion_anchors: tuple[str, ...] = ()
    constitution_id: str | None = None
    verdicts: tuple[JudgeVerdict, ...] = ()
    judge_failures: tuple[JudgeExecutionFailure, ...] = ()
    guardrail_verdicts: tuple[GuardrailVerdict, ...] = ()
    guardrail_failures: tuple[GuardrailExecutionFailure, ...] = ()


class JudgeTaskOutput(ContractModel):
    verdicts: tuple[JudgeVerdict, ...]
    judge_failures: tuple[JudgeExecutionFailure, ...]
    guardrail_verdicts: tuple[GuardrailVerdict, ...] = ()
    guardrail_failures: tuple[GuardrailExecutionFailure, ...] = ()


EvaluationGraph = CompiledStateGraph[
    EvaluationState,
    EvaluationContext,
    EvaluationInput,
    EvaluationOutput,
]


def build_evaluation_graph(
    *,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
) -> EvaluationGraph:
    """Build the target -> intake -> isolated panel -> aggregate graph."""

    builder = StateGraph(
        EvaluationState,
        context_schema=EvaluationContext,
        input_schema=EvaluationInput,
        output_schema=EvaluationOutput,
    )
    # LangGraph injects Runtime and RunnableConfig together at runtime, but its
    # public StateNode type alias does not yet describe that supported signature.
    builder.add_node("target_answer", _target_answer)  # type: ignore[call-overload]
    builder.add_node("prepare_evaluation", _prepare_evaluation)  # type: ignore[call-overload]
    builder.add_node("request_grounding", _request_grounding)  # type: ignore[call-overload]
    builder.add_node("request_analysis", _request_analysis)  # type: ignore[call-overload]
    builder.add_node(  # type: ignore[call-overload]
        "response_category_enrichment", _response_category_enrichment
    )
    builder.add_node("constitution_route", _constitution_route)  # type: ignore[call-overload]
    builder.add_node("route_terminal", _route_terminal)  # type: ignore[call-overload]
    for axis in JudgeAxis:
        builder.add_node(
            _judge_node_name(axis),
            _build_judge_subgraph(axis),
            input_schema=JudgeTaskInput,
        )
    builder.add_node("aggregate", _aggregate)  # type: ignore[call-overload]
    builder.add_node("arbitrate", _arbitrate)  # type: ignore[call-overload]
    builder.add_node("finalize", _finalize)  # type: ignore[call-overload]

    builder.add_edge(START, "target_answer")
    builder.add_edge("target_answer", "prepare_evaluation")
    builder.add_edge("prepare_evaluation", "request_grounding")
    builder.add_edge("request_grounding", "request_analysis")
    builder.add_edge("request_analysis", "response_category_enrichment")
    builder.add_edge("response_category_enrichment", "constitution_route")
    builder.add_conditional_edges(
        "constitution_route",
        _route_after_constitution,
        [*[_judge_node_name(axis) for axis in JudgeAxis], "route_terminal"],
    )
    builder.add_edge("route_terminal", "finalize")
    for axis in JudgeAxis:
        builder.add_edge(_judge_node_name(axis), "aggregate")
    builder.add_conditional_edges(
        "aggregate",
        _route_after_aggregate,
        {"arbitrate": "arbitrate", "finalize": "finalize"},
    )
    builder.add_edge("arbitrate", "finalize")
    builder.add_edge("finalize", END)
    return builder.compile(checkpointer=checkpointer, name="safejudge-m3-evaluation")


async def _target_answer(
    state: EvaluationState,
    runtime: Runtime[EvaluationContext],
    config: RunnableConfig,
) -> dict[str, TargetResponse]:
    async with _track_node(
        runtime=runtime,
        config=config,
        node_name="target_answer",
        sample_id=state.sample.sample_id,
        evaluation_key=None,
        input_value=state,
    ) as span:
        if state.target_response is not None:
            if state.target_response.sample_id != state.sample.sample_id:
                raise ContractValidationError(
                    "provided target response does not belong to the canonical sample"
                )
            return _record_output(span, {})
        runner = runtime.context.target_runner
        if runner is None:
            raise ConfigurationError(
                "target_runner is required when EvaluationInput has no target_response"
            )
        response = await runner.run(
            state.sample,
            context=runtime.context.invocation,
        )
        return _record_output(span, {"target_response": response})


async def _prepare_evaluation(
    state: EvaluationState,
    runtime: Runtime[EvaluationContext],
    config: RunnableConfig,
) -> dict[str, RequestSnapshot | EvaluationSpec]:
    async with _track_node(
        runtime=runtime,
        config=config,
        node_name="prepare_evaluation",
        sample_id=state.sample.sample_id,
        evaluation_key=None,
        input_value=state,
    ) as span:
        response = _require_target_response(state)
        snapshot, spec = create_evaluation_identity(
            sample=state.sample,
            target_response=response,
            context=runtime.context,
        )
        return _record_output(
            span,
            {"request_snapshot": snapshot, "evaluation_spec": spec},
        )


def create_evaluation_identity(
    *,
    sample: CanonicalMultimodalSample,
    target_response: TargetResponse,
    context: EvaluationContext,
) -> tuple[RequestSnapshot, EvaluationSpec]:
    """Build the immutable request packet and semantic idempotency identity."""

    snapshot = RequestSnapshot.from_sample(sample)
    spec = EvaluationSpec.create(
        sample_hash=_canonical_hash(sample.model_dump(mode="json")),
        request_snapshot_hash=snapshot.sha256,
        target_response_hash=_target_response_hash(target_response),
        target_provider=target_response.model.provider,
        target_model=target_response.model.model,
        target_revision=target_response.model.revision,
        jury=context.jury.identity,
        jury_hash=context.jury.identity.fingerprint,
        constitution_id=context.constitution_pack.constitution_id,
        constitution_version=context.constitution_pack.version,
        constitution_hash=context.constitution_pack.constitution_hash,
        constitution_router_version="constitution-router-v2",
        scope_id=context.constitution_pack.scope_id,
        taxonomy_id=(
            context.taxonomy_pack.taxonomy_id if context.taxonomy_pack is not None else None
        ),
        taxonomy_version=(
            context.taxonomy_pack.taxonomy_version if context.taxonomy_pack is not None else None
        ),
        taxonomy_hash=(
            context.taxonomy_pack.taxonomy_hash if context.taxonomy_pack is not None else None
        ),
        standard_id=(
            context.taxonomy_pack.standard_id if context.taxonomy_pack is not None else None
        ),
        grounding_mode=context.grounding_pipeline.mode.value,
        grounding_pipeline_id=context.grounding_pipeline.pipeline_id,
        grounding_pipeline_version=context.grounding_pipeline.pipeline_version,
        grounding_pipeline_hash=context.grounding_pipeline.fingerprint,
        prompt_bundle_version=PROMPT_BUNDLE_VERSION,
        prompt_bundle_hash=prompt_bundle_hash(),
        rubric_version=context.rubric_version,
        rubric_hash=rubric_hash(),
        aggregator_id=context.aggregation_policy.aggregator_id,
        aggregator_version=context.aggregation_policy.aggregator_version,
        aggregator_hash=context.aggregation_policy.fingerprint,
        judge_parameters_hash=_canonical_hash(context.judge_parameters),
    )
    return snapshot, spec


async def _request_grounding(
    state: EvaluationState,
    runtime: Runtime[EvaluationContext],
    config: RunnableConfig,
) -> dict[str, GroundingArtifact]:
    async with _track_node(
        runtime=runtime,
        config=config,
        node_name="request_grounding",
        sample_id=state.sample.sample_id,
        evaluation_key=_require_evaluation_spec(state).evaluation_key,
        input_value=state,
    ) as span:
        artifact = await runtime.context.grounding_pipeline.run(
            state.sample,
            context=runtime.context.invocation,
        )
        return _record_output(span, {"grounding_artifact": artifact})


async def _request_analysis(
    state: EvaluationState,
    runtime: Runtime[EvaluationContext],
    config: RunnableConfig,
) -> dict[str, IntentAnalysis | CategoryAnalysis]:
    async with _track_node(
        runtime=runtime,
        config=config,
        node_name="request_analysis",
        sample_id=state.sample.sample_id,
        evaluation_key=_require_evaluation_spec(state).evaluation_key,
        input_value=state,
    ) as span:
        request_snapshot = _require_request_snapshot(state)
        grounding = _require_grounding(state)
        taxonomy = runtime.context.taxonomy_pack
        intent_constitution = compile_constitution(
            runtime.context.constitution_pack,
            axis="intent",
            scenarios=frozenset({grounding.mode.value}),
        )
        intent = await runtime.context.jury.intent.analyze_intent(
            request_snapshot,
            grounding=grounding,
            constitution=intent_constitution,
            taxonomy=taxonomy,
            context=runtime.context.invocation,
            parameters=runtime.context.judge_parameters,
        )
        if (
            intent.ambiguity_kind is AmbiguityKind.INSUFFICIENT_GROUNDING
            and grounding.status is GroundingStatus.COMPLETE
            and grounding.observations
        ):
            intent = await runtime.context.jury.intent.review_insufficient_grounding(
                request_snapshot,
                previous_intent=intent,
                grounding=grounding,
                constitution=intent_constitution,
                taxonomy=taxonomy,
                context=runtime.context.invocation,
                parameters=runtime.context.judge_parameters,
            )
        if (
            taxonomy is not None
            and intent.scope_status is ScopeStatus.OUT_OF_SCOPE
            and grounding.status is GroundingStatus.COMPLETE
        ):
            intent = await runtime.context.jury.intent.review_out_of_scope(
                request_snapshot,
                previous_intent=intent,
                grounding=grounding,
                constitution=intent_constitution,
                taxonomy=taxonomy,
                context=runtime.context.invocation,
                parameters=runtime.context.judge_parameters,
            )
        output: dict[str, IntentAnalysis | CategoryAnalysis] = {"intent_analysis": intent}
        if taxonomy is None:
            return _record_output(span, output)
        category_analysis = CategoryAnalysis(
            category_ids=intent.request_category_ids,
            request_category_ids=intent.request_category_ids,
            taxonomy_id=taxonomy.taxonomy_id,
            taxonomy_version=taxonomy.taxonomy_version,
            standard_id=taxonomy.standard_id,
            prompt_version=intent.category_prompt_version or intent.prompt_version,
            trace=intent.category_trace or intent.trace,
        )
        if (
            not category_analysis.category_ids
            and intent.request_intent is RequestIntent.HARMFUL
            and intent.scope_status is ScopeStatus.IN_SCOPE
            and grounding.status is GroundingStatus.COMPLETE
        ):
            category_analysis = await runtime.context.jury.intent.review_empty_request_categories(
                request_snapshot,
                intent=intent,
                grounding=grounding,
                taxonomy=taxonomy,
                context=runtime.context.invocation,
                parameters=runtime.context.judge_parameters,
            )
            intent = intent.model_copy(
                update={
                    "request_category_ids": category_analysis.category_ids,
                    "category_prompt_version": category_analysis.prompt_version,
                    "category_trace": category_analysis.trace,
                }
            )
            output["intent_analysis"] = intent
        output["category_analysis"] = category_analysis
        return _record_output(span, output)


async def _response_category_enrichment(
    state: EvaluationState,
    runtime: Runtime[EvaluationContext],
    config: RunnableConfig,
) -> dict[str, CategoryAnalysis]:
    taxonomy = runtime.context.taxonomy_pack
    if taxonomy is None:
        return {}
    async with _track_node(
        runtime=runtime,
        config=config,
        node_name="response_category_enrichment",
        sample_id=state.sample.sample_id,
        evaluation_key=_require_evaluation_spec(state).evaluation_key,
        input_value=state,
    ) as span:
        analysis = await runtime.context.jury.intent.classify_response_added_categories(
            _require_request_snapshot(state),
            target_response=_require_target_response(state),
            intent=_require_intent(state),
            grounding=_require_grounding(state),
            taxonomy=taxonomy,
            context=runtime.context.invocation,
            parameters=runtime.context.judge_parameters,
        )
        return _record_output(span, {"category_analysis": analysis})


async def _constitution_route(
    state: EvaluationState,
    runtime: Runtime[EvaluationContext],
    config: RunnableConfig,
) -> dict[
    str,
    ConstitutionRoute | MultiCategoryRoute | tuple[JudgeAxis, ...] | None,
]:
    async with _track_node(
        runtime=runtime,
        config=config,
        node_name="constitution_route",
        sample_id=state.sample.sample_id,
        evaluation_key=_require_evaluation_spec(state).evaluation_key,
        input_value=state,
    ) as span:
        intent = _require_intent(state)
        grounding = _require_grounding(state)
        route = route_constitution(
            runtime.context.constitution_pack,
            intent=intent,
            grounding=grounding,
        )
        axes = [JudgeAxis.COMPLIANCE, JudgeAxis.HARM_ENABLEMENT]
        if intent.request_intent is RequestIntent.BENIGN:
            axes.append(JudgeAxis.OVERSENSITIVITY)
        category_route: MultiCategoryRoute | None = None
        if runtime.context.taxonomy_pack is not None:
            registry = runtime.context.constitution_registry
            analysis = state.category_analysis
            if registry is None or analysis is None:
                raise ContractValidationError(
                    "taxonomy evaluation requires category analysis and registry"
                )
            if tuple(sorted(intent.request_category_ids)) != tuple(
                sorted(analysis.request_category_ids)
            ):
                raise ContractValidationError(
                    "Request Analyzer categories differ from compatibility route view"
                )
            category_route = route_categories(
                runtime.context.taxonomy_pack,
                registry,
                intent=intent,
                grounding=grounding,
                category_ids=analysis.category_ids,
                response_added_category_ids=analysis.response_added_category_ids,
            )
        return _record_output(
            span,
            {
                "constitution_route": route,
                "category_route": category_route,
                "routed_axes": tuple(axes),
            },
        )


def _fan_out_panel(state: EvaluationState) -> list[Send]:
    response = _require_target_response(state)
    intent = _require_intent(state)
    if state.category_route is not None:
        tasks = [
            Send(
                _judge_node_name(axis),
                JudgeTaskInput(
                    sample_id=state.sample.sample_id,
                    evaluation_key=_require_evaluation_spec(state).evaluation_key,
                    request_snapshot=_require_request_snapshot(state),
                    target_response=response,
                    grounding_artifact=_require_grounding(state),
                    intent_analysis=intent,
                    constitution_scenarios=state.category_route.scenarios,
                    category_id=binding.category_id,
                    category_origin=binding.category_origin,
                    category_name=binding.category_name,
                    parent_id=binding.parent_id,
                    parent_name=binding.parent_name,
                    standard_clause=binding.standard_clause,
                    operational_definition=binding.operational_definition,
                    inclusion_anchors=binding.inclusion_anchors,
                    exclusion_anchors=binding.exclusion_anchors,
                    constitution_id=binding.constitution_id,
                ),
            )
            for binding in state.category_route.bindings
            for axis in (JudgeAxis.COMPLIANCE, JudgeAxis.HARM_ENABLEMENT)
        ]
        if intent.request_intent is RequestIntent.BENIGN:
            tasks.append(
                Send(
                    _judge_node_name(JudgeAxis.OVERSENSITIVITY),
                    JudgeTaskInput(
                        sample_id=state.sample.sample_id,
                        evaluation_key=_require_evaluation_spec(state).evaluation_key,
                        request_snapshot=_require_request_snapshot(state),
                        target_response=response,
                        grounding_artifact=_require_grounding(state),
                        intent_analysis=intent,
                        constitution_scenarios=_require_constitution_route(state).scenarios,
                    ),
                )
            )
        return tasks
    return [
        Send(
            _judge_node_name(axis),
            JudgeTaskInput(
                sample_id=state.sample.sample_id,
                evaluation_key=_require_evaluation_spec(state).evaluation_key,
                request_snapshot=_require_request_snapshot(state),
                target_response=response,
                grounding_artifact=_require_grounding(state),
                intent_analysis=intent,
                constitution_scenarios=_require_constitution_route(state).scenarios,
            ),
        )
        for axis in state.routed_axes
    ]


def _route_after_constitution(state: EvaluationState) -> list[Send] | str:
    if _effective_route_action(state) is ConstitutionRouteAction.EVALUATE:
        return _fan_out_panel(state)
    return "route_terminal"


async def _route_terminal(
    state: EvaluationState,
    runtime: Runtime[EvaluationContext],
    config: RunnableConfig,
) -> dict[str, AggregateDecision]:
    action = _effective_route_action(state)
    decision_status = (
        DecisionStatus.NOT_EVALUATED
        if action is ConstitutionRouteAction.NOT_EVALUATED
        else DecisionStatus.REVIEW_REQUIRED
    )
    async with _track_node(
        runtime=runtime,
        config=config,
        node_name="route_terminal",
        sample_id=state.sample.sample_id,
        evaluation_key=_require_evaluation_spec(state).evaluation_key,
        input_value=state,
    ) as span:
        conflicts: tuple[str, ...] = ()
        if decision_status is DecisionStatus.REVIEW_REQUIRED:
            grounding_errors = tuple(_require_grounding(state).error_codes)
            if grounding_errors:
                conflicts = grounding_errors
            elif _require_intent(state).ambiguity_kind is not AmbiguityKind.NONE:
                conflicts = (
                    {
                        AmbiguityKind.SEMANTIC_INTENT: "AMBIGUOUS_INTENT",
                        AmbiguityKind.INSUFFICIENT_GROUNDING: "INSUFFICIENT_GROUNDING",
                        AmbiguityKind.CONTRADICTORY_EVIDENCE: "CONTRADICTORY_EVIDENCE",
                    }[_require_intent(state).ambiguity_kind],
                )
            elif (
                state.category_route is not None and not state.category_route.selected_category_ids
            ):
                conflicts = ("CATEGORY_ROUTING_EMPTY",)
            else:
                conflicts = ("INSUFFICIENT_GROUNDING",)
        aggregate = AggregateDecision(
            decision_status=decision_status,
            response_compliance_level=None,
            provisional_level=None,
            oversensitive=None,
            conflict_codes=conflicts,
            resolution_source="review" if conflicts else "deterministic",
        )
        return _record_output(span, {"aggregate": aggregate})


def _build_judge_subgraph(
    axis: JudgeAxis,
) -> CompiledStateGraph[
    JudgeTaskState,
    EvaluationContext,
    JudgeTaskInput,
    JudgeTaskOutput,
]:
    async def judge_node(
        state: JudgeTaskState,
        runtime: Runtime[EvaluationContext],
        config: RunnableConfig,
    ) -> dict[
        str,
        tuple[JudgeVerdict, ...]
        | tuple[JudgeExecutionFailure, ...]
        | tuple[GuardrailVerdict, ...]
        | tuple[GuardrailExecutionFailure, ...],
    ]:
        node_name = _judge_node_name(axis)
        async with _track_node(
            runtime=runtime,
            config=config,
            node_name=node_name,
            sample_id=state.sample_id,
            evaluation_key=state.evaluation_key,
            input_value=state,
        ) as span:
            output: dict[
                str,
                tuple[JudgeVerdict, ...]
                | tuple[JudgeExecutionFailure, ...]
                | tuple[GuardrailVerdict, ...]
                | tuple[GuardrailExecutionFailure, ...],
            ] = {}
            compiled: CompiledConstitution | None = None
            try:
                policy_pack = runtime.context.constitution_pack
                if state.constitution_id is not None:
                    registry = runtime.context.constitution_registry
                    if registry is None:
                        raise ConfigurationError("category judge requires Constitution registry")
                    policy_pack = registry.get(state.constitution_id)
                compiled = compile_constitution(
                    policy_pack,
                    axis=axis,
                    scenarios=frozenset(state.constitution_scenarios),
                    category_id=state.category_id,
                )
            except SafeJudgeError as error:
                output["judge_failures"] = (
                    _judge_failure(
                        axis,
                        state.sample_id,
                        error,
                        category_id=state.category_id,
                        constitution_id=state.constitution_id,
                    ),
                )
                return _record_output(span, output)

            guardrail = runtime.context.jury.guardrail
            if (
                axis is JudgeAxis.COMPLIANCE
                and guardrail is not None
                and state.category_id is not None
                and state.constitution_id is not None
            ):
                category_id = state.category_id
                constitution_id = state.constitution_id
                try:
                    guardrail_verdict = await guardrail.judge_category(
                        sample_id=state.sample_id,
                        request_snapshot=state.request_snapshot,
                        target_response=state.target_response,
                        grounding=state.grounding_artifact,
                        intent=state.intent_analysis,
                        constitution=compiled,
                        category_id=category_id,
                        constitution_id=constitution_id,
                        category_name=state.category_name,
                        parent_id=state.parent_id,
                        parent_name=state.parent_name,
                        standard_clause=state.standard_clause,
                        operational_definition=state.operational_definition,
                        inclusion_anchors=state.inclusion_anchors,
                        exclusion_anchors=state.exclusion_anchors,
                        context_mode=runtime.context.jury.subjudge_context_mode,
                        context=runtime.context.invocation,
                        parameters=runtime.context.judge_parameters,
                    )
                    compliance_verdict = _guardrail_compliance_verdict(
                        guardrail_verdict,
                        constitution=compiled,
                    )
                except SafeJudgeError as error:
                    output["judge_failures"] = (
                        _judge_failure(
                            axis,
                            state.sample_id,
                            error,
                            category_id=category_id,
                            constitution_id=constitution_id,
                        ),
                    )
                    output["guardrail_failures"] = (
                        _guardrail_failure(
                            state.sample_id,
                            category_id,
                            constitution_id,
                            error,
                        ),
                    )
                else:
                    output["verdicts"] = (compliance_verdict,)
                    output["guardrail_verdicts"] = (guardrail_verdict,)
                return _record_output(span, output)

            try:
                verdict = await runtime.context.jury.for_axis(axis).judge(
                    axis=axis,
                    sample_id=state.sample_id,
                    request_snapshot=state.request_snapshot,
                    target_response=state.target_response,
                    grounding=state.grounding_artifact,
                    intent=state.intent_analysis,
                    constitution=compiled,
                    category_id=state.category_id,
                    category_origin=state.category_origin,
                    category_name=state.category_name,
                    parent_id=state.parent_id,
                    parent_name=state.parent_name,
                    standard_clause=state.standard_clause,
                    operational_definition=state.operational_definition,
                    inclusion_anchors=state.inclusion_anchors,
                    exclusion_anchors=state.exclusion_anchors,
                    constitution_id=state.constitution_id,
                    context_mode=runtime.context.jury.subjudge_context_mode,
                    context=runtime.context.invocation,
                    parameters=runtime.context.judge_parameters,
                )
            except SafeJudgeError as error:
                failure = _judge_failure(
                    axis,
                    state.sample_id,
                    error,
                    category_id=state.category_id,
                    constitution_id=state.constitution_id,
                )
                output["judge_failures"] = (failure,)
            else:
                output["verdicts"] = (verdict,)
            return _record_output(span, output)

    builder = StateGraph(
        JudgeTaskState,
        context_schema=EvaluationContext,
        input_schema=JudgeTaskInput,
        output_schema=JudgeTaskOutput,
    )
    builder.add_node("judge", judge_node)  # type: ignore[call-overload]
    builder.add_edge(START, "judge")
    builder.add_edge("judge", END)
    return builder.compile(name=f"isolated-{axis.value}-judge")


async def _aggregate(
    state: EvaluationState,
    runtime: Runtime[EvaluationContext],
    config: RunnableConfig,
) -> dict[str, AggregateDecision | tuple[CategoryEvaluationResult, ...]]:
    async with _track_node(
        runtime=runtime,
        config=config,
        node_name="aggregate",
        sample_id=state.sample.sample_id,
        evaluation_key=_require_evaluation_spec(state).evaluation_key,
        input_value=state,
    ) as span:
        if state.category_route is not None:
            category_results = tuple(
                _aggregate_category_binding(state, runtime.context, binding)
                for binding in state.category_route.bindings
            )
            top_aggregate = _aggregate_category_results(state, category_results)
            return _record_output(
                span,
                {
                    "category_results": category_results,
                    "aggregate": top_aggregate,
                },
            )
        if state.judge_failures:
            failures = tuple(sorted(state.judge_failures, key=lambda item: item.axis.value))
            aggregate = AggregateDecision(
                decision_status=DecisionStatus.REVIEW_REQUIRED,
                response_compliance_level=None,
                provisional_level=None,
                oversensitive=None,
                conflict_codes=tuple(
                    f"JUDGE_FAILURE:{failure.axis.value}:{failure.failure_code.value}"
                    for failure in failures
                ),
                requires_arbitration=False,
                resolution_source="review",
            )
            return _record_output(span, {"aggregate": aggregate})
        verdicts = _ordered_verdicts(state.verdicts)
        by_axis = {verdict.axis: verdict for verdict in verdicts}
        enablement = by_axis[JudgeAxis.HARM_ENABLEMENT]
        compliance = by_axis[JudgeAxis.COMPLIANCE]
        oversensitivity = by_axis.get(JudgeAxis.OVERSENSITIVITY)
        score, level, conflicts = _score_compliance_facets(
            runtime.context.aggregation_policy,
            compliance_label=compliance.label,
            facets=enablement.enablement_scores,
        )
        aggregate_policy = compile_constitution(
            runtime.context.constitution_pack,
            axis="aggregation",
            scenarios=frozenset(_require_constitution_route(state).scenarios),
        )
        if level is not None:
            level = _apply_enablement_lower_bound(
                level,
                original_enablement_label=enablement.label,
                minimum_level=aggregate_policy.deterministic_minimum_level,
            )
        oversensitive: bool | None
        if (
            _require_intent(state).request_intent is not RequestIntent.BENIGN
            or oversensitivity is None
            or oversensitivity.label is JudgeLabel.NOT_APPLICABLE
        ):
            oversensitive = None
        else:
            oversensitive = oversensitivity.label is JudgeLabel.OVERSENSITIVE
        return _record_output(
            span,
            {
                "aggregate": AggregateDecision(
                    decision_status=(
                        DecisionStatus.REVIEW_REQUIRED if conflicts else DecisionStatus.RESOLVED
                    ),
                    response_compliance_level=None if conflicts else level,
                    provisional_level=None,
                    oversensitive=oversensitive,
                    overall_score=None if conflicts else score,
                    conflict_codes=conflicts,
                    requires_arbitration=bool(conflicts),
                    resolution_source="review" if conflicts else "deterministic",
                )
            },
        )


def _aggregate_category_binding(
    state: EvaluationState,
    context: EvaluationContext,
    binding: CategoryConstitutionBinding,
) -> CategoryEvaluationResult:
    verdicts = tuple(
        verdict
        for verdict in state.verdicts
        if verdict.category_id == binding.category_id
        and verdict.constitution_id == binding.constitution_id
    )
    failures = tuple(
        failure
        for failure in state.judge_failures
        if failure.category_id == binding.category_id
        and failure.constitution_id == binding.constitution_id
    )
    guardrail_verdicts = tuple(
        verdict
        for verdict in state.guardrail_verdicts
        if verdict.category_id == binding.category_id
        and verdict.constitution_id == binding.constitution_id
    )
    guardrail_failures = tuple(
        failure
        for failure in state.guardrail_failures
        if failure.category_id == binding.category_id
        and failure.constitution_id == binding.constitution_id
    )
    guardrail_outcomes = len(guardrail_verdicts) + len(guardrail_failures)
    expected_guardrail_outcomes = 1 if context.jury.guardrail is not None else 0
    if guardrail_outcomes != expected_guardrail_outcomes:
        raise ContractValidationError(
            f"category guardrail {binding.category_id} has missing or duplicate outcomes"
        )
    axes = {verdict.axis for verdict in verdicts}
    failed_axes = {failure.axis for failure in failures}
    expected = {JudgeAxis.COMPLIANCE, JudgeAxis.HARM_ENABLEMENT}
    if axes.intersection(failed_axes) or axes.union(failed_axes) != expected:
        raise ContractValidationError(
            f"category panel {binding.category_id} has missing or duplicate axes"
        )
    if failures:
        aggregate = AggregateDecision(
            decision_status=DecisionStatus.REVIEW_REQUIRED,
            response_compliance_level=None,
            provisional_level=None,
            oversensitive=None,
            conflict_codes=tuple(
                f"JUDGE_FAILURE:{failure.axis.value}:{failure.failure_code.value}"
                for failure in failures
            ),
            requires_arbitration=False,
            resolution_source="review",
        )
    else:
        by_axis = {verdict.axis: verdict for verdict in verdicts}
        compliance = by_axis[JudgeAxis.COMPLIANCE]
        enablement = by_axis[JudgeAxis.HARM_ENABLEMENT]
        score, level, conflicts = _score_compliance_facets(
            context.aggregation_policy,
            compliance_label=compliance.label,
            facets=enablement.enablement_scores,
        )
        registry = context.constitution_registry
        if registry is None:
            raise ConfigurationError("category aggregation requires Constitution registry")
        aggregate_policy = compile_constitution(
            registry.get(binding.constitution_id),
            axis="aggregation",
            scenarios=frozenset(_require_category_route(state).scenarios),
            category_id=binding.category_id,
        )
        if level is not None:
            level = _apply_enablement_lower_bound(
                level,
                original_enablement_label=enablement.label,
                minimum_level=aggregate_policy.deterministic_minimum_level,
            )
        provisional_level = None
        if conflicts:
            provisional_level = _conservative_conflict_level(
                context.aggregation_policy,
                facets=enablement.enablement_scores,
                enablement_label=enablement.label,
                minimum_level=aggregate_policy.deterministic_minimum_level,
            )
        aggregate = AggregateDecision(
            decision_status=(
                DecisionStatus.REVIEW_REQUIRED if conflicts else DecisionStatus.RESOLVED
            ),
            response_compliance_level=None if conflicts else level,
            provisional_level=provisional_level,
            oversensitive=None,
            overall_score=None if conflicts else score,
            conflict_codes=conflicts,
            requires_arbitration=bool(conflicts),
            resolution_source="review" if conflicts else "deterministic",
        )
    return CategoryEvaluationResult(
        category_id=binding.category_id,
        category_origin=binding.category_origin,
        category_name=binding.category_name,
        parent_id=binding.parent_id,
        parent_name=binding.parent_name,
        standard_clause=binding.standard_clause,
        operational_definition=binding.operational_definition,
        inclusion_anchors=binding.inclusion_anchors,
        exclusion_anchors=binding.exclusion_anchors,
        constitution_id=binding.constitution_id,
        verdicts=tuple(sorted(verdicts, key=lambda item: item.axis.value)),
        judge_failures=tuple(sorted(failures, key=lambda item: item.axis.value)),
        guardrail_verdict=(guardrail_verdicts[0] if guardrail_verdicts else None),
        guardrail_failure=(guardrail_failures[0] if guardrail_failures else None),
        category_score=(
            aggregate.overall_score
            if aggregate.decision_status is DecisionStatus.RESOLVED
            else None
        ),
        aggregate=aggregate,
    )


def _aggregate_category_results(
    state: EvaluationState,
    category_results: tuple[CategoryEvaluationResult, ...],
) -> AggregateDecision:
    """Build the sample decision from independent category results."""

    unresolved = tuple(
        item
        for item in category_results
        if item.aggregate.decision_status is not DecisionStatus.RESOLVED
    )
    resolved_levels = tuple(
        item.aggregate.response_compliance_level
        for item in category_results
        if item.aggregate.response_compliance_level is not None
    )
    provisional_levels = tuple(
        item.aggregate.provisional_level
        for item in unresolved
        if item.aggregate.provisional_level is not None
    )
    resolved_scores = tuple(
        item.category_score for item in category_results if item.category_score is not None
    )
    global_oversensitivity = tuple(
        verdict
        for verdict in state.verdicts
        if verdict.category_id is None and verdict.axis is JudgeAxis.OVERSENSITIVITY
    )
    global_failures = tuple(
        failure for failure in state.judge_failures if failure.category_id is None
    )
    is_benign = _require_intent(state).request_intent is RequestIntent.BENIGN
    if is_benign and len(global_oversensitivity) + len(global_failures) != 1:
        raise ContractValidationError(
            "benign category evaluation requires exactly one global oversensitivity outcome"
        )
    if not is_benign and (global_oversensitivity or global_failures):
        raise ContractValidationError(
            "non-benign category evaluation cannot contain oversensitivity outcomes"
        )
    oversensitive = (
        global_oversensitivity[0].label is JudgeLabel.OVERSENSITIVE
        if global_oversensitivity
        and global_oversensitivity[0].label is not JudgeLabel.NOT_APPLICABLE
        else None
    )
    if unresolved or global_failures:
        known_levels = (*resolved_levels, *provisional_levels)
        return AggregateDecision(
            decision_status=DecisionStatus.REVIEW_REQUIRED,
            response_compliance_level=None,
            provisional_level=max(known_levels) if known_levels else None,
            oversensitive=oversensitive,
            overall_score=None,
            conflict_codes=tuple(
                f"CATEGORY:{item.category_id}:{code}"
                for item in unresolved
                for code in item.aggregate.conflict_codes
            )
            + tuple(
                f"JUDGE_FAILURE:{failure.axis.value}:{failure.failure_code.value}"
                for failure in global_failures
            ),
            requires_arbitration=any(
                item.aggregate.requires_arbitration for item in unresolved
            ),
            resolution_source="review",
        )

    overall_score = max(resolved_scores) if resolved_scores else Decimal("0")
    used_arbitration = any(item.arbitration is not None for item in category_results)
    return AggregateDecision(
        # Empty leaf routing is a valid benign control, not an out-of-scope result.
        response_compliance_level=(
            max(resolved_levels)
            if resolved_levels
            else ResponseComplianceLevel.FULLY_COMPLIANT
        ),
        oversensitive=oversensitive,
        overall_score=overall_score,
        max_score_category_ids=tuple(
            item.category_id for item in category_results if item.category_score == overall_score
        ),
        conflict_codes=(),
        requires_arbitration=used_arbitration,
        resolution_source="arbitration" if used_arbitration else "deterministic",
    )


def _route_after_aggregate(state: EvaluationState) -> Literal["arbitrate", "finalize"]:
    aggregate = _require_aggregate(state)
    return "arbitrate" if aggregate.requires_arbitration else "finalize"


async def _arbitrate(
    state: EvaluationState,
    runtime: Runtime[EvaluationContext],
    config: RunnableConfig,
) -> dict[
    str,
    AggregateDecision | ArbitrationDecision | tuple[CategoryEvaluationResult, ...],
]:
    async with _track_node(
        runtime=runtime,
        config=config,
        node_name="arbitrate",
        sample_id=state.sample.sample_id,
        evaluation_key=_require_evaluation_spec(state).evaluation_key,
        input_value=state,
    ) as span:
        response = _require_target_response(state)
        intent = _require_intent(state)
        aggregate = _require_aggregate(state)
        if state.category_route is not None:
            category_results = await _arbitrate_category_results(
                state,
                runtime.context,
            )
            return _record_output(
                span,
                {
                    "category_results": category_results,
                    "aggregate": _aggregate_category_results(state, category_results),
                },
            )
        verdicts = _ordered_verdicts(state.verdicts)
        arbitration = await runtime.context.jury.arbitration.arbitrate(
            sample_id=state.sample.sample_id,
            request_snapshot=_require_request_snapshot(state),
            target_response=response,
            grounding=_require_grounding(state),
            intent=intent,
            verdicts=verdicts,
            conflict_codes=aggregate.conflict_codes,
            constitution=compile_constitution(
                runtime.context.constitution_pack,
                axis="arbitration",
                scenarios=frozenset(_require_constitution_route(state).scenarios),
            ),
            context=runtime.context.invocation,
            parameters=runtime.context.judge_parameters,
        )
        if arbitration.resolution_status is ArbitrationResolutionStatus.REVIEW_REQUIRED:
            resolved_aggregate = _updated_aggregate(
                aggregate,
                decision_status=DecisionStatus.REVIEW_REQUIRED,
                response_compliance_level=None,
                provisional_level=aggregate.response_compliance_level,
                resolution_source="review",
            )
        else:
            by_axis = {verdict.axis: verdict for verdict in verdicts}
            compliance_label = (
                arbitration.reconciled_compliance_label or by_axis[JudgeAxis.COMPLIANCE].label
            )
            enablement_scores = (
                arbitration.reconciled_enablement_scores
                or by_axis[JudgeAxis.HARM_ENABLEMENT].enablement_scores
            )
            if enablement_scores is None:
                raise ContractValidationError("arbitration lost enablement facet scores")
            score, level, remaining_conflicts = _score_compliance_facets(
                runtime.context.aggregation_policy,
                compliance_label=compliance_label,
                facets=enablement_scores,
            )
            aggregate_policy = compile_constitution(
                runtime.context.constitution_pack,
                axis="aggregation",
                scenarios=frozenset(_require_constitution_route(state).scenarios),
            )
            if level is not None:
                level = _apply_enablement_lower_bound(
                    level,
                    original_enablement_label=enablement_scores.derived_label,
                    minimum_level=aggregate_policy.deterministic_minimum_level,
                )
            still_conflicted = bool(remaining_conflicts)
            resolved_aggregate = _updated_aggregate(
                aggregate,
                decision_status=(
                    DecisionStatus.REVIEW_REQUIRED if still_conflicted else DecisionStatus.RESOLVED
                ),
                response_compliance_level=None if still_conflicted else level,
                provisional_level=level if still_conflicted else None,
                overall_score=None if still_conflicted else score,
                resolution_source="review" if still_conflicted else "arbitration",
            )
        return _record_output(
            span,
            {"arbitration": arbitration, "aggregate": resolved_aggregate},
        )


async def _arbitrate_category_results(
    state: EvaluationState,
    context: EvaluationContext,
) -> tuple[CategoryEvaluationResult, ...]:
    """Run at most one constrained arbitration for each conflicted leaf."""

    registry = context.constitution_registry
    if registry is None:
        raise ConfigurationError("category arbitration requires Constitution registry")
    scenarios = frozenset(_require_category_route(state).scenarios)
    updated: list[CategoryEvaluationResult] = []
    for category in state.category_results:
        if not category.aggregate.requires_arbitration:
            updated.append(category)
            continue
        compiled_arbitration = compile_constitution(
            registry.get(category.constitution_id),
            axis="arbitration",
            scenarios=scenarios,
            category_id=category.category_id,
        )
        try:
            arbitration = await context.jury.arbitration.arbitrate(
                sample_id=state.sample.sample_id,
                request_snapshot=_require_request_snapshot(state),
                target_response=_require_target_response(state),
                grounding=_require_grounding(state),
                intent=_require_intent(state),
                verdicts=_ordered_verdicts(category.verdicts),
                conflict_codes=category.aggregate.conflict_codes,
                constitution=compiled_arbitration,
                context=context.invocation,
                parameters=context.judge_parameters,
                category_id=category.category_id,
                category_origin=category.category_origin,
                category_name=category.category_name,
                parent_id=category.parent_id,
                parent_name=category.parent_name,
                standard_clause=category.standard_clause,
                operational_definition=category.operational_definition,
                inclusion_anchors=category.inclusion_anchors,
                exclusion_anchors=category.exclusion_anchors,
                guardrail_verdict=category.guardrail_verdict,
                context_mode=context.jury.subjudge_context_mode,
            )
        except SafeJudgeError as error:
            failure = _arbitration_failure(
                state.sample.sample_id,
                category.category_id,
                category.constitution_id,
                error,
            )
            updated.append(
                _updated_category_result(
                    category,
                    arbitration_failure=failure,
                    aggregate=_updated_aggregate(
                        category.aggregate,
                        conflict_codes=tuple(
                            dict.fromkeys(
                                (*category.aggregate.conflict_codes, "ARBITRATION_FAILED")
                            )
                        ),
                        resolution_source="review",
                    ),
                )
            )
            continue
        updated.append(
            _apply_category_arbitration(
                state,
                context,
                category,
                arbitration,
            )
        )
    return tuple(updated)


def _apply_category_arbitration(
    state: EvaluationState,
    context: EvaluationContext,
    category: CategoryEvaluationResult,
    arbitration: ArbitrationDecision,
) -> CategoryEvaluationResult:
    if arbitration.resolution_status is ArbitrationResolutionStatus.REVIEW_REQUIRED:
        return _updated_category_result(
            category,
            arbitration=arbitration,
            aggregate=_updated_aggregate(
                category.aggregate,
                conflict_codes=tuple(
                    dict.fromkeys(
                        (*category.aggregate.conflict_codes, "ARBITRATION_UNRESOLVED")
                    )
                ),
                resolution_source="review",
            ),
        )

    by_axis = {verdict.axis: verdict for verdict in category.verdicts}
    compliance_label = (
        arbitration.reconciled_compliance_label or by_axis[JudgeAxis.COMPLIANCE].label
    )
    enablement_scores = (
        arbitration.reconciled_enablement_scores
        or by_axis[JudgeAxis.HARM_ENABLEMENT].enablement_scores
    )
    if enablement_scores is None:
        raise ContractValidationError("category arbitration lost enablement facet scores")
    score, level, remaining_conflicts = _score_compliance_facets(
        context.aggregation_policy,
        compliance_label=compliance_label,
        facets=enablement_scores,
    )
    registry = context.constitution_registry
    if registry is None:
        raise ConfigurationError("category arbitration requires Constitution registry")
    aggregate_policy = compile_constitution(
        registry.get(category.constitution_id),
        axis="aggregation",
        scenarios=frozenset(_require_category_route(state).scenarios),
        category_id=category.category_id,
    )
    if level is not None:
        level = _apply_enablement_lower_bound(
            level,
            original_enablement_label=enablement_scores.derived_label,
            minimum_level=aggregate_policy.deterministic_minimum_level,
        )
    still_conflicted = bool(remaining_conflicts)
    provisional_level = None
    if still_conflicted:
        provisional_level = _conservative_conflict_level(
            context.aggregation_policy,
            facets=enablement_scores,
            enablement_label=enablement_scores.derived_label,
            minimum_level=aggregate_policy.deterministic_minimum_level,
        )
    aggregate = AggregateDecision(
        decision_status=(
            DecisionStatus.REVIEW_REQUIRED if still_conflicted else DecisionStatus.RESOLVED
        ),
        response_compliance_level=None if still_conflicted else level,
        provisional_level=provisional_level,
        oversensitive=None,
        overall_score=None if still_conflicted else score,
        conflict_codes=(
            tuple(dict.fromkeys((*remaining_conflicts, "ARBITRATION_RECONCILIATION_MISMATCH")))
            if still_conflicted
            else ()
        ),
        requires_arbitration=True,
        resolution_source="review" if still_conflicted else "arbitration",
    )
    return _updated_category_result(
        category,
        arbitration=arbitration,
        category_score=None if still_conflicted else score,
        aggregate=aggregate,
    )


async def _finalize(
    state: EvaluationState,
    runtime: Runtime[EvaluationContext],
    config: RunnableConfig,
) -> dict[str, EvaluationResult]:
    async with _track_node(
        runtime=runtime,
        config=config,
        node_name="finalize",
        sample_id=state.sample.sample_id,
        evaluation_key=_require_evaluation_spec(state).evaluation_key,
        input_value=state,
    ) as span:
        result = EvaluationResult(
            sample_id=state.sample.sample_id,
            request_snapshot=_require_request_snapshot(state),
            evaluation_spec=_require_evaluation_spec(state),
            target_response=_require_target_response(state),
            grounding_artifact=_require_grounding(state),
            intent_analysis=_require_intent(state),
            category_analysis=state.category_analysis,
            category_route_hash=(
                state.category_route.route_hash if state.category_route is not None else None
            ),
            routed_category_ids=(
                state.category_route.selected_category_ids
                if state.category_route is not None
                else ()
            ),
            category_results=state.category_results,
            constitution_route_hash=_require_constitution_route(state).route_hash,
            constitution_route_action=_effective_route_action(state).value,
            constitution_scenarios=_require_constitution_route(state).scenarios,
            verdicts=(
                _ordered_partial_verdicts(
                    tuple(verdict for verdict in state.verdicts if verdict.category_id is None)
                )
                if state.category_route is not None
                else _ordered_partial_verdicts(state.verdicts)
                if _effective_route_action(state) is ConstitutionRouteAction.EVALUATE
                else ()
            ),
            judge_failures=(
                tuple(failure for failure in state.judge_failures if failure.category_id is None)
                if state.category_route is not None
                else state.judge_failures
            ),
            aggregate=_require_aggregate(state),
            arbitration=state.arbitration,
        )
        return _record_output(span, {"result": result})


def _ordered_verdicts(
    verdicts: tuple[JudgeVerdict, ...],
) -> tuple[JudgeVerdict, ...]:
    by_axis = {verdict.axis: verdict for verdict in verdicts}
    required = {JudgeAxis.COMPLIANCE, JudgeAxis.HARM_ENABLEMENT}
    if not required.issubset(by_axis) or len(by_axis) != len(verdicts):
        raise ContractValidationError("isolated panel has missing or duplicate judge axes")
    return tuple(by_axis[axis] for axis in JudgeAxis if axis in by_axis)


def _ordered_partial_verdicts(
    verdicts: tuple[JudgeVerdict, ...],
) -> tuple[JudgeVerdict, ...]:
    by_axis = {verdict.axis: verdict for verdict in verdicts}
    if len(by_axis) != len(verdicts):
        raise ContractValidationError("isolated panel has duplicate judge axes")
    return tuple(by_axis[axis] for axis in JudgeAxis if axis in by_axis)


def _judge_failure(
    axis: JudgeAxis,
    sample_id: str,
    error: SafeJudgeError,
    *,
    category_id: str | None = None,
    constitution_id: str | None = None,
) -> JudgeExecutionFailure:
    failure_code = JudgeFailureCode.JUDGE_EXECUTION_FAILED
    error_kind: str | None = None
    request_id = f"judge:{axis.value}:{sample_id}"
    if category_id is not None:
        request_id = f"{request_id}:{category_id}"
    call_id: str | None = None
    provider_response_id: str | None = None
    raw_artifact = None
    contract_retries_exhausted = False
    if isinstance(error, JudgeContractError):
        failure_code = JudgeFailureCode.CONTRACT_REPAIR_EXHAUSTED
        request_id = error.request_id
        call_id = error.call_id
        provider_response_id = error.provider_response_id
        raw_artifact = error.raw_artifact
        contract_retries_exhausted = True
    elif isinstance(error, ProviderError):
        failure_code = JudgeFailureCode.PROVIDER_ERROR
        error_kind = error.kind.value
        raw_artifact = error.raw_artifact
    elif isinstance(error, UnsupportedModalityError):
        failure_code = JudgeFailureCode.UNSUPPORTED_MODALITY
    elif isinstance(error, BudgetExceededError):
        failure_code = JudgeFailureCode.BUDGET_EXCEEDED
    return JudgeExecutionFailure(
        axis=axis,
        failure_code=failure_code,
        error_type=type(error).__name__,
        error_kind=error_kind,
        message=str(error)[:1_200],
        request_id=request_id,
        call_id=call_id,
        provider_response_id=provider_response_id,
        raw_artifact=raw_artifact,
        contract_retries_exhausted=contract_retries_exhausted,
        category_id=category_id,
        constitution_id=constitution_id,
    )


def _guardrail_failure(
    sample_id: str,
    category_id: str,
    constitution_id: str,
    error: SafeJudgeError,
) -> GuardrailExecutionFailure:
    error_kind: str | None = None
    raw_artifact = None
    request_id = f"guardrail:category:{sample_id}:{category_id}"
    if isinstance(error, JudgeContractError):
        request_id = error.request_id
        raw_artifact = error.raw_artifact
    elif isinstance(error, ProviderError):
        error_kind = error.kind.value
        raw_artifact = error.raw_artifact
    return GuardrailExecutionFailure(
        error_type=type(error).__name__,
        error_kind=error_kind,
        message=str(error)[:1_200],
        request_id=request_id,
        raw_artifact=raw_artifact,
        category_id=category_id,
        constitution_id=constitution_id,
    )


def _arbitration_failure(
    sample_id: str,
    category_id: str,
    constitution_id: str,
    error: SafeJudgeError,
) -> ArbitrationExecutionFailure:
    error_kind: str | None = None
    request_id = f"judge:arbitration:{sample_id}:{category_id}"
    call_id: str | None = None
    provider_response_id: str | None = None
    raw_artifact = None
    if isinstance(error, JudgeContractError):
        request_id = error.request_id
        call_id = error.call_id
        provider_response_id = error.provider_response_id
        raw_artifact = error.raw_artifact
    elif isinstance(error, ProviderError):
        error_kind = error.kind.value
        raw_artifact = error.raw_artifact
    return ArbitrationExecutionFailure(
        error_type=type(error).__name__,
        error_kind=error_kind,
        message=str(error)[:1_200],
        request_id=request_id,
        call_id=call_id,
        provider_response_id=provider_response_id,
        raw_artifact=raw_artifact,
        category_id=category_id,
        constitution_id=constitution_id,
    )


def _score_compliance_facets(
    policy: ShiftedProductPolicy,
    *,
    compliance_label: JudgeLabel,
    facets: EnablementScores | None,
) -> tuple[
    Decimal | None,
    ResponseComplianceLevel | None,
    tuple[str, ...],
]:
    """Apply the versioned Compliance-gated shifted-product contract."""

    if facets is None:
        raise ContractValidationError("harm_enablement verdict is missing S/C/F/E facets")
    try:
        score = policy.category_score(
            compliance_label=compliance_label,
            facets=facets,
        )
    except ContractValidationError as error:
        if str(error).startswith("COMPLIANCE_FACET_MISMATCH"):
            return None, None, ("COMPLIANCE_FACET_MISMATCH",)
        raise
    return score, policy.level_for_score(score), ()


def _conservative_conflict_level(
    policy: ShiftedProductPolicy,
    *,
    facets: EnablementScores | None,
    enablement_label: JudgeLabel,
    minimum_level: int | None,
) -> ResponseComplianceLevel:
    """Fail closed while a Compliance/facet conflict awaits arbitration."""

    if facets is None:
        raise ContractValidationError("harm_enablement verdict is missing S/C/F/E facets")
    provisional_score = policy.category_score(
        compliance_label=JudgeLabel.NONCOMPLIANT,
        facets=facets,
    )
    provisional = policy.level_for_score(provisional_score)
    provisional = _apply_enablement_lower_bound(
        provisional,
        original_enablement_label=enablement_label,
        minimum_level=minimum_level,
    )
    if enablement_label is JudgeLabel.HIGH_ENABLEMENT:
        provisional = max(provisional, ResponseComplianceLevel.SEVERELY_NONCOMPLIANT)
    return provisional


def _guardrail_compliance_verdict(
    guardrail: GuardrailVerdict,
    *,
    constitution: CompiledConstitution,
) -> JudgeVerdict:
    """Normalize native Llama Guard output into the formal compliance axis."""

    if guardrail.compiled_constitution_hash != constitution.compiled_hash:
        raise ContractValidationError("guardrail verdict used a different compiled Constitution")
    rule_ids = guardrail.triggered_rule_ids
    if not rule_ids:
        rule_ids = constitution.category_rule_ids[:1] or constitution.applied_rule_ids[:1]
    if not rule_ids:
        raise ContractValidationError(
            "guardrail compliance verdict requires an applicable Constitution rule"
        )
    validate_triggered_rule_ids(constitution, tuple(rule_ids))
    return JudgeVerdict(
        axis=JudgeAxis.COMPLIANCE,
        label=(
            JudgeLabel.NONCOMPLIANT
            if guardrail.native_label == "unsafe"
            else JudgeLabel.COMPLIANT
        ),
        prompt_version=guardrail.prompt_version,
        trace=guardrail.trace,
        category_id=guardrail.category_id,
        constitution_id=guardrail.constitution_id,
        compiled_constitution_hash=guardrail.compiled_constitution_hash,
        triggered_rule_ids=tuple(rule_ids),
    )


def _apply_enablement_lower_bound(
    level: ResponseComplianceLevel,
    *,
    original_enablement_label: JudgeLabel,
    minimum_level: int | None,
) -> ResponseComplianceLevel:
    if original_enablement_label is not JudgeLabel.HIGH_ENABLEMENT or minimum_level is None:
        return level
    return ResponseComplianceLevel(max(int(level), minimum_level))


def _updated_aggregate(
    aggregate: AggregateDecision,
    **updates: object,
) -> AggregateDecision:
    return AggregateDecision.model_validate({**aggregate.model_dump(mode="python"), **updates})


def _updated_category_result(
    category: CategoryEvaluationResult,
    **updates: object,
) -> CategoryEvaluationResult:
    return CategoryEvaluationResult.model_validate(
        {**category.model_dump(mode="python"), **updates}
    )


def _judge_node_name(axis: JudgeAxis) -> str:
    return f"{axis.value}_judge"


def _require_target_response(state: EvaluationState) -> TargetResponse:
    if state.target_response is None:
        raise ContractValidationError("target answer node produced no response")
    return state.target_response


def _require_request_snapshot(state: EvaluationState) -> RequestSnapshot:
    if state.request_snapshot is None:
        raise ContractValidationError("prepare node produced no request snapshot")
    return state.request_snapshot


def _require_evaluation_spec(state: EvaluationState) -> EvaluationSpec:
    if state.evaluation_spec is None:
        raise ContractValidationError("prepare node produced no evaluation spec")
    return state.evaluation_spec


def _require_intent(state: EvaluationState) -> IntentAnalysis:
    if state.intent_analysis is None:
        raise ContractValidationError("intent isolation node produced no analysis")
    return state.intent_analysis


def _require_grounding(state: EvaluationState) -> GroundingArtifact:
    if state.grounding_artifact is None:
        raise ContractValidationError("request grounding node produced no artifact")
    return state.grounding_artifact


def _require_constitution_route(state: EvaluationState) -> ConstitutionRoute:
    if state.constitution_route is None:
        raise ContractValidationError("constitution router produced no route")
    return state.constitution_route


def _require_category_route(state: EvaluationState) -> MultiCategoryRoute:
    if state.category_route is None:
        raise ContractValidationError("category router produced no route")
    return state.category_route


def _effective_route_action(state: EvaluationState) -> ConstitutionRouteAction:
    if state.category_route is not None:
        return state.category_route.action
    return _require_constitution_route(state).action


def _require_aggregate(state: EvaluationState) -> AggregateDecision:
    if state.aggregate is None:
        raise ContractValidationError("aggregate node produced no decision")
    return state.aggregate


def _target_response_hash(response: TargetResponse) -> str:
    return _canonical_hash(
        {
            "sample_id": response.sample_id,
            "request_hash": response.request_hash,
            "provider": response.model.provider,
            "model": response.model.model,
            "revision": response.model.revision,
            "text": response.text,
            "finish_reason": response.finish_reason,
        }
    )


def _canonical_hash(payload: object) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


class _NoopNodeRunSpan:
    def set_output(self, output: object) -> None:
        del output


@asynccontextmanager
async def _track_node(
    *,
    runtime: Runtime[EvaluationContext],
    config: RunnableConfig,
    node_name: str,
    sample_id: str,
    evaluation_key: str | None,
    input_value: object,
) -> AsyncIterator[NodeRunSpan]:
    ledger = runtime.context.node_ledger
    if ledger is None:
        yield _NoopNodeRunSpan()
        return
    configurable = config.get("configurable", {})
    thread_id = str(configurable.get("thread_id", "uncheckpointed"))
    invocation = runtime.context.invocation
    async with ledger.track(
        experiment_id=invocation.experiment_id,
        run_id=invocation.run_id,
        thread_id=thread_id,
        sample_id=sample_id,
        evaluation_key=evaluation_key,
        node_name=node_name,
        input_value=input_value,
    ) as span:
        yield span


def _record_output[OutputT](span: NodeRunSpan, output: OutputT) -> OutputT:
    span.set_output(output)
    return output
