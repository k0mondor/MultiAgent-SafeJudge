"""Deterministic ConstitutionPack routing from grounded request intent."""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Literal

from pydantic import Field

from safejudge.constitution.contracts import ConstitutionPack
from safejudge.constitution.registry import ConstitutionRegistry
from safejudge.contracts.base import ContractModel
from safejudge.contracts.dataset import RequestIntent
from safejudge.contracts.judging import IntentAnalysis, ScopeStatus
from safejudge.grounding.contracts import (
    GroundingArtifact,
    GroundingMode,
    GroundingStatus,
    ObservationModality,
)
from safejudge.taxonomy.contracts import TaxonomyPack


class ConstitutionRouteAction(StrEnum):
    EVALUATE = "evaluate"
    REVIEW_REQUIRED = "review_required"
    NOT_EVALUATED = "not_evaluated"


class ConstitutionRoute(ContractModel):
    schema_version: Literal["1.0"] = "1.0"
    router_version: Literal["constitution-router-v2"] = "constitution-router-v2"
    constitution_id: str
    constitution_version: str
    constitution_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    scope_id: str
    scope_status: ScopeStatus
    action: ConstitutionRouteAction
    scenarios: tuple[str, ...]
    grounding_artifact_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    route_hash: str = Field(pattern=r"^[a-f0-9]{64}$")


class CategoryConstitutionBinding(ContractModel):
    """One selected standard category bound to one executable policy pack."""

    category_id: str
    category_name: str
    parent_id: str | None = None
    standard_clause: str
    constitution_id: str
    constitution_version: str
    constitution_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    scope_id: str


class MultiCategoryRoute(ContractModel):
    """A deterministic, multi-label routing result for one evaluation item."""

    schema_version: Literal["1.0"] = "1.0"
    router_version: Literal["category-router-v1"] = "category-router-v1"
    taxonomy_id: str
    taxonomy_version: str
    taxonomy_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    standard_id: str
    scope_status: ScopeStatus
    action: ConstitutionRouteAction
    selected_category_ids: tuple[str, ...]
    bindings: tuple[CategoryConstitutionBinding, ...]
    scenarios: tuple[str, ...]
    grounding_artifact_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    route_hash: str = Field(pattern=r"^[a-f0-9]{64}$")


def route_constitution(
    pack: ConstitutionPack,
    *,
    intent: IntentAnalysis,
    grounding: GroundingArtifact,
) -> ConstitutionRoute:
    if (
        intent.scope_id != pack.scope_id
        or grounding.status is not GroundingStatus.COMPLETE
    ):
        scope_status = ScopeStatus.AMBIGUOUS
    elif grounding.mode is GroundingMode.BENCHMARK_ASSISTED:
        has_label = any(
            item.modality is ObservationModality.BENCHMARK_LABEL
            for item in grounding.observations
        )
        scope_status = (
            ScopeStatus.IN_SCOPE
            if has_label
            and (
                intent.request_intent in {RequestIntent.BENIGN, RequestIntent.HARMFUL}
                or intent.scope_status is ScopeStatus.IN_SCOPE
            )
            else ScopeStatus.AMBIGUOUS
        )
    else:
        scope_status = intent.scope_status
    if scope_status is ScopeStatus.OUT_OF_SCOPE:
        action = ConstitutionRouteAction.NOT_EVALUATED
    elif scope_status is ScopeStatus.AMBIGUOUS:
        action = ConstitutionRouteAction.REVIEW_REQUIRED
    else:
        action = ConstitutionRouteAction.EVALUATE
    scenarios = {
        grounding.mode.value,
        "benign"
        if intent.request_intent is RequestIntent.BENIGN
        else "non_benign",
    }
    payload = {
        "router_version": "constitution-router-v2",
        "constitution_id": pack.constitution_id,
        "constitution_version": pack.version,
        "constitution_hash": pack.constitution_hash,
        "scope_id": pack.scope_id,
        "scope_status": scope_status.value,
        "action": action.value,
        "scenarios": sorted(scenarios),
        "grounding_artifact_hash": grounding.artifact_hash,
    }
    route_hash = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return ConstitutionRoute(**payload, route_hash=route_hash)


def route_categories(
    taxonomy: TaxonomyPack,
    registry: ConstitutionRegistry,
    *,
    intent: IntentAnalysis,
    grounding: GroundingArtifact,
    category_ids: tuple[str, ...],
) -> MultiCategoryRoute:
    """Route zero or more request/response leaf categories to Constitution packs.

    The model may select several categories for the same evaluation. This function
    never guesses or collapses them: it validates every ID against the enabled
    taxonomy and emits one binding per category/Constitution pair.
    """

    enabled = {
        category.category_id: category for category in taxonomy.routed_categories
    }
    unknown = sorted(set(category_ids) - set(enabled))
    if unknown:
        raise ValueError(
            "intent selected categories that are not routing-enabled: "
            + ", ".join(unknown)
        )
    selected_ids = tuple(sorted(set(category_ids)))
    if grounding.status is not GroundingStatus.COMPLETE or (
        grounding.mode is GroundingMode.BENCHMARK_ASSISTED
        and not any(
            item.modality is ObservationModality.BENCHMARK_LABEL
            for item in grounding.observations
        )
    ):
        scope_status = ScopeStatus.AMBIGUOUS
    else:
        scope_status = intent.scope_status

    if scope_status is ScopeStatus.AMBIGUOUS:
        action = ConstitutionRouteAction.REVIEW_REQUIRED
    elif scope_status is ScopeStatus.OUT_OF_SCOPE:
        action = ConstitutionRouteAction.NOT_EVALUATED
    elif not selected_ids and intent.request_intent is RequestIntent.HARMFUL:
        action = ConstitutionRouteAction.REVIEW_REQUIRED
    elif not selected_ids:
        action = ConstitutionRouteAction.NOT_EVALUATED
    else:
        action = ConstitutionRouteAction.EVALUATE

    bindings: list[dict[str, object]] = []
    for category_id in selected_ids:
        category = enabled[category_id]
        for constitution_id in category.constitution_ids:
            pack = registry.get(constitution_id)
            bindings.append(
                {
                    "category_id": category.category_id,
                    "category_name": category.category_name,
                    "parent_id": category.parent_id,
                    "standard_clause": category.standard_clause,
                    "constitution_id": pack.constitution_id,
                    "constitution_version": pack.version,
                    "constitution_hash": pack.constitution_hash,
                    "scope_id": pack.scope_id,
                }
            )
    scenarios = {
        grounding.mode.value,
        "benign" if intent.request_intent is RequestIntent.BENIGN else "non_benign",
    }
    payload = {
        "router_version": "category-router-v1",
        "taxonomy_id": taxonomy.taxonomy_id,
        "taxonomy_version": taxonomy.taxonomy_version,
        "taxonomy_hash": taxonomy.taxonomy_hash,
        "standard_id": taxonomy.standard_id,
        "scope_status": scope_status.value,
        "action": action.value,
        "selected_category_ids": selected_ids,
        "bindings": bindings,
        "scenarios": sorted(scenarios),
        "grounding_artifact_hash": grounding.artifact_hash,
    }
    route_hash = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return MultiCategoryRoute(**payload, route_hash=route_hash)
