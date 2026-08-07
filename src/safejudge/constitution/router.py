"""Deterministic ConstitutionPack routing from grounded request intent."""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Literal

from pydantic import Field

from safejudge.constitution.contracts import ConstitutionPack
from safejudge.contracts.base import ContractModel
from safejudge.contracts.dataset import RequestIntent
from safejudge.contracts.judging import IntentAnalysis, ScopeStatus
from safejudge.grounding.contracts import (
    GroundingArtifact,
    GroundingMode,
    GroundingStatus,
    ObservationModality,
)


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
