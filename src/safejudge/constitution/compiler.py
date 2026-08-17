"""Deterministically select and compile only the rules needed by one judge axis."""

from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import Field

from safejudge.constitution.contracts import ConstitutionPack, RuleType
from safejudge.contracts.base import ContractModel
from safejudge.contracts.judging import JudgeAxis
from safejudge.core.errors import ContractValidationError


class CompiledConstitution(ContractModel):
    schema_version: Literal["1.0"] = "1.0"
    constitution_id: str
    constitution_version: str
    constitution_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    scope_id: str
    category_id: str | None = None
    axis: str
    scenarios: tuple[str, ...]
    applied_rule_ids: tuple[str, ...]
    category_rule_ids: tuple[str, ...] = ()
    applied_rule_hashes: tuple[str, ...]
    skipped_rule_ids: tuple[str, ...]
    prompt_fragments: tuple[str, ...]
    deterministic_minimum_level: int | None = None
    compiled_hash: str = Field(pattern=r"^[a-f0-9]{64}$")


def compile_constitution(
    pack: ConstitutionPack,
    *,
    axis: JudgeAxis | Literal["intent", "aggregation", "arbitration"],
    scenarios: frozenset[str] = frozenset(),
    category_id: str | None = None,
) -> CompiledConstitution:
    axis_name = axis.value if isinstance(axis, JudgeAxis) else axis
    selected = tuple(
        rule
        for rule in pack.rules
        if (axis_name in rule.applies_to or "all" in rule.applies_to)
        and set(rule.scenarios).issubset(scenarios)
        and (not rule.category_ids or category_id in rule.category_ids)
    )
    selected = tuple(sorted(selected, key=lambda rule: (-rule.priority, rule.rule_id)))
    selected_ids = {rule.rule_id for rule in selected}
    skipped = tuple(rule.rule_id for rule in pack.rules if rule.rule_id not in selected_ids)
    minimums = [
        int(rule.effect.minimum_level)
        for rule in selected
        if rule.rule_type is RuleType.DETERMINISTIC
        and rule.effect.minimum_level is not None
    ]
    payload = {
        "constitution_id": pack.constitution_id,
        "constitution_version": pack.version,
        "constitution_hash": pack.constitution_hash,
        "scope_id": pack.scope_id,
        "category_id": category_id,
        "axis": axis_name,
        "scenarios": sorted(scenarios),
        "applied_rule_ids": [rule.rule_id for rule in selected],
        "category_rule_ids": [rule.rule_id for rule in selected if rule.category_ids],
        "applied_rule_hashes": [rule.rule_hash for rule in selected],
        "skipped_rule_ids": skipped,
        "prompt_fragments": [
            rule.effect.prompt_text for rule in selected if rule.effect.prompt_text is not None
        ],
        "deterministic_minimum_level": max(minimums) if minimums else None,
    }
    compiled_hash = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return CompiledConstitution(**payload, compiled_hash=compiled_hash)


def validate_triggered_rule_ids(
    constitution: CompiledConstitution,
    triggered_rule_ids: tuple[str, ...],
) -> None:
    """Require a unique, non-empty subset of the rules compiled for this panel."""

    if not triggered_rule_ids:
        raise ContractValidationError("judge must cite at least one triggered rule")
    if len(set(triggered_rule_ids)) != len(triggered_rule_ids):
        raise ContractValidationError("judge cited duplicate triggered rule IDs")
    unknown = sorted(set(triggered_rule_ids).difference(constitution.applied_rule_ids))
    if unknown:
        raise ContractValidationError(
            f"judge cited rules absent from the compiled Constitution: {unknown}"
        )
    if constitution.category_rule_ids and not set(triggered_rule_ids).intersection(
        constitution.category_rule_ids
    ):
        raise ContractValidationError(
            "category verdict must cite at least one category-specific Constitution rule"
        )
