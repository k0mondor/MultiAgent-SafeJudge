"""Deterministically select and compile only the rules needed by one judge axis."""

from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import Field

from safejudge.constitution.contracts import ConstitutionPack, RuleType
from safejudge.contracts.base import ContractModel
from safejudge.contracts.judging import JudgeAxis


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
    applied_rule_hashes: tuple[str, ...]
    skipped_rule_ids: tuple[str, ...]
    prompt_fragments: tuple[str, ...]
    allowed_reason_codes: tuple[str, ...]
    required_evidence_sources: tuple[str, ...]
    allowed_evidence_sources: tuple[str, ...]
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
        "applied_rule_hashes": [rule.rule_hash for rule in selected],
        "skipped_rule_ids": skipped,
        "prompt_fragments": [
            rule.effect.prompt_text for rule in selected if rule.effect.prompt_text is not None
        ],
        "allowed_reason_codes": sorted(
            {code for rule in selected for code in rule.allowed_reason_codes}
        ),
        "required_evidence_sources": sorted(
            {source for rule in selected for source in rule.required_evidence_sources}
        ),
        "allowed_evidence_sources": sorted(
            {source for rule in selected for source in rule.allowed_evidence_sources}
        ),
        "deterministic_minimum_level": max(minimums) if minimums else None,
    }
    compiled_hash = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return CompiledConstitution(**payload, compiled_hash=compiled_hash)
