"""Contracts for executable ConstitutionPack rules."""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from safejudge.contracts.base import ContractModel
from safejudge.contracts.evaluation import ResponseComplianceLevel


class RuleType(StrEnum):
    SEMANTIC = "semantic"
    DETERMINISTIC = "deterministic"


class RuleEffect(ContractModel):
    minimum_level: ResponseComplianceLevel | None = None
    prompt_text: str | None = Field(default=None, min_length=1)
    validator_policy_id: str | None = Field(default=None, min_length=1)


class ConstitutionRule(ContractModel):
    rule_id: str = Field(pattern=r"^[A-Z][A-Z0-9_]*$")
    rule_version: str = Field(min_length=1)
    rule_type: RuleType
    applies_to: tuple[str, ...] = Field(min_length=1)
    category_ids: tuple[str, ...] = ()
    scenarios: tuple[str, ...] = ()
    effect: RuleEffect = Field(default_factory=RuleEffect)
    priority: int = Field(default=0, ge=0, le=1_000)

    @model_validator(mode="after")
    def category_ids_are_unique(self) -> ConstitutionRule:
        if len(set(self.category_ids)) != len(self.category_ids):
            raise ValueError("constitution rule category IDs must be unique")
        return self

    @property
    def rule_hash(self) -> str:
        payload = self.model_dump(mode="json")
        if not self.category_ids:
            payload.pop("category_ids")
        return _canonical_hash(payload)


class ConstitutionPack(ContractModel):
    schema_version: Literal["1.0"] = "1.0"
    constitution_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    version: str = Field(min_length=1)
    scope_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    extends: tuple[str, ...] = ()
    core_rules: tuple[ConstitutionRule, ...] = ()
    domain_rules: tuple[ConstitutionRule, ...] = ()
    scenario_rules: tuple[ConstitutionRule, ...] = ()

    @property
    def rules(self) -> tuple[ConstitutionRule, ...]:
        return (*self.core_rules, *self.domain_rules, *self.scenario_rules)

    @property
    def constitution_hash(self) -> str:
        payload = self.model_dump(mode="json")
        for group_name in ("core_rules", "domain_rules", "scenario_rules"):
            for rule in payload[group_name]:
                if not rule["category_ids"]:
                    rule.pop("category_ids")
        return _canonical_hash(payload)

    @model_validator(mode="after")
    def rules_are_unambiguous(self) -> ConstitutionPack:
        by_id = {rule.rule_id: rule for rule in self.rules}
        if len(by_id) != len(self.rules):
            raise ValueError("constitution rule IDs must be unique")
        return self


def _canonical_hash(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
