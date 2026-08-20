"""Versioned mappings between native guardrail and SafeJudge taxonomies."""

from __future__ import annotations

import hashlib
import json
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from safejudge.contracts.base import ContractModel
from safejudge.core.errors import ConfigurationError

LLAMA_GUARD_ADAPTER_VERSION = "llama-guard-compliance-adapter-v1"
LLAMA_GUARD_PROMPT_VERSION = "llama-guard-native-chat-v2-controlled-context"


class NativeCategoryMapping(ContractModel):
    native_category: str = Field(pattern=r"^S(?:1[0-4]|[1-9])$")
    category_ids: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def category_ids_are_unique(self) -> NativeCategoryMapping:
        if len(set(self.category_ids)) != len(self.category_ids):
            raise ValueError("guardrail category mappings must be unique")
        return self


class GuardrailPolicy(ContractModel):
    """Mapping used only after SafeJudge has routed applicable GB/T categories."""

    schema_version: Literal["1.0"] = "1.0"
    policy_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    version: str = Field(min_length=1)
    model_taxonomy: str = Field(min_length=1)
    mappings: tuple[NativeCategoryMapping, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def native_categories_are_unique(self) -> GuardrailPolicy:
        native = [mapping.native_category for mapping in self.mappings]
        if len(set(native)) != len(native):
            raise ValueError("duplicate native guardrail category mapping")
        return self

    @classmethod
    def load(cls, path: Path) -> GuardrailPolicy:
        source = path.resolve()
        if not source.is_file():
            raise ConfigurationError(f"guardrail policy does not exist: {source}")
        try:
            return cls.model_validate(tomllib.loads(source.read_text(encoding="utf-8")))
        except (OSError, tomllib.TOMLDecodeError, ValueError) as error:
            raise ConfigurationError(f"invalid guardrail policy {source}: {error}") from error

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def mapped_categories(self, native_categories: tuple[str, ...]) -> frozenset[str]:
        selected = set(native_categories)
        return frozenset(
            category_id
            for mapping in self.mappings
            if mapping.native_category in selected
            for category_id in mapping.category_ids
        )
