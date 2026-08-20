"""Versioned model profiles and the approved/candidate model registry."""

from __future__ import annotations

import hashlib
import json
import tomllib
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import Field, JsonValue, model_validator

from safejudge.contracts.base import ContractModel
from safejudge.contracts.model import ModelRole
from safejudge.core.errors import ConfigurationError


class StructuredOutputMode(StrEnum):
    NONE = "none"
    NATIVE_JSON_SCHEMA = "native_json_schema"
    PROMPTED_JSON = "prompted_json"


class JudgeAdapterKind(StrEnum):
    CHAT_JSON = "chat_json"
    LLAMA_GUARD = "llama_guard"


class QualificationStatus(StrEnum):
    CANDIDATE = "candidate"
    APPROVED = "approved"


class ModelProfile(ContractModel):
    """How one model is called for one or more SafeJudge roles."""

    schema_version: Literal["1.0"] = "1.0"
    profile_id: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9._-]*$")
    profile_version: str = Field(min_length=1)
    provider: Literal["local-openai", "openrouter"]
    model_id: str = Field(min_length=1)
    roles: frozenset[ModelRole] = Field(min_length=1)
    structured_output_mode: StructuredOutputMode = StructuredOutputMode.NONE
    judge_adapter: JudgeAdapterKind = JudgeAdapterKind.CHAT_JSON
    thinking_field: Literal["reasoning", "reasoning_content", "content_tags"] | None = None
    thinking_always_enabled: bool = False
    default_parameters: dict[str, JsonValue] = Field(default_factory=dict)
    retry_parameters: tuple[dict[str, JsonValue], ...] = ()
    max_contract_retries: int = Field(default=1, ge=0, le=3)
    openrouter_require_parameters: bool = True
    openrouter_response_healing: bool = False
    qualification_status: QualificationStatus = QualificationStatus.CANDIDATE
    qualification_report: str | None = None

    @model_validator(mode="after")
    def role_and_output_mode_are_compatible(self) -> ModelProfile:
        structured_roles = {ModelRole.JUDGE, ModelRole.GROUNDING}
        has_structured_role = bool(structured_roles.intersection(self.roles))
        guardrail_judge = (
            ModelRole.JUDGE in self.roles
            and self.judge_adapter is JudgeAdapterKind.LLAMA_GUARD
        )
        missing_structured_output = (
            has_structured_role
            and self.structured_output_mode is StructuredOutputMode.NONE
        )
        if missing_structured_output and not (
            guardrail_judge and self.roles == {ModelRole.JUDGE}
        ):
            raise ValueError("judge and grounding profiles require a structured output mode")
        if (
            not has_structured_role
            and self.structured_output_mode is not StructuredOutputMode.NONE
        ):
            raise ValueError(
                "structured output mode is only valid for judge or grounding profiles"
            )
        if self.judge_adapter is not JudgeAdapterKind.CHAT_JSON and not guardrail_judge:
            raise ValueError("specialized judge adapters require a judge-only profile")
        if guardrail_judge and self.structured_output_mode is not StructuredOutputMode.NONE:
            raise ValueError("Llama Guard uses its native text classification output")
        return self

    def supports_role(self, role: ModelRole) -> bool:
        return role in self.roles

    @property
    def fingerprint(self) -> str:
        payload = self.model_dump(
            mode="json",
            exclude={"qualification_status", "qualification_report"},
        )
        if self.judge_adapter is JudgeAdapterKind.CHAT_JSON:
            # Preserve historical fingerprints created before adapters were configurable.
            payload.pop("judge_adapter")
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


class ModelRegistry:
    def __init__(self, profiles: tuple[ModelProfile, ...], *, source: Path) -> None:
        self.source = source
        self._profiles = {profile.profile_id: profile for profile in profiles}
        if len(self._profiles) != len(profiles):
            raise ConfigurationError(f"duplicate model profile_id in {source}")

    @classmethod
    def load(cls, path: Path) -> ModelRegistry:
        source = path.resolve()
        if not source.is_file():
            raise ConfigurationError(f"model registry does not exist: {source}")
        try:
            data = tomllib.loads(source.read_text(encoding="utf-8"))
            raw_profiles = data.get("models", [])
            if not isinstance(raw_profiles, list):
                raise ValueError("top-level models must be an array of tables")
            profiles = tuple(ModelProfile.model_validate(item) for item in raw_profiles)
        except (OSError, tomllib.TOMLDecodeError, ValueError) as error:
            raise ConfigurationError(f"invalid model registry {source}: {error}") from error
        return cls(profiles, source=source)

    def get(
        self,
        profile_id: str,
        *,
        role: ModelRole,
        allow_unqualified: bool = False,
    ) -> ModelProfile:
        try:
            profile = self._profiles[profile_id]
        except KeyError as error:
            raise ConfigurationError(f"unknown model profile: {profile_id!r}") from error
        if not profile.supports_role(role):
            raise ConfigurationError(
                f"model profile {profile_id!r} is not registered for role {role.value!r}"
            )
        if (
            not allow_unqualified
            and profile.qualification_status is not QualificationStatus.APPROVED
        ):
            raise ConfigurationError(
                f"model profile {profile_id!r} is {profile.qualification_status.value!r}; "
                "review its real-API qualification report and mark it approved, or use "
                "--allow-unqualified-model for an explicit development run"
            )
        return profile

    def list(self, *, role: ModelRole | None = None) -> tuple[ModelProfile, ...]:
        values = tuple(self._profiles.values())
        if role is not None:
            values = tuple(profile for profile in values if profile.supports_role(role))
        return tuple(sorted(values, key=lambda item: item.profile_id))
