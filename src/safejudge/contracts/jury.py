"""Versioned contract for one Judge model used by every judging stage."""

from __future__ import annotations

import hashlib
import json
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from safejudge.contracts.base import ContractModel
from safejudge.contracts.dataset import NonEmptyString
from safejudge.contracts.model import ModelRole
from safejudge.core.errors import ConfigurationError
from safejudge.guardrails.policy import (
    LLAMA_GUARD_ADAPTER_VERSION,
    LLAMA_GUARD_PROMPT_VERSION,
    GuardrailPolicy,
)
from safejudge.models.profiles import JudgeAdapterKind, ModelProfile, ModelRegistry

SubjudgeContextMode = Literal["full", "compact"]


class GuardrailIdentity(ContractModel):
    provider: NonEmptyString
    model: NonEmptyString
    profile_id: NonEmptyString
    profile_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    policy_id: NonEmptyString
    policy_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    adapter_version: NonEmptyString
    prompt_version: NonEmptyString


class JuryPlan(ContractModel):
    """Main Judge plus an optional per-category guardrail Compliance judge."""

    schema_version: Literal["2.0"] = "2.0"
    jury_id: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9._-]*$")
    version: NonEmptyString
    judge_profile: NonEmptyString
    category_guardrail_profile: NonEmptyString | None = None
    category_guardrail_policy: NonEmptyString | None = None
    subjudge_context_mode: SubjudgeContextMode | None = None

    @model_validator(mode="after")
    def guardrail_fields_are_paired(self) -> JuryPlan:
        if (self.category_guardrail_profile is None) != (self.category_guardrail_policy is None):
            raise ValueError(
                "category_guardrail_profile and category_guardrail_policy "
                "must be configured together"
            )
        return self

    @classmethod
    def load(cls, path: Path) -> JuryPlan:
        source = path.resolve()
        if not source.is_file():
            raise ConfigurationError(f"judge plan does not exist: {source}")
        try:
            data = tomllib.loads(source.read_text(encoding="utf-8"))
            return cls.model_validate(data)
        except (OSError, tomllib.TOMLDecodeError, ValueError) as error:
            raise ConfigurationError(f"invalid judge plan {source}: {error}") from error

    def resolve(
        self,
        registry: ModelRegistry,
        *,
        allow_unqualified: bool = False,
    ) -> JuryDefinition:
        profile = registry.get(
            self.judge_profile,
            role=ModelRole.JUDGE,
            allow_unqualified=allow_unqualified,
        )
        guardrail_profile: ModelProfile | None = None
        guardrail_policy: GuardrailPolicy | None = None
        if self.category_guardrail_profile is not None:
            guardrail_profile = registry.get(
                self.category_guardrail_profile,
                role=ModelRole.JUDGE,
                allow_unqualified=allow_unqualified,
            )
            if guardrail_profile.judge_adapter is not JudgeAdapterKind.LLAMA_GUARD:
                raise ConfigurationError(
                    "category guardrail profile must use judge_adapter='llama_guard'"
                )
            guardrail_policy = GuardrailPolicy.load(Path(str(self.category_guardrail_policy)))
        identity = JuryIdentity.from_profiles(
            jury_id=self.jury_id,
            version=self.version,
            profile=profile,
            guardrail_profile=guardrail_profile,
            guardrail_policy=guardrail_policy,
            subjudge_context_mode=self.subjudge_context_mode,
        )
        return JuryDefinition(
            identity=identity,
            profile=profile,
            category_guardrail_profile=guardrail_profile,
            category_guardrail_policy=guardrail_policy,
        )


class JuryIdentity(ContractModel):
    """Reproducible identity of the single Judge model used for the whole run."""

    schema_version: Literal["2.0"] = "2.0"
    jury_id: NonEmptyString
    version: NonEmptyString
    provider: NonEmptyString
    model: NonEmptyString
    profile_id: NonEmptyString
    profile_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    category_guardrail: GuardrailIdentity | None = None
    subjudge_context_mode: SubjudgeContextMode | None = None

    @classmethod
    def from_profiles(
        cls,
        *,
        jury_id: str,
        version: str,
        profile: ModelProfile,
        guardrail_profile: ModelProfile | None = None,
        guardrail_policy: GuardrailPolicy | None = None,
        subjudge_context_mode: SubjudgeContextMode | None = None,
    ) -> JuryIdentity:
        if (guardrail_profile is None) != (guardrail_policy is None):
            raise ConfigurationError("guardrail profile and policy must be supplied together")
        guardrail_identity = None
        if guardrail_profile is not None and guardrail_policy is not None:
            guardrail_identity = GuardrailIdentity(
                provider=guardrail_profile.provider,
                model=guardrail_profile.model_id,
                profile_id=guardrail_profile.profile_id,
                profile_hash=guardrail_profile.fingerprint,
                policy_id=guardrail_policy.policy_id,
                policy_hash=guardrail_policy.fingerprint,
                adapter_version=LLAMA_GUARD_ADAPTER_VERSION,
                prompt_version=LLAMA_GUARD_PROMPT_VERSION,
            )
        return cls(
            jury_id=jury_id,
            version=version,
            provider=profile.provider,
            model=profile.model_id,
            profile_id=profile.profile_id,
            profile_hash=profile.fingerprint,
            category_guardrail=guardrail_identity,
            subjudge_context_mode=subjudge_context_mode,
        )

    @classmethod
    def from_profile(
        cls,
        *,
        jury_id: str,
        version: str,
        profile: ModelProfile,
    ) -> JuryIdentity:
        """Backward-compatible constructor for a main-Judge-only plan."""

        return cls.from_profiles(jury_id=jury_id, version=version, profile=profile)

    @property
    def fingerprint(self) -> str:
        payload = self.model_dump(mode="json")
        # Keep historical main-Judge-only identities stable when the new policy is
        # unspecified. New plans opt in explicitly and therefore get a new identity.
        if self.subjudge_context_mode is None:
            payload.pop("subjudge_context_mode", None)
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


class JuryDefinition:
    """Resolved single-Judge plan used to build the runtime runner."""

    def __init__(
        self,
        *,
        identity: JuryIdentity,
        profile: ModelProfile,
        category_guardrail_profile: ModelProfile | None = None,
        category_guardrail_policy: GuardrailPolicy | None = None,
    ) -> None:
        if (
            identity.profile_id != profile.profile_id
            or identity.profile_hash != profile.fingerprint
        ):
            raise ConfigurationError("Judge profile does not match run identity")
        if (category_guardrail_profile is None) != (category_guardrail_policy is None):
            raise ConfigurationError(
                "category guardrail profile and policy must be supplied together"
            )
        guardrail_identity = identity.category_guardrail
        if category_guardrail_profile is None:
            if guardrail_identity is not None:
                raise ConfigurationError(
                    "Jury identity declares a guardrail but the definition does not"
                )
        elif (
            guardrail_identity is None
            or guardrail_identity.profile_id != category_guardrail_profile.profile_id
            or guardrail_identity.profile_hash != category_guardrail_profile.fingerprint
            or guardrail_identity.policy_id
            != _required_guardrail_policy(category_guardrail_policy).policy_id
            or guardrail_identity.policy_hash
            != _required_guardrail_policy(category_guardrail_policy).fingerprint
            or guardrail_identity.adapter_version != LLAMA_GUARD_ADAPTER_VERSION
            or guardrail_identity.prompt_version != LLAMA_GUARD_PROMPT_VERSION
        ):
            raise ConfigurationError("category guardrail does not match run identity")
        self.identity = identity
        self.profile = profile
        self.category_guardrail_profile = category_guardrail_profile
        self.category_guardrail_policy = category_guardrail_policy


def _required_guardrail_policy(policy: GuardrailPolicy | None) -> GuardrailPolicy:
    if policy is None:
        raise ConfigurationError("category guardrail policy is required")
    return policy
