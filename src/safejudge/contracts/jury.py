"""Versioned contract for one Judge model used by every judging stage."""

from __future__ import annotations

import hashlib
import json
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import Field

from safejudge.contracts.base import ContractModel
from safejudge.contracts.dataset import NonEmptyString
from safejudge.contracts.model import ModelRole
from safejudge.core.errors import ConfigurationError
from safejudge.models.profiles import ModelProfile, ModelRegistry


class JuryPlan(ContractModel):
    """Single-model plan shared by intent, category, panel, and arbitration calls."""

    schema_version: Literal["2.0"] = "2.0"
    jury_id: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9._-]*$")
    version: NonEmptyString
    judge_profile: NonEmptyString

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
        identity = JuryIdentity.from_profile(
            jury_id=self.jury_id,
            version=self.version,
            profile=profile,
        )
        return JuryDefinition(identity=identity, profile=profile)


class JuryIdentity(ContractModel):
    """Reproducible identity of the single Judge model used for the whole run."""

    schema_version: Literal["2.0"] = "2.0"
    jury_id: NonEmptyString
    version: NonEmptyString
    provider: NonEmptyString
    model: NonEmptyString
    profile_id: NonEmptyString
    profile_hash: str = Field(pattern=r"^[a-f0-9]{64}$")

    @classmethod
    def from_profile(
        cls,
        *,
        jury_id: str,
        version: str,
        profile: ModelProfile,
    ) -> JuryIdentity:
        return cls(
            jury_id=jury_id,
            version=version,
            provider=profile.provider,
            model=profile.model_id,
            profile_id=profile.profile_id,
            profile_hash=profile.fingerprint,
        )

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.model_dump(mode="json"),
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
    ) -> None:
        if (
            identity.profile_id != profile.profile_id
            or identity.profile_hash != profile.fingerprint
        ):
            raise ConfigurationError("Judge profile does not match run identity")
        self.identity = identity
        self.profile = profile
