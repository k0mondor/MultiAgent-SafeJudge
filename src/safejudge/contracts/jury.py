"""Versioned contracts for assigning heterogeneous models to Jury seats."""

from __future__ import annotations

import hashlib
import json
import tomllib
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from safejudge.contracts.base import ContractModel
from safejudge.contracts.dataset import NonEmptyString
from safejudge.contracts.model import ModelRole
from safejudge.core.errors import ConfigurationError
from safejudge.models.profiles import ModelProfile, ModelRegistry


class JurySeat(StrEnum):
    INTENT = "intent"
    COMPLIANCE = "compliance"
    HARM_ENABLEMENT = "harm_enablement"
    OVERSENSITIVITY = "oversensitivity"
    ARBITRATION = "arbitration"


class JuryPlan(ContractModel):
    """Small, explicit configuration mapping every Jury seat to one profile."""

    schema_version: Literal["1.0"] = "1.0"
    jury_id: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9._-]*$")
    version: NonEmptyString
    seats: dict[JurySeat, NonEmptyString]

    @model_validator(mode="after")
    def all_seats_are_assigned(self) -> JuryPlan:
        if set(self.seats) != set(JurySeat):
            missing = sorted(seat.value for seat in set(JurySeat) - set(self.seats))
            extra = sorted(str(seat) for seat in set(self.seats) - set(JurySeat))
            raise ValueError(f"jury seats must be exact; missing={missing}, extra={extra}")
        return self

    @classmethod
    def load(cls, path: Path) -> JuryPlan:
        source = path.resolve()
        if not source.is_file():
            raise ConfigurationError(f"jury plan does not exist: {source}")
        try:
            data = tomllib.loads(source.read_text(encoding="utf-8"))
            return cls.model_validate(data)
        except (OSError, tomllib.TOMLDecodeError, ValueError) as error:
            raise ConfigurationError(f"invalid jury plan {source}: {error}") from error

    def resolve(
        self,
        registry: ModelRegistry,
        *,
        allow_unqualified: bool = False,
    ) -> JuryDefinition:
        profiles = {
            seat: registry.get(
                profile_id,
                role=ModelRole.JUDGE,
                allow_unqualified=allow_unqualified,
            )
            for seat, profile_id in self.seats.items()
        }
        identity = JuryIdentity(
            jury_id=self.jury_id,
            version=self.version,
            seats=tuple(
                JurySeatIdentity.from_profile(seat, profiles[seat]) for seat in JurySeat
            ),
        )
        return JuryDefinition(identity=identity, profiles=profiles)


class JurySeatIdentity(ContractModel):
    seat: JurySeat
    provider: NonEmptyString
    model: NonEmptyString
    profile_id: NonEmptyString
    profile_hash: str = Field(pattern=r"^[a-f0-9]{64}$")

    @classmethod
    def from_profile(cls, seat: JurySeat, profile: ModelProfile) -> JurySeatIdentity:
        return cls(
            seat=seat,
            provider=profile.provider,
            model=profile.model_id,
            profile_id=profile.profile_id,
            profile_hash=profile.fingerprint,
        )


class JuryIdentity(ContractModel):
    schema_version: Literal["1.0"] = "1.0"
    jury_id: NonEmptyString
    version: NonEmptyString
    seats: tuple[JurySeatIdentity, ...] = Field(min_length=5, max_length=5)

    @model_validator(mode="after")
    def seats_are_complete_and_ordered(self) -> JuryIdentity:
        actual = tuple(item.seat for item in self.seats)
        if actual != tuple(JurySeat):
            raise ValueError("jury identity seats must contain every seat in canonical order")
        return self

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def seat(self, seat: JurySeat) -> JurySeatIdentity:
        return next(item for item in self.seats if item.seat is seat)


class JuryDefinition:
    """Resolved plan used to build runners; profiles are runtime objects, not output data."""

    def __init__(
        self,
        *,
        identity: JuryIdentity,
        profiles: dict[JurySeat, ModelProfile],
    ) -> None:
        if set(profiles) != set(JurySeat):
            raise ConfigurationError("resolved Jury must provide a profile for every seat")
        for seat, profile in profiles.items():
            expected = identity.seat(seat)
            if (
                expected.profile_id != profile.profile_id
                or expected.profile_hash != profile.fingerprint
            ):
                raise ConfigurationError(f"Jury profile does not match identity for {seat.value}")
        self.identity = identity
        self.profiles = dict(profiles)
