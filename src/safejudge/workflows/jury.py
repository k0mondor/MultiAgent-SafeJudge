"""Runtime wiring for the five explicit heterogeneous Jury seats."""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict, model_validator

from safejudge.contracts.judging import JudgeAxis
from safejudge.contracts.jury import JuryDefinition, JuryIdentity, JurySeat
from safejudge.core.errors import ConfigurationError
from safejudge.models.base import ModelProvider
from safejudge.models.cache import SQLiteModelStore
from safejudge.models.invocation import InvocationPolicy, ModelInvoker
from safejudge.workflows.judge import JudgeRunner


class JuryRuntime(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    identity: JuryIdentity
    runners: dict[JurySeat, JudgeRunner]

    @model_validator(mode="after")
    def runners_match_identity(self) -> JuryRuntime:
        if set(self.runners) != set(JurySeat):
            raise ValueError("Jury runtime must provide exactly one runner per seat")
        for seat, runner in self.runners.items():
            expected = self.identity.seat(seat)
            if (
                runner.profile.profile_id != expected.profile_id
                or runner.profile.fingerprint != expected.profile_hash
            ):
                raise ValueError(f"Jury runner does not match identity for {seat.value}")
        return self

    def for_axis(self, axis: JudgeAxis) -> JudgeRunner:
        return self.runners[JurySeat(axis.value)]

    @property
    def intent(self) -> JudgeRunner:
        return self.runners[JurySeat.INTENT]

    @property
    def arbitration(self) -> JudgeRunner:
        return self.runners[JurySeat.ARBITRATION]


def build_jury_runtime(
    definition: JuryDefinition,
    *,
    providers: Mapping[JurySeat, ModelProvider],
    store: SQLiteModelStore,
    policy: InvocationPolicy,
) -> JuryRuntime:
    if set(providers) != set(JurySeat):
        raise ConfigurationError("Jury providers must contain every seat exactly once")
    runners: dict[JurySeat, JudgeRunner] = {}
    for seat in JurySeat:
        profile = definition.profiles[seat]
        provider = providers[seat]
        expected_provider = (
            "local-openai" if provider.provider_name == "local" else provider.provider_name
        )
        if expected_provider != profile.provider or provider.model_id != profile.model_id:
            raise ConfigurationError(f"provider does not match Jury profile for {seat.value}")
        runners[seat] = JudgeRunner(
            ModelInvoker(provider=provider, store=store, policy=policy),
            profile=profile,
        )
    return JuryRuntime(identity=definition.identity, runners=runners)
