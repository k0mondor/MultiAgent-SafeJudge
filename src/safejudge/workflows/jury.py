"""Runtime wiring for one Judge model shared by all judging stages."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, model_validator

from safejudge.contracts.judging import JudgeAxis
from safejudge.contracts.jury import JuryDefinition, JuryIdentity
from safejudge.core.errors import ConfigurationError
from safejudge.models.base import ModelProvider
from safejudge.models.cache import SQLiteModelStore
from safejudge.models.invocation import InvocationPolicy, ModelInvoker
from safejudge.workflows.judge import JudgeRunner


class JuryRuntime(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    identity: JuryIdentity
    runner: JudgeRunner

    @model_validator(mode="after")
    def runner_matches_identity(self) -> JuryRuntime:
        if (
            self.runner.profile.profile_id != self.identity.profile_id
            or self.runner.profile.fingerprint != self.identity.profile_hash
        ):
            raise ValueError("Judge runner does not match identity")
        return self

    def for_axis(self, axis: JudgeAxis) -> JudgeRunner:
        del axis
        return self.runner

    @property
    def intent(self) -> JudgeRunner:
        return self.runner

    @property
    def arbitration(self) -> JudgeRunner:
        return self.runner


def build_jury_runtime(
    definition: JuryDefinition,
    *,
    provider: ModelProvider,
    store: SQLiteModelStore,
    policy: InvocationPolicy,
) -> JuryRuntime:
    profile = definition.profile
    expected_provider = (
        "local-openai" if provider.provider_name == "local" else provider.provider_name
    )
    if expected_provider != profile.provider or provider.model_id != profile.model_id:
        raise ConfigurationError("provider does not match Judge profile")
    runner = JudgeRunner(
        ModelInvoker(provider=provider, store=store, policy=policy),
        profile=profile,
    )
    return JuryRuntime(identity=definition.identity, runner=runner)
