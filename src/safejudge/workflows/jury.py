"""Runtime wiring for one Judge model shared by all judging stages."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, model_validator

from safejudge.contracts.judging import JudgeAxis
from safejudge.contracts.jury import (
    JuryDefinition,
    JuryIdentity,
    SubjudgeContextMode,
)
from safejudge.core.errors import ConfigurationError
from safejudge.guardrails.llama_guard import LlamaGuardCategoryRunner
from safejudge.guardrails.policy import (
    LLAMA_GUARD_ADAPTER_VERSION,
    LLAMA_GUARD_PROMPT_VERSION,
)
from safejudge.models.base import ModelProvider
from safejudge.models.cache import SQLiteModelStore
from safejudge.models.invocation import InvocationPolicy, ModelInvoker
from safejudge.workflows.judge import JudgeRunner


class JuryRuntime(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    identity: JuryIdentity
    runner: JudgeRunner
    category_guardrail: LlamaGuardCategoryRunner | None = None

    @model_validator(mode="after")
    def runner_matches_identity(self) -> JuryRuntime:
        if (
            self.runner.profile.profile_id != self.identity.profile_id
            or self.runner.profile.fingerprint != self.identity.profile_hash
        ):
            raise ValueError("Judge runner does not match identity")
        guardrail_identity = self.identity.category_guardrail
        if self.category_guardrail is None:
            if guardrail_identity is not None:
                raise ValueError("Jury identity requires a category guardrail")
        elif (
            guardrail_identity is None
            or self.category_guardrail.profile.profile_id != guardrail_identity.profile_id
            or self.category_guardrail.profile.fingerprint != guardrail_identity.profile_hash
            or self.category_guardrail.policy.policy_id != guardrail_identity.policy_id
            or self.category_guardrail.policy.fingerprint != guardrail_identity.policy_hash
            or guardrail_identity.adapter_version != LLAMA_GUARD_ADAPTER_VERSION
            or guardrail_identity.prompt_version != LLAMA_GUARD_PROMPT_VERSION
        ):
            raise ValueError("category guardrail does not match Jury identity")
        return self

    def for_axis(
        self,
        axis: JudgeAxis,
    ) -> JudgeRunner:
        del axis
        return self.runner

    @property
    def guardrail(self) -> LlamaGuardCategoryRunner | None:
        return self.category_guardrail

    @property
    def subjudge_context_mode(self) -> SubjudgeContextMode:
        """Resolve legacy plans to their historical full-context behavior."""

        return self.identity.subjudge_context_mode or "full"

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
    guardrail_provider: ModelProvider | None = None,
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
    guardrail_runner = None
    guardrail_profile = definition.category_guardrail_profile
    guardrail_policy = definition.category_guardrail_policy
    if guardrail_profile is not None and guardrail_policy is not None:
        if guardrail_provider is None:
            raise ConfigurationError("category guardrail provider is required")
        expected_guardrail_provider = (
            "local-openai"
            if guardrail_provider.provider_name == "local"
            else guardrail_provider.provider_name
        )
        if (
            expected_guardrail_provider != guardrail_profile.provider
            or guardrail_provider.model_id != guardrail_profile.model_id
        ):
            raise ConfigurationError("provider does not match category guardrail profile")
        guardrail_runner = LlamaGuardCategoryRunner(
            ModelInvoker(
                provider=guardrail_provider,
                store=store,
                policy=policy,
            ),
            profile=guardrail_profile,
            policy=guardrail_policy,
        )
    elif guardrail_provider is not None:
        raise ConfigurationError(
            "category guardrail provider supplied without a configured profile"
        )
    return JuryRuntime(
        identity=definition.identity,
        runner=runner,
        category_guardrail=guardrail_runner,
    )
