from __future__ import annotations

from pathlib import Path

from safejudge.contracts.jury import JuryPlan
from safejudge.contracts.model import ModelRole
from safejudge.models.profiles import ModelRegistry
from safejudge.workflows.judge import _CategoryPayload, _judge_parameters


def test_routed_profile_excludes_unstable_endpoint_and_has_new_fingerprint() -> None:
    registry = ModelRegistry.load(Path("config/models.toml"))
    old = registry.get(
        "deepseek-v4-flash-no-reasoning-judge-v1",
        role=ModelRole.JUDGE,
        allow_unqualified=True,
    )
    routed = registry.get(
        "deepseek-v4-flash-routed-judge-v3",
        role=ModelRole.JUDGE,
        allow_unqualified=True,
    )
    parameters = _judge_parameters(profile=routed, parameters=None, payload_schema=_CategoryPayload)
    assert parameters["provider"] == {
        "only": ["deepinfra/fp8", "alibaba/fp8", "venice"],
        "allow_fallbacks": False,
        "require_parameters": True,
    }
    assert routed.fingerprint != old.fingerprint


def test_routed_coordinator_enters_jury_fingerprint() -> None:
    registry = ModelRegistry.load(Path("config/models.toml"))
    old = JuryPlan.load(
        Path("config/juries/m3-deepseek-gemma4-enable-llamaguard-intent-on-compact-v1.toml")
    ).resolve(registry, allow_unqualified=True)
    routed = JuryPlan.load(
        Path("config/juries/m3-deepseek-routed-gemma4-enable-llamaguard-intent-on-compact-v3.toml")
    ).resolve(registry, allow_unqualified=True)
    assert routed.identity.fingerprint != old.identity.fingerprint
    assert routed.profile.profile_id == "deepseek-v4-flash-routed-judge-v3"
