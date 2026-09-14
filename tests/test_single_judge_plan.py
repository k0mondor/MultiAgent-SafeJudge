from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from safejudge.contracts.judging import JudgeAxis
from safejudge.contracts.jury import JuryPlan
from safejudge.models.cache import SQLiteModelStore
from safejudge.models.invocation import InvocationPolicy
from safejudge.models.profiles import ModelRegistry
from safejudge.workflows.jury import build_jury_runtime


class _Provider:
    provider_name = "openrouter"

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id


class SingleJudgePlanTests(unittest.TestCase):
    def test_plan_accepts_exactly_one_judge_profile(self) -> None:
        plan = JuryPlan.model_validate(
            {
                "schema_version": "2.0",
                "jury_id": "single-judge-test",
                "version": "1",
                "judge_profile": "judge-v1",
            }
        )
        self.assertEqual(plan.judge_profile, "judge-v1")

    def test_legacy_heterogeneous_seats_are_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            JuryPlan.model_validate(
                {
                    "schema_version": "2.0",
                    "jury_id": "legacy-heterogeneous",
                    "version": "1",
                    "seats": {"intent": "judge-a", "compliance": "judge-b"},
                }
            )

    def test_guardrail_profile_and_policy_must_be_paired(self) -> None:
        with self.assertRaises(ValidationError):
            JuryPlan.model_validate(
                {
                    "schema_version": "2.0",
                    "jury_id": "guardrail-test",
                    "version": "1",
                    "judge_profile": "judge-v1",
                    "category_guardrail_profile": "guard-v1",
                }
            )

    def test_default_deepseek_guardrail_plans_resolve(self) -> None:
        registry = ModelRegistry.load(Path("config/models.toml"))
        for name, provider in (
            ("m3-deepseek-llamaguard-remote-v1.toml", "openrouter"),
            ("m3-deepseek-llamaguard-local-v1.toml", "local-openai"),
        ):
            definition = JuryPlan.load(Path("config/juries") / name).resolve(
                registry,
                allow_unqualified=True,
            )
            self.assertEqual(
                definition.profile.model_id,
                "deepseek/deepseek-v4-flash",
            )
            self.assertEqual(
                definition.identity.subjudge_context_mode,
                "compact",
            )
            self.assertIsNotNone(definition.category_guardrail_profile)
            assert definition.category_guardrail_profile is not None
            self.assertEqual(definition.category_guardrail_profile.provider, provider)

    def test_enablement_profile_is_resolved_as_a_separate_role(self) -> None:
        registry = ModelRegistry.load(Path("config/models.toml"))
        definition = JuryPlan.model_validate(
            {
                "schema_version": "2.0",
                "jury_id": "role-separated-test",
                "version": "1",
                "judge_profile": "deepseek-v4-flash-no-reasoning-judge-v1",
                "enablement_profile": "seed-2.0-mini-judge-v1",
                "subjudge_context_mode": "compact",
            }
        ).resolve(registry, allow_unqualified=True)

        self.assertEqual(definition.profile.model_id, "deepseek/deepseek-v4-flash")
        self.assertIsNotNone(definition.enablement_profile)
        assert definition.enablement_profile is not None
        self.assertEqual(definition.enablement_profile.model_id, "bytedance-seed/seed-2.0-mini")
        self.assertIsNotNone(definition.identity.enablement_judge)
        assert definition.identity.enablement_judge is not None
        self.assertEqual(
            definition.identity.enablement_judge.profile_id,
            "seed-2.0-mini-judge-v1",
        )
        self.assertIsNotNone(definition.identity.coordinator)
        assert definition.identity.coordinator is not None
        self.assertEqual(
            definition.identity.coordinator.profile_id,
            "deepseek-v4-flash-no-reasoning-judge-v1",
        )

    def test_intent_ablation_modes_produce_distinct_jury_fingerprints(self) -> None:
        registry = ModelRegistry.load(Path("config/models.toml"))
        fingerprints = {
            JuryPlan.model_validate(
                {
                    "jury_id": "intent-mode-fingerprint-test",
                    "version": "1",
                    "judge_profile": "deepseek-v4-flash-no-reasoning-judge-v1",
                    "enablement_profile": "seed-2.0-mini-judge-v1",
                    "enablement_intent_mode": mode,
                }
            ).resolve(registry, allow_unqualified=True).identity.fingerprint
            for mode in (
                "intent_on_compact",
                "intent_off_raw",
                "intent_off_masked",
            )
        }
        self.assertEqual(len(fingerprints), 3)

    def test_intent_ablation_requires_enablement_profile(self) -> None:
        with self.assertRaisesRegex(ValidationError, "enablement_profile"):
            JuryPlan.model_validate(
                {
                    "jury_id": "invalid-intent-ablation",
                    "version": "1",
                    "judge_profile": "deepseek-v4-flash-no-reasoning-judge-v1",
                    "enablement_intent_mode": "intent_off_raw",
                }
            )

    def test_optional_enablement_role_preserves_existing_jury_hash(self) -> None:
        registry = ModelRegistry.load(Path("config/models.toml"))
        definition = JuryPlan.load(
            Path("config/juries/m3-deepseek-llamaguard-remote-v1.toml")
        ).resolve(registry, allow_unqualified=True)
        self.assertEqual(
            definition.identity.fingerprint,
            "6d67aa48d9247848f54b5390da8fdb59b0954d1e496520ba6fdb3f16041bdf48",
        )

    def test_runtime_dispatches_only_harm_enablement_to_specialized_runner(self) -> None:
        registry = ModelRegistry.load(Path("config/models.toml"))
        definition = JuryPlan.model_validate(
            {
                "schema_version": "2.0",
                "jury_id": "role-dispatch-test",
                "version": "1",
                "judge_profile": "deepseek-v4-flash-no-reasoning-judge-v1",
                "enablement_profile": "seed-2.0-mini-judge-v1",
            }
        ).resolve(registry, allow_unqualified=True)
        with tempfile.TemporaryDirectory() as directory:
            runtime = build_jury_runtime(
                definition,
                provider=_Provider("deepseek/deepseek-v4-flash"),
                enablement_provider=_Provider("bytedance-seed/seed-2.0-mini"),
                store=SQLiteModelStore(Path(directory) / "calls.sqlite3"),
                policy=InvocationPolicy(),
            )

        self.assertEqual(runtime.intent.profile.model_id, "deepseek/deepseek-v4-flash")
        self.assertEqual(
            runtime.for_axis(JudgeAxis.HARM_ENABLEMENT).profile.model_id,
            "bytedance-seed/seed-2.0-mini",
        )
        self.assertEqual(
            runtime.for_axis(JudgeAxis.COMPLIANCE).profile.model_id,
            "deepseek/deepseek-v4-flash",
        )
        self.assertEqual(runtime.arbitration.profile.model_id, "deepseek/deepseek-v4-flash")

    def test_legacy_plan_keeps_full_context_behavior(self) -> None:
        registry = ModelRegistry.load(Path("config/models.toml"))
        definition = JuryPlan.load(
            Path("config/juries/m3-single-judge-v1.toml")
        ).resolve(registry, allow_unqualified=True)
        self.assertIsNone(definition.identity.subjudge_context_mode)


if __name__ == "__main__":
    unittest.main()
