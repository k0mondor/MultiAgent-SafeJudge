from __future__ import annotations

import unittest
from pathlib import Path

from pydantic import ValidationError

from safejudge.contracts.jury import JuryPlan
from safejudge.models.profiles import ModelRegistry


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

    def test_legacy_plan_keeps_full_context_behavior(self) -> None:
        registry = ModelRegistry.load(Path("config/models.toml"))
        definition = JuryPlan.load(
            Path("config/juries/m3-single-judge-v1.toml")
        ).resolve(registry, allow_unqualified=True)
        self.assertIsNone(definition.identity.subjudge_context_mode)


if __name__ == "__main__":
    unittest.main()
