from __future__ import annotations

import unittest

from pydantic import ValidationError

from safejudge.contracts.jury import JuryPlan


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


if __name__ == "__main__":
    unittest.main()
