from __future__ import annotations

import unittest
from pathlib import Path

from safejudge.constitution import (
    ConstitutionRegistry,
    ConstitutionRouteAction,
    compile_constitution,
    route_categories,
    validate_triggered_rule_ids,
)
from safejudge.contracts.dataset import RequestIntent
from safejudge.contracts.judging import IntentAnalysis, JudgeAxis, ScopeStatus
from safejudge.core.errors import ContractValidationError
from safejudge.grounding.contracts import (
    GroundingArtifact,
    GroundingMode,
    GroundingStatus,
)
from safejudge.taxonomy import TaxonomyRegistry


class CategoryRouterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.constitutions = ConstitutionRegistry.load(Path("config/constitutions"))
        self.taxonomy = TaxonomyRegistry.load(Path("config/taxonomies")).get(
            "gb-t-45654-2025-safejudge-v1",
            version="1.0",
        )
        self.grounding = GroundingArtifact.create(
            sample_id="sample-1",
            mode=GroundingMode.BLIND,
            status=GroundingStatus.COMPLETE,
            pipeline_id="test",
            pipeline_version="1",
            pipeline_hash="0" * 64,
        )

    @staticmethod
    def _intent() -> IntentAnalysis:
        # The routing unit only consumes intake fields, so no provider trace is needed.
        return IntentAnalysis.model_construct(
            scope_status=ScopeStatus.IN_SCOPE,
            request_intent=RequestIntent.HARMFUL,
            scope_id="taxonomy-router",
            requested_action="disclose a trade secret and personal data",
            intent_basis="request_text",
            prompt_version="test",
            trace=None,  # type: ignore[arg-type]  # deliberately bypassed test trace
        )

    def test_one_response_can_route_to_multiple_leaf_categories(self) -> None:
        route = route_categories(
            self.taxonomy,
            self.constitutions,
            intent=self._intent(),
            grounding=self.grounding,
            category_ids=("A.4.f", "A.3.c"),
        )

        self.assertEqual(route.action, ConstitutionRouteAction.EVALUATE)
        self.assertEqual(route.selected_category_ids, ("A.3.c", "A.4.f"))
        self.assertEqual(
            {(item.category_id, item.constitution_id) for item in route.bindings},
            {("A.3.c", "gbt45654-a3-v1"), ("A.4.f", "gbt45654-a4-v1")},
        )

    def test_compiler_applies_only_rules_for_current_category(self) -> None:
        pack = self.constitutions.get("gbt45654-a3-v1")
        matching = compile_constitution(
            pack,
            axis=JudgeAxis.COMPLIANCE,
            category_id="A.3.c",
        )
        nonmatching = compile_constitution(
            pack,
            axis=JudgeAxis.COMPLIANCE,
            category_id="A.4.f",
        )

        self.assertIn("GBT45654_A3_CATEGORY_SCOPE", matching.applied_rule_ids)
        self.assertNotIn("GBT45654_A3_CATEGORY_SCOPE", nonmatching.applied_rule_ids)
        self.assertEqual(
            matching.category_rule_ids,
            ("GBT45654_A3_CATEGORY_SCOPE",),
        )

    def test_triggered_rules_must_come_from_compiled_category_policy(self) -> None:
        compiled = compile_constitution(
            self.constitutions.get("gbt45654-a3-v1"),
            axis=JudgeAxis.COMPLIANCE,
            category_id="A.3.c",
        )
        validate_triggered_rule_ids(
            compiled,
            ("GBT45654_A3_CATEGORY_SCOPE",),
        )
        with self.assertRaisesRegex(ContractValidationError, "absent"):
            validate_triggered_rule_ids(compiled, ("MADE_UP_RULE",))
        with self.assertRaisesRegex(ContractValidationError, "category-specific"):
            validate_triggered_rule_ids(compiled, ("COMPLIANCE_DECISION_ANCHORS",))

    def test_harmful_in_scope_empty_selection_requires_review(self) -> None:
        route = route_categories(
            self.taxonomy,
            self.constitutions,
            intent=self._intent(),
            grounding=self.grounding,
            category_ids=(),
        )

        self.assertEqual(route.action, ConstitutionRouteAction.REVIEW_REQUIRED)
        self.assertEqual(route.bindings, ())

    def test_benign_in_scope_empty_selection_is_not_evaluated(self) -> None:
        intent = self._intent().model_copy(
            update={"request_intent": RequestIntent.BENIGN}
        )
        route = route_categories(
            self.taxonomy,
            self.constitutions,
            intent=intent,
            grounding=self.grounding,
            category_ids=(),
        )

        self.assertEqual(route.action, ConstitutionRouteAction.NOT_EVALUATED)
        self.assertEqual(route.bindings, ())

    def test_disabled_or_unknown_category_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "not routing-enabled"):
            route_categories(
                self.taxonomy,
                self.constitutions,
                intent=self._intent(),
                grounding=self.grounding,
                category_ids=("A.3",),
            )


if __name__ == "__main__":
    unittest.main()
