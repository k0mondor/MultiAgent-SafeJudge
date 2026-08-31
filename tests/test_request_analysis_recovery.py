from __future__ import annotations

import hashlib
import unittest
from pathlib import Path

from safejudge.constitution import ConstitutionRegistry, compile_constitution
from safejudge.contracts.dataset import RequestIntent
from safejudge.contracts.judging import (
    AmbiguityKind,
    AnalysisStatus,
    IntentAnalysis,
    RequestSnapshot,
    ScopeStatus,
)
from safejudge.grounding.contracts import (
    GroundingArtifact,
    GroundingMode,
    GroundingObservation,
    GroundingStatus,
    ObservationModality,
    RawGroundingObservation,
)
from safejudge.taxonomy import TaxonomyRegistry
from safejudge.workflows.judge import (
    _IntentPayload,
    _normalized_response_added_category_ids,
    _ResponseCategoryPayload,
)
from safejudge.workflows.prompts import (
    category_router_prompt,
    grounding_sufficiency_review_prompt,
    scope_boundary_review_prompt,
)


class RequestAnalysisRecoveryTests(unittest.TestCase):
    def test_infers_insufficient_grounding_when_review_omits_ambiguity_kind(self) -> None:
        payload = _IntentPayload.model_validate(
            {
                "analysis_status": "review_required",
                "scope_status": "unknown",
                "request_intent": "unknown",
                "requested_action": "perform the activity shown in the image",
                "intent_basis": "insufficient_grounding",
                "request_category_ids": [],
            }
        )

        self.assertEqual(payload.ambiguity_kind, AmbiguityKind.INSUFFICIENT_GROUNDING)

    def test_filters_request_categories_from_response_added_categories(self) -> None:
        payload = _ResponseCategoryPayload(
            response_added_category_ids=("A.1.h", "A.4.a"),
        )

        self.assertEqual(
            _normalized_response_added_category_ids(
                payload,
                request_category_ids=("A.1.h",),
            ),
            ("A.4.a",),
        )

    def test_recovery_prompts_separate_sufficiency_from_risk_without_target(self) -> None:
        content = (
            '{"parts":[{"kind":"text","text":"List the object shown in the image."}],'
            '"sample_id":"sample-1"}'
        )
        snapshot = RequestSnapshot(
            sample_id="sample-1",
            content=content,
            sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
        )
        observation = GroundingObservation.from_raw(
            RawGroundingObservation(
                modality=ObservationModality.IMAGE_VLM,
                text="Mrs.",
                confidence=0.9,
                tool_id="test-grounder",
                tool_version="1",
            ),
            evidence_id="G000",
        )
        grounding = GroundingArtifact.create(
            sample_id="sample-1",
            mode=GroundingMode.BLIND,
            status=GroundingStatus.COMPLETE,
            pipeline_id="test",
            pipeline_version="1",
            pipeline_hash="0" * 64,
            observations=(observation,),
        )
        previous = IntentAnalysis.model_construct(
            analysis_status=AnalysisStatus.REVIEW_REQUIRED,
            ambiguity_kind=AmbiguityKind.INSUFFICIENT_GROUNDING,
            scope_status=ScopeStatus.UNKNOWN,
            request_intent=RequestIntent.UNKNOWN,
            scope_id="taxonomy-router",
            requested_action="List the object shown in the image.",
            intent_basis="insufficient_grounding",
            request_category_ids=(),
            prompt_version="test",
            trace=None,
        )
        constitution = compile_constitution(
            ConstitutionRegistry.load(Path("config/constitutions")).get(
                "illegal-enablement-v1"
            ),
            axis="intent",
            scenarios=frozenset({"blind"}),
        )
        taxonomy = TaxonomyRegistry.load(Path("config/taxonomies")).get(
            "gb-t-45654-2025-safejudge-v1",
            version="1.1",
        )

        sufficiency_prompt = grounding_sufficiency_review_prompt(
            request_snapshot=snapshot,
            previous_intent=previous,
            grounding=grounding,
            constitution=constitution,
            taxonomy=taxonomy,
        )
        category_prompt = category_router_prompt(
            request_snapshot=snapshot,
            intent=previous.model_copy(
                update={
                    "analysis_status": AnalysisStatus.RESOLVED,
                    "ambiguity_kind": AmbiguityKind.NONE,
                    "scope_status": ScopeStatus.IN_SCOPE,
                    "request_intent": RequestIntent.HARMFUL,
                    "intent_basis": "mixed",
                }
            ),
            grounding=grounding,
            taxonomy=taxonomy,
        )
        scope_prompt = scope_boundary_review_prompt(
            request_snapshot=snapshot,
            previous_intent=previous.model_copy(
                update={
                    "analysis_status": AnalysisStatus.RESOLVED,
                    "ambiguity_kind": AmbiguityKind.NONE,
                    "scope_status": ScopeStatus.OUT_OF_SCOPE,
                    "request_intent": RequestIntent.BENIGN,
                    "intent_basis": "mixed",
                }
            ),
            grounding=grounding,
            constitution=constitution,
            taxonomy=taxonomy,
        )

        self.assertIn("Evidence sufficiency is about whether", sufficiency_prompt)
        self.assertIn("resolved benign, in-scope", sufficiency_prompt)
        self.assertIn("Catch-all leaves are nevertheless legitimate", category_prompt)
        self.assertIn("create or distribute malware", category_prompt)
        self.assertIn("in-scope benign control", scope_prompt)
        self.assertIn("never neutralizes an explicit act", scope_prompt)
        self.assertNotIn('"target_response":', sufficiency_prompt)
        self.assertNotIn('"target_response":', category_prompt)
        self.assertNotIn('"target_response":', scope_prompt)


if __name__ == "__main__":
    unittest.main()
