from __future__ import annotations

import pytest
from pydantic import ValidationError

from safejudge.contracts.dataset import (
    CanonicalMultimodalSample,
    MediaPart,
    MediaRef,
    MediaType,
    RequestContext,
    RequestIntent,
    SourceRecord,
    TextPart,
)
from safejudge.contracts.evaluation import (
    ResponseComplianceLevel,
    TargetModelRef,
    TargetResponse,
    TokenUsage,
)


def image_text_sample() -> CanonicalMultimodalSample:
    return CanonicalMultimodalSample(
        sample_id="sample_demo",
        parts=(
            MediaPart(
                media=MediaRef(
                    media_type=MediaType.IMAGE,
                    uri="media/demo.png",
                    mime_type="image/png",
                    sha256="a" * 64,
                )
            ),
            TextPart(text="Describe the image."),
        ),
        source=SourceRecord(
            dataset_name="fixture",
            dataset_version="1",
            original_id="demo-1",
        ),
        request_context=RequestContext(intent=RequestIntent.BENIGN),
    )


def test_sample_round_trips_with_discriminated_multimodal_parts() -> None:
    sample = image_text_sample()

    restored = CanonicalMultimodalSample.model_validate_json(sample.model_dump_json())

    assert restored == sample
    assert restored.parts[0].kind == "media"
    assert restored.parts[1].kind == "text"


def test_request_cannot_contain_a_response_score() -> None:
    payload = image_text_sample().model_dump(mode="json")
    payload["response_compliance_level"] = 2

    with pytest.raises(ValidationError):
        CanonicalMultimodalSample.model_validate(payload)


def test_image_rejects_duration() -> None:
    with pytest.raises(ValidationError):
        MediaRef(
            media_type=MediaType.IMAGE,
            uri="media/demo.png",
            mime_type="image/png",
            duration_ms=100,
        )


def test_target_response_is_separate_and_traceable() -> None:
    response = TargetResponse(
        response_id="response_demo",
        sample_id=image_text_sample().sample_id,
        model=TargetModelRef(provider="fake", model="safejudge-fixture"),
        text="I cannot help with that request.",
        token_usage=TokenUsage(input_tokens=10, output_tokens=7),
        latency_ms=4,
    )

    assert response.model.role == "target"
    assert response.token_usage is not None
    assert response.token_usage.total_tokens == 17


def test_response_compliance_level_has_explicit_order() -> None:
    assert ResponseComplianceLevel.FULLY_COMPLIANT.value == 0
    assert ResponseComplianceLevel.PARTIALLY_COMPLIANT.value == 1
    assert ResponseComplianceLevel.SEVERELY_NONCOMPLIANT.value == 2
