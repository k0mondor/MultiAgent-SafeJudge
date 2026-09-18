"""Response-facing observations from blind request grounding."""

from __future__ import annotations

from safejudge.grounding.contracts import GroundingArtifact


def response_grounding_observations(grounding: GroundingArtifact) -> list[dict[str, str]]:
    """Forward only observed media facts when judging a Target response."""

    return [
        {"modality": item.modality.value, "text": item.text}
        for item in grounding.observations
    ]
