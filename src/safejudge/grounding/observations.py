"""Response-facing grounding observations without assisted benchmark risk labels."""

from __future__ import annotations

import json

from safejudge.grounding.contracts import GroundingArtifact


def response_grounding_observations(grounding: GroundingArtifact) -> list[dict[str, str]]:
    """Keep media referents, not request-risk labels, when judging a Target response."""

    observations: list[dict[str, str]] = []
    for item in grounding.observations:
        if item.modality.value != "benchmark_label":
            observations.append({"modality": item.modality.value, "text": item.text})
            continue
        try:
            label = json.loads(item.text)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(label, dict):
            continue
        corresponding_text = label.get("official_corresponding_text")
        if isinstance(corresponding_text, str) and corresponding_text.strip():
            observations.append(
                {"modality": "benchmark_corresponding_text", "text": corresponding_text.strip()}
            )
    return observations
