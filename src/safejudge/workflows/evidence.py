"""Deterministic evidence candidates selected by judges through stable IDs."""

from __future__ import annotations

import hashlib
import re
from typing import Literal

from pydantic import Field

from safejudge.contracts.base import ContractModel
from safejudge.contracts.evaluation import TargetResponse
from safejudge.contracts.judging import EvidenceRef, RequestSnapshot
from safejudge.grounding.contracts import GroundingArtifact


class EvidenceCandidate(ContractModel):
    evidence_id: str = Field(pattern=r"^[RTG][0-9]{3}$")
    source: Literal["request_snapshot", "target_response", "grounding_observation"]
    source_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    text: str = Field(min_length=1, max_length=500)

    def to_ref(self) -> EvidenceRef:
        return EvidenceRef(
            source=self.source,
            source_sha256=self.source_sha256,
            start=self.start,
            end=self.end,
            text=self.text,
        )


def build_evidence_candidates(
    *,
    request_snapshot: RequestSnapshot,
    target_response: TargetResponse,
    grounding_artifact: GroundingArtifact | None = None,
    maximum_per_source: int = 24,
) -> tuple[EvidenceCandidate, ...]:
    candidates: list[EvidenceCandidate] = []
    candidates.extend(
        _source_candidates(
            source="request_snapshot",
            prefix="R",
            text=request_snapshot.content,
            maximum=maximum_per_source,
        )
    )
    candidates.extend(
        _source_candidates(
            source="target_response",
            prefix="T",
            text=target_response.text,
            maximum=maximum_per_source,
        )
    )
    if grounding_artifact is not None:
        candidates.extend(
            EvidenceCandidate(
                evidence_id=observation.evidence_id,
                source="grounding_observation",
                source_sha256=observation.observation_sha256,
                start=0,
                end=len(observation.text),
                text=observation.text,
            )
            for observation in grounding_artifact.observations
        )
    return tuple(candidates)


def resolve_evidence_ids(
    evidence_ids: tuple[str, ...],
    candidates: tuple[EvidenceCandidate, ...],
) -> tuple[EvidenceRef, ...]:
    by_id = {candidate.evidence_id: candidate for candidate in candidates}
    missing = sorted(set(evidence_ids).difference(by_id))
    if missing:
        raise ValueError(f"unknown evidence IDs: {missing}")
    return tuple(by_id[evidence_id].to_ref() for evidence_id in evidence_ids)


def _source_candidates(
    *,
    source: Literal["request_snapshot", "target_response"],
    prefix: Literal["R", "T"],
    text: str,
    maximum: int,
) -> tuple[EvidenceCandidate, ...]:
    spans: list[tuple[int, int]] = []
    if len(text) <= 500:
        spans.append(_trim_span(text, 0, len(text)))
    for match in re.finditer(r"[^.!?\u3002\uFF01\uFF1F\n]+[.!?\u3002\uFF01\uFF1F]?", text):
        start, end = match.span()
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if 0 < end - start <= 500:
            spans.append((start, end))
    for start in range(0, len(text), 400):
        end = min(len(text), start + 400)
        if end > start:
            spans.append(_trim_span(text, start, end))

    source_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    unique: list[tuple[int, int]] = []
    for span in spans:
        if span[1] <= span[0]:
            continue
        if span not in unique:
            unique.append(span)
        if len(unique) >= maximum:
            break
    return tuple(
        EvidenceCandidate(
            evidence_id=f"{prefix}{index:03d}",
            source=source,
            source_sha256=source_hash,
            start=start,
            end=end,
            text=text[start:end],
        )
        for index, (start, end) in enumerate(unique)
    )


def _trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    """Keep stored offsets aligned with ContractModel's stripped text value."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end
