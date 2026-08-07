"""Provider-neutral contracts for target and judge model calls."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, JsonValue, StringConstraints, computed_field, model_validator

from safejudge.contracts.artifact import ArtifactRef
from safejudge.contracts.base import ContractModel
from safejudge.contracts.dataset import MediaRef, MediaType, NonEmptyString
from safejudge.contracts.evaluation import ModelCost, TokenUsage

MODEL_SCHEMA_VERSION: Literal["1.0"] = "1.0"
RequestHash = Annotated[str, StringConstraints(pattern=r"^[a-f0-9]{64}$")]


class ModelRole(StrEnum):
    """A model's trust role in an evaluation."""

    TARGET = "target"
    JUDGE = "judge"
    GROUNDING = "grounding"


class InvocationContext(ContractModel):
    """Trusted budget and accounting scope for one logical model invocation."""

    experiment_id: NonEmptyString
    run_id: NonEmptyString


class InputModality(StrEnum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"


class ModelTextPart(ContractModel):
    kind: Literal["text"] = "text"
    text: NonEmptyString


class ModelMediaPart(ContractModel):
    kind: Literal["media"] = "media"
    media: MediaRef


ModelInputPart = Annotated[ModelTextPart | ModelMediaPart, Field(discriminator="kind")]


class ModalityCombination(ContractModel):
    """One exact input-modality combination accepted by a model."""

    modalities: frozenset[InputModality] = Field(min_length=1)


class ModelCapabilities(ContractModel):
    """Explicit combinations, avoiding assumptions based on individual modalities."""

    input_combinations: tuple[ModalityCombination, ...] = Field(min_length=1)

    def supports(self, modalities: frozenset[InputModality]) -> bool:
        return any(item.modalities == modalities for item in self.input_combinations)


class ModelRequest(ContractModel):
    schema_version: Literal["1.0"] = MODEL_SCHEMA_VERSION
    request_id: NonEmptyString
    role: ModelRole
    parts: tuple[ModelInputPart, ...] = Field(min_length=1)
    parameters: dict[str, JsonValue] = Field(default_factory=dict)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def modalities(self) -> frozenset[InputModality]:
        values: set[InputModality] = set()
        for part in self.parts:
            if isinstance(part, ModelTextPart):
                values.add(InputModality.TEXT)
            else:
                values.add(InputModality(part.media.media_type.value))
        return frozenset(values)

    @model_validator(mode="after")
    def require_text_instruction(self) -> ModelRequest:
        if InputModality.TEXT not in self.modalities:
            raise ValueError("model requests must contain at least one text part")
        return self


class ModelResponse(ContractModel):
    schema_version: Literal["1.0"] = MODEL_SCHEMA_VERSION
    response_id: NonEmptyString
    request_hash: RequestHash
    role: ModelRole
    provider: NonEmptyString
    model: NonEmptyString
    model_version: str | None = None
    answer: NonEmptyString
    finish_reason: str | None = None
    token_usage: TokenUsage | None = None
    latency_ms: int = Field(ge=0)
    cost: ModelCost | None = None
    raw_artifact: ArtifactRef | None = None

    @model_validator(mode="after")
    def cost_requires_usage(self) -> ModelResponse:
        if self.cost is not None and self.token_usage is None:
            raise ValueError("cost requires token_usage")
        return self


def modalities_from_media_types(
    media_types: set[MediaType], *, include_text: bool = True
) -> frozenset[InputModality]:
    values = {InputModality(item.value) for item in media_types}
    if include_text:
        values.add(InputModality.TEXT)
    return frozenset(values)
