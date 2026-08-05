from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from safejudge.contracts.dataset import MediaRef, MediaType
from safejudge.contracts.model import (
    InputModality,
    ModalityCombination,
    ModelCapabilities,
    ModelMediaPart,
    ModelRequest,
    ModelRole,
    ModelTextPart,
)
from safejudge.models.artifacts import InMemoryArtifactStore
from safejudge.models.local_transformers import LocalGeneration, LocalTransformersProvider


class StubRuntime:
    def __init__(self) -> None:
        self.conversation: Sequence[Mapping[str, Any]] | None = None
        self.max_new_tokens: int | None = None

    def generate(
        self,
        conversation: Sequence[Mapping[str, Any]],
        *,
        max_new_tokens: int,
        temperature: float,
    ) -> LocalGeneration:
        self.conversation = conversation
        self.max_new_tokens = max_new_tokens
        assert temperature == 0
        return LocalGeneration(
            answer="normalized local answer",
            input_tokens=18,
            output_tokens=4,
            finish_reason="stop",
            metrics={"device": "cuda:0"},
        )


def test_local_transformers_provider_resolves_media_and_normalizes_response(
    tmp_path: Path,
) -> None:
    media = tmp_path / "image.png"
    media.write_bytes(b"image")
    runtime = StubRuntime()
    artifacts = InMemoryArtifactStore()
    provider = LocalTransformersProvider(
        model_path=tmp_path / "model",
        media_root=tmp_path,
        capabilities=ModelCapabilities(
            input_combinations=(
                ModalityCombination(
                    modalities=frozenset({InputModality.TEXT, InputModality.IMAGE})
                ),
            )
        ),
        artifact_store=artifacts,
        runtime=runtime,
        max_new_tokens=32,
    )
    request = ModelRequest(
        request_id="local-transformers-test",
        role=ModelRole.TARGET,
        parts=(
            ModelMediaPart(
                media=MediaRef(
                    media_type=MediaType.IMAGE,
                    uri="image.png",
                    mime_type="image/png",
                )
            ),
            ModelTextPart(text="inspect this image"),
        ),
        parameters={"temperature": 0, "max_new_tokens": 12},
    )

    response = asyncio.run(provider.generate(request, request_hash="a" * 64))

    assert response.provider == "local-transformers"
    assert response.answer == "normalized local answer"
    assert response.token_usage is not None
    assert response.token_usage.total_tokens == 22
    assert response.cost is not None and response.cost.actual_usd == 0
    assert response.raw_artifact is not None
    assert response.raw_artifact.uri in artifacts.contents
    assert runtime.max_new_tokens == 12
    assert runtime.conversation is not None
    content = runtime.conversation[0]["content"]
    assert content[0]["type"] == "image"
    assert Path(content[0]["image"]) == media
    assert content[1] == {"type": "text", "text": "inspect this image"}
