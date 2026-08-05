from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

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
from safejudge.core.errors import ConfigurationError, ProviderError, ProviderErrorKind
from safejudge.models.artifacts import InMemoryArtifactStore
from safejudge.models.local_openai import LocalOpenAIProvider, LocalOpenAISettings

CAPABILITIES = ModelCapabilities(
    input_combinations=(
        ModalityCombination(modalities=frozenset({InputModality.TEXT})),
        ModalityCombination(modalities=frozenset({InputModality.TEXT, InputModality.IMAGE})),
    )
)


def settings(**overrides: Any) -> LocalOpenAISettings:
    return LocalOpenAISettings(
        _env_file=None,
        model_id="local/test-model",
        **overrides,
    )


def request(
    parts: tuple[ModelTextPart | ModelMediaPart, ...] | None = None,
    *,
    parameters: dict[str, Any] | None = None,
) -> ModelRequest:
    return ModelRequest(
        request_id="local-request",
        role=ModelRole.TARGET,
        parts=parts or (ModelTextPart(text="hello"),),
        parameters=parameters or {"temperature": 0},
    )


def test_local_settings_ignore_ambient_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCAL_MODEL_API_KEY", "ambient-secret")

    loaded = settings()

    assert loaded.api_key is None


def test_local_provider_serializes_image_and_normalizes_content_blocks(tmp_path: Path) -> None:
    (tmp_path / "image.png").write_bytes(b"image")
    captured: list[httpx.Request] = []

    def handler(http_request: httpx.Request) -> httpx.Response:
        captured.append(http_request)
        return httpx.Response(
            200,
            json={
                "model": "local/test-model-revision",
                "choices": [
                    {
                        "message": {
                            "content": [
                                {"type": "text", "text": "local"},
                                {"type": "text", "text": " answer"},
                            ]
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"input_tokens": 8, "output_tokens": 2},
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    artifacts = InMemoryArtifactStore()
    provider = LocalOpenAIProvider(
        settings=settings(api_key="local-secret"),
        capabilities=CAPABILITIES,
        artifact_store=artifacts,
        media_root=tmp_path,
        client=client,
    )
    model_request = request(
        (
            ModelMediaPart(
                media=MediaRef(
                    media_type=MediaType.IMAGE,
                    uri="image.png",
                    mime_type="image/png",
                )
            ),
            ModelTextPart(text="inspect"),
        )
    )

    result = asyncio.run(provider.generate(model_request, request_hash="a" * 64))
    asyncio.run(client.aclose())

    payload = json.loads(captured[0].content)
    assert str(captured[0].url) == "http://127.0.0.1:8000/v1/chat/completions"
    assert captured[0].headers["Authorization"] == "Bearer local-secret"
    assert payload["messages"][0]["content"][0]["image_url"]["url"].startswith(
        "data:image/png;base64,"
    )
    assert result.response_id == "local_aaaaaaaaaaaaaaaaaaaaaaaa"
    assert result.answer == "local answer"
    assert result.model_version == "local/test-model-revision"
    assert result.token_usage is not None and result.token_usage.total_tokens == 10
    assert result.cost is not None and result.cost.actual_usd == 0
    assert result.raw_artifact is not None


def test_local_provider_uses_refusal_when_content_is_null() -> None:
    def handler(http_request: httpx.Request) -> httpx.Response:
        assert "Authorization" not in http_request.headers
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-refusal",
                "choices": [
                    {
                        "message": {"content": None, "refusal": "I cannot help with that."},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = LocalOpenAIProvider(
        settings=settings(),
        capabilities=CAPABILITIES,
        artifact_store=InMemoryArtifactStore(),
        client=client,
    )

    result = asyncio.run(provider.generate(request(), request_hash="b" * 64))
    asyncio.run(client.aclose())

    assert result.answer == "I cannot help with that."
    assert result.token_usage is None
    assert result.cost is None


def test_local_provider_classifies_server_error_and_preserves_raw_response() -> None:
    def handler(http_request: httpx.Request) -> httpx.Response:
        del http_request
        return httpx.Response(503, json={"error": {"message": "model is loading"}})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = LocalOpenAIProvider(
        settings=settings(),
        capabilities=CAPABILITIES,
        artifact_store=InMemoryArtifactStore(),
        client=client,
    )

    with pytest.raises(ProviderError) as caught:
        asyncio.run(provider.generate(request(), request_hash="c" * 64))
    asyncio.run(client.aclose())

    assert caught.value.kind is ProviderErrorKind.UNAVAILABLE
    assert caught.value.retryable is True
    assert caught.value.raw_artifact is not None


def test_local_provider_rejects_reserved_parameter_override() -> None:
    provider = LocalOpenAIProvider(
        settings=settings(),
        capabilities=CAPABILITIES,
        artifact_store=InMemoryArtifactStore(),
    )

    with pytest.raises(ConfigurationError, match="reserved local provider fields"):
        asyncio.run(
            provider.generate(
                request(parameters={"messages": []}),
                request_hash="d" * 64,
            )
        )
