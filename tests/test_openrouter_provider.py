from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import ValidationError

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
from safejudge.models.openrouter import OpenRouterProvider, OpenRouterSettings

CAPABILITIES = ModelCapabilities(
    input_combinations=(
        ModalityCombination(modalities=frozenset({InputModality.TEXT})),
        ModalityCombination(modalities=frozenset({InputModality.TEXT, InputModality.IMAGE})),
        ModalityCombination(modalities=frozenset({InputModality.TEXT, InputModality.AUDIO})),
        ModalityCombination(modalities=frozenset({InputModality.TEXT, InputModality.VIDEO})),
    )
)


def settings(**overrides: Any) -> OpenRouterSettings:
    return OpenRouterSettings(
        _env_file=None,
        api_key="test-secret",
        model_id="vendor/model",
        **overrides,
    )


def request(parts: tuple[ModelTextPart | ModelMediaPart, ...] | None = None) -> ModelRequest:
    return ModelRequest(
        request_id="request_1",
        role=ModelRole.TARGET,
        parts=parts or (ModelTextPart(text="hello"),),
        parameters={"temperature": 0},
    )


def response_payload() -> dict[str, Any]:
    return {
        "id": "generation_1",
        "model": "vendor/model-2026-01",
        "choices": [{"message": {"content": "answer"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.0002},
    }


def test_api_key_is_not_read_from_ambient_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "ambient-secret")
    with pytest.raises(ValidationError):
        OpenRouterSettings(_env_file=None, model_id="vendor/model")


def test_settings_load_model_and_secret_from_dotenv(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "OPENROUTER_API_KEY=dotenv-secret\nOPENROUTER_MODEL_ID=vendor/model\n",
        encoding="utf-8",
    )

    loaded = OpenRouterSettings(_env_file=env_file)

    assert loaded.api_key.get_secret_value() == "dotenv-secret"
    assert loaded.model_id == "vendor/model"
    assert "dotenv-secret" not in repr(loaded)


def test_openrouter_serializes_image_audio_and_video(tmp_path: Path) -> None:
    (tmp_path / "image.png").write_bytes(b"image")
    (tmp_path / "audio.wav").write_bytes(b"audio")
    (tmp_path / "video.mp4").write_bytes(b"video")
    captured: list[dict[str, Any]] = []

    def handler(http_request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(http_request.content))
        return httpx.Response(200, json=response_payload())

    client = httpx.AsyncClient(
        base_url="https://openrouter.test/api/v1",
        transport=httpx.MockTransport(handler),
    )
    artifacts = InMemoryArtifactStore()
    provider = OpenRouterProvider(
        settings=settings(),
        capabilities=CAPABILITIES,
        artifact_store=artifacts,
        media_root=tmp_path,
        client=client,
    )
    parts = (
        ModelMediaPart(
            media=MediaRef(
                media_type=MediaType.IMAGE,
                uri="image.png",
                mime_type="image/png",
            )
        ),
        ModelMediaPart(
            media=MediaRef(
                media_type=MediaType.AUDIO,
                uri="audio.wav",
                mime_type="audio/wav",
            )
        ),
        ModelMediaPart(
            media=MediaRef(
                media_type=MediaType.VIDEO,
                uri="video.mp4",
                mime_type="video/mp4",
            )
        ),
        ModelTextPart(text="inspect all media"),
    )

    result = asyncio.run(provider.generate(request(parts), request_hash="a" * 64))
    asyncio.run(client.aclose())

    content = captured[0]["messages"][0]["content"]
    assert [item["type"] for item in content] == [
        "image_url",
        "input_audio",
        "video_url",
        "text",
    ]
    assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert content[1]["input_audio"]["format"] == "wav"
    assert content[2]["video_url"]["url"].startswith("data:video/mp4;base64,")
    assert result.model_version == "vendor/model-2026-01"
    assert result.cost is not None and str(result.cost.actual_usd) == "0.0002"
    assert result.raw_artifact is not None
    assert json.loads(artifacts.contents[result.raw_artifact.uri]) == response_payload()


@pytest.mark.parametrize(
    ("status", "error_type", "expected_kind", "retryable"),
    [
        (429, "rate_limit_exceeded", ProviderErrorKind.RATE_LIMIT, True),
        (503, "provider_overloaded", ProviderErrorKind.UNAVAILABLE, True),
        (401, "authentication", ProviderErrorKind.AUTHENTICATION, False),
        (400, "invalid_request", ProviderErrorKind.INVALID_REQUEST, False),
    ],
)
def test_openrouter_classifies_errors(
    status: int,
    error_type: str,
    expected_kind: ProviderErrorKind,
    retryable: bool,
) -> None:
    def handler(http_request: httpx.Request) -> httpx.Response:
        del http_request
        return httpx.Response(
            status,
            json={"error": {"message": "failed", "metadata": {"error_type": error_type}}},
        )

    client = httpx.AsyncClient(
        base_url="https://openrouter.test/api/v1",
        transport=httpx.MockTransport(handler),
    )
    provider = OpenRouterProvider(
        settings=settings(),
        capabilities=CAPABILITIES,
        artifact_store=InMemoryArtifactStore(),
        client=client,
    )

    with pytest.raises(ProviderError) as caught:
        asyncio.run(provider.generate(request(), request_hash="a" * 64))
    asyncio.run(client.aclose())

    assert caught.value.kind is expected_kind
    assert caught.value.retryable is retryable
    assert caught.value.raw_artifact is not None


def test_openrouter_returns_retryable_error_after_exactly_one_attempt() -> None:
    calls = 0

    def handler(http_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        del http_request
        calls += 1
        return httpx.Response(
            429,
            headers={"Retry-After": "0"},
            json={
                "error": {
                    "message": "limited",
                    "metadata": {"error_type": "rate_limit_exceeded"},
                }
            },
        )

    client = httpx.AsyncClient(
        base_url="https://openrouter.test/api/v1",
        transport=httpx.MockTransport(handler),
    )
    provider = OpenRouterProvider(
        settings=settings(),
        capabilities=CAPABILITIES,
        artifact_store=InMemoryArtifactStore(),
        client=client,
    )

    with pytest.raises(ProviderError) as caught:
        asyncio.run(provider.generate(request(), request_hash="a" * 64))
    asyncio.run(client.aclose())

    assert caught.value.retryable is True
    assert calls == 1


def test_openrouter_does_not_retry_non_retryable_error() -> None:
    calls = 0

    def handler(http_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        del http_request
        calls += 1
        return httpx.Response(401, json={"error": {"message": "bad key"}})

    client = httpx.AsyncClient(
        base_url="https://openrouter.test/api/v1",
        transport=httpx.MockTransport(handler),
    )
    provider = OpenRouterProvider(
        settings=settings(),
        capabilities=CAPABILITIES,
        artifact_store=InMemoryArtifactStore(),
        client=client,
    )

    with pytest.raises(ProviderError):
        asyncio.run(provider.generate(request(), request_hash="a" * 64))
    asyncio.run(client.aclose())

    assert calls == 1


@pytest.mark.parametrize("reserved", ["model", "messages", "stream"])
def test_openrouter_rejects_reserved_parameter_overrides(reserved: str) -> None:
    provider = OpenRouterProvider(
        settings=settings(),
        capabilities=CAPABILITIES,
        artifact_store=InMemoryArtifactStore(),
    )
    unsafe_request = ModelRequest(
        request_id="request_reserved",
        role=ModelRole.TARGET,
        parts=(ModelTextPart(text="hello"),),
        parameters={reserved: "attacker-controlled"},
    )

    with pytest.raises(ConfigurationError, match="reserved OpenRouter fields"):
        asyncio.run(provider.generate(unsafe_request, request_hash="a" * 64))


def test_openrouter_rejects_local_media_path_escape(tmp_path: Path) -> None:
    media_root = tmp_path / "media"
    media_root.mkdir()
    (tmp_path / "outside.png").write_bytes(b"secret")
    provider = OpenRouterProvider(
        settings=settings(),
        capabilities=CAPABILITIES,
        artifact_store=InMemoryArtifactStore(),
        media_root=media_root,
    )
    unsafe_request = request(
        (
            ModelMediaPart(
                media=MediaRef(
                    media_type=MediaType.IMAGE,
                    uri="../outside.png",
                    mime_type="image/png",
                )
            ),
            ModelTextPart(text="inspect"),
        )
    )

    with pytest.raises(ConfigurationError, match="escapes media_root"):
        asyncio.run(provider.generate(unsafe_request, request_hash="a" * 64))


def test_openrouter_rejects_oversized_local_media(tmp_path: Path) -> None:
    (tmp_path / "large.png").write_bytes(b"12345")
    provider = OpenRouterProvider(
        settings=settings(max_local_media_bytes=4),
        capabilities=CAPABILITIES,
        artifact_store=InMemoryArtifactStore(),
        media_root=tmp_path,
    )
    oversized_request = request(
        (
            ModelMediaPart(
                media=MediaRef(
                    media_type=MediaType.IMAGE,
                    uri="large.png",
                    mime_type="image/png",
                )
            ),
            ModelTextPart(text="inspect"),
        )
    )

    with pytest.raises(ConfigurationError, match="exceeds 4 bytes"):
        asyncio.run(provider.generate(oversized_request, request_hash="a" * 64))
