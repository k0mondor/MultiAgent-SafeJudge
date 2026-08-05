"""Local OpenAI-compatible chat-completions provider."""

from __future__ import annotations

import base64
from collections.abc import Mapping
from decimal import Decimal
from pathlib import Path
from time import monotonic
from typing import Any, ClassVar

import httpx
from pydantic import Field, SecretStr
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from safejudge.contracts.artifact import ArtifactRef
from safejudge.contracts.dataset import MediaType
from safejudge.contracts.evaluation import ModelCost, PriceSnapshot, TokenUsage
from safejudge.contracts.model import (
    ModelCapabilities,
    ModelMediaPart,
    ModelRequest,
    ModelResponse,
    ModelTextPart,
)
from safejudge.core.errors import (
    ArtifactError,
    ConfigurationError,
    ProviderError,
    ProviderErrorKind,
)
from safejudge.models.artifacts import ArtifactStore


class LocalOpenAISettings(BaseSettings):
    """Local endpoint settings loaded from the project ``.env`` only."""

    model_config = SettingsConfigDict(
        env_prefix="LOCAL_MODEL_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    base_url: str = Field(default="http://127.0.0.1:8000/v1", min_length=1)
    model_id: str = Field(min_length=1, validation_alias="LOCAL_MODEL_ID")
    api_key: SecretStr | None = None
    timeout_seconds: float = Field(default=120, gt=0)
    max_local_media_bytes: int = Field(default=100 * 1024 * 1024, gt=0)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        del settings_cls, env_settings, file_secret_settings
        return init_settings, dotenv_settings


class JudgeOpenAISettings(LocalOpenAISettings):
    """Independent local endpoint settings for a staged text judge service."""

    model_config = SettingsConfigDict(
        env_prefix="JUDGE_MODEL_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    base_url: str = Field(
        default="http://127.0.0.1:8000/v1",
        min_length=1,
        validation_alias="JUDGE_MODEL_BASE_URL",
    )
    model_id: str = Field(min_length=1, validation_alias="JUDGE_MODEL_ID")
    api_key: SecretStr | None = Field(
        default=None,
        validation_alias="JUDGE_MODEL_API_KEY",
    )
    timeout_seconds: float = Field(
        default=120,
        gt=0,
        validation_alias="JUDGE_MODEL_TIMEOUT_SECONDS",
    )
    max_local_media_bytes: int = Field(
        default=100 * 1024 * 1024,
        gt=0,
        validation_alias="JUDGE_MODEL_MAX_LOCAL_MEDIA_BYTES",
    )


class LocalOpenAIProvider:
    """Calls a local server that implements OpenAI chat completions."""

    _RESERVED_PARAMETERS: ClassVar[frozenset[str]] = frozenset(
        {"model", "messages", "stream"}
    )

    def __init__(
        self,
        *,
        settings: LocalOpenAISettings,
        capabilities: ModelCapabilities,
        artifact_store: ArtifactStore,
        media_root: Path | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.settings = settings
        self._capabilities = capabilities
        self._artifact_store = artifact_store
        self._media_root = media_root
        self._client = client

    @property
    def provider_name(self) -> str:
        return "local-openai"

    @property
    def model_id(self) -> str:
        return self.settings.model_id

    @property
    def capabilities(self) -> ModelCapabilities:
        return self._capabilities

    @property
    def price_snapshot(self) -> PriceSnapshot:
        return PriceSnapshot(source="local-inference")

    async def generate(self, request: ModelRequest, *, request_hash: str) -> ModelResponse:
        payload = self._build_payload(request)
        started = monotonic()
        response = await self._post(payload)
        try:
            raw_artifact = self._artifact_store.write_bytes(
                category="local-openai-responses",
                content=response.content,
                content_type=response.headers.get("Content-Type", "application/octet-stream"),
            )
        except ArtifactError as error:
            raise ProviderError(
                str(error),
                retryable=False,
                kind=ProviderErrorKind.ARTIFACT_PERSISTENCE,
            ) from error
        data = self._decode_response(response, raw_artifact=raw_artifact)
        return self._to_model_response(
            data,
            request=request,
            request_hash=request_hash,
            latency_ms=max(0, round((monotonic() - started) * 1000)),
            raw_artifact=raw_artifact,
        )

    async def _post(self, payload: dict[str, Any]) -> httpx.Response:
        headers = {"Content-Type": "application/json"}
        if self.settings.api_key is not None:
            api_key = self.settings.api_key.get_secret_value().strip()
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
        url = f"{self.settings.base_url.rstrip('/')}/chat/completions"
        try:
            if self._client is not None:
                return await self._client.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=self.settings.timeout_seconds,
                )
            async with httpx.AsyncClient() as client:
                return await client.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=self.settings.timeout_seconds,
                )
        except httpx.TimeoutException as error:
            raise ProviderError(
                "local model request timed out",
                retryable=True,
                kind=ProviderErrorKind.TIMEOUT,
            ) from error
        except httpx.TransportError as error:
            raise ProviderError(
                "cannot connect to local model server",
                retryable=True,
                kind=ProviderErrorKind.UNAVAILABLE,
            ) from error

    def _build_payload(self, request: ModelRequest) -> dict[str, Any]:
        reserved = sorted(self._RESERVED_PARAMETERS.intersection(request.parameters))
        if reserved:
            raise ConfigurationError(
                f"model parameters cannot override reserved local provider fields: {reserved}"
            )
        content: list[dict[str, Any]] = []
        for part in request.parts:
            if isinstance(part, ModelTextPart):
                content.append({"type": "text", "text": part.text})
            elif isinstance(part, ModelMediaPart):
                content.append(self._media_content(part))
        return {
            **request.parameters,
            "model": self.model_id,
            "messages": [{"role": "user", "content": content}],
            "stream": False,
        }

    def _media_content(self, part: ModelMediaPart) -> dict[str, Any]:
        media = part.media
        if media.media_type is MediaType.IMAGE:
            url = self._data_or_url(media.uri, media.mime_type)
            return {"type": "image_url", "image_url": {"url": url}}
        if media.media_type is MediaType.VIDEO:
            url = self._data_or_url(media.uri, media.mime_type)
            return {"type": "video_url", "video_url": {"url": url}}
        if media.uri.startswith(("https://", "http://", "data:")):
            raise ConfigurationError("local audio input must be a validated local file")
        audio_path = self._local_path(media.uri)
        encoded = base64.b64encode(audio_path.read_bytes()).decode("ascii")
        audio_format = _audio_format(media.mime_type, audio_path.suffix)
        return {"type": "input_audio", "input_audio": {"data": encoded, "format": audio_format}}

    def _data_or_url(self, uri: str, mime_type: str) -> str:
        if uri.startswith(("https://", "http://", "data:")):
            return uri
        path = self._local_path(uri)
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    def _local_path(self, uri: str) -> Path:
        path = Path(uri)
        if path.is_absolute():
            raise ConfigurationError(f"absolute local media paths are not allowed: {uri}")
        if self._media_root is None:
            raise ConfigurationError("media_root is required for local media inputs")
        root = self._media_root.resolve()
        physical = (root / path).resolve()
        try:
            physical.relative_to(root)
        except ValueError as error:
            raise ConfigurationError(f"local media path escapes media_root: {uri}") from error
        if not physical.is_file():
            raise ConfigurationError(f"local media file does not exist: {uri}")
        if physical.stat().st_size > self.settings.max_local_media_bytes:
            raise ConfigurationError(
                f"local media file exceeds {self.settings.max_local_media_bytes} bytes: {uri}"
            )
        return physical

    def _decode_response(
        self,
        response: httpx.Response,
        *,
        raw_artifact: ArtifactRef,
    ) -> dict[str, Any]:
        try:
            data = response.json()
        except ValueError as error:
            raise ProviderError(
                "local model server returned non-JSON content",
                retryable=response.status_code >= 500,
                kind=ProviderErrorKind.MALFORMED_RESPONSE,
                status_code=response.status_code,
                raw_artifact=raw_artifact,
            ) from error
        if not isinstance(data, dict):
            raise ProviderError(
                "local model server returned a non-object JSON response",
                retryable=False,
                kind=ProviderErrorKind.MALFORMED_RESPONSE,
                status_code=response.status_code,
                raw_artifact=raw_artifact,
            )
        if response.is_error or "error" in data:
            raise _classified_error(response, data, raw_artifact=raw_artifact)
        return data

    def _to_model_response(
        self,
        data: dict[str, Any],
        *,
        request: ModelRequest,
        request_hash: str,
        latency_ms: int,
        raw_artifact: ArtifactRef,
    ) -> ModelResponse:
        try:
            choice = data["choices"][0]
            message = choice["message"]
        except (KeyError, IndexError, TypeError) as error:
            raise ProviderError(
                "local model response is missing a completion choice",
                retryable=False,
                kind=ProviderErrorKind.MALFORMED_RESPONSE,
                raw_artifact=raw_artifact,
            ) from error
        if not isinstance(choice, dict) or not isinstance(message, dict):
            raise ProviderError(
                "local model response contains an invalid completion choice",
                retryable=False,
                kind=ProviderErrorKind.MALFORMED_RESPONSE,
                raw_artifact=raw_artifact,
            )
        answer = _completion_text(message)
        if not answer.strip():
            raise ProviderError(
                "local model response contains an empty completion answer",
                retryable=True,
                kind=ProviderErrorKind.MALFORMED_RESPONSE,
                raw_artifact=raw_artifact,
            )
        raw_response_id = data.get("id")
        response_id = (
            raw_response_id
            if isinstance(raw_response_id, str) and raw_response_id.strip()
            else f"local_{request_hash[:24]}"
        )
        usage = _token_usage(data.get("usage"))
        returned_model = data.get("model")
        finish_reason = choice.get("finish_reason")
        return ModelResponse(
            response_id=response_id,
            request_hash=request_hash,
            role=request.role,
            provider=self.provider_name,
            model=self.model_id,
            model_version=returned_model if isinstance(returned_model, str) else None,
            answer=answer,
            finish_reason=finish_reason if isinstance(finish_reason, str) else None,
            token_usage=usage,
            latency_ms=latency_ms,
            cost=(
                ModelCost(
                    price_snapshot=self.price_snapshot,
                    estimated_usd=Decimal("0"),
                    actual_usd=Decimal("0"),
                )
                if usage is not None
                else None
            ),
            raw_artifact=raw_artifact,
        )


def _completion_text(message: Mapping[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if isinstance(text, str):
                pieces.append(text)
        if pieces:
            return "".join(pieces)
    refusal = message.get("refusal")
    return refusal if isinstance(refusal, str) else ""


def _token_usage(raw_usage: Any) -> TokenUsage | None:
    if not isinstance(raw_usage, dict):
        return None
    input_tokens = raw_usage.get("prompt_tokens", raw_usage.get("input_tokens"))
    output_tokens = raw_usage.get("completion_tokens", raw_usage.get("output_tokens"))
    if not isinstance(input_tokens, int) or not isinstance(output_tokens, int):
        return None
    return TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens)


def _classified_error(
    response: httpx.Response,
    data: dict[str, Any],
    *,
    raw_artifact: ArtifactRef,
) -> ProviderError:
    error_data = data.get("error")
    if not isinstance(error_data, dict):
        error_data = {}
    message = error_data.get("message")
    if not isinstance(message, str):
        message = f"local model request failed with HTTP {response.status_code}"
    kind = _kind_from_status(response.status_code)
    return ProviderError(
        message,
        retryable=kind
        in {ProviderErrorKind.TIMEOUT, ProviderErrorKind.RATE_LIMIT, ProviderErrorKind.UNAVAILABLE},
        kind=kind,
        status_code=response.status_code,
        retry_after_seconds=_retry_after(response.headers.get("Retry-After")),
        raw_artifact=raw_artifact,
    )


def _kind_from_status(status: int) -> ProviderErrorKind:
    if status in {408, 504}:
        return ProviderErrorKind.TIMEOUT
    if status == 429:
        return ProviderErrorKind.RATE_LIMIT
    if status >= 500:
        return ProviderErrorKind.UNAVAILABLE
    if status == 401:
        return ProviderErrorKind.AUTHENTICATION
    if status == 402:
        return ProviderErrorKind.INSUFFICIENT_CREDITS
    if status == 403:
        return ProviderErrorKind.CONTENT_POLICY
    if 400 <= status < 500:
        return ProviderErrorKind.INVALID_REQUEST
    return ProviderErrorKind.UNKNOWN


def _retry_after(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return max(0, float(value))
    except ValueError:
        return None


def _audio_format(mime_type: str, suffix: str) -> str:
    by_mime = {
        "audio/wav": "wav",
        "audio/x-wav": "wav",
        "audio/mpeg": "mp3",
        "audio/mp3": "mp3",
        "audio/ogg": "ogg",
        "audio/flac": "flac",
    }
    return by_mime.get(mime_type, suffix.removeprefix(".").lower() or "wav")
