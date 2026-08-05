"""OpenRouter adapter using its OpenAI-compatible chat-completions endpoint."""

from __future__ import annotations

import base64
from collections.abc import Mapping
from decimal import Decimal, InvalidOperation
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


class OpenRouterSettings(BaseSettings):
    """Secrets and model selection loaded from ``.env``, never ambient process env."""

    model_config = SettingsConfigDict(
        env_prefix="OPENROUTER_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    api_key: SecretStr
    model_id: str = Field(min_length=1)
    base_url: str = "https://openrouter.ai/api/v1"
    timeout_seconds: float = Field(default=30, gt=0)
    max_local_media_bytes: int = Field(default=100 * 1024 * 1024, gt=0)
    input_per_million_tokens: Decimal = Field(default=Decimal("0"), ge=0)
    output_per_million_tokens: Decimal = Field(default=Decimal("0"), ge=0)

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


class OpenRouterProvider:
    _RESERVED_PARAMETERS: ClassVar[frozenset[str]] = frozenset(
        {"model", "messages", "stream"}
    )
    _ERROR_KIND_BY_TYPE: ClassVar[Mapping[str, ProviderErrorKind]] = {
        "authentication": ProviderErrorKind.AUTHENTICATION,
        "payment_required": ProviderErrorKind.INSUFFICIENT_CREDITS,
        "rate_limit_exceeded": ProviderErrorKind.RATE_LIMIT,
        "provider_overloaded": ProviderErrorKind.UNAVAILABLE,
        "provider_unavailable": ProviderErrorKind.UNAVAILABLE,
        "timeout": ProviderErrorKind.TIMEOUT,
        "invalid_request": ProviderErrorKind.INVALID_REQUEST,
        "invalid_prompt": ProviderErrorKind.INVALID_REQUEST,
        "content_policy_violation": ProviderErrorKind.CONTENT_POLICY,
        "refusal": ProviderErrorKind.CONTENT_POLICY,
        "server": ProviderErrorKind.UNAVAILABLE,
        "unmapped": ProviderErrorKind.UNKNOWN,
    }

    def __init__(
        self,
        *,
        settings: OpenRouterSettings,
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
        return "openrouter"

    @property
    def model_id(self) -> str:
        return self.settings.model_id

    @property
    def capabilities(self) -> ModelCapabilities:
        return self._capabilities

    @property
    def price_snapshot(self) -> PriceSnapshot:
        return PriceSnapshot(
            input_per_million_tokens=self.settings.input_per_million_tokens,
            output_per_million_tokens=self.settings.output_per_million_tokens,
            source="openrouter-.env",
        )

    async def generate(self, request: ModelRequest, *, request_hash: str) -> ModelResponse:
        payload = self._build_payload(request)
        started = monotonic()
        response = await self._post(payload, request_hash=request_hash)
        try:
            raw_artifact = self._artifact_store.write_bytes(
                category="openrouter-responses",
                content=response.content,
                content_type=response.headers.get(
                    "Content-Type",
                    "application/octet-stream",
                ),
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

    async def _post(self, payload: dict[str, Any], *, request_hash: str) -> httpx.Response:
        headers = {
            "Authorization": f"Bearer {self.settings.api_key.get_secret_value()}",
            "Content-Type": "application/json",
            "Idempotency-Key": request_hash,
        }
        try:
            if self._client is not None:
                return await self._client.post(
                    "/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=self.settings.timeout_seconds,
                )
            async with httpx.AsyncClient(base_url=self.settings.base_url) as client:
                return await client.post(
                    "/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=self.settings.timeout_seconds,
                )
        except httpx.TimeoutException as error:
            raise ProviderError(
                "OpenRouter request timed out",
                retryable=True,
                kind=ProviderErrorKind.TIMEOUT,
            ) from error
        except httpx.TransportError as error:
            raise ProviderError(
                "OpenRouter transport failed",
                retryable=True,
                kind=ProviderErrorKind.UNAVAILABLE,
            ) from error

    def _build_payload(self, request: ModelRequest) -> dict[str, Any]:
        reserved = sorted(self._RESERVED_PARAMETERS.intersection(request.parameters))
        if reserved:
            raise ConfigurationError(
                f"model parameters cannot override reserved OpenRouter fields: {reserved}"
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
            raise ConfigurationError("OpenRouter audio input must be a validated local file")
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
        size_bytes = physical.stat().st_size
        if size_bytes > self.settings.max_local_media_bytes:
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
                "OpenRouter returned non-JSON content",
                retryable=response.status_code >= 500,
                kind=ProviderErrorKind.MALFORMED_RESPONSE,
                status_code=response.status_code,
                raw_artifact=raw_artifact,
            ) from error
        if not isinstance(data, dict):
            raise ProviderError(
                "OpenRouter returned a non-object JSON response",
                retryable=False,
                kind=ProviderErrorKind.MALFORMED_RESPONSE,
                status_code=response.status_code,
                raw_artifact=raw_artifact,
            )
        if response.is_error or "error" in data:
            raise self._classified_error(response, data, raw_artifact=raw_artifact)
        return data

    def _classified_error(
        self,
        response: httpx.Response,
        data: dict[str, Any],
        *,
        raw_artifact: ArtifactRef,
    ) -> ProviderError:
        error_data = data.get("error")
        if not isinstance(error_data, dict):
            error_data = {}
        metadata = error_data.get("metadata")
        error_type = metadata.get("error_type") if isinstance(metadata, dict) else None
        if not isinstance(error_type, str):
            top_error_type = data.get("error_type")
            error_type = top_error_type if isinstance(top_error_type, str) else ""
        kind = self._ERROR_KIND_BY_TYPE.get(error_type, _kind_from_status(response.status_code))
        retryable = kind in {
            ProviderErrorKind.TIMEOUT,
            ProviderErrorKind.RATE_LIMIT,
            ProviderErrorKind.UNAVAILABLE,
        }
        message = error_data.get("message")
        if not isinstance(message, str):
            message = f"OpenRouter request failed with HTTP {response.status_code}"
        retry_after = _retry_after(response.headers.get("Retry-After"))
        return ProviderError(
            message,
            retryable=retryable,
            kind=kind,
            status_code=response.status_code,
            retry_after_seconds=retry_after,
            raw_artifact=raw_artifact,
        )

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
            answer = choice["message"]["content"]
            response_id = str(data["id"])
        except (KeyError, IndexError, TypeError) as error:
            raise ProviderError(
                "OpenRouter response is missing a completion answer",
                retryable=False,
                kind=ProviderErrorKind.MALFORMED_RESPONSE,
                raw_artifact=raw_artifact,
            ) from error
        if not isinstance(answer, str) or not answer.strip():
            raise ProviderError(
                "OpenRouter response contains an empty completion answer",
                retryable=True,
                kind=ProviderErrorKind.MALFORMED_RESPONSE,
                raw_artifact=raw_artifact,
            )
        usage = _token_usage(data.get("usage"))
        cost = self._model_cost(data.get("usage"), usage)
        returned_model = data.get("model")
        return ModelResponse(
            response_id=response_id,
            request_hash=request_hash,
            role=request.role,
            provider=self.provider_name,
            model=self.model_id,
            model_version=returned_model if isinstance(returned_model, str) else None,
            answer=answer,
            finish_reason=choice.get("finish_reason"),
            token_usage=usage,
            latency_ms=latency_ms,
            cost=cost,
            raw_artifact=raw_artifact,
        )

    def _model_cost(self, raw_usage: Any, usage: TokenUsage | None) -> ModelCost | None:
        if usage is None:
            return None
        estimated = self.price_snapshot.estimate(usage)
        actual: Decimal | None = None
        if isinstance(raw_usage, dict) and raw_usage.get("cost") is not None:
            try:
                actual = Decimal(str(raw_usage["cost"]))
            except InvalidOperation:
                actual = None
        return ModelCost(
            price_snapshot=self.price_snapshot,
            estimated_usd=estimated,
            actual_usd=actual,
        )


def _token_usage(raw_usage: Any) -> TokenUsage | None:
    if not isinstance(raw_usage, dict):
        return None
    prompt = raw_usage.get("prompt_tokens")
    completion = raw_usage.get("completion_tokens")
    if not isinstance(prompt, int) or not isinstance(completion, int):
        return None
    return TokenUsage(input_tokens=prompt, output_tokens=completion)


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
        parsed = float(value)
    except ValueError:
        return None
    return max(0, parsed)


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
