"""Local OpenAI-compatible chat-completions provider."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from time import monotonic
from typing import Any

import httpx
from pydantic import Field, SecretStr
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from safejudge.contracts.artifact import ArtifactRef
from safejudge.contracts.evaluation import ModelCost, PriceSnapshot
from safejudge.contracts.model import (
    ModelCapabilities,
    ModelRequest,
    ModelResponse,
)
from safejudge.core.errors import (
    ArtifactError,
    ProviderError,
    ProviderErrorKind,
)
from safejudge.models.artifacts import ArtifactStore
from safejudge.models.openai_compat import (
    build_chat_payload,
    decode_json_object,
    error_kind_from_status,
    parse_chat_completion,
    retry_after,
)


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
            async with httpx.AsyncClient(trust_env=False) as client:
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
        return build_chat_payload(
            request,
            model_id=self.model_id,
            media_root=self._media_root,
            max_local_media_bytes=self.settings.max_local_media_bytes,
            reserved_fields_label="local provider",
            audio_label="local",
        )

    def _decode_response(
        self,
        response: httpx.Response,
        *,
        raw_artifact: ArtifactRef,
    ) -> dict[str, Any]:
        data = decode_json_object(
            response,
            provider_label="local model server",
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
        parsed = parse_chat_completion(
            data,
            provider_label="local model",
            raw_artifact=raw_artifact,
        )
        return ModelResponse(
            response_id=parsed.response_id,
            request_hash=request_hash,
            role=request.role,
            provider=self.provider_name,
            model=self.model_id,
            model_version=parsed.model_version,
            answer=parsed.answer,
            finish_reason=parsed.finish_reason,
            token_usage=parsed.token_usage,
            latency_ms=latency_ms,
            cost=(
                ModelCost(
                    price_snapshot=self.price_snapshot,
                    estimated_usd=Decimal("0"),
                    actual_usd=Decimal("0"),
                )
                if parsed.token_usage is not None
                else None
            ),
            raw_artifact=raw_artifact,
        )
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
    kind = error_kind_from_status(response.status_code)
    return ProviderError(
        message,
        retryable=kind
        in {ProviderErrorKind.TIMEOUT, ProviderErrorKind.RATE_LIMIT, ProviderErrorKind.UNAVAILABLE},
        kind=kind,
        status_code=response.status_code,
        retry_after_seconds=retry_after(response.headers.get("Retry-After")),
        raw_artifact=raw_artifact,
    )
