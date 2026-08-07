"""OpenRouter adapter using its OpenAI-compatible chat-completions endpoint."""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal, InvalidOperation
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
from safejudge.contracts.evaluation import ModelCost, PriceSnapshot, TokenUsage
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
    _ERROR_KIND_BY_TYPE: Mapping[str, ProviderErrorKind] = {
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
        return build_chat_payload(
            request,
            model_id=self.model_id,
            media_root=self._media_root,
            max_local_media_bytes=self.settings.max_local_media_bytes,
            reserved_fields_label="OpenRouter",
            audio_label="OpenRouter",
        )

    def _decode_response(
        self,
        response: httpx.Response,
        *,
        raw_artifact: ArtifactRef,
    ) -> dict[str, Any]:
        data = decode_json_object(
            response,
            provider_label="OpenRouter",
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
        message = error_data.get("message")
        if not isinstance(message, str):
            message = f"OpenRouter request failed with HTTP {response.status_code}"
        normalized_message = message.casefold()
        if (
            "key limit exceeded" in normalized_message
            or "insufficient credits" in normalized_message
        ):
            # OpenRouter may return account/key spending-limit exhaustion as HTTP
            # 403 without a specific error_type. It is a billing/quota failure, not
            # a provider content-policy refusal.
            kind = ProviderErrorKind.INSUFFICIENT_CREDITS
        else:
            kind = self._ERROR_KIND_BY_TYPE.get(
                error_type, error_kind_from_status(response.status_code)
            )
        retryable = kind in {
            ProviderErrorKind.TIMEOUT,
            ProviderErrorKind.RATE_LIMIT,
            ProviderErrorKind.UNAVAILABLE,
        }
        retry_delay = retry_after(response.headers.get("Retry-After"))
        return ProviderError(
            message,
            retryable=retryable,
            kind=kind,
            status_code=response.status_code,
            retry_after_seconds=retry_delay,
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
        parsed = parse_chat_completion(
            data,
            provider_label="OpenRouter",
            raw_artifact=raw_artifact,
        )
        cost = self._model_cost(parsed.raw_usage, parsed.token_usage)
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
