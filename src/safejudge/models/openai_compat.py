"""Shared mechanics for OpenAI-compatible chat-completions providers."""

from __future__ import annotations

import base64
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from safejudge.contracts.artifact import ArtifactRef
from safejudge.contracts.dataset import MediaType
from safejudge.contracts.evaluation import TokenUsage
from safejudge.contracts.model import ModelMediaPart, ModelRequest, ModelTextPart
from safejudge.core.errors import ConfigurationError, ProviderError, ProviderErrorKind

_RESERVED_CHAT_PARAMETERS = frozenset({"model", "messages", "stream"})


@dataclass(frozen=True, slots=True)
class ParsedChatCompletion:
    response_id: str
    model_version: str | None
    answer: str
    finish_reason: str | None
    token_usage: TokenUsage | None
    raw_usage: Any


def build_chat_payload(
    request: ModelRequest,
    *,
    model_id: str,
    media_root: Path | None,
    max_local_media_bytes: int,
    reserved_fields_label: str,
    audio_label: str,
) -> dict[str, Any]:
    """Build the common OpenAI chat payload and validate local media once."""

    reserved = sorted(_RESERVED_CHAT_PARAMETERS.intersection(request.parameters))
    if reserved:
        raise ConfigurationError(
            f"model parameters cannot override reserved {reserved_fields_label} fields: "
            f"{reserved}"
        )
    content: list[dict[str, Any]] = []
    for part in request.parts:
        if isinstance(part, ModelTextPart):
            content.append({"type": "text", "text": part.text})
        elif isinstance(part, ModelMediaPart):
            content.append(
                _media_content(
                    part,
                    media_root=media_root,
                    max_local_media_bytes=max_local_media_bytes,
                    audio_label=audio_label,
                )
            )
    return {
        **request.parameters,
        "model": model_id,
        "messages": [{"role": "user", "content": content}],
        "stream": False,
    }


def decode_json_object(
    response: httpx.Response,
    *,
    provider_label: str,
    raw_artifact: ArtifactRef,
) -> dict[str, Any]:
    try:
        data = response.json()
    except ValueError as error:
        raise ProviderError(
            f"{provider_label} returned non-JSON content",
            retryable=response.status_code >= 500,
            kind=ProviderErrorKind.MALFORMED_RESPONSE,
            status_code=response.status_code,
            raw_artifact=raw_artifact,
        ) from error
    if not isinstance(data, dict):
        raise ProviderError(
            f"{provider_label} returned a non-object JSON response",
            retryable=False,
            kind=ProviderErrorKind.MALFORMED_RESPONSE,
            status_code=response.status_code,
            raw_artifact=raw_artifact,
        )
    return data


def parse_chat_completion(
    data: dict[str, Any],
    *,
    provider_label: str,
    raw_artifact: ArtifactRef,
) -> ParsedChatCompletion:
    try:
        choice = data["choices"][0]
        message = choice["message"]
    except (KeyError, IndexError, TypeError) as error:
        raise ProviderError(
            f"{provider_label} response is missing a completion answer",
            retryable=False,
            kind=ProviderErrorKind.MALFORMED_RESPONSE,
            raw_artifact=raw_artifact,
        ) from error
    if not isinstance(choice, dict) or not isinstance(message, dict):
        raise ProviderError(
            f"{provider_label} response contains an invalid completion choice",
            retryable=False,
            kind=ProviderErrorKind.MALFORMED_RESPONSE,
            raw_artifact=raw_artifact,
        )
    finish_reason = choice.get("finish_reason")
    if finish_reason in {"length", "max_tokens"}:
        raise ProviderError(
            f"{provider_label} completion was truncated before a usable final answer",
            retryable=True,
            kind=ProviderErrorKind.TRUNCATED_RESPONSE,
            raw_artifact=raw_artifact,
        )
    answer = _completion_text(message)
    if not answer.strip():
        if _reasoning_text(message).strip():
            raise ProviderError(
                f"{provider_label} returned reasoning but no final answer",
                retryable=True,
                kind=ProviderErrorKind.REASONING_ONLY,
                raw_artifact=raw_artifact,
            )
        raise ProviderError(
            f"{provider_label} returned neither a final answer nor reasoning",
            retryable=True,
            kind=ProviderErrorKind.EMPTY_RESPONSE,
            raw_artifact=raw_artifact,
        )
    response_id = data.get("id")
    if not isinstance(response_id, str) or not response_id.strip():
        raise ProviderError(
            f"{provider_label} response is missing an id",
            retryable=False,
            kind=ProviderErrorKind.MALFORMED_RESPONSE,
            raw_artifact=raw_artifact,
        )
    model_version = data.get("model")
    raw_usage = data.get("usage")
    return ParsedChatCompletion(
        response_id=response_id,
        model_version=model_version if isinstance(model_version, str) else None,
        answer=answer,
        finish_reason=finish_reason if isinstance(finish_reason, str) else None,
        token_usage=token_usage(raw_usage),
        raw_usage=raw_usage,
    )


def token_usage(raw_usage: Any) -> TokenUsage | None:
    if not isinstance(raw_usage, dict):
        return None
    input_tokens = raw_usage.get("prompt_tokens", raw_usage.get("input_tokens"))
    output_tokens = raw_usage.get("completion_tokens", raw_usage.get("output_tokens"))
    if not isinstance(input_tokens, int) or not isinstance(output_tokens, int):
        return None
    return TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens)


def error_kind_from_status(status: int) -> ProviderErrorKind:
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


def retry_after(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return max(0, float(value))
    except ValueError:
        return None


def _media_content(
    part: ModelMediaPart,
    *,
    media_root: Path | None,
    max_local_media_bytes: int,
    audio_label: str,
) -> dict[str, Any]:
    media = part.media
    if media.media_type is MediaType.IMAGE:
        url = _data_or_url(
            media.uri,
            media.mime_type,
            media_root=media_root,
            max_local_media_bytes=max_local_media_bytes,
        )
        return {"type": "image_url", "image_url": {"url": url}}
    if media.media_type is MediaType.VIDEO:
        url = _data_or_url(
            media.uri,
            media.mime_type,
            media_root=media_root,
            max_local_media_bytes=max_local_media_bytes,
        )
        return {"type": "video_url", "video_url": {"url": url}}
    if media.uri.startswith(("https://", "http://", "data:")):
        raise ConfigurationError(f"{audio_label} audio input must be a validated local file")
    audio_path = _local_path(
        media.uri,
        media_root=media_root,
        max_local_media_bytes=max_local_media_bytes,
    )
    encoded = base64.b64encode(audio_path.read_bytes()).decode("ascii")
    return {
        "type": "input_audio",
        "input_audio": {
            "data": encoded,
            "format": _audio_format(media.mime_type, audio_path.suffix),
        },
    }


def _data_or_url(
    uri: str,
    mime_type: str,
    *,
    media_root: Path | None,
    max_local_media_bytes: int,
) -> str:
    if uri.startswith(("https://", "http://", "data:")):
        return uri
    path = _local_path(
        uri,
        media_root=media_root,
        max_local_media_bytes=max_local_media_bytes,
    )
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _local_path(
    uri: str,
    *,
    media_root: Path | None,
    max_local_media_bytes: int,
) -> Path:
    path = Path(uri)
    if path.is_absolute():
        raise ConfigurationError(f"absolute local media paths are not allowed: {uri}")
    if media_root is None:
        raise ConfigurationError("media_root is required for local media inputs")
    root = media_root.resolve()
    physical = (root / path).resolve()
    try:
        physical.relative_to(root)
    except ValueError as error:
        raise ConfigurationError(f"local media path escapes media_root: {uri}") from error
    if not physical.is_file():
        raise ConfigurationError(f"local media file does not exist: {uri}")
    if physical.stat().st_size > max_local_media_bytes:
        raise ConfigurationError(
            f"local media file exceeds {max_local_media_bytes} bytes: {uri}"
        )
    return physical


def _completion_text(message: Mapping[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces = [
            item["text"]
            for item in content
            if isinstance(item, dict) and isinstance(item.get("text"), str)
        ]
        if pieces:
            return "".join(pieces)
    refusal = message.get("refusal")
    return refusal if isinstance(refusal, str) else ""


def _reasoning_text(message: Mapping[str, Any]) -> str:
    for field in ("reasoning", "reasoning_content"):
        value = message.get(field)
        if isinstance(value, str):
            return value
    return ""


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
