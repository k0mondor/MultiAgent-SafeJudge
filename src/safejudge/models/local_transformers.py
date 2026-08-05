"""In-process Qwen2.5-Omni provider for local GPU acceptance runs."""

from __future__ import annotations

import asyncio
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from time import monotonic
from typing import Any, Protocol, cast

from pydantic import JsonValue

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


@dataclass(frozen=True, slots=True)
class LocalGeneration:
    answer: str
    input_tokens: int
    output_tokens: int
    finish_reason: str
    metrics: dict[str, JsonValue]


class LocalGenerationRuntime(Protocol):
    def generate(
        self,
        conversation: Sequence[Mapping[str, Any]],
        *,
        max_new_tokens: int,
        temperature: float,
    ) -> LocalGeneration: ...


class QwenOmniTransformersRuntime:
    """Heavy runtime loaded only in the dedicated GPU environment."""

    def __init__(self, *, model_path: Path, device: str) -> None:
        try:
            import torch  # type: ignore[import-not-found]
            from transformers import (  # type: ignore[import-not-found]
                Qwen2_5OmniForConditionalGeneration,
                Qwen2_5OmniProcessor,
            )
        except ImportError as error:
            raise ConfigurationError(
                "local Transformers inference requires torch, transformers, "
                "torchvision, and qwen-omni-utils"
            ) from error

        if device == "cuda" and not torch.cuda.is_available():
            raise ConfigurationError(
                "CUDA was requested, but torch.cuda.is_available() is false"
            )
        resolved_device = "cuda:0" if device == "cuda" else "cpu"
        self._torch = torch
        self._device = resolved_device
        self._processor = Qwen2_5OmniProcessor.from_pretrained(
            model_path,
            local_files_only=True,
        )
        self._model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_path,
            device_map={"": resolved_device},
            low_cpu_mem_usage=True,
            local_files_only=True,
        )

    def generate(
        self,
        conversation: Sequence[Mapping[str, Any]],
        *,
        max_new_tokens: int,
        temperature: float,
    ) -> LocalGeneration:
        try:
            from qwen_omni_utils import process_mm_info  # type: ignore[import-not-found]
        except ImportError as error:
            raise ConfigurationError(
                "qwen-omni-utils is required for multimodal preprocessing"
            ) from error

        torch = self._torch
        prepared_conversation, decoded_video_frames = self._prepare_video_inputs(conversation)
        if self._device == "cuda:0":
            torch.cuda.reset_peak_memory_stats()
        preprocessing_started = monotonic()
        prompt = self._processor.apply_chat_template(
            prepared_conversation,
            add_generation_prompt=True,
            tokenize=False,
        )
        audios, images, videos = process_mm_info(
            prepared_conversation,
            use_audio_in_video=False,
        )
        inputs = self._processor(
            text=prompt,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=False,
        ).to(self._model.device)
        preprocessing_seconds = monotonic() - preprocessing_started

        generation_started = monotonic()
        do_sample = temperature > 0
        generation_parameters: dict[str, Any] = {
            "generation_mode": "text",
            "thinker_max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
        }
        if do_sample:
            generation_parameters["temperature"] = temperature
        with torch.inference_mode():
            generated = self._model.generate(**inputs, **generation_parameters)
        if self._device == "cuda:0":
            torch.cuda.synchronize()
        generation_seconds = monotonic() - generation_started
        input_tokens = int(inputs.input_ids.numel())
        generated_tokens = generated[:, inputs.input_ids.shape[1] :]
        output_tokens = int(generated_tokens.numel())
        answer = self._processor.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        metrics: dict[str, JsonValue] = {
            "device": str(self._model.device),
            "preprocessing_seconds": round(preprocessing_seconds, 3),
            "generation_seconds": round(generation_seconds, 3),
            "tokens_per_second": round(output_tokens / generation_seconds, 3),
            "decoded_video_frames": cast(JsonValue, decoded_video_frames),
            "input_shapes": {
                key: list(value.shape)
                for key, value in inputs.items()
                if hasattr(value, "shape")
            },
        }
        if self._device == "cuda:0":
            metrics["cuda_peak_allocated_mib"] = round(
                torch.cuda.max_memory_allocated() / 1024**2,
                1,
            )
            metrics["cuda_peak_reserved_mib"] = round(
                torch.cuda.max_memory_reserved() / 1024**2,
                1,
            )
        return LocalGeneration(
            answer=answer,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            finish_reason="length" if output_tokens >= max_new_tokens else "stop",
            metrics=metrics,
        )

    @staticmethod
    def _prepare_video_inputs(
        conversation: Sequence[Mapping[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[int]]:
        """Decode video paths with PyAV because torchvision removed read_video()."""

        prepared: list[dict[str, Any]] = []
        decoded_counts: list[int] = []
        for message in conversation:
            content = message.get("content")
            if not isinstance(content, Sequence) or isinstance(content, str):
                prepared.append(dict(message))
                continue
            prepared_content: list[Any] = []
            for raw_item in content:
                if not isinstance(raw_item, Mapping):
                    prepared_content.append(raw_item)
                    continue
                item = dict(raw_item)
                video = item.get("video")
                if item.get("type") == "video" and isinstance(video, str):
                    frames, raw_fps, sample_fps = _decode_video_frames(
                        Path(video),
                        requested_fps=float(item.pop("fps", 1.0)),
                        max_frames=int(item.pop("max_frames", 16)),
                    )
                    item["video"] = frames
                    item["raw_fps"] = raw_fps
                    item["sample_fps"] = sample_fps
                    decoded_counts.append(len(frames))
                prepared_content.append(item)
            prepared.append({**message, "content": prepared_content})
        return prepared, decoded_counts


class LocalTransformersProvider:
    """Runs a local multimodal model in-process and emits provider-neutral responses."""

    def __init__(
        self,
        *,
        model_path: Path,
        media_root: Path,
        capabilities: ModelCapabilities,
        artifact_store: ArtifactStore,
        device: str = "cuda",
        max_new_tokens: int = 64,
        runtime: LocalGenerationRuntime | None = None,
    ) -> None:
        resolved_model = model_path.resolve()
        if not (resolved_model / "config.json").is_file() and runtime is None:
            raise ConfigurationError(f"invalid local model checkpoint: {resolved_model}")
        if device not in {"cuda", "cpu"}:
            raise ConfigurationError("device must be 'cuda' or 'cpu'")
        if max_new_tokens < 1:
            raise ConfigurationError("max_new_tokens must be at least 1")
        self.model_path = resolved_model
        self.media_root = media_root.resolve()
        self._capabilities = capabilities
        self._artifact_store = artifact_store
        self._max_new_tokens = max_new_tokens
        self._runtime = runtime or QwenOmniTransformersRuntime(
            model_path=resolved_model,
            device=device,
        )
        self._lock = asyncio.Lock()

    @property
    def provider_name(self) -> str:
        return "local-transformers"

    @property
    def model_id(self) -> str:
        return self.model_path.name

    @property
    def capabilities(self) -> ModelCapabilities:
        return self._capabilities

    @property
    def price_snapshot(self) -> PriceSnapshot:
        return PriceSnapshot(source="local-inference")

    async def generate(self, request: ModelRequest, *, request_hash: str) -> ModelResponse:
        max_new_tokens, temperature = self._parameters(request.parameters)
        conversation = [
            {
                "role": "user",
                "content": [self._content(part) for part in request.parts],
            }
        ]
        started = monotonic()
        try:
            async with self._lock:
                generated = await asyncio.to_thread(
                    self._runtime.generate,
                    conversation,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )
        except (ConfigurationError, ProviderError):
            raise
        except Exception as error:
            raise ProviderError(
                f"local Transformers generation failed: {error}",
                retryable=False,
                kind=ProviderErrorKind.INVALID_REQUEST,
            ) from error
        if not generated.answer:
            raise ProviderError(
                "local Transformers model returned an empty answer",
                retryable=False,
                kind=ProviderErrorKind.MALFORMED_RESPONSE,
            )
        usage = TokenUsage(
            input_tokens=generated.input_tokens,
            output_tokens=generated.output_tokens,
        )
        payload = {
            "answer": generated.answer,
            "finish_reason": generated.finish_reason,
            "metrics": generated.metrics,
            "model": self.model_id,
            "provider": self.provider_name,
            "request_id": request.request_id,
            "usage": {
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
            },
        }
        try:
            raw_artifact = self._artifact_store.write_bytes(
                category="local-transformers-responses",
                content=json.dumps(
                    payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8"),
                content_type="application/json",
            )
        except ArtifactError as error:
            raise ProviderError(
                str(error),
                retryable=False,
                kind=ProviderErrorKind.ARTIFACT_PERSISTENCE,
            ) from error
        return ModelResponse(
            response_id=f"local_transformers_{request_hash[:24]}",
            request_hash=request_hash,
            role=request.role,
            provider=self.provider_name,
            model=self.model_id,
            model_version="bitsandbytes-nf4",
            answer=generated.answer,
            finish_reason=generated.finish_reason,
            token_usage=usage,
            latency_ms=max(0, round((monotonic() - started) * 1000)),
            cost=ModelCost(
                price_snapshot=self.price_snapshot,
                estimated_usd=Decimal("0"),
                actual_usd=Decimal("0"),
            ),
            raw_artifact=raw_artifact,
        )

    def _content(self, part: ModelTextPart | ModelMediaPart) -> dict[str, Any]:
        if isinstance(part, ModelTextPart):
            return {"type": "text", "text": part.text}
        physical = self._local_path(part.media.uri)
        media_type = part.media.media_type.value
        content: dict[str, Any] = {"type": media_type, media_type: str(physical)}
        if media_type == "image":
            content["max_pixels"] = 512 * 28 * 28
        elif media_type == "video":
            content.update({"fps": 1.0, "max_frames": 16, "max_pixels": 256 * 28 * 28})
        return content

    def _local_path(self, uri: str) -> Path:
        path = Path(uri)
        if path.is_absolute():
            raise ConfigurationError(f"absolute local media paths are not allowed: {uri}")
        physical = (self.media_root / path).resolve()
        try:
            physical.relative_to(self.media_root)
        except ValueError as error:
            raise ConfigurationError(f"local media path escapes media_root: {uri}") from error
        if not physical.is_file():
            raise ConfigurationError(f"local media file does not exist: {uri}")
        return physical

    def _parameters(self, parameters: Mapping[str, JsonValue]) -> tuple[int, float]:
        max_new_tokens = parameters.get("max_new_tokens", self._max_new_tokens)
        temperature = parameters.get("temperature", 0)
        if not isinstance(max_new_tokens, int) or isinstance(max_new_tokens, bool):
            raise ConfigurationError("max_new_tokens must be an integer")
        if max_new_tokens < 1:
            raise ConfigurationError("max_new_tokens must be at least 1")
        if not isinstance(temperature, int | float) or isinstance(temperature, bool):
            raise ConfigurationError("temperature must be numeric")
        if temperature < 0:
            raise ConfigurationError("temperature must be non-negative")
        return max_new_tokens, float(temperature)


def _decode_video_frames(
    path: Path,
    *,
    requested_fps: float,
    max_frames: int,
) -> tuple[list[Any], float, float]:
    try:
        import av  # type: ignore[import-not-found]
    except ImportError as error:
        raise ConfigurationError("PyAV is required for local video preprocessing") from error
    if requested_fps <= 0:
        raise ConfigurationError("video fps must be positive")
    try:
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            raw_fps = float(stream.average_rate or stream.base_rate or requested_fps)
            decoded = [frame.to_image().convert("RGB") for frame in container.decode(stream)]
    except Exception as error:
        raise ConfigurationError(f"cannot decode local video {path}: {error}") from error
    if not decoded:
        raise ConfigurationError(f"local video contains no decodable frames: {path}")
    duration_seconds = len(decoded) / raw_fps if raw_fps > 0 else 0
    desired = max(2, math.ceil(duration_seconds * requested_fps))
    desired = min(max_frames, desired, len(decoded))
    if desired == 1:
        indices = [0]
    else:
        indices = [round(index * (len(decoded) - 1) / (desired - 1)) for index in range(desired)]
    return [decoded[index] for index in indices], raw_fps, requested_fps
