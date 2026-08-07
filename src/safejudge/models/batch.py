"""Batch bridge from canonical benchmark JSONL to immutable target responses."""

from __future__ import annotations

import asyncio
import hashlib
import os
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import Field, ValidationError

from safejudge.contracts.base import ContractModel
from safejudge.contracts.dataset import (
    CanonicalMultimodalSample,
    MediaPart,
    Sha256,
)
from safejudge.contracts.evaluation import TargetResponse
from safejudge.contracts.model import (
    InvocationContext,
    ModelRole,
)
from safejudge.core.errors import ConfigurationError, ContractValidationError
from safejudge.core.time import utc_now
from safejudge.models.base import ModelProvider
from safejudge.models.cache import SQLiteModelStore
from safejudge.models.invocation import InvocationPolicy, ModelInvoker
from safejudge.models.profiles import ModelProfile
from safejudge.models.target import TargetRunner


class BatchFileDigest(ContractModel):
    name: str
    sha256: Sha256


class TargetBatchManifest(ContractModel):
    manifest_version: Literal["1.1"] = "1.1"
    target_response_schema_version: Literal["1.1"] = "1.1"
    input: BatchFileDigest
    output: BatchFileDigest
    provider: str
    model: str
    experiment_id: str
    run_id: str
    input_sample_count: int = Field(default=0, ge=0)
    sample_count: int = Field(ge=0)
    failure_count: int = Field(default=0, ge=0)
    failures: BatchFileDigest | None = None
    verified_media_count: int = Field(ge=0)
    modality_counts: dict[str, int]
    cache_hit_count: int = Field(ge=0)
    cache_miss_count: int = Field(ge=0)
    logical_call_count: int = Field(default=0, ge=0)
    provider_attempt_count: int = Field(default=0, ge=0)
    successful_call_count: int = Field(default=0, ge=0)
    failed_call_count: int = Field(default=0, ge=0)
    billed_cost_usd: Decimal = Field(ge=0)
    created_at: datetime = Field(default_factory=utc_now)


@dataclass(frozen=True, slots=True)
class TargetBatchResult:
    output_path: Path
    manifest_path: Path
    store_path: Path
    manifest: TargetBatchManifest
    failure_path: Path | None = None


class TargetBatchFailure(ContractModel):
    schema_version: Literal["1.0"] = "1.0"
    sample_id: str
    error_type: str
    error_message: str


async def run_target_batch(
    *,
    input_path: Path,
    media_root: Path,
    output_path: Path,
    store_path: Path,
    context: InvocationContext,
    provider: ModelProvider,
    profile: ModelProfile,
    max_concurrency: int = 4,
    max_local_media_bytes: int = 100 * 1024 * 1024,
    limit: int | None = None,
    overwrite: bool = False,
) -> TargetBatchResult:
    """Validate canonical input and persist normalized responses from any provider."""

    input_path = input_path.resolve()
    media_root = media_root.resolve()
    output_path = output_path.resolve()
    store_path = store_path.resolve()
    manifest_path = output_path.with_suffix(f"{output_path.suffix}.manifest.json")
    failure_path = output_path.with_name(f"{output_path.stem}.failures{output_path.suffix}")

    if not input_path.is_file():
        raise ConfigurationError(f"canonical input JSONL does not exist: {input_path}")
    if not media_root.is_dir():
        raise ConfigurationError(f"media root does not exist: {media_root}")
    if max_concurrency < 1:
        raise ConfigurationError("max concurrency must be at least 1")
    if max_local_media_bytes < 1:
        raise ConfigurationError("max local media bytes must be at least 1")
    if limit is not None and limit < 1:
        raise ConfigurationError("limit must be at least 1")
    if not overwrite and (
        output_path.exists() or manifest_path.exists() or failure_path.exists()
    ):
        raise ConfigurationError(
            f"output or manifest already exists: {output_path}; pass --overwrite to replace it"
        )

    samples = _limited_samples(_read_canonical_jsonl(input_path), limit=limit)
    modality_counts: Counter[str] = Counter()
    verified_media_count = 0
    for sample in samples:
        verified_media_count += _verify_sample_media(
            sample,
            media_root=media_root,
            max_local_media_bytes=max_local_media_bytes,
            modality_counts=modality_counts,
        )

    store = SQLiteModelStore(store_path)
    runner = TargetRunner(
        ModelInvoker(
            provider=provider,
            store=store,
            policy=InvocationPolicy(max_concurrency=max_concurrency),
        ),
        profile=profile,
    )
    batch_outputs = await asyncio.gather(
        *(
            runner.run(sample, context=context)
            for sample in samples
        ),
        return_exceptions=True,
    )
    responses: list[TargetResponse] = []
    failures: list[TargetBatchFailure] = []
    for sample, output in zip(samples, batch_outputs, strict=True):
        if isinstance(output, BaseException):
            failures.append(
                TargetBatchFailure(
                    sample_id=sample.sample_id,
                    error_type=type(output).__name__,
                    error_message=str(output),
                )
            )
        else:
            responses.append(output)
    return _persist_batch(
        input_path=input_path,
        output_path=output_path,
        manifest_path=manifest_path,
        failure_path=failure_path,
        store_path=store_path,
        context=context,
        provider=provider,
        responses=responses,
        failures=failures,
        input_sample_count=len(samples),
        verified_media_count=verified_media_count,
        modality_counts=modality_counts,
    )


def _limited_samples(
    samples: tuple[CanonicalMultimodalSample, ...],
    *,
    limit: int | None,
) -> tuple[CanonicalMultimodalSample, ...]:
    if limit is not None and limit < 1:
        raise ConfigurationError("limit must be at least 1")
    return samples if limit is None else samples[:limit]


def _read_canonical_jsonl(path: Path) -> tuple[CanonicalMultimodalSample, ...]:
    samples: list[CanonicalMultimodalSample] = []
    sample_ids: set[str] = set()
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    sample = CanonicalMultimodalSample.model_validate_json(line)
                except ValidationError as error:
                    raise ContractValidationError(
                        f"invalid canonical sample at {path.name}:{line_number}: {error}"
                    ) from error
                if sample.sample_id in sample_ids:
                    raise ContractValidationError(
                        f"duplicate canonical sample_id at {path.name}:{line_number}: "
                        f"{sample.sample_id}"
                    )
                sample_ids.add(sample.sample_id)
                samples.append(sample)
    except OSError as error:
        raise ContractValidationError(f"cannot read canonical JSONL {path}: {error}") from error
    if not samples:
        raise ContractValidationError(f"canonical input JSONL is empty: {path}")
    return tuple(samples)


def _verify_sample_media(
    sample: CanonicalMultimodalSample,
    *,
    media_root: Path,
    max_local_media_bytes: int,
    modality_counts: Counter[str],
) -> int:
    count = 0
    for part in sample.parts:
        if not isinstance(part, MediaPart):
            continue
        media = part.media
        normalized = media.uri.strip().replace("\\", "/")
        relative = PurePosixPath(normalized)
        if not normalized or relative.is_absolute() or ".." in relative.parts:
            raise ContractValidationError(
                f"sample {sample.sample_id!r} has unsafe media path: {media.uri!r}"
            )
        physical = (media_root / Path(*relative.parts)).resolve()
        try:
            physical.relative_to(media_root)
        except ValueError as error:
            raise ContractValidationError(
                f"sample {sample.sample_id!r} media escapes media root: {media.uri!r}"
            ) from error
        if not physical.is_file():
            raise ContractValidationError(
                f"sample {sample.sample_id!r} media file does not exist: {media.uri}"
            )
        size_bytes = physical.stat().st_size
        if size_bytes > max_local_media_bytes:
            raise ContractValidationError(
                f"sample {sample.sample_id!r} media exceeds {max_local_media_bytes} bytes: "
                f"{media.uri}"
            )
        if media.size_bytes is None or media.size_bytes != size_bytes:
            raise ContractValidationError(
                f"sample {sample.sample_id!r} media size mismatch: {media.uri}"
            )
        if media.sha256 is None or media.sha256.lower() != _file_sha256(physical):
            raise ContractValidationError(
                f"sample {sample.sample_id!r} media SHA-256 mismatch: {media.uri}"
            )
        modality_counts[media.media_type.value] += 1
        count += 1
    return count


def _persist_batch(
    *,
    input_path: Path,
    output_path: Path,
    manifest_path: Path,
    failure_path: Path,
    store_path: Path,
    context: InvocationContext,
    provider: ModelProvider,
    responses: list[TargetResponse],
    failures: list[TargetBatchFailure],
    input_sample_count: int,
    verified_media_count: int,
    modality_counts: Counter[str],
) -> TargetBatchResult:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output: Path | None = None
    temporary_manifest: Path | None = None
    temporary_failures: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            dir=output_path.parent,
            delete=False,
        ) as stream:
            temporary_output = Path(stream.name)
            for response in responses:
                stream.write(response.model_dump_json())
                stream.write("\n")

        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{failure_path.name}.",
            suffix=".tmp",
            dir=output_path.parent,
            delete=False,
        ) as stream:
            temporary_failures = Path(stream.name)
            for failure in failures:
                stream.write(failure.model_dump_json())
                stream.write("\n")

        run_stats = SQLiteModelStore(store_path).run_stats(
            context=context,
            role=ModelRole.TARGET,
        )

        manifest = TargetBatchManifest(
            input=BatchFileDigest(name=input_path.name, sha256=_file_sha256(input_path)),
            output=BatchFileDigest(
                name=output_path.name,
                sha256=_file_sha256(temporary_output),
            ),
            provider=provider.provider_name,
            model=provider.model_id,
            experiment_id=context.experiment_id,
            run_id=context.run_id,
            input_sample_count=input_sample_count,
            sample_count=len(responses),
            failure_count=len(failures),
            failures=BatchFileDigest(
                name=failure_path.name,
                sha256=_file_sha256(temporary_failures),
            ),
            verified_media_count=verified_media_count,
            modality_counts=dict(sorted(modality_counts.items())),
            cache_hit_count=run_stats.cache_hit_count,
            cache_miss_count=run_stats.cache_miss_count,
            logical_call_count=run_stats.logical_call_count,
            provider_attempt_count=run_stats.provider_attempt_count,
            successful_call_count=run_stats.successful_call_count,
            failed_call_count=run_stats.failed_call_count,
            billed_cost_usd=run_stats.billed_cost_usd,
        )
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{manifest_path.name}.",
            suffix=".tmp",
            dir=manifest_path.parent,
            delete=False,
        ) as stream:
            temporary_manifest = Path(stream.name)
            stream.write(manifest.model_dump_json(indent=2))
            stream.write("\n")

        os.replace(temporary_output, output_path)
        temporary_output = None
        os.replace(temporary_failures, failure_path)
        temporary_failures = None
        os.replace(temporary_manifest, manifest_path)
        temporary_manifest = None
        return TargetBatchResult(
            output_path=output_path,
            manifest_path=manifest_path,
            store_path=store_path,
            manifest=manifest,
            failure_path=failure_path,
        )
    finally:
        for temporary_path in (
            temporary_output,
            temporary_failures,
            temporary_manifest,
        ):
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
