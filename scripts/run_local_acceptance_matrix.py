"""Run one small sample from every downloaded dataset through local Qwen."""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass
from pathlib import Path

from safejudge.contracts.dataset import CanonicalMultimodalSample, MediaPart
from safejudge.contracts.evaluation import TargetResponse
from safejudge.contracts.model import (
    InputModality,
    InvocationContext,
    ModalityCombination,
    ModelCapabilities,
)
from safejudge.models.artifacts import FileArtifactStore
from safejudge.models.batch import run_target_batch
from safejudge.models.local_transformers import LocalTransformersProvider


@dataclass(frozen=True, slots=True)
class AcceptanceGroup:
    name: str
    input_path: Path
    media_prefix: Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(r"C:\models\Qwen2.5-Omni-7B-NF4-Text"),
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/local-qwen-acceptance"),
    )
    parser.add_argument("--samples-per-group", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.samples_per_group < 1:
        raise SystemExit("samples-per-group must be at least 1")
    project_root = args.project_root.resolve()
    output_dir = (project_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_input = output_dir / "canonical-matrix.jsonl"
    response_path = output_dir / "target-responses.jsonl"
    store_path = output_dir / "model-calls.sqlite3"
    artifact_root = output_dir / "artifacts"

    samples, group_by_sample = _combined_samples(
        _groups(project_root),
        samples_per_group=args.samples_per_group,
    )
    combined_input.write_text(
        "".join(f"{sample.model_dump_json()}\n" for sample in samples),
        encoding="utf-8",
    )
    capabilities = ModelCapabilities(
        input_combinations=tuple(
            ModalityCombination(modalities=frozenset({InputModality.TEXT, modality}))
            for modality in (
                InputModality.IMAGE,
                InputModality.AUDIO,
                InputModality.VIDEO,
            )
        )
    )
    provider = LocalTransformersProvider(
        model_path=args.model,
        media_root=project_root,
        capabilities=capabilities,
        artifact_store=FileArtifactStore(artifact_root),
        device=args.device,
        max_new_tokens=args.max_new_tokens,
    )
    result = asyncio.run(
        run_target_batch(
            input_path=combined_input,
            media_root=project_root,
            output_path=response_path,
            store_path=store_path,
            context=InvocationContext(
                experiment_id="local-qwen-acceptance",
                run_id="all-downloaded-datasets",
            ),
            provider=provider,
            max_concurrency=1,
            parameters={
                "temperature": 0,
                "max_new_tokens": args.max_new_tokens,
            },
            overwrite=args.overwrite,
        )
    )
    responses = [
        TargetResponse.model_validate_json(line)
        for line in response_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    summary = {
        "output": str(result.output_path),
        "manifest": str(result.manifest_path),
        "store": str(result.store_path),
        "samples": result.manifest.sample_count,
        "verified_media": result.manifest.verified_media_count,
        "modalities": result.manifest.modality_counts,
        "cache_hits": result.manifest.cache_hit_count,
        "answers": [
            {
                "group": group_by_sample[response.sample_id],
                "sample_id": response.sample_id,
                "schema_version": response.schema_version,
                "provider": response.model.provider,
                "text": response.text,
                "input_tokens": (
                    response.token_usage.input_tokens if response.token_usage else None
                ),
                "output_tokens": (
                    response.token_usage.output_tokens if response.token_usage else None
                ),
                "latency_ms": response.latency_ms,
                "raw_artifact_uri": response.raw_artifact_uri,
            }
            for response in responses
        ],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _groups(project_root: Path) -> tuple[AcceptanceGroup, ...]:
    acceptance = project_root / "data" / "acceptance"
    return (
        AcceptanceGroup(
            name="mm-safetybench",
            input_path=acceptance / "mm-safetybench.jsonl",
            media_prefix=Path("vendor/MM-SafetyBench/data/imgs"),
        ),
        AcceptanceGroup(
            name="mossbench",
            input_path=acceptance / "mossbench.jsonl",
            media_prefix=Path("vendor/MOSSBench"),
        ),
        AcceptanceGroup(
            name="omni-image-text-typo",
            input_path=acceptance / "omni-image-text-typo.jsonl",
            media_prefix=Path("vendor/Omni-SafetyBench"),
        ),
        AcceptanceGroup(
            name="omni-audio-text-tts",
            input_path=acceptance / "omni-audio-text-tts.jsonl",
            media_prefix=Path("vendor/Omni-SafetyBench"),
        ),
        AcceptanceGroup(
            name="omni-video-text-typo",
            input_path=acceptance / "omni-video-text-typo.jsonl",
            media_prefix=Path("vendor/Omni-SafetyBench"),
        ),
    )


def _combined_samples(
    groups: tuple[AcceptanceGroup, ...],
    *,
    samples_per_group: int,
) -> tuple[tuple[CanonicalMultimodalSample, ...], dict[str, str]]:
    combined: list[CanonicalMultimodalSample] = []
    group_by_sample: dict[str, str] = {}
    for group in groups:
        if not group.input_path.is_file():
            raise SystemExit(f"missing acceptance input: {group.input_path}")
        loaded = [
            CanonicalMultimodalSample.model_validate_json(line)
            for line in group.input_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(loaded) < samples_per_group:
            raise SystemExit(
                f"{group.name} contains only {len(loaded)} samples, "
                f"requested {samples_per_group}"
            )
        for sample in loaded[:samples_per_group]:
            parts = tuple(
                part.model_copy(
                    update={
                        "media": part.media.model_copy(
                            update={"uri": (group.media_prefix / part.media.uri).as_posix()}
                        )
                    }
                )
                if isinstance(part, MediaPart)
                else part
                for part in sample.parts
            )
            combined.append(sample.model_copy(update={"parts": parts}))
            group_by_sample[sample.sample_id] = group.name
    return tuple(combined), group_by_sample


if __name__ == "__main__":
    raise SystemExit(main())
