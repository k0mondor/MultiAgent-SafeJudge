"""Command-line entry points for data conversion and target-model runs."""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from safejudge import __version__
from safejudge.contracts.dataset import DatasetSplit
from safejudge.contracts.model import (
    InputModality,
    InvocationContext,
    ModalityCombination,
    ModelCapabilities,
)
from safejudge.core.config import Settings
from safejudge.core.errors import SafeJudgeError
from safejudge.datasets.base import AdapterContext
from safejudge.datasets.conversion import convert_dataset
from safejudge.datasets.registry import list_adapters
from safejudge.models.artifacts import FileArtifactStore
from safejudge.models.batch import run_fake_target_batch, run_target_batch
from safejudge.models.local_openai import (
    JudgeOpenAISettings,
    LocalOpenAIProvider,
    LocalOpenAISettings,
)
from safejudge.models.openrouter import OpenRouterProvider, OpenRouterSettings
from safejudge.workflows.batch import run_evaluation_batch, run_fake_evaluation_batch

_TARGET_CAPABILITIES = ("text", "text+image", "text+audio", "text+video")
_DEFAULT_MAX_LOCAL_MEDIA_BYTES = 100 * 1024 * 1024


def _doctor(settings: Settings) -> int:
    report = {
        "application": "multiagent-safejudge",
        "version": __version__,
        "environment": settings.environment.value,
        "log_level": settings.log_level,
        "data_dir": str(settings.data_dir),
        "artifact_dir": str(settings.artifact_dir),
        "max_concurrency": settings.max_concurrency,
        "network_used": False,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="safejudge")
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("doctor", help="validate local configuration without using network")

    data_parser = subparsers.add_parser("data", help="inspect and convert benchmark data")
    data_commands = data_parser.add_subparsers(dest="data_command", required=True)
    data_commands.add_parser("adapters", help="list built-in dataset adapters")

    convert_parser = data_commands.add_parser("convert", help="convert metadata to canonical JSONL")
    convert_parser.add_argument("--adapter", required=True, choices=list_adapters())
    convert_parser.add_argument("--input", required=True, type=Path)
    convert_parser.add_argument("--media-root", required=True, type=Path)
    convert_parser.add_argument("--output", required=True, type=Path)
    convert_parser.add_argument("--dataset-version", default="main")
    convert_parser.add_argument(
        "--split",
        choices=[split.value for split in DatasetSplit],
        default=DatasetSplit.TEST.value,
    )
    convert_parser.add_argument("--subset")
    convert_parser.add_argument(
        "--variant",
        action="append",
        default=[],
        help="adapter-specific variant; repeat the flag to select multiple",
    )
    convert_parser.add_argument("--skip-media-verification", action="store_true")
    convert_parser.add_argument("--hash-media", action="store_true")
    convert_parser.add_argument(
        "--limit",
        type=int,
        help="convert only the first N canonical samples for a small acceptance run",
    )
    convert_parser.add_argument("--overwrite", action="store_true")

    target_parser = subparsers.add_parser("target", help="run target-model acceptance jobs")
    target_commands = target_parser.add_subparsers(dest="target_command", required=True)
    target_run_parser = target_commands.add_parser(
        "run-jsonl",
        help="run canonical JSONL through a fake, local, or OpenRouter target",
    )
    target_run_parser.add_argument("--input", required=True, type=Path)
    target_run_parser.add_argument("--media-root", required=True, type=Path)
    target_run_parser.add_argument("--output", required=True, type=Path)
    target_run_parser.add_argument("--store", required=True, type=Path)
    target_run_parser.add_argument("--artifact-root", required=True, type=Path)
    target_run_parser.add_argument(
        "--provider",
        choices=("fake", "local", "openrouter"),
        default="fake",
    )
    target_run_parser.add_argument(
        "--capability",
        action="append",
        choices=_TARGET_CAPABILITIES,
        help="exact input combination supported by the selected model; repeat as needed",
    )
    target_run_parser.add_argument(
        "--experiment-id",
        default="real-data-acceptance",
    )
    target_run_parser.add_argument("--run-id", required=True)
    target_run_parser.add_argument("--max-concurrency", type=int, default=4)
    target_run_parser.add_argument(
        "--max-local-media-bytes",
        type=int,
        default=None,
    )
    target_run_parser.add_argument(
        "--limit",
        type=int,
        help="run only the first N validated samples for a low-cost pilot",
    )
    target_run_parser.add_argument("--overwrite", action="store_true")

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="judge frozen target responses with the recoverable M3 graph",
    )
    evaluate_commands = evaluate_parser.add_subparsers(
        dest="evaluate_command",
        required=True,
    )
    evaluate_run_parser = evaluate_commands.add_parser(
        "run-jsonl",
        help="run canonical samples and frozen TargetResponse JSONL through M3 judges",
    )
    evaluate_run_parser.add_argument("--input", required=True, type=Path)
    evaluate_run_parser.add_argument("--target-responses", required=True, type=Path)
    evaluate_run_parser.add_argument("--output", required=True, type=Path)
    evaluate_run_parser.add_argument("--store", required=True, type=Path)
    evaluate_run_parser.add_argument("--checkpoint", required=True, type=Path)
    evaluate_run_parser.add_argument("--node-ledger", required=True, type=Path)
    evaluate_run_parser.add_argument("--artifact-root", required=True, type=Path)
    evaluate_run_parser.add_argument(
        "--provider",
        choices=("fake", "local"),
        default="fake",
    )
    evaluate_run_parser.add_argument("--experiment-id", default="m3-evaluation")
    evaluate_run_parser.add_argument("--run-id", required=True)
    evaluate_run_parser.add_argument("--max-concurrency", type=int, default=3)
    evaluate_run_parser.add_argument("--max-sample-concurrency", type=int, default=1)
    evaluate_run_parser.add_argument("--max-retries", type=int, default=1)
    evaluate_run_parser.add_argument(
        "--judge-env-file",
        type=Path,
        default=Path(".env"),
        help="dotenv file containing JUDGE_MODEL_* settings for the local provider",
    )
    evaluate_run_parser.add_argument("--limit", type=int)
    evaluate_run_parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "doctor":
        return _doctor(Settings())
    if args.command == "data" and args.data_command == "adapters":
        print(json.dumps({"adapters": list_adapters()}, indent=2))
        return 0
    if args.command == "data" and args.data_command == "convert":
        try:
            conversion_result = convert_dataset(
                adapter_name=args.adapter,
                source_file=args.input,
                output_path=args.output,
                context=AdapterContext(
                    media_root=args.media_root,
                    dataset_version=args.dataset_version,
                    split=DatasetSplit(args.split),
                    subset=args.subset,
                    verify_media=not args.skip_media_verification,
                    hash_media=args.hash_media,
                    variants=tuple(args.variant),
                ),
                overwrite=args.overwrite,
                limit=args.limit,
            )
        except SafeJudgeError as error:
            build_parser().error(str(error))
        print(
            json.dumps(
                {
                    "output": str(conversion_result.output_path),
                    "manifest": str(conversion_result.manifest_path),
                    "samples": conversion_result.manifest.sample_count,
                    "modalities": conversion_result.manifest.modality_counts,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.command == "target" and args.target_command == "run-jsonl":
        try:
            context = InvocationContext(
                experiment_id=args.experiment_id,
                run_id=args.run_id,
            )
            if args.provider == "fake":
                target_result = asyncio.run(
                    run_fake_target_batch(
                        input_path=args.input,
                        media_root=args.media_root,
                        output_path=args.output,
                        store_path=args.store,
                        artifact_root=args.artifact_root,
                        context=context,
                        max_concurrency=args.max_concurrency,
                        max_local_media_bytes=(
                            args.max_local_media_bytes or _DEFAULT_MAX_LOCAL_MEDIA_BYTES
                        ),
                        limit=args.limit,
                        overwrite=args.overwrite,
                    )
                )
            elif args.provider == "local":
                local_settings = (
                    LocalOpenAISettings()
                    if args.max_local_media_bytes is None
                    else LocalOpenAISettings(
                        max_local_media_bytes=args.max_local_media_bytes,
                    )
                )
                local_provider = LocalOpenAIProvider(
                    settings=local_settings,
                    capabilities=_target_capabilities(args.capability),
                    artifact_store=FileArtifactStore(args.artifact_root.resolve()),
                    media_root=args.media_root,
                )
                target_result = asyncio.run(
                    run_target_batch(
                        input_path=args.input,
                        media_root=args.media_root,
                        output_path=args.output,
                        store_path=args.store,
                        context=context,
                        provider=local_provider,
                        max_concurrency=args.max_concurrency,
                        max_local_media_bytes=local_settings.max_local_media_bytes,
                        limit=args.limit,
                        overwrite=args.overwrite,
                    )
                )
            else:
                openrouter_settings = (
                    OpenRouterSettings()
                    if args.max_local_media_bytes is None
                    else OpenRouterSettings(
                        max_local_media_bytes=args.max_local_media_bytes,
                    )
                )
                openrouter_provider = OpenRouterProvider(
                    settings=openrouter_settings,
                    capabilities=_target_capabilities(args.capability),
                    artifact_store=FileArtifactStore(args.artifact_root.resolve()),
                    media_root=args.media_root,
                )
                target_result = asyncio.run(
                    run_target_batch(
                        input_path=args.input,
                        media_root=args.media_root,
                        output_path=args.output,
                        store_path=args.store,
                        context=context,
                        provider=openrouter_provider,
                        max_concurrency=args.max_concurrency,
                        max_local_media_bytes=openrouter_settings.max_local_media_bytes,
                        limit=args.limit,
                        overwrite=args.overwrite,
                    )
                )
        except (SafeJudgeError, ValidationError) as error:
            build_parser().error(str(error))
        print(
            json.dumps(
                {
                    "output": str(target_result.output_path),
                    "manifest": str(target_result.manifest_path),
                    "store": str(target_result.store_path),
                    "provider": target_result.manifest.provider,
                    "model": target_result.manifest.model,
                    "samples": target_result.manifest.sample_count,
                    "verified_media": target_result.manifest.verified_media_count,
                    "cache_hits": target_result.manifest.cache_hit_count,
                    "cache_misses": target_result.manifest.cache_miss_count,
                    "billed_cost_usd": str(target_result.manifest.billed_cost_usd),
                    "modalities": target_result.manifest.modality_counts,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.command == "evaluate" and args.evaluate_command == "run-jsonl":
        try:
            context = InvocationContext(
                experiment_id=args.experiment_id,
                run_id=args.run_id,
            )
            common = {
                "samples_path": args.input,
                "target_responses_path": args.target_responses,
                "output_path": args.output,
                "store_path": args.store,
                "checkpoint_path": args.checkpoint,
                "node_ledger_path": args.node_ledger,
                "context": context,
                "max_concurrency": args.max_concurrency,
                "max_sample_concurrency": args.max_sample_concurrency,
                "limit": args.limit,
                "overwrite": args.overwrite,
            }
            if args.provider == "fake":
                evaluation_result = asyncio.run(
                    run_fake_evaluation_batch(
                        **common,
                        artifact_root=args.artifact_root,
                    )
                )
            else:
                local_provider = LocalOpenAIProvider(
                    settings=JudgeOpenAISettings(_env_file=args.judge_env_file),
                    capabilities=_judge_capabilities(),
                    artifact_store=FileArtifactStore(args.artifact_root.resolve()),
                )
                evaluation_result = asyncio.run(
                    run_evaluation_batch(
                        **common,
                        provider=local_provider,
                        max_retries=args.max_retries,
                    )
                )
        except (SafeJudgeError, ValidationError) as error:
            build_parser().error(str(error))
        print(
            json.dumps(
                {
                    "output": str(evaluation_result.output_path),
                    "manifest": str(evaluation_result.manifest_path),
                    "store": str(evaluation_result.store_path),
                    "checkpoint": str(evaluation_result.checkpoint_path),
                    "node_ledger": str(evaluation_result.node_ledger_path),
                    "provider": evaluation_result.manifest.judge_provider,
                    "model": evaluation_result.manifest.judge_model,
                    "samples": evaluation_result.manifest.sample_count,
                    "arbitrations": evaluation_result.manifest.arbitration_count,
                    "cache_hits": evaluation_result.manifest.cache_hit_count,
                    "cache_misses": evaluation_result.manifest.cache_miss_count,
                    "billed_cost_usd": str(evaluation_result.manifest.billed_cost_usd),
                    "levels": evaluation_result.manifest.compliance_level_counts,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


def _target_capabilities(values: list[str] | None) -> ModelCapabilities:
    selected = values or ["text"]
    combinations = tuple(
        ModalityCombination(modalities=frozenset(InputModality(item) for item in value.split("+")))
        for value in selected
    )
    return ModelCapabilities(input_combinations=combinations)


def _judge_capabilities() -> ModelCapabilities:
    return ModelCapabilities(
        input_combinations=(ModalityCombination(modalities=frozenset({InputModality.TEXT})),)
    )


if __name__ == "__main__":
    raise SystemExit(main())
