"""Command-line entry points for data conversion and target-model runs."""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from pydantic import ValidationError

from safejudge import __version__
from safejudge.constitution import ConstitutionRegistry
from safejudge.contracts.dataset import DatasetSplit, MediaType
from safejudge.contracts.jury import (
    JuryDefinition,
    JuryPlan,
    JurySeat,
)
from safejudge.contracts.model import (
    InputModality,
    InvocationContext,
    ModalityCombination,
    ModelCapabilities,
    ModelRole,
)
from safejudge.core.config import Settings
from safejudge.core.errors import ConfigurationError, SafeJudgeError
from safejudge.datasets.base import AdapterContext
from safejudge.datasets.conversion import convert_dataset
from safejudge.datasets.registry import list_adapters
from safejudge.grounding.contracts import GroundingMode, ObservationModality
from safejudge.grounding.pipeline import GroundingPipeline, GroundingTool
from safejudge.grounding.tools import (
    ModelGroundingTool,
    SidecarGroundingTool,
    load_sidecar_observations,
)
from safejudge.models.artifacts import FileArtifactStore
from safejudge.models.base import ModelProvider
from safejudge.models.batch import run_target_batch
from safejudge.models.cache import SQLiteModelStore
from safejudge.models.invocation import InvocationPolicy, ModelInvoker
from safejudge.models.local_openai import (
    JudgeOpenAISettings,
    LocalOpenAIProvider,
    LocalOpenAISettings,
)
from safejudge.models.openrouter import OpenRouterProvider, OpenRouterSettings
from safejudge.models.profiles import ModelProfile, ModelRegistry
from safejudge.taxonomy import TaxonomyRegistry
from safejudge.workflows.batch import run_evaluation_batch

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
        help="run canonical JSONL through a registered real target model",
    )
    target_run_parser.add_argument("--input", required=True, type=Path)
    target_run_parser.add_argument("--media-root", required=True, type=Path)
    target_run_parser.add_argument("--output", required=True, type=Path)
    target_run_parser.add_argument("--store", required=True, type=Path)
    target_run_parser.add_argument("--artifact-root", required=True, type=Path)
    _add_profile_options(target_run_parser)
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
        "--media-root",
        type=Path,
        default=Path("."),
        help="trusted root used to resolve local media for blind grounding",
    )
    evaluate_run_parser.add_argument(
        "--jury-plan",
        required=True,
        type=Path,
        help="TOML plan assigning an explicit model profile to every Jury seat",
    )
    evaluate_run_parser.add_argument(
        "--model-registry",
        type=Path,
        default=Path("config/models.toml"),
    )
    evaluate_run_parser.add_argument("--allow-unqualified-model", action="store_true")
    evaluate_run_parser.add_argument(
        "--taxonomy",
        help=(
            "taxonomy ID to enable multi-label Category Router "
            "(for example gb-t-45654-2025-safejudge-v1)"
        ),
    )
    evaluate_run_parser.add_argument("--taxonomy-version")
    evaluate_run_parser.add_argument(
        "--taxonomy-registry",
        type=Path,
        default=Path("config/taxonomies"),
    )
    evaluate_run_parser.add_argument(
        "--constitution-registry",
        type=Path,
        default=Path("config/constitutions"),
    )
    evaluate_run_parser.add_argument(
        "--grounding-sidecar",
        type=Path,
        help="JSONL RawGroundingObservation records keyed by media_sha256",
    )
    evaluate_run_parser.add_argument(
        "--grounding-model-profile",
        help="multimodal grounding profile used to create blind OCR/VLM observations",
    )
    evaluate_run_parser.add_argument(
        "--grounding-store",
        type=Path,
        help="SQLite call store for grounding; defaults beside the Judge store",
    )
    evaluate_run_parser.add_argument(
        "--grounding-max-concurrency",
        type=int,
        default=1,
    )
    evaluate_run_parser.add_argument("--experiment-id", default="m3-evaluation")
    evaluate_run_parser.add_argument("--run-id", required=True)
    evaluate_run_parser.add_argument("--max-concurrency", type=int, default=3)
    evaluate_run_parser.add_argument("--max-sample-concurrency", type=int, default=1)
    evaluate_run_parser.add_argument("--max-retries", type=int, default=1)
    evaluate_run_parser.add_argument(
        "--grounding-mode",
        choices=tuple(item.value for item in GroundingMode),
        default=GroundingMode.BENCHMARK_ASSISTED.value,
        help="blind requires configured media grounding tools; failures become review_required",
    )
    evaluate_run_parser.add_argument(
        "--judge-env-file",
        type=Path,
        default=Path(".env"),
        help="dotenv file containing JUDGE_MODEL_* settings for the local provider",
    )
    evaluate_run_parser.add_argument("--limit", type=int)
    evaluate_run_parser.add_argument("--overwrite", action="store_true")

    models_parser = subparsers.add_parser("models", help="inspect model profiles")
    models_commands = models_parser.add_subparsers(dest="models_command", required=True)
    models_list_parser = models_commands.add_parser("list", help="list model-pool profiles")
    models_list_parser.add_argument(
        "--model-registry",
        type=Path,
        default=Path("config/models.toml"),
    )
    models_list_parser.add_argument("--role", choices=("target", "judge", "grounding"))
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
    if args.command == "models" and args.models_command == "list":
        try:
            role = ModelRole(args.role) if args.role is not None else None
            profiles = ModelRegistry.load(args.model_registry).list(role=role)
        except (SafeJudgeError, ValidationError) as error:
            build_parser().error(str(error))
        print(
            json.dumps(
                {"profiles": [profile.model_dump(mode="json") for profile in profiles]},
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.command == "target" and args.target_command == "run-jsonl":
        try:
            target_profile = _selected_profile(
                args,
                role=ModelRole.TARGET,
            )
            selected_provider = target_profile.provider
            context = InvocationContext(
                experiment_id=args.experiment_id,
                run_id=args.run_id,
            )
            if selected_provider == "local-openai":
                local_settings = _local_target_settings(
                    target_profile,
                    max_local_media_bytes=args.max_local_media_bytes,
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
                        profile=target_profile,
                        limit=args.limit,
                        overwrite=args.overwrite,
                    )
                )
            else:
                openrouter_settings = _openrouter_profile_settings(
                    target_profile,
                    max_local_media_bytes=args.max_local_media_bytes,
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
                        profile=target_profile,
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
                    "failures": (
                        str(target_result.failure_path)
                        if target_result.failure_path is not None
                        else None
                    ),
                    "store": str(target_result.store_path),
                    "provider": target_result.manifest.provider,
                    "model": target_result.manifest.model,
                    "samples": target_result.manifest.sample_count,
                    "input_samples": target_result.manifest.input_sample_count,
                    "batch_failures": target_result.manifest.failure_count,
                    "verified_media": target_result.manifest.verified_media_count,
                    "cache_hits": target_result.manifest.cache_hit_count,
                    "cache_misses": target_result.manifest.cache_miss_count,
                    "logical_calls": target_result.manifest.logical_call_count,
                    "provider_attempts": target_result.manifest.provider_attempt_count,
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
                "grounding_pipeline": _grounding_pipeline(args),
            }
            if args.taxonomy:
                constitution_registry = ConstitutionRegistry.load(
                    args.constitution_registry
                )
                taxonomy_registry = TaxonomyRegistry.load(args.taxonomy_registry)
                taxonomy_pack = taxonomy_registry.get(
                    args.taxonomy,
                    version=args.taxonomy_version,
                )
                taxonomy_registry.validate_constitutions(constitution_registry)
                common.update(
                    taxonomy_pack=taxonomy_pack,
                    constitution_registry=constitution_registry,
                )
            definition = JuryPlan.load(args.jury_plan).resolve(
                ModelRegistry.load(args.model_registry),
                allow_unqualified=args.allow_unqualified_model,
            )
            evaluation_result = asyncio.run(
                run_evaluation_batch(
                    **common,
                    jury_definition=definition,
                    jury_providers=_jury_providers(
                        definition,
                        artifact_root=args.artifact_root.resolve(),
                        judge_env_file=args.judge_env_file,
                    ),
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
                    "failures": str(evaluation_result.failure_path),
                    "store": str(evaluation_result.store_path),
                    "checkpoint": str(evaluation_result.checkpoint_path),
                    "node_ledger": str(evaluation_result.node_ledger_path),
                    "jury_id": evaluation_result.manifest.jury.jury_id,
                    "jury_hash": evaluation_result.manifest.jury_hash,
                    "jury_seats": {
                        item.seat.value: item.model
                        for item in evaluation_result.manifest.jury.seats
                    },
                    "samples": evaluation_result.manifest.sample_count,
                    "input_samples": evaluation_result.manifest.input_sample_count,
                    "batch_failures": evaluation_result.manifest.failure_count,
                    "judge_failures": evaluation_result.manifest.judge_failure_count,
                    "arbitrations": evaluation_result.manifest.arbitration_count,
                    "logical_calls": evaluation_result.manifest.logical_call_count,
                    "provider_attempts": evaluation_result.manifest.provider_attempt_count,
                    "contract_repairs": (
                        evaluation_result.manifest.contract_repair_call_count
                    ),
                    "cache_hits": evaluation_result.manifest.cache_hit_count,
                    "cache_misses": evaluation_result.manifest.cache_miss_count,
                    "billed_cost_usd": str(evaluation_result.manifest.billed_cost_usd),
                    "levels": evaluation_result.manifest.compliance_level_counts,
                    "taxonomy_id": evaluation_result.manifest.taxonomy_id,
                    "standard_id": evaluation_result.manifest.standard_id,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


def _add_profile_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model-registry",
        type=Path,
        default=Path("config/models.toml"),
    )
    parser.add_argument(
        "--model-profile",
        required=True,
        help="versioned profile ID; provider and model ID are taken from the registry",
    )
    parser.add_argument(
        "--allow-unqualified-model",
        action="store_true",
        help="allow a candidate profile for an explicit development run",
    )


def _grounding_pipeline(args: argparse.Namespace) -> GroundingPipeline:
    sidecar = getattr(args, "grounding_sidecar", None)
    grounding_profile_id = getattr(args, "grounding_model_profile", None)
    if sidecar is not None and grounding_profile_id is not None:
        raise ConfigurationError(
            "choose either --grounding-sidecar or --grounding-model-profile"
        )
    tools: tuple[GroundingTool, ...] = ()
    if sidecar is not None:
        tools = (SidecarGroundingTool(load_sidecar_observations(sidecar)),)
    elif grounding_profile_id is not None:
        if GroundingMode(args.grounding_mode) is not GroundingMode.BLIND:
            raise ConfigurationError("a grounding model profile requires --grounding-mode blind")
        profile = ModelRegistry.load(args.model_registry).get(
            grounding_profile_id,
            role=ModelRole.GROUNDING,
            allow_unqualified=args.allow_unqualified_model,
        )
        if profile.provider != "openrouter":
            raise ConfigurationError("model grounding currently supports OpenRouter profiles")
        provider = OpenRouterProvider(
            settings=_openrouter_profile_settings(profile),
            capabilities=_grounding_capabilities(),
            artifact_store=FileArtifactStore(
                args.artifact_root.resolve() / "grounding"
            ),
            media_root=args.media_root.resolve(),
        )
        invoker = ModelInvoker(
            provider=provider,
            store=SQLiteModelStore(_grounding_store_path(args)),
            policy=InvocationPolicy(
                max_concurrency=args.grounding_max_concurrency,
                max_retries=args.max_retries,
                retry_backoff_seconds=0.5,
            ),
        )
        tools = (
            ModelGroundingTool(
                invoker,
                tool_id=profile.profile_id,
                tool_version=profile.profile_version,
                modality_by_media_type={
                    MediaType.IMAGE: ObservationModality.IMAGE_VLM,
                },
                instruction=(
                    "Transcribe all visible text in the image exactly, then briefly "
                    "describe only directly observable visual context relevant to the "
                    "user request. Do not infer intent, harmfulness, policy, or motives."
                ),
                parameters=profile.default_parameters,
                retry_parameters=profile.retry_parameters,
                max_contract_retries=profile.max_contract_retries,
            ),
        )
    return GroundingPipeline(mode=GroundingMode(args.grounding_mode), tools=tools)


def _grounding_store_path(args: argparse.Namespace) -> Path:
    configured = getattr(args, "grounding_store", None)
    if configured is not None:
        return cast(Path, configured).resolve()
    judge_store = cast(Path, args.store).resolve()
    return judge_store.with_name(f"{judge_store.stem}-grounding.sqlite3")


def _selected_profile(args: argparse.Namespace, *, role: ModelRole) -> ModelProfile:
    return ModelRegistry.load(args.model_registry).get(
        args.model_profile,
        role=role,
        allow_unqualified=args.allow_unqualified_model,
    )


def _local_target_settings(
    profile: ModelProfile,
    *,
    max_local_media_bytes: int | None,
) -> LocalOpenAISettings:
    if max_local_media_bytes is None:
        return LocalOpenAISettings(model_id=profile.model_id)
    return LocalOpenAISettings(
        model_id=profile.model_id,
        max_local_media_bytes=max_local_media_bytes,
    )


def _local_judge_settings(
    profile: ModelProfile,
    env_file: Path,
) -> JudgeOpenAISettings:
    return JudgeOpenAISettings(_env_file=env_file, model_id=profile.model_id)


def _openrouter_profile_settings(
    profile: ModelProfile,
    *,
    max_local_media_bytes: int | None = None,
) -> OpenRouterSettings:
    if max_local_media_bytes is None:
        return OpenRouterSettings(model_id=profile.model_id)
    return OpenRouterSettings(
        model_id=profile.model_id,
        max_local_media_bytes=max_local_media_bytes,
    )


def _jury_providers(
    definition: JuryDefinition,
    *,
    artifact_root: Path,
    judge_env_file: Path,
) -> dict[JurySeat, ModelProvider]:
    providers: dict[JurySeat, ModelProvider] = {}
    for seat, profile in definition.profiles.items():
        seat_artifacts = FileArtifactStore(artifact_root / "jury" / seat.value)
        if profile.provider == "local-openai":
            providers[seat] = LocalOpenAIProvider(
                settings=_local_judge_settings(profile, judge_env_file),
                capabilities=_judge_capabilities(),
                artifact_store=seat_artifacts,
            )
        else:
            providers[seat] = OpenRouterProvider(
                settings=_openrouter_profile_settings(profile),
                capabilities=_judge_capabilities(),
                artifact_store=seat_artifacts,
            )
    return providers


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


def _grounding_capabilities() -> ModelCapabilities:
    return ModelCapabilities(
        input_combinations=(
            ModalityCombination(
                modalities=frozenset({InputModality.TEXT, InputModality.IMAGE})
            ),
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
