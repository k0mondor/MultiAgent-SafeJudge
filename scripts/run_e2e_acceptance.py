"""Run the one formal SafeJudge acceptance path with real configured models."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from safejudge.cli import main as safejudge_main
from safejudge.contracts.judging import DecisionStatus, EvaluationResult
from safejudge.models.batch import TargetBatchManifest
from safejudge.workflows.batch import EvaluationBatchManifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Canonical data -> Target -> blind grounding -> heterogeneous Jury -> result"
        )
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--media-root", required=True, type=Path)
    parser.add_argument("--target-profile", required=True)
    parser.add_argument("--grounding-profile", required=True)
    parser.add_argument(
        "--jury-plan",
        type=Path,
        default=Path("config/juries/m3-heterogeneous-v1.toml"),
    )
    parser.add_argument(
        "--model-registry",
        type=Path,
        default=Path("config/models.toml"),
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--capability",
        action="append",
        choices=("text", "text+image", "text+audio", "text+video"),
        default=None,
    )
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--run-id")
    parser.add_argument("--allow-unqualified-model", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.limit < 1:
        raise SystemExit("--limit must be at least 1")
    input_path = args.input.resolve()
    media_root = args.media_root.resolve()
    registry = args.model_registry.resolve()
    jury_plan = args.jury_plan.resolve()
    if not input_path.is_file():
        raise SystemExit(f"input does not exist: {input_path}")
    if not media_root.is_dir():
        raise SystemExit(f"media root does not exist: {media_root}")
    if not registry.is_file():
        raise SystemExit(f"model registry does not exist: {registry}")
    if not jury_plan.is_file():
        raise SystemExit(f"jury plan does not exist: {jury_plan}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = args.run_id or datetime.now(UTC).strftime("e2e-%Y%m%dT%H%M%SZ")
    target_output = output_dir / "target-responses.jsonl"
    evaluation_output = output_dir / "evaluations.jsonl"
    artifacts = output_dir / "artifacts"

    common_flags = ["--allow-unqualified-model"] if args.allow_unqualified_model else []
    overwrite_flags = ["--overwrite"] if args.overwrite else []
    capability_flags = [
        item
        for capability in (args.capability or ["text+image"])
        for item in ("--capability", capability)
    ]

    safejudge_main(
        [
            "target",
            "run-jsonl",
            "--input",
            str(input_path),
            "--media-root",
            str(media_root),
            "--output",
            str(target_output),
            "--store",
            str(output_dir / "target-calls.sqlite3"),
            "--artifact-root",
            str(artifacts),
            "--model-registry",
            str(registry),
            "--model-profile",
            args.target_profile,
            "--experiment-id",
            "formal-e2e-acceptance",
            "--run-id",
            f"{run_id}-target",
            "--max-concurrency",
            "1",
            "--limit",
            str(args.limit),
            *capability_flags,
            *common_flags,
            *overwrite_flags,
        ]
    )

    safejudge_main(
        [
            "evaluate",
            "run-jsonl",
            "--input",
            str(input_path),
            "--target-responses",
            str(target_output),
            "--output",
            str(evaluation_output),
            "--store",
            str(output_dir / "judge-calls.sqlite3"),
            "--checkpoint",
            str(output_dir / "checkpoints.sqlite3"),
            "--node-ledger",
            str(output_dir / "node-ledger.sqlite3"),
            "--artifact-root",
            str(artifacts),
            "--media-root",
            str(media_root),
            "--model-registry",
            str(registry),
            "--jury-plan",
            str(jury_plan),
            "--grounding-mode",
            "blind",
            "--grounding-model-profile",
            args.grounding_profile,
            "--experiment-id",
            "formal-e2e-acceptance",
            "--run-id",
            f"{run_id}-jury",
            "--max-concurrency",
            "3",
            "--max-sample-concurrency",
            "1",
            "--limit",
            str(args.limit),
            *common_flags,
            *overwrite_flags,
        ]
    )

    target_manifest = TargetBatchManifest.model_validate_json(
        target_output.with_suffix(".jsonl.manifest.json").read_text(encoding="utf-8")
    )
    evaluation_manifest = EvaluationBatchManifest.model_validate_json(
        evaluation_output.with_suffix(".jsonl.manifest.json").read_text(encoding="utf-8")
    )
    results = tuple(
        EvaluationResult.model_validate_json(line)
        for line in evaluation_output.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if target_manifest.failure_count or (
        target_manifest.sample_count != target_manifest.input_sample_count
    ):
        raise SystemExit("formal acceptance failed during Target generation")
    if evaluation_manifest.failure_count or evaluation_manifest.judge_failure_count:
        raise SystemExit("formal acceptance contains batch or Judge failures")
    if evaluation_manifest.sample_count != evaluation_manifest.input_sample_count:
        raise SystemExit("formal acceptance did not produce one result per input")
    if not results or any(
        item.aggregate.decision_status is not DecisionStatus.RESOLVED for item in results
    ):
        raise SystemExit("formal acceptance contains unresolved results")

    report = {
        "schema_version": "1.0",
        "status": "passed",
        "real_models": True,
        "grounding_mode": "blind",
        "run_id": run_id,
        "input": str(input_path),
        "samples": len(results),
        "target_profile": args.target_profile,
        "grounding_profile": args.grounding_profile,
        "jury_id": evaluation_manifest.jury.jury_id,
        "jury_hash": evaluation_manifest.jury_hash,
        "levels": evaluation_manifest.compliance_level_counts,
        "arbitrations": evaluation_manifest.arbitration_count,
        "provider_attempts": {
            "target": target_manifest.provider_attempt_count,
            "jury": evaluation_manifest.provider_attempt_count,
        },
        "billed_cost_usd": {
            "target": str(target_manifest.billed_cost_usd),
            "jury": str(evaluation_manifest.billed_cost_usd),
        },
        "target_manifest": str(target_output.with_suffix(".jsonl.manifest.json")),
        "evaluation_manifest": str(
            evaluation_output.with_suffix(".jsonl.manifest.json")
        ),
    }
    report_path = output_dir / "e2e-acceptance-report.json"
    report_path.write_text(
        f"{json.dumps(report, ensure_ascii=False, indent=2)}\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
