"""Run the one formal SafeJudge acceptance path with real configured models."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from safejudge.cli import main as safejudge_main
from safejudge.contracts.judging import DecisionStatus, EvaluationResult
from safejudge.models.batch import TargetBatchManifest
from safejudge.reporting import write_evaluation_markdown_report
from safejudge.workflows.batch import EvaluationBatchManifest

DEFAULT_TAXONOMY_ID = "gb-t-45654-2025-safejudge-v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Canonical data -> Target -> grounding -> DeepSeek Judge + category guardrail -> result"
        )
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--media-root", required=True, type=Path)
    parser.add_argument("--target-profile", required=True)
    parser.add_argument("--grounding-profile", required=True)
    parser.add_argument(
        "--jury-plan",
        type=Path,
        default=Path("config/juries/m3-deepseek-llamaguard-remote-v1.toml"),
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
    taxonomy_group = parser.add_mutually_exclusive_group()
    taxonomy_group.add_argument(
        "--taxonomy",
        default=DEFAULT_TAXONOMY_ID,
        help=f"taxonomy ID (default: {DEFAULT_TAXONOMY_ID})",
    )
    taxonomy_group.add_argument(
        "--no-taxonomy",
        dest="taxonomy",
        action="store_const",
        const=None,
        help="disable taxonomy routing for this run",
    )
    parser.add_argument(
        "--taxonomy-version",
        help="taxonomy version (default: latest registered version)",
    )
    parser.add_argument("--allow-unqualified-model", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--acceptance-mode",
        choices=("strict", "batch"),
        default="strict",
        help=(
            "strict fails when human review is required; batch completes and records "
            "completed_with_review"
        ),
    )
    parser.add_argument(
        "--replay",
        action="store_true",
        help=(
            "reuse the existing frozen TargetResponse artifact and forbid all new "
            "grounding, Judge, and guardrail provider calls"
        ),
    )
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
    target_manifest_path = target_output.with_suffix(".jsonl.manifest.json")
    evaluation_output = output_dir / "evaluations.jsonl"
    artifacts = output_dir / "artifacts"

    common_flags = ["--allow-unqualified-model"] if args.allow_unqualified_model else []
    overwrite_flags = ["--overwrite"] if args.overwrite else []
    capability_flags = [
        item
        for capability in (args.capability or ["text+image"])
        for item in ("--capability", capability)
    ]
    taxonomy_flags = ["--no-taxonomy"]
    if args.taxonomy is not None:
        taxonomy_flags = ["--taxonomy", args.taxonomy]
        if args.taxonomy_version is not None:
            taxonomy_flags.extend(["--taxonomy-version", args.taxonomy_version])

    if args.replay:
        if not args.overwrite:
            raise SystemExit("--replay requires --overwrite for evaluation outputs")
        _validate_frozen_target(
            input_path=input_path,
            target_output=target_output,
            target_manifest_path=target_manifest_path,
            expected_samples=args.limit,
        )
        print(f"Replay: using frozen TargetResponse artifact {target_output}")
    else:
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

    target_manifest = TargetBatchManifest.model_validate_json(
        target_manifest_path.read_text(encoding="utf-8")
    )
    if target_manifest.failure_count or (
        target_manifest.sample_count != target_manifest.input_sample_count
    ):
        raise SystemExit("formal acceptance failed during Target generation")

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
            *taxonomy_flags,
            *common_flags,
            *overwrite_flags,
            *(["--cache-only"] if args.replay else []),
        ]
    )

    evaluation_manifest = EvaluationBatchManifest.model_validate_json(
        evaluation_output.with_suffix(".jsonl.manifest.json").read_text(encoding="utf-8")
    )
    results = tuple(
        EvaluationResult.model_validate_json(line)
        for line in evaluation_output.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    readable_report_path = output_dir / "evaluation-report.md"
    write_evaluation_markdown_report(
        readable_report_path,
        manifest=evaluation_manifest,
        results=results,
    )
    print(f"Human-readable report: {readable_report_path}")
    if evaluation_manifest.failure_count or evaluation_manifest.judge_failure_count:
        raise SystemExit("formal acceptance contains batch or Judge failures")
    if evaluation_manifest.jury.category_guardrail is not None and (
        evaluation_manifest.guardrail_failure_count
        or evaluation_manifest.guardrail_verdict_count
        != evaluation_manifest.category_result_count
    ):
        raise SystemExit("formal acceptance contains missing or failed guardrail results")
    if evaluation_manifest.sample_count != evaluation_manifest.input_sample_count:
        raise SystemExit("formal acceptance did not produce one result per input")
    if not results or any(
        item.aggregate.decision_status is DecisionStatus.NOT_EVALUATED for item in results
    ):
        raise SystemExit("formal acceptance contains unevaluated results")

    review_required_count = sum(
        item.aggregate.decision_status is DecisionStatus.REVIEW_REQUIRED for item in results
    )
    status = _acceptance_status(
        review_required_count=review_required_count,
        acceptance_mode=args.acceptance_mode,
    )

    report = {
        "schema_version": "1.0",
        "status": status,
        "acceptance_mode": args.acceptance_mode,
        "execution_mode": "replay" if args.replay else "fresh",
        "real_models": True,
        "grounding_mode": "blind",
        "taxonomy_id": evaluation_manifest.taxonomy_id,
        "taxonomy_version": evaluation_manifest.taxonomy_version,
        "standard_id": evaluation_manifest.standard_id,
        "run_id": run_id,
        "input": str(input_path),
        "samples": len(results),
        "review_required_samples": review_required_count,
        "target_profile": args.target_profile,
        "grounding_profile": args.grounding_profile,
        "jury_id": evaluation_manifest.jury.jury_id,
        "jury_hash": evaluation_manifest.jury_hash,
        "levels": evaluation_manifest.compliance_level_counts,
        "arbitrations": evaluation_manifest.arbitration_count,
        "guardrail_verdicts": evaluation_manifest.guardrail_verdict_count,
        "guardrail_triggers": evaluation_manifest.guardrail_trigger_counts,
        "provider_attempts": {
            "target": 0 if args.replay else target_manifest.provider_attempt_count,
            "jury": evaluation_manifest.provider_attempt_count,
        },
        "billed_cost_usd": {
            "target": "0" if args.replay else str(target_manifest.billed_cost_usd),
            "jury": str(evaluation_manifest.billed_cost_usd),
        },
        "target_manifest": str(target_output.with_suffix(".jsonl.manifest.json")),
        "evaluation_manifest": str(
            evaluation_output.with_suffix(".jsonl.manifest.json")
        ),
        "human_readable_report": str(readable_report_path),
    }
    report_path = output_dir / "e2e-acceptance-report.json"
    report_path.write_text(
        f"{json.dumps(report, ensure_ascii=False, indent=2)}\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if review_required_count and args.acceptance_mode == "strict":
        raise SystemExit("formal acceptance contains results requiring human review")
    return 0


def _validate_frozen_target(
    *,
    input_path: Path,
    target_output: Path,
    target_manifest_path: Path,
    expected_samples: int,
) -> None:
    if not target_output.is_file() or not target_manifest_path.is_file():
        raise SystemExit(
            "replay requires existing target-responses.jsonl and its manifest"
        )
    manifest = TargetBatchManifest.model_validate_json(
        target_manifest_path.read_text(encoding="utf-8")
    )
    if manifest.input.sha256 != _file_sha256(input_path):
        raise SystemExit("frozen TargetResponse input hash does not match --input")
    if manifest.output.sha256 != _file_sha256(target_output):
        raise SystemExit("frozen TargetResponse artifact hash does not match its manifest")
    if manifest.input_sample_count != expected_samples:
        raise SystemExit(
            "replay sample count differs from frozen TargetResponse manifest: "
            f"expected {manifest.input_sample_count}, got --limit {expected_samples}"
        )
    if manifest.failure_count or manifest.sample_count != expected_samples:
        raise SystemExit("frozen TargetResponse artifact is incomplete")


def _acceptance_status(*, review_required_count: int, acceptance_mode: str) -> str:
    if review_required_count < 0:
        raise ValueError("review_required_count cannot be negative")
    if acceptance_mode not in {"strict", "batch"}:
        raise ValueError(f"unknown acceptance mode: {acceptance_mode!r}")
    if not review_required_count:
        return "passed"
    return "completed_with_review" if acceptance_mode == "batch" else "review_required"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
