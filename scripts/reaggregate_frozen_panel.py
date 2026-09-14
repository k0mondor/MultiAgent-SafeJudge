"""Recompute final levels from frozen panel verdicts without model calls."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from safejudge.aggregation import load_default_aggregation_policy
from safejudge.contracts.judging import (
    AggregateDecision,
    DecisionStatus,
    EvaluationResult,
    EvaluationSpec,
    JudgeAxis,
)
from safejudge.workflows.decision import (
    conservative_conflict_level,
    score_compliance_facets,
    updated_aggregate,
    updated_category_result,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument(
        "--baseline-comparison",
        type=Path,
        help="Reconstruct GPT prelabels from an earlier comparison on the same frozen panel.",
    )
    return parser


def _effective_panel(category: Any) -> tuple[Any, Any]:
    by_axis = {verdict.axis: verdict for verdict in category.verdicts}
    compliance = by_axis[JudgeAxis.COMPLIANCE].label
    facets = by_axis[JudgeAxis.HARM_ENABLEMENT].enablement_scores
    arbitration = category.arbitration
    if arbitration is not None and arbitration.decision_status is DecisionStatus.RESOLVED:
        compliance = arbitration.reconciled_compliance_label or compliance
        facets = arbitration.reconciled_enablement_scores or facets
    if facets is None:
        raise ValueError(f"missing enablement facets for {category.category_id}")
    return compliance, facets


def _reaggregate_category(category: Any, policy: Any) -> Any:
    if category.judge_failures or category.guardrail_failure is not None:
        return category
    compliance, facets = _effective_panel(category)
    score, level, conflicts = score_compliance_facets(
        policy,
        compliance_label=compliance,
        facets=facets,
        guardrail_native_label=(
            category.guardrail_verdict.native_label
            if category.guardrail_verdict is not None
            else None
        ),
    )
    if conflicts:
        existing = category.aggregate.conflict_codes
        aggregate = updated_aggregate(
            category.aggregate,
            decision_status=DecisionStatus.REVIEW_REQUIRED,
            response_compliance_level=None,
            provisional_level=conservative_conflict_level(policy, facets=facets),
            overall_score=None,
            conflict_codes=tuple(dict.fromkeys((*existing, *conflicts))),
            requires_arbitration=True,
            resolution_source="review",
        )
        return updated_category_result(category, aggregate=aggregate, category_score=None)
    aggregate = AggregateDecision(
        response_compliance_level=level,
        oversensitive=None,
        overall_score=score,
        requires_arbitration=category.arbitration is not None,
        resolution_source="arbitration" if category.arbitration is not None else "deterministic",
    )
    return updated_category_result(category, aggregate=aggregate, category_score=score)


def _reaggregate_sample(result: EvaluationResult, policy: Any) -> EvaluationResult:
    categories = tuple(_reaggregate_category(item, policy) for item in result.category_results)
    if not categories:
        aggregate = result.aggregate
    else:
        unresolved = tuple(
            item
            for item in categories
            if item.aggregate.decision_status is not DecisionStatus.RESOLVED
        )
        resolved_levels = tuple(
            item.aggregate.response_compliance_level
            for item in categories
            if item.aggregate.response_compliance_level is not None
        )
        provisional_levels = tuple(
            item.aggregate.provisional_level
            for item in unresolved
            if item.aggregate.provisional_level is not None
        )
        if unresolved or result.judge_failures:
            known_levels = (*resolved_levels, *provisional_levels)
            aggregate = AggregateDecision(
                decision_status=DecisionStatus.REVIEW_REQUIRED,
                response_compliance_level=None,
                provisional_level=max(known_levels) if known_levels else None,
                oversensitive=result.aggregate.oversensitive,
                conflict_codes=tuple(
                    f"CATEGORY:{item.category_id}:{code}"
                    for item in unresolved
                    for code in item.aggregate.conflict_codes
                ),
                requires_arbitration=any(
                    item.aggregate.requires_arbitration for item in unresolved
                ),
                resolution_source="review",
            )
        else:
            scores = tuple(
                item.category_score for item in categories if item.category_score is not None
            )
            overall_score = max(scores) if scores else 0
            aggregate = AggregateDecision(
                response_compliance_level=max(resolved_levels),
                oversensitive=result.aggregate.oversensitive,
                overall_score=overall_score,
                max_score_category_ids=tuple(
                    item.category_id for item in categories if item.category_score == overall_score
                ),
                requires_arbitration=any(item.arbitration is not None for item in categories),
                resolution_source=(
                    "arbitration"
                    if any(item.arbitration is not None for item in categories)
                    else "deterministic"
                ),
            )

    spec_values = result.evaluation_spec.model_dump(
        mode="python", exclude={"schema_version", "evaluation_key"}
    )
    spec_values.update(
        aggregator_id=policy.aggregator_id,
        aggregator_version=policy.aggregator_version,
        aggregator_hash=policy.fingerprint,
    )
    spec = EvaluationSpec.create(**spec_values)
    return EvaluationResult.model_validate(
        {
            **result.model_dump(mode="python"),
            "evaluation_spec": spec,
            "category_results": categories,
            "aggregate": aggregate,
        }
    )


def _macro_f1(gold: list[str], predictions: list[str]) -> float:
    values = []
    for label in ("L0", "L1", "L2"):
        tp = sum(g == label and p == label for g, p in zip(gold, predictions, strict=True))
        fp = sum(g != label and p == label for g, p in zip(gold, predictions, strict=True))
        fn = sum(g == label and p != label for g, p in zip(gold, predictions, strict=True))
        denominator = 2 * tp + fp + fn
        values.append(0.0 if denominator == 0 else 2 * tp / denominator)
    return sum(values) / len(values)


def _prediction(result: EvaluationResult) -> str:
    if result.aggregate.decision_status is not DecisionStatus.RESOLVED:
        return "review_required"
    level = result.aggregate.response_compliance_level
    if level is None:
        raise ValueError("resolved result has no level")
    return f"L{int(level)}"


def _comparison(
    old: tuple[EvaluationResult, ...],
    new: tuple[EvaluationResult, ...],
    annotations: dict[str, str],
) -> dict[str, Any]:
    old_by_id = {item.sample_id: item for item in old}
    gold: list[str] = []
    predictions: list[str] = []
    resolved_gold: list[str] = []
    resolved_predictions: list[str] = []
    differences = []
    changed = []
    for item in new:
        expected = annotations[item.sample_id]
        predicted = _prediction(item)
        previous = _prediction(old_by_id[item.sample_id])
        gold.append(expected)
        predictions.append(predicted)
        if predicted != "review_required":
            resolved_gold.append(expected)
            resolved_predictions.append(predicted)
        if predicted != expected:
            differences.append(
                {
                    "sample_id": item.sample_id,
                    "gpt_prelabel": expected,
                    "framework": predicted,
                    "provisional_level": (
                        None
                        if item.aggregate.provisional_level is None
                        else f"L{int(item.aggregate.provisional_level)}"
                    ),
                }
            )
        if previous != predicted:
            changed.append({"sample_id": item.sample_id, "old": previous, "new": predicted})
    correct_resolved = sum(g == p for g, p in zip(resolved_gold, resolved_predictions, strict=True))
    correct_all = sum(g == p for g, p in zip(gold, predictions, strict=True))
    return {
        "reference": "GPT-assisted prelabels; not human-confirmed Gold",
        "samples": len(new),
        "distribution": dict(Counter(predictions)),
        "gpt_distribution": dict(Counter(gold)),
        "automatic_coverage": len(resolved_predictions) / len(new),
        "selective_accuracy": correct_resolved / len(resolved_predictions),
        "overall_accuracy_review_as_incorrect": correct_all / len(new),
        "resolved_macro_f1": _macro_f1(resolved_gold, resolved_predictions),
        "all_macro_f1_review_as_abstention": _macro_f1(gold, predictions),
        "differences": differences,
        "changed_from_old_aggregation": changed,
    }


def _reference_labels_from_annotations(path: Path) -> dict[str, str]:
    return {
        item["sample_id"]: item["level"] for item in json.loads(path.read_text(encoding="utf-8"))
    }


def _reference_labels_from_baseline(
    old: tuple[EvaluationResult, ...], path: Path
) -> dict[str, str]:
    baseline = json.loads(path.read_text(encoding="utf-8"))
    if baseline.get("samples") != len(old):
        raise ValueError("baseline comparison does not describe the same sample count")
    differences = {item["sample_id"]: item["gpt_prelabel"] for item in baseline["differences"]}
    return {item.sample_id: differences.get(item.sample_id, _prediction(item)) for item in old}


def _write_markdown(path: Path, comparison: dict[str, Any]) -> None:
    lines = [
        "# Frozen-panel reaggregation vs GPT prelabels",
        "",
        "> GPT-assisted prelabels are not human-confirmed Gold.",
        "",
        f"- Samples: {comparison['samples']}",
        f"- Distribution: `{comparison['distribution']}`",
        f"- GPT distribution: `{comparison['gpt_distribution']}`",
        f"- Automatic coverage: {comparison['automatic_coverage']:.3%}",
        f"- Selective accuracy: {comparison['selective_accuracy']:.3%}",
        "- Overall accuracy (review as incorrect): "
        f"{comparison['overall_accuracy_review_as_incorrect']:.3%}",
        f"- Resolved Macro-F1: {comparison['resolved_macro_f1']:.4f}",
        "- All-sample Macro-F1 (review as abstention): "
        f"{comparison['all_macro_f1_review_as_abstention']:.4f}",
        "- Changed final predictions from old aggregation: "
        f"{len(comparison['changed_from_old_aggregation'])}",
        "",
        "## Differences",
        "",
        "| sample_id | GPT prelabel | framework | provisional |",
        "|---|---:|---:|---:|",
    ]
    lines.extend(
        f"| `{item['sample_id']}` | {item['gpt_prelabel']} | "
        f"{item['framework']} | {item['provisional_level'] or '—'} |"
        for item in comparison["differences"]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = build_parser().parse_args()
    if args.annotations is not None and args.baseline_comparison is not None:
        raise ValueError("choose annotations or baseline comparison, not both")
    old = tuple(
        EvaluationResult.model_validate_json(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    policy = load_default_aggregation_policy()
    new = tuple(_reaggregate_sample(item, policy) for item in old)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "evaluations.jsonl"
    output.write_text("".join(item.model_dump_json() + "\n" for item in new), encoding="utf-8")
    metadata = {
        "schema_version": "1.0",
        "execution_mode": "frozen_panel_reaggregation",
        "source": str(args.input.resolve()),
        "source_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "output": str(output.resolve()),
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "created_at": datetime.now(UTC).isoformat(),
        "model_calls": 0,
        "aggregator_id": policy.aggregator_id,
        "aggregator_version": policy.aggregator_version,
        "aggregator_hash": policy.fingerprint,
    }
    (args.output_dir / "reaggregation-manifest.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    reference_labels = None
    if args.annotations is not None:
        reference_labels = _reference_labels_from_annotations(args.annotations)
    elif args.baseline_comparison is not None:
        reference_labels = _reference_labels_from_baseline(old, args.baseline_comparison)
    if reference_labels is not None:
        comparison = _comparison(old, new, reference_labels)
        (args.output_dir / "comparison-with-gpt-prelabels.json").write_text(
            json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        _write_markdown(args.output_dir / "comparison-with-gpt-prelabels.md", comparison)
        print(json.dumps(comparison, ensure_ascii=False, indent=2))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
