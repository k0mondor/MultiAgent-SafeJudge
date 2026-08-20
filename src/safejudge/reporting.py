"""Human-readable reports for completed SafeJudge evaluations."""

# ruff: noqa: RUF001

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path

from safejudge.contracts.evaluation import ResponseComplianceLevel
from safejudge.contracts.judging import (
    AggregateDecision,
    CategoryEvaluationResult,
    EvaluationResult,
    JudgeExecutionFailure,
    JudgeVerdict,
)
from safejudge.workflows.batch import EvaluationBatchManifest

_LEVEL_LABELS = {
    ResponseComplianceLevel.FULLY_COMPLIANT: "0（完全合规）",
    ResponseComplianceLevel.PARTIALLY_COMPLIANT: "1（部分合规）",
    ResponseComplianceLevel.SEVERELY_NONCOMPLIANT: "2（严重不合规）",
}


def write_evaluation_markdown_report(
    path: Path,
    *,
    manifest: EvaluationBatchManifest,
    results: Sequence[EvaluationResult],
) -> None:
    """Write one readable report for an evaluation batch."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        render_evaluation_markdown_report(manifest=manifest, results=results),
        encoding="utf-8",
    )


def render_evaluation_markdown_report(
    *,
    manifest: EvaluationBatchManifest,
    results: Sequence[EvaluationResult],
) -> str:
    """Render batch results without hiding unresolved or failed Judge decisions."""

    review_count = sum(
        result.aggregate.decision_status.value == "review_required" for result in results
    )
    resolved_count = sum(
        result.aggregate.decision_status.value == "resolved" for result in results
    )
    lines = [
        "# SafeJudge 实验结果",
        "",
        "> 这是由框架自动生成的人工可读报告；机器可复现的原始结果仍保存在 `evaluations.jsonl`。",
        "",
        "## 实验概览",
        "",
        "| 项目 | 内容 |",
        "| --- | --- |",
        f"| 实验 ID | `{_escape_cell(manifest.experiment_id)}` |",
        f"| 运行 ID | `{_escape_cell(manifest.run_id)}` |",
        f"| 样本数 | {len(results)} |",
        f"| 已得出结论 | {resolved_count} |",
        f"| 需要人工复核 | {review_count} |",
        f"| Taxonomy | `{_escape_cell(manifest.taxonomy_id or '未启用')}` |",
        f"| 国标 | `{_escape_cell(manifest.standard_id or '无')}` |",
        f"| 主 Judge | `{_escape_cell(manifest.jury.model)}` |",
        f"| Judge 阶段计费 | `${manifest.billed_cost_usd}` |",
        "",
    ]
    if not results:
        lines.extend(["> 本次运行没有生成有效评估结果。", ""])
    for index, result in enumerate(results, start=1):
        _append_result(lines, result=result, index=index)
    return "\n".join(lines).rstrip() + "\n"


def _append_result(lines: list[str], *, result: EvaluationResult, index: int) -> None:
    aggregate = result.aggregate
    lines.extend(
        [
            f"## 样本 {index}：`{_escape_inline(result.sample_id)}`",
            "",
            "### 最终结论",
            "",
            "| 项目 | 内容 |",
            "| --- | --- |",
            f"| 状态 | **{_status_label(aggregate)}** |",
        ]
    )
    if aggregate.response_compliance_level is not None:
        lines.append(
            f"| 最终等级 | {_level_label(aggregate.response_compliance_level)} |"
        )
    if result.routed_category_ids:
        lines.append(f"| 命中小类 | {_inline_codes(result.routed_category_ids)} |")
    if aggregate.conflict_codes:
        lines.append(f"| 复核原因 | {_inline_codes(aggregate.conflict_codes)} |")
    lines.append("")
    if aggregate.decision_status.value == "review_required":
        lines.extend(
            [
                (
                    "> **这条样本需要人工复核。** 请重点比较 Target 回答、"
                    "各分轴 Judge 结论和安全护栏结论。"
                ),
                "",
            ]
        )

    lines.extend(["### 原始请求", "", _code_block(_request_text(result)), ""])
    lines.extend(["### 上游媒体理解", ""])
    if result.grounding_artifact.observations:
        for observation in result.grounding_artifact.observations:
            lines.append(
                f"- `{observation.evidence_id}` / `{observation.modality.value}`："
                f"{observation.text}"
            )
    else:
        lines.append("- 无媒体观察结果。")
    if result.grounding_artifact.error_codes:
        lines.append(
            f"- Grounding 错误：{_inline_codes(result.grounding_artifact.error_codes)}"
        )
    lines.extend(["", "### Target 完整回答", "", _code_block(result.target_response.text), ""])
    lines.extend(
        [
            "### 请求理解与 Taxonomy 路由",
            "",
            "| 项目 | 内容 |",
            "| --- | --- |",
            f"| 请求意图 | `{result.intent_analysis.request_intent.value}` |",
            f"| 是否在范围内 | `{result.intent_analysis.scope_status.value}` |",
            f"| 请求行为 | {_escape_cell(result.intent_analysis.requested_action)} |",
            f"| 路由小类 | {_inline_codes(result.routed_category_ids)} |",
            "",
        ]
    )

    if result.category_results:
        lines.extend(
            [
                "### 分小类裁判结果",
                "",
                (
                    "> 分数说明：`compliance` 使用二元离散值（0=合规，1=不合规）；"
                    "`harm_enablement` 的 specificity、completeness、feasibility、"
                    "evasion_help 各自使用 0/1/2 三档。无法可靠合并的情况显示为"
                    "“需要人工复核”，不伪造成连续分数。"
                ),
                "",
            ]
        )
        for category in result.category_results:
            _append_category(lines, category)
    else:
        lines.extend(["### 裁判结果", ""])
        if result.verdicts:
            _append_verdict_table(lines, result.verdicts)
        elif not result.judge_failures:
            lines.extend(
                [
                    "> 未进入子裁判阶段；请根据上面的复核原因检查 Grounding、"
                    "Intent 或 Taxonomy 路由。",
                    "",
                ]
            )
        if result.judge_failures:
            _append_failures(lines, result.judge_failures)

    lines.extend(
        [
            "### 模型与溯源",
            "",
            f"- Target：`{result.target_response.model.model}`",
            f"- Target call ID：`{result.target_response.call_id or '无'}`",
            f"- Target 本次计费：`${result.target_response.billed_cost_usd}`",
        ]
    )
    if result.target_response.raw_artifact_uri:
        artifact_uri = result.target_response.raw_artifact_uri.replace("\\", "/")
        lines.append(f"- Target 原始返回：[`{artifact_uri}`](artifacts/{artifact_uri})")
    lines.extend(["", "---", ""])


def _append_category(lines: list[str], category: CategoryEvaluationResult) -> None:
    lines.extend(
        [
            f"#### `{_escape_inline(category.category_id)}` {category.category_name}",
            "",
            f"- 国标条款：{category.standard_clause}",
            f"- Constitution：`{category.constitution_id}`",
            f"- 小类状态：**{_status_label(category.aggregate)}**",
        ]
    )
    if category.aggregate.response_compliance_level is not None:
        lines.append(
            "- 小类最终等级："
            f"{_level_label(category.aggregate.response_compliance_level)}"
        )
    if category.aggregate.conflict_codes:
        lines.append(
            f"- 复核原因：{_inline_codes(category.aggregate.conflict_codes)}"
        )
    lines.append("")
    _append_verdict_table(lines, category.verdicts)
    if category.judge_failures:
        _append_failures(lines, category.judge_failures)
    if category.guardrail_verdict is not None:
        guardrail = category.guardrail_verdict
        lines.extend(
            [
                "安全护栏：",
                "",
                "| 模型 | 原生结论 | 映射结论 | 原生类别 | 触发规则 |",
                "| --- | --- | --- | --- | --- |",
                (
                    f"| `{_escape_cell(guardrail.trace.model.model)}` "
                    f"| `{guardrail.native_label}` | `{guardrail.label}` "
                    f"| {_inline_codes(guardrail.native_categories)} "
                    f"| {_inline_codes(guardrail.triggered_rule_ids)} |"
                ),
                "",
            ]
        )
    elif category.guardrail_failure is not None:
        lines.extend(
            [
                f"> 安全护栏调用失败：{category.guardrail_failure.message}",
                "",
            ]
        )


def _append_verdict_table(
    lines: list[str], verdicts: Sequence[JudgeVerdict]
) -> None:
    lines.extend(
        [
            "| 裁判轴 | 结论 | 模型 | 离散分数 | 触发规则 |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for verdict in verdicts:
        scores = verdict.enablement_scores
        if verdict.axis.value == "compliance":
            score_text = "0（合规）" if verdict.label.value == "compliant" else "1（不合规）"
        elif scores is not None:
            score_text = (
                f"specificity={scores.specificity}, completeness={scores.completeness}, "
                f"feasibility={scores.feasibility}, evasion_help={scores.evasion_help}"
            )
        else:
            score_text = "—"
        lines.append(
            f"| `{verdict.axis.value}` | `{verdict.label.value}` "
            f"| `{_escape_cell(verdict.trace.model.model)}` | {_escape_cell(score_text)} "
            f"| {_inline_codes(verdict.triggered_rule_ids)} |"
        )
    if not verdicts:
        lines.append("| — | — | — | — | — |")
    lines.append("")


def _append_failures(
    lines: list[str], failures: Sequence[JudgeExecutionFailure]
) -> None:
    lines.extend(["裁判调用失败：", ""])
    for failure in failures:
        lines.append(
            f"- `{failure.axis.value}` / `{failure.failure_code.value}`：{failure.message}"
        )
    lines.append("")


def _request_text(result: EvaluationResult) -> str:
    try:
        snapshot = json.loads(result.request_snapshot.content)
    except json.JSONDecodeError:
        return result.request_snapshot.content
    rendered: list[str] = []
    for part in snapshot.get("parts", []):
        if not isinstance(part, dict):
            continue
        if part.get("kind") == "text" and isinstance(part.get("text"), str):
            rendered.append(part["text"])
        elif part.get("kind") == "media":
            rendered.append(
                f"[媒体：{part.get('media_type', 'unknown')}，"
                f"SHA-256={part.get('sha256', 'unknown')}]"
            )
    return "\n\n".join(rendered) or result.request_snapshot.content


def _status_label(aggregate: AggregateDecision) -> str:
    return {
        "resolved": "已得出结论",
        "review_required": "需要人工复核",
        "not_evaluated": "未评估",
    }[aggregate.decision_status.value]


def _level_label(level: ResponseComplianceLevel | None) -> str:
    return _LEVEL_LABELS[level] if level is not None else "—"


def _inline_codes(values: Sequence[str]) -> str:
    return "、".join(f"`{_escape_inline(value)}`" for value in values) or "—"


def _escape_inline(value: str) -> str:
    return value.replace("`", "\\`")


def _escape_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", "<br>")


def _code_block(value: str) -> str:
    longest_run = max((len(run) for run in re.findall(r"`+", value)), default=0)
    fence = "`" * max(3, longest_run + 1)
    return f"{fence}text\n{value}\n{fence}"
