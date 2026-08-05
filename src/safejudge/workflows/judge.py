"""Provider-neutral runner for structured main, panel, and arbitration judges."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from typing import Any, Literal, cast

from pydantic import Field, JsonValue, ValidationError

from safejudge.contracts.base import ContractModel
from safejudge.contracts.dataset import RequestIntent
from safejudge.contracts.evaluation import ResponseComplianceLevel, TargetResponse
from safejudge.contracts.judging import (
    ArbitrationDecision,
    EnablementScores,
    EvidenceRef,
    IntentAnalysis,
    JudgeAxis,
    JudgeCallTrace,
    JudgeLabel,
    JudgeModelRef,
    JudgeVerdict,
    RequestSnapshot,
)
from safejudge.contracts.model import (
    InputModality,
    InvocationContext,
    ModelRequest,
    ModelRole,
    ModelTextPart,
)
from safejudge.core.errors import ContractValidationError, UnsupportedModalityError
from safejudge.models.invocation import InvocationResult, ModelInvoker
from safejudge.workflows.prompts import (
    ARBITRATION_PROMPT_VERSION,
    INTENT_PROMPT_VERSION,
    PANEL_PROMPT_VERSION,
    arbitration_prompt,
    intent_prompt,
    panel_prompt,
)


class _IntentPayload(ContractModel):
    request_intent: RequestIntent
    risk_category: str | None = None
    request_summary: str = Field(min_length=1, max_length=1_000)
    response_summary: str = Field(min_length=1, max_length=1_000)
    expanded_task_context: str = Field(min_length=1, max_length=2_000)
    confidence: float = Field(ge=0, le=1)


class _EvidenceQuote(ContractModel):
    source: Literal["request_snapshot", "target_response"]
    text: str = Field(min_length=1, max_length=500)


class _VerdictPayload(ContractModel):
    label: JudgeLabel
    confidence: float = Field(ge=0, le=1)
    reason_codes: tuple[str, ...] = Field(min_length=1)
    evidence: tuple[_EvidenceQuote, ...] = Field(min_length=1)


class _EnablementPayload(ContractModel):
    specificity: int = Field(ge=0, le=2)
    completeness: int = Field(ge=0, le=2)
    feasibility: int = Field(ge=0, le=2)
    evasion_help: int = Field(ge=0, le=2)
    confidence: float = Field(ge=0, le=1)
    reason_codes: tuple[str, ...] = Field(min_length=1)
    evidence: tuple[_EvidenceQuote, ...] = Field(min_length=1)


class _ArbitrationPayload(ContractModel):
    response_compliance_level: ResponseComplianceLevel
    confidence: float = Field(ge=0, le=1)
    reason_codes: tuple[str, ...] = Field(min_length=1)
    evidence: tuple[_EvidenceQuote, ...] = Field(min_length=1)


class JudgeRunner:
    """Invokes a JUDGE model and accepts only strict, versioned JSON results."""

    def __init__(self, invoker: ModelInvoker) -> None:
        self.invoker = invoker

    async def analyze_intent(
        self,
        request_snapshot: RequestSnapshot,
        target_response: TargetResponse,
        *,
        context: InvocationContext,
        parameters: Mapping[str, JsonValue] | None = None,
    ) -> IntentAnalysis:
        result = await self._invoke(
            request_id=f"judge:intent:{request_snapshot.sample_id}",
            prompt=intent_prompt(request_snapshot, target_response),
            context=context,
            parameters=parameters,
            payload_schema=_IntentPayload,
            evidence_text=None,
        )
        payload = self._validate_payload(_IntentPayload, result.response.answer)
        return IntentAnalysis(
            **payload.model_dump(),
            prompt_version=INTENT_PROMPT_VERSION,
            trace=_trace(result),
        )

    async def judge(
        self,
        *,
        axis: JudgeAxis,
        sample_id: str,
        request_snapshot: RequestSnapshot,
        target_response: TargetResponse,
        intent: IntentAnalysis,
        context: InvocationContext,
        parameters: Mapping[str, JsonValue] | None = None,
    ) -> JudgeVerdict:
        payload_schema: type[_EnablementPayload] | type[_VerdictPayload]
        payload_schema = (
            _EnablementPayload if axis is JudgeAxis.HARM_ENABLEMENT else _VerdictPayload
        )
        result = await self._invoke(
            request_id=f"judge:{axis.value}:{sample_id}",
            prompt=panel_prompt(
                axis=axis,
                sample_id=sample_id,
                request_snapshot=request_snapshot,
                target_response=target_response,
                intent=intent,
            ),
            context=context,
            parameters=parameters,
            payload_schema=payload_schema,
            evidence_text=target_response.text,
        )
        try:
            if axis is JudgeAxis.HARM_ENABLEMENT:
                enablement_payload = self._validate_payload(
                    _EnablementPayload, result.response.answer
                )
                scores = EnablementScores(
                    specificity=enablement_payload.specificity,
                    completeness=enablement_payload.completeness,
                    feasibility=enablement_payload.feasibility,
                    evasion_help=enablement_payload.evasion_help,
                )
                return JudgeVerdict(
                    axis=axis,
                    label=scores.derived_label,
                    confidence=enablement_payload.confidence,
                    reason_codes=enablement_payload.reason_codes,
                    evidence=_verified_evidence(
                        enablement_payload.evidence,
                        request_snapshot=request_snapshot,
                        target_response=target_response,
                    ),
                    enablement_scores=scores,
                    prompt_version=PANEL_PROMPT_VERSION,
                    trace=_trace(result),
                )
            payload = self._validate_payload(_VerdictPayload, result.response.answer)
            return JudgeVerdict(
                axis=axis,
                label=payload.label,
                confidence=payload.confidence,
                reason_codes=payload.reason_codes,
                evidence=_verified_evidence(
                    payload.evidence,
                    request_snapshot=request_snapshot,
                    target_response=target_response,
                ),
                prompt_version=PANEL_PROMPT_VERSION,
                trace=_trace(result),
            )
        except ValidationError as error:
            raise ContractValidationError(
                f"judge returned an invalid {axis.value} label: {error}"
            ) from error

    async def arbitrate(
        self,
        *,
        sample_id: str,
        request_snapshot: RequestSnapshot,
        target_response: TargetResponse,
        intent: IntentAnalysis,
        verdicts: tuple[JudgeVerdict, ...],
        conflict_codes: tuple[str, ...],
        context: InvocationContext,
        parameters: Mapping[str, JsonValue] | None = None,
    ) -> ArbitrationDecision:
        result = await self._invoke(
            request_id=f"judge:arbitration:{sample_id}",
            prompt=arbitration_prompt(
                sample_id=sample_id,
                request_snapshot=request_snapshot,
                target_response=target_response,
                intent=intent,
                verdicts=verdicts,
                conflict_codes=conflict_codes,
            ),
            context=context,
            parameters=parameters,
            payload_schema=_ArbitrationPayload,
            evidence_text=target_response.text,
        )
        payload = self._validate_payload(_ArbitrationPayload, result.response.answer)
        return ArbitrationDecision(
            response_compliance_level=payload.response_compliance_level,
            confidence=payload.confidence,
            reason_codes=payload.reason_codes,
            evidence=_verified_evidence(
                payload.evidence,
                request_snapshot=request_snapshot,
                target_response=target_response,
            ),
            prompt_version=ARBITRATION_PROMPT_VERSION,
            trace=_trace(result),
        )

    async def _invoke(
        self,
        *,
        request_id: str,
        prompt: str,
        context: InvocationContext,
        parameters: Mapping[str, JsonValue] | None,
        payload_schema: type[ContractModel],
        evidence_text: str | None,
    ) -> InvocationResult:
        resolved_parameters = dict(parameters or {})
        if (
            self.invoker.provider.provider_name == "local-openai"
            and "response_format" not in resolved_parameters
        ):
            resolved_parameters["response_format"] = cast(
                JsonValue,
                _structured_response_format(payload_schema, evidence_text=evidence_text),
            )
        request = ModelRequest(
            request_id=request_id,
            role=ModelRole.JUDGE,
            parts=(ModelTextPart(text=prompt),),
            parameters=resolved_parameters,
        )
        if not self.invoker.provider.capabilities.supports(frozenset({InputModality.TEXT})):
            raise UnsupportedModalityError(
                f"judge model {self.invoker.provider.model_id!r} does not support text input"
            )
        return await self.invoker.invoke(request, context=context)

    @staticmethod
    def _validate_payload[PayloadT: ContractModel](schema: type[PayloadT], raw: str) -> PayloadT:
        try:
            payload = _decode_json_object(raw)
            return schema.model_validate(payload)
        except (json.JSONDecodeError, ValidationError, ValueError) as error:
            raise ContractValidationError(
                f"judge response is not valid {schema.__name__} JSON: {error}"
            ) from error


def _decode_json_object(raw: str) -> dict[str, Any]:
    value = raw.strip()
    if value.startswith("```") and value.endswith("```"):
        lines = value.splitlines()
        if len(lines) >= 3:
            value = "\n".join(lines[1:-1]).strip()
    decoded = json.loads(value)
    if not isinstance(decoded, dict):
        raise ValueError("top-level judge response must be an object")
    return decoded


def _structured_response_format(
    payload_schema: type[ContractModel],
    *,
    evidence_text: str | None,
) -> dict[str, Any]:
    schema = payload_schema.model_json_schema()
    if evidence_text is not None:
        evidence_schema = schema["$defs"]["_EvidenceQuote"]["properties"]
        evidence_schema["source"] = {"const": "target_response", "type": "string"}
        evidence_schema["text"] = {
            "enum": list(_evidence_candidates(evidence_text)),
            "type": "string",
        }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": payload_schema.__name__.lstrip("_").lower(),
            "schema": schema,
            "strict": True,
        },
    }


def _evidence_candidates(text: str) -> tuple[str, ...]:
    candidates: list[str] = []
    if len(text) <= 500:
        candidates.append(text)
    for match in re.finditer(
        r"[^.!?\u3002\uFF01\uFF1F\n]+[.!?\u3002\uFF01\uFF1F]?",
        text,
    ):
        candidate = match.group(0).strip()
        if 0 < len(candidate) <= 500 and candidate not in candidates:
            candidates.append(candidate)
    for start in range(0, len(text), 400):
        candidate = text[start : start + 400]
        if candidate and candidate not in candidates:
            candidates.append(candidate)
    return tuple(candidates[:12])


def _trace(result: InvocationResult) -> JudgeCallTrace:
    response = result.response
    return JudgeCallTrace(
        call_id=result.call_id,
        provider_response_id=response.response_id,
        request_hash=response.request_hash,
        model=JudgeModelRef(
            provider=response.provider,
            model=response.model,
            revision=response.model_version,
        ),
        token_usage=response.token_usage,
        latency_ms=response.latency_ms,
        billed_cost_usd=result.billed_usd,
        cache_hit=result.cache_hit,
    )


def _verified_evidence(
    quotes: tuple[_EvidenceQuote, ...],
    *,
    request_snapshot: RequestSnapshot,
    target_response: TargetResponse,
) -> tuple[EvidenceRef, ...]:
    sources = {
        "request_snapshot": request_snapshot.content,
        "target_response": target_response.text,
    }
    verified: list[EvidenceRef] = []
    for quote in quotes:
        resolved_source = quote.source
        source_text = sources[resolved_source]
        start = source_text.find(quote.text)
        if start < 0:
            alternate_source: Literal["request_snapshot", "target_response"] = (
                "target_response" if resolved_source == "request_snapshot" else "request_snapshot"
            )
            alternate_text = sources[alternate_source]
            alternate_start = alternate_text.find(quote.text)
            if alternate_start < 0:
                raise ContractValidationError(
                    "judge evidence is not a verbatim span of either allowed source"
                )
            resolved_source = alternate_source
            source_text = alternate_text
            start = alternate_start
        verified.append(
            EvidenceRef(
                source=resolved_source,
                source_sha256=hashlib.sha256(source_text.encode("utf-8")).hexdigest(),
                start=start,
                end=start + len(quote.text),
                text=quote.text,
            )
        )
    return tuple(verified)
