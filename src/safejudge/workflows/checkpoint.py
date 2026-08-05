"""Development checkpoint configuration for M3 LangGraph runs."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import aiosqlite
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from safejudge.contracts.artifact import ArtifactRef
from safejudge.contracts.dataset import (
    CanonicalMultimodalSample,
    DatasetSplit,
    MediaPart,
    MediaRef,
    MediaType,
    RequestContext,
    RequestIntent,
    SourceRecord,
    TextPart,
)
from safejudge.contracts.evaluation import (
    ModelCost,
    PriceSnapshot,
    ResponseComplianceLevel,
    TargetModelRef,
    TargetResponse,
    TokenUsage,
)
from safejudge.contracts.judging import (
    AggregateDecision,
    ArbitrationDecision,
    EnablementScores,
    EvaluationResult,
    EvaluationSpec,
    EvidenceRef,
    IntentAnalysis,
    JudgeAxis,
    JudgeCallTrace,
    JudgeLabel,
    JudgeModelRef,
    JudgeVerdict,
    RequestSnapshot,
)
from safejudge.workflows.graph import (
    EvaluationInput,
    EvaluationOutput,
    EvaluationState,
    JudgeTaskInput,
    JudgeTaskOutput,
    JudgeTaskState,
)

_CHECKPOINT_TYPES = (
    ArtifactRef,
    CanonicalMultimodalSample,
    DatasetSplit,
    MediaPart,
    MediaRef,
    MediaType,
    RequestContext,
    RequestIntent,
    SourceRecord,
    TextPart,
    ModelCost,
    PriceSnapshot,
    ResponseComplianceLevel,
    TargetModelRef,
    TargetResponse,
    TokenUsage,
    AggregateDecision,
    ArbitrationDecision,
    EnablementScores,
    EvaluationSpec,
    EvaluationResult,
    EvidenceRef,
    IntentAnalysis,
    JudgeAxis,
    JudgeCallTrace,
    JudgeLabel,
    JudgeModelRef,
    JudgeVerdict,
    RequestSnapshot,
    EvaluationInput,
    EvaluationOutput,
    EvaluationState,
    JudgeTaskInput,
    JudgeTaskOutput,
    JudgeTaskState,
)


@asynccontextmanager
async def sqlite_checkpointer(path: Path) -> AsyncIterator[AsyncSqliteSaver]:
    """Open a development-only SQLite checkpointer and close it reliably."""

    resolved = path.resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    serializer = JsonPlusSerializer(allowed_msgpack_modules=_CHECKPOINT_TYPES)
    async with aiosqlite.connect(str(resolved)) as connection:
        saver = AsyncSqliteSaver(connection, serde=serializer)
        yield saver
