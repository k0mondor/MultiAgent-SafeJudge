"""Stable request hashing plus a small persistent SQLite call/cache store."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from safejudge.contracts.artifact import ArtifactRef
from safejudge.contracts.model import InvocationContext, ModelRequest, ModelResponse, ModelRole
from safejudge.core.errors import ProviderErrorKind


@dataclass(frozen=True, slots=True)
class ModelRunStats:
    logical_call_count: int
    provider_attempt_count: int
    contract_repair_call_count: int
    successful_call_count: int
    failed_call_count: int
    cache_hit_count: int
    cache_miss_count: int
    billed_cost_usd: Decimal


def stable_request_hash(*, provider: str, model: str, request: ModelRequest) -> str:
    """Hash semantic input only; run-specific request IDs intentionally do not participate."""

    payload = {
        "schema_version": request.schema_version,
        "role": request.role.value,
        "provider": provider,
        "model": model,
        # New optional transport hints must not invalidate historical requests when absent.
        "parts": [
            part.model_dump(mode="json", exclude_none=True)
            for part in request.parts
        ],
        "parameters": request.parameters,
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


class SQLiteModelStore:
    """Stores successful responses separately from the append-only invocation ledger."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.execute("PRAGMA journal_mode=WAL")
        try:
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS model_cache (
                    cache_key TEXT PRIMARY KEY,
                    response_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS model_calls (
                    call_id TEXT PRIMARY KEY,
                    request_id TEXT NOT NULL,
                    cache_key TEXT NOT NULL,
                    experiment_id TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    status TEXT NOT NULL,
                    cache_hit INTEGER NOT NULL,
                    attempts INTEGER NOT NULL,
                    estimated_usd TEXT NOT NULL,
                    actual_usd TEXT,
                    billed_usd TEXT NOT NULL,
                    response_id TEXT,
                    raw_artifact_uri TEXT,
                    raw_artifact_sha256 TEXT,
                    raw_artifact_content_type TEXT,
                    raw_artifact_size_bytes INTEGER,
                    error_kind TEXT,
                    error_message TEXT,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS ix_model_calls_cache_key
                    ON model_calls(cache_key);
                CREATE TABLE IF NOT EXISTS model_attempts (
                    attempt_id TEXT PRIMARY KEY,
                    call_id TEXT NOT NULL,
                    attempt_number INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    response_id TEXT,
                    error_kind TEXT,
                    error_message TEXT,
                    raw_artifact_uri TEXT,
                    raw_artifact_sha256 TEXT,
                    raw_artifact_content_type TEXT,
                    raw_artifact_size_bytes INTEGER,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS ix_model_attempts_call_id
                    ON model_attempts(call_id);
                """
            )
            columns = {
                str(row[1]) for row in connection.execute("PRAGMA table_info(model_calls)")
            }
            if "experiment_id" not in columns:
                connection.execute(
                    "ALTER TABLE model_calls ADD COLUMN experiment_id "
                    "TEXT NOT NULL DEFAULT 'legacy'"
                )
            if "run_id" not in columns:
                connection.execute(
                    "ALTER TABLE model_calls ADD COLUMN run_id TEXT NOT NULL DEFAULT 'legacy'"
                )
            if "request_id" not in columns:
                connection.execute(
                    "ALTER TABLE model_calls ADD COLUMN request_id "
                    "TEXT NOT NULL DEFAULT 'legacy'"
                )
            artifact_columns = {
                "raw_artifact_uri": "TEXT",
                "raw_artifact_sha256": "TEXT",
                "raw_artifact_content_type": "TEXT",
                "raw_artifact_size_bytes": "INTEGER",
            }
            for column, sql_type in artifact_columns.items():
                if column not in columns:
                    connection.execute(
                        f"ALTER TABLE model_calls ADD COLUMN {column} {sql_type}"
                    )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS ix_model_calls_budget_scope
                ON model_calls(experiment_id, run_id, role)
                """
            )

    def get(self, cache_key: str) -> ModelResponse | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT response_json FROM model_cache WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
        if row is None:
            return None
        return ModelResponse.model_validate_json(row[0])

    def put(self, cache_key: str, response: ModelResponse) -> None:
        now = datetime.now(UTC).isoformat()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO model_cache(cache_key, response_json, created_at)
                VALUES (?, ?, ?)
                ON CONFLICT(cache_key) DO NOTHING
                """,
                (cache_key, response.model_dump_json(), now),
            )

    def record_success(
        self,
        *,
        call_id: str,
        cache_key: str,
        context: InvocationContext,
        request: ModelRequest,
        response: ModelResponse,
        cache_hit: bool,
        attempts: int,
        billed_usd: Decimal,
    ) -> None:
        estimated, actual = _response_cost(response)
        self._record(
            call_id=call_id,
            cache_key=cache_key,
            context=context,
            request=request,
            provider=response.provider,
            model=response.model,
            status="success",
            cache_hit=cache_hit,
            attempts=attempts,
            estimated_usd=estimated,
            actual_usd=actual,
            billed_usd=billed_usd,
            response_id=response.response_id,
            raw_artifact=response.raw_artifact,
            error_kind=None,
            error_message=None,
        )

    def record_attempt(
        self,
        *,
        call_id: str,
        attempt_number: int,
        response: ModelResponse | None = None,
        error_kind: ProviderErrorKind | None = None,
        error_message: str | None = None,
        raw_artifact: ArtifactRef | None = None,
    ) -> None:
        if (response is None) == (error_kind is None):
            raise ValueError("attempt must contain exactly one response or error")
        artifact = response.raw_artifact if response is not None else raw_artifact
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO model_attempts(
                    attempt_id, call_id, attempt_number, status, response_id,
                    error_kind, error_message, raw_artifact_uri,
                    raw_artifact_sha256, raw_artifact_content_type,
                    raw_artifact_size_bytes, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    f"{call_id}:{attempt_number}",
                    call_id,
                    attempt_number,
                    "success" if response is not None else "failed",
                    response.response_id if response is not None else None,
                    error_kind.value if error_kind is not None else None,
                    error_message,
                    artifact.uri if artifact is not None else None,
                    artifact.sha256 if artifact is not None else None,
                    artifact.content_type if artifact is not None else None,
                    artifact.size_bytes if artifact is not None else None,
                    datetime.now(UTC).isoformat(),
                ),
            )

    def record_failure(
        self,
        *,
        call_id: str,
        cache_key: str,
        context: InvocationContext,
        request: ModelRequest,
        provider: str,
        model: str,
        attempts: int,
        error_kind: ProviderErrorKind,
        error_message: str,
        raw_artifact: ArtifactRef | None,
    ) -> None:
        self._record(
            call_id=call_id,
            cache_key=cache_key,
            context=context,
            request=request,
            provider=provider,
            model=model,
            status="failed",
            cache_hit=False,
            attempts=attempts,
            estimated_usd=Decimal("0"),
            actual_usd=None,
            billed_usd=Decimal("0"),
            response_id=None,
            raw_artifact=raw_artifact,
            error_kind=error_kind.value,
            error_message=error_message,
        )

    def _record(
        self,
        *,
        call_id: str,
        cache_key: str,
        context: InvocationContext,
        request: ModelRequest,
        provider: str,
        model: str,
        status: str,
        cache_hit: bool,
        attempts: int,
        estimated_usd: Decimal,
        actual_usd: Decimal | None,
        billed_usd: Decimal,
        response_id: str | None,
        raw_artifact: ArtifactRef | None,
        error_kind: str | None,
        error_message: str | None,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO model_calls(
                    call_id, request_id, cache_key, experiment_id, run_id, role,
                    provider, model, status, cache_hit,
                    attempts, estimated_usd, actual_usd, billed_usd, response_id,
                    raw_artifact_uri, raw_artifact_sha256,
                    raw_artifact_content_type, raw_artifact_size_bytes,
                    error_kind, error_message, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    call_id,
                    request.request_id,
                    cache_key,
                    context.experiment_id,
                    context.run_id,
                    request.role.value,
                    provider,
                    model,
                    status,
                    int(cache_hit),
                    attempts,
                    str(estimated_usd),
                    str(actual_usd) if actual_usd is not None else None,
                    str(billed_usd),
                    response_id,
                    raw_artifact.uri if raw_artifact is not None else None,
                    raw_artifact.sha256 if raw_artifact is not None else None,
                    raw_artifact.content_type if raw_artifact is not None else None,
                    raw_artifact.size_bytes if raw_artifact is not None else None,
                    error_kind,
                    error_message,
                    datetime.now(UTC).isoformat(),
                ),
            )

    def total_billed_usd(
        self,
        *,
        context: InvocationContext,
        role: ModelRole,
    ) -> Decimal:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT billed_usd FROM model_calls
                WHERE experiment_id = ? AND run_id = ? AND role = ?
                """,
                (context.experiment_id, context.run_id, role.value),
            ).fetchall()
        return sum((Decimal(row[0]) for row in rows), start=Decimal("0"))

    def call_count(self, *, cache_hit: bool | None = None) -> int:
        query = "SELECT COUNT(*) FROM model_calls"
        parameters: tuple[int, ...] = ()
        if cache_hit is not None:
            query += " WHERE cache_hit = ?"
            parameters = (int(cache_hit),)
        with self._connect() as connection:
            row = connection.execute(query, parameters).fetchone()
        assert row is not None
        return int(row[0])

    def run_stats(
        self,
        *,
        context: InvocationContext,
        role: ModelRole,
    ) -> ModelRunStats:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT request_id, status, cache_hit, attempts, billed_usd
                FROM model_calls
                WHERE experiment_id = ? AND run_id = ? AND role = ?
                """,
                (context.experiment_id, context.run_id, role.value),
            ).fetchall()
        return ModelRunStats(
            logical_call_count=len(rows),
            provider_attempt_count=sum(int(row[3]) for row in rows),
            contract_repair_call_count=sum(":repair:" in str(row[0]) for row in rows),
            successful_call_count=sum(row[1] == "success" for row in rows),
            failed_call_count=sum(row[1] == "failed" for row in rows),
            cache_hit_count=sum(bool(row[2]) for row in rows),
            cache_miss_count=sum(not bool(row[2]) for row in rows),
            billed_cost_usd=sum(
                (Decimal(str(row[4])) for row in rows),
                start=Decimal("0"),
            ),
        )


def _response_cost(response: ModelResponse) -> tuple[Decimal, Decimal | None]:
    if response.cost is None:
        return Decimal("0"), None
    return response.cost.estimated_usd, response.cost.actual_usd
