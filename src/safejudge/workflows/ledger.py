"""Append-only, content-safe execution ledger for LangGraph business nodes."""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
from collections.abc import AsyncIterator, Iterator, Mapping
from contextlib import AbstractAsyncContextManager, asynccontextmanager, contextmanager
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from time import monotonic
from typing import Protocol, runtime_checkable
from uuid import uuid4

from pydantic import BaseModel

from safejudge.core.time import utc_now


class NodeRunSpan(Protocol):
    def set_output(self, output: object) -> None: ...


@runtime_checkable
class NodeLedger(Protocol):
    def track(
        self,
        *,
        experiment_id: str,
        run_id: str,
        thread_id: str,
        sample_id: str,
        evaluation_key: str | None,
        node_name: str,
        input_value: object,
    ) -> AbstractAsyncContextManager[NodeRunSpan]: ...


@dataclass(slots=True)
class _Span:
    output: object | None = None

    def set_output(self, output: object) -> None:
        self.output = output


class SQLiteNodeLedger:
    """Stores hashes and metadata only; raw node inputs and outputs stay in checkpoints."""

    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
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
                CREATE TABLE IF NOT EXISTS node_runs (
                    node_run_id TEXT PRIMARY KEY,
                    experiment_id TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    thread_id TEXT NOT NULL,
                    sample_id TEXT NOT NULL,
                    evaluation_key TEXT,
                    node_name TEXT NOT NULL,
                    attempt INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    input_hash TEXT NOT NULL,
                    output_hash TEXT,
                    model_call_ids_json TEXT NOT NULL,
                    error_kind TEXT,
                    error_message_hash TEXT,
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    latency_ms INTEGER
                );
                CREATE INDEX IF NOT EXISTS ix_node_runs_thread_node
                    ON node_runs(thread_id, node_name, attempt);
                """
            )

    @asynccontextmanager
    async def track(
        self,
        *,
        experiment_id: str,
        run_id: str,
        thread_id: str,
        sample_id: str,
        evaluation_key: str | None,
        node_name: str,
        input_value: object,
    ) -> AsyncIterator[_Span]:
        node_run_id = f"node_{uuid4().hex}"
        started_at = utc_now()
        started = monotonic()
        input_hash = _content_hash(input_value)
        async with self._lock:
            with self._connect() as connection:
                attempt = int(
                    connection.execute(
                        """
                        SELECT COUNT(*) + 1 FROM node_runs
                        WHERE thread_id = ? AND node_name = ?
                        """,
                        (thread_id, node_name),
                    ).fetchone()[0]
                )
                connection.execute(
                    """
                    INSERT INTO node_runs (
                        node_run_id, experiment_id, run_id, thread_id, sample_id,
                        evaluation_key, node_name, attempt, status, input_hash,
                        model_call_ids_json, started_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'running', ?, '[]', ?)
                    """,
                    (
                        node_run_id,
                        experiment_id,
                        run_id,
                        thread_id,
                        sample_id,
                        evaluation_key,
                        node_name,
                        attempt,
                        input_hash,
                        started_at.isoformat(),
                    ),
                )
        span = _Span()
        try:
            yield span
        except BaseException as error:
            await self._finish(
                node_run_id=node_run_id,
                started=started,
                status="failed",
                output=None,
                error=error,
            )
            raise
        else:
            await self._finish(
                node_run_id=node_run_id,
                started=started,
                status="success",
                output=span.output,
                error=None,
            )

    async def _finish(
        self,
        *,
        node_run_id: str,
        started: float,
        status: str,
        output: object | None,
        error: BaseException | None,
    ) -> None:
        finished_at = utc_now()
        output_hash = _content_hash(output) if output is not None else None
        call_ids = sorted(_collect_call_ids(_jsonable(output))) if output is not None else []
        error_kind = type(error).__name__ if error is not None else None
        error_message_hash = (
            hashlib.sha256(str(error).encode("utf-8")).hexdigest() if error is not None else None
        )
        async with self._lock:
            with self._connect() as connection:
                connection.execute(
                    """
                    UPDATE node_runs SET
                        status = ?, output_hash = ?, model_call_ids_json = ?,
                        error_kind = ?, error_message_hash = ?, finished_at = ?,
                        latency_ms = ?
                    WHERE node_run_id = ?
                    """,
                    (
                        status,
                        output_hash,
                        json.dumps(call_ids, separators=(",", ":")),
                        error_kind,
                        error_message_hash,
                        finished_at.isoformat(),
                        max(0, round((monotonic() - started) * 1000)),
                        node_run_id,
                    ),
                )


def _content_hash(value: object) -> str:
    canonical = json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _jsonable(value: object) -> object:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _collect_call_ids(value: object) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key == "call_id" and isinstance(item, str):
                found.add(item)
            found.update(_collect_call_ids(item))
    elif isinstance(value, list):
        for item in value:
            found.update(_collect_call_ids(item))
    return found
