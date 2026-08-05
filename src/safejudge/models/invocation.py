"""Cache, retry, accounting, and budget policy around any model provider."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from decimal import Decimal
from uuid import uuid4

from safejudge.contracts.model import InvocationContext, ModelRequest, ModelResponse
from safejudge.core.errors import BudgetExceededError, ProviderError
from safejudge.models.base import ModelProvider
from safejudge.models.cache import SQLiteModelStore, stable_request_hash

Sleeper = Callable[[float], Awaitable[None]]
BudgetKey = tuple[str, str, str]


@dataclass(frozen=True, slots=True)
class InvocationResult:
    call_id: str
    response: ModelResponse
    cache_hit: bool
    attempts: int
    billed_usd: Decimal


@dataclass(frozen=True, slots=True)
class InvocationPolicy:
    max_retries: int = 0
    retry_backoff_seconds: float = 0
    max_retry_delay_seconds: float = 5
    max_concurrency: int = 4
    max_total_cost_usd: Decimal | None = None
    uncached_call_reservation_usd: Decimal = Decimal("0")

    def __post_init__(self) -> None:
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if self.retry_backoff_seconds < 0:
            raise ValueError("retry_backoff_seconds must be non-negative")
        if self.max_retry_delay_seconds < 0:
            raise ValueError("max_retry_delay_seconds must be non-negative")
        if self.max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1")
        if self.max_total_cost_usd is not None and self.max_total_cost_usd < 0:
            raise ValueError("max_total_cost_usd must be non-negative")
        if self.uncached_call_reservation_usd < 0:
            raise ValueError("uncached_call_reservation_usd must be non-negative")


class ModelInvoker:
    def __init__(
        self,
        *,
        provider: ModelProvider,
        store: SQLiteModelStore,
        policy: InvocationPolicy | None = None,
        sleeper: Sleeper = asyncio.sleep,
    ) -> None:
        self.provider = provider
        self.store = store
        self.policy = policy or InvocationPolicy()
        self._sleeper = sleeper
        self._budget_lock = asyncio.Lock()
        self._reserved_budget_usd: dict[BudgetKey, Decimal] = {}
        self._concurrency = asyncio.Semaphore(self.policy.max_concurrency)
        self._singleflight = _SingleflightLocks()

    async def invoke(
        self,
        request: ModelRequest,
        *,
        context: InvocationContext,
    ) -> InvocationResult:
        cache_key = stable_request_hash(
            provider=self.provider.provider_name,
            model=self.provider.model_id,
            request=request,
        )
        call_id = f"call_{uuid4().hex}"
        cached = self.store.get(cache_key)
        if cached is not None:
            self.store.record_success(
                call_id=call_id,
                cache_key=cache_key,
                context=context,
                request=request,
                response=cached,
                cache_hit=True,
                attempts=0,
                billed_usd=Decimal("0"),
            )
            return InvocationResult(
                call_id=call_id,
                response=cached,
                cache_hit=True,
                attempts=0,
                billed_usd=Decimal("0"),
            )

        async with self._singleflight.hold(cache_key):
            # Another task may have populated the same key while this task waited.
            cached = self.store.get(cache_key)
            if cached is not None:
                self.store.record_success(
                    call_id=call_id,
                    cache_key=cache_key,
                    context=context,
                    request=request,
                    response=cached,
                    cache_hit=True,
                    attempts=0,
                    billed_usd=Decimal("0"),
                )
                return InvocationResult(
                    call_id=call_id,
                    response=cached,
                    cache_hit=True,
                    attempts=0,
                    billed_usd=Decimal("0"),
                )
            reservation = await self._reserve_budget(context=context, request=request)
            try:
                attempts = 0
                while True:
                    attempts += 1
                    try:
                        async with self._concurrency:
                            response = await self.provider.generate(
                                request,
                                request_hash=cache_key,
                            )
                        self.store.record_attempt(
                            call_id=call_id,
                            attempt_number=attempts,
                            response=response,
                        )
                        break
                    except ProviderError as error:
                        self.store.record_attempt(
                            call_id=call_id,
                            attempt_number=attempts,
                            error_kind=error.kind,
                            error_message=str(error),
                            raw_artifact=error.raw_artifact,
                        )
                        if not error.retryable or attempts > self.policy.max_retries:
                            self.store.record_failure(
                                call_id=call_id,
                                cache_key=cache_key,
                                context=context,
                                request=request,
                                provider=self.provider.provider_name,
                                model=self.provider.model_id,
                                attempts=attempts,
                                error_kind=error.kind,
                                error_message=str(error),
                                raw_artifact=error.raw_artifact,
                            )
                            raise
                        delay = error.retry_after_seconds
                        if delay is None:
                            delay = self.policy.retry_backoff_seconds * (2 ** (attempts - 1))
                        await self._sleeper(min(delay, self.policy.max_retry_delay_seconds))

                self.store.put(cache_key, response)
                billed = _billed_cost(response)
                self.store.record_success(
                    call_id=call_id,
                    cache_key=cache_key,
                    context=context,
                    request=request,
                    response=response,
                    cache_hit=False,
                    attempts=attempts,
                    billed_usd=billed,
                )
                return InvocationResult(
                    call_id=call_id,
                    response=response,
                    cache_hit=False,
                    attempts=attempts,
                    billed_usd=billed,
                )
            finally:
                await self._release_budget(reservation)

    async def _reserve_budget(
        self,
        *,
        context: InvocationContext,
        request: ModelRequest,
    ) -> tuple[BudgetKey, Decimal] | None:
        maximum = self.policy.max_total_cost_usd
        if maximum is None:
            return None
        reservation = self.policy.uncached_call_reservation_usd
        budget_key = (context.experiment_id, context.run_id, request.role.value)
        async with self._budget_lock:
            projected = (
                self.store.total_billed_usd(context=context, role=request.role)
                + self._reserved_budget_usd.get(budget_key, Decimal("0"))
                + reservation
            )
            if projected > maximum:
                raise BudgetExceededError(
                    f"model call blocked: projected cost USD {projected} "
                    f"exceeds budget USD {maximum}"
                )
            self._reserved_budget_usd[budget_key] = (
                self._reserved_budget_usd.get(budget_key, Decimal("0")) + reservation
            )
        return budget_key, reservation

    async def _release_budget(
        self,
        reservation: tuple[BudgetKey, Decimal] | None,
    ) -> None:
        if reservation is None:
            return
        budget_key, amount = reservation
        async with self._budget_lock:
            remaining = self._reserved_budget_usd[budget_key] - amount
            if remaining == 0:
                self._reserved_budget_usd.pop(budget_key)
            else:
                self._reserved_budget_usd[budget_key] = remaining


@dataclass(slots=True)
class _LockEntry:
    lock: asyncio.Lock
    users: int = 0


class _SingleflightLocks:
    """Serialize identical cache keys while allowing different keys to run concurrently."""

    def __init__(self) -> None:
        self._guard = asyncio.Lock()
        self._entries: dict[str, _LockEntry] = {}

    @asynccontextmanager
    async def hold(self, key: str) -> AsyncIterator[None]:
        async with self._guard:
            entry = self._entries.setdefault(key, _LockEntry(lock=asyncio.Lock()))
            entry.users += 1
        try:
            await entry.lock.acquire()
        except BaseException:
            await self._drop_user(key, entry)
            raise
        try:
            yield
        finally:
            entry.lock.release()
            await self._drop_user(key, entry)

    async def _drop_user(self, key: str, entry: _LockEntry) -> None:
        async with self._guard:
            entry.users -= 1
            if entry.users == 0:
                self._entries.pop(key, None)


def _billed_cost(response: ModelResponse) -> Decimal:
    if response.cost is None:
        return Decimal("0")
    if response.cost.actual_usd is not None:
        return response.cost.actual_usd
    return response.cost.estimated_usd
