"""The stable provider boundary consumed by runners and future graphs."""

from __future__ import annotations

from typing import Protocol

from safejudge.contracts.evaluation import PriceSnapshot
from safejudge.contracts.model import ModelCapabilities, ModelRequest, ModelResponse


class ModelProvider(Protocol):
    @property
    def provider_name(self) -> str: ...

    @property
    def model_id(self) -> str: ...

    @property
    def capabilities(self) -> ModelCapabilities: ...

    @property
    def price_snapshot(self) -> PriceSnapshot: ...

    async def generate(self, request: ModelRequest, *, request_hash: str) -> ModelResponse: ...
