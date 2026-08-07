"""Stable error taxonomy shared across adapters, providers, and workflows."""

from __future__ import annotations

from enum import StrEnum

from safejudge.contracts.artifact import ArtifactRef


class SafeJudgeError(Exception):
    """Base class for expected application failures."""


class ConfigurationError(SafeJudgeError):
    """Configuration is missing, inconsistent, or unsafe."""


class ContractValidationError(SafeJudgeError):
    """External data does not satisfy a versioned project contract."""


class JudgeContractError(ContractValidationError):
    """A judge exhausted contract repairs; carries the last auditable response."""

    def __init__(
        self,
        message: str,
        *,
        request_id: str,
        call_id: str,
        provider_response_id: str,
        raw_artifact: ArtifactRef | None,
    ) -> None:
        super().__init__(message)
        self.request_id = request_id
        self.call_id = call_id
        self.provider_response_id = provider_response_id
        self.raw_artifact = raw_artifact


class AdapterError(SafeJudgeError):
    """A dataset adapter could not convert a source record."""


class ArtifactError(SafeJudgeError):
    """A raw provider response could not be persisted safely."""


class ProviderErrorKind(StrEnum):
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    UNAVAILABLE = "unavailable"
    AUTHENTICATION = "authentication"
    INSUFFICIENT_CREDITS = "insufficient_credits"
    INVALID_REQUEST = "invalid_request"
    CONTENT_POLICY = "content_policy"
    REASONING_ONLY = "reasoning_only"
    EMPTY_RESPONSE = "empty_response"
    TRUNCATED_RESPONSE = "truncated_response"
    MALFORMED_RESPONSE = "malformed_response"
    ARTIFACT_PERSISTENCE = "artifact_persistence"
    UNKNOWN = "unknown"


class ProviderError(SafeJudgeError):
    """A model provider call failed in a classified way."""

    def __init__(
        self,
        message: str,
        *,
        retryable: bool,
        kind: ProviderErrorKind = ProviderErrorKind.UNKNOWN,
        status_code: int | None = None,
        retry_after_seconds: float | None = None,
        raw_artifact: ArtifactRef | None = None,
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.kind = kind
        self.status_code = status_code
        self.retry_after_seconds = retry_after_seconds
        self.raw_artifact = raw_artifact


class UnsupportedModalityError(SafeJudgeError):
    """The configured model cannot accept a sample's exact modality combination."""


class BudgetExceededError(SafeJudgeError):
    """A model call was blocked before spending beyond its configured budget."""
