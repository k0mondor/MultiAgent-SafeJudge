"""Application configuration loaded from explicit ``SAFEJUDGE_`` variables."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(StrEnum):
    DEVELOPMENT = "development"
    TEST = "test"
    PRODUCTION = "production"


class Settings(BaseSettings):
    """Non-secret application settings.

    Provider credentials will be represented by separate provider-specific settings so
    they cannot accidentally be serialized with this object.
    """

    model_config = SettingsConfigDict(
        env_prefix="SAFEJUDGE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    environment: Environment = Environment.DEVELOPMENT
    log_level: str = "INFO"
    data_dir: Path = Path("data")
    artifact_dir: Path = Path("artifacts")
    max_concurrency: int = Field(default=4, ge=1, le=64)
