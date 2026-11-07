"""Configuration from the environment (12-factor), validated by pydantic-settings."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """INDEX_PATH=... LOG_LEVEL=... (a .env file is read too — for local development only)."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    index_path: Path = Path("index.bin")
    embeddings_path: Path | None = None  # EMBEDDINGS_PATH: enables mode=semantic|hybrid
    log_level: str = "INFO"
