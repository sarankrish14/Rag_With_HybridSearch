"""Validated application settings loaded from environment / .env only."""

from __future__ import annotations

from functools import lru_cache
from urllib.parse import quote_plus

from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Secrets and connection settings — missing required values fail at import."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    groq_api_key: str = Field(validation_alias="GROQ_API_KEY")
    api_key: str = Field(validation_alias=AliasChoices("RAG_API_KEY", "API_KEY"))

    postgres_host: str = Field(default="", validation_alias="POSTGRES_HOST")
    postgres_port: str = Field(default="5432", validation_alias="POSTGRES_PORT")
    postgres_db: str = Field(default="", validation_alias="POSTGRES_DB")
    postgres_user: str = Field(default="", validation_alias="POSTGRES_USER")
    postgres_password: str = Field(default="", validation_alias="POSTGRES_PASSWORD")
    database_url: str = Field(default="", validation_alias="DATABASE_URL")

    redis_host: str = Field(default="localhost", validation_alias="RAG_REDIS_HOST")
    redis_port: int = Field(default=6379, validation_alias="RAG_REDIS_PORT")

    @model_validator(mode="after")
    def _resolve_database_url(self) -> Settings:
        if self.database_url and "${" not in self.database_url:
            return self
        if self.postgres_host and self.postgres_db and self.postgres_user:
            object.__setattr__(
                self,
                "database_url",
                (
                    f"postgresql://{quote_plus(self.postgres_user)}:"
                    f"{quote_plus(self.postgres_password)}"
                    f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
                ),
            )
        return self

    @property
    def resolved_database_url(self) -> str:
        if not self.database_url:
            raise ValueError(
                "DATABASE_URL is not set. Configure DATABASE_URL or POSTGRES_* in .env."
            )
        return self.database_url


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
