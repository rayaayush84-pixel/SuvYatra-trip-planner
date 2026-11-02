from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """All configuration comes from environment variables or the .env file."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # App
    app_name: str = "Nepal Travel Bot API"
    app_version: str = "1.0.0"
    environment: str = "development"  # development | production
    log_level: str = "INFO"
    cors_origins: str = "*"  # comma separated, e.g. https://mysite.com,http://localhost:3000
    suvyatra_api_url: str | None = None

    # Security
    admin_api_key: str | None = None  # protects /api/v1/ingest/* via X-API-Key header

    # Gemini
    google_api_key: str
    chat_model: str = "gemini-3.6-flash"
    embed_model: str = "models/gemini-embedding-001"
    temperature: float = 0.3
    max_output_tokens: int = 1000
    thinking_budget: int = 0  # 0 = no thinking (fastest). -1 = don't send this setting

    # RAG / ChromaDB
    chroma_dir: str = str(PROJECT_ROOT / "chroma_db")
    collection_name: str = "nepal_travel"
    data_dir: str = str(PROJECT_ROOT / "data")
    top_k: int = 3
    chunk_size: int = 800
    chunk_overlap: int = 100
    auto_ingest: bool = True  # load ./data on startup when the database is empty

    @field_validator("chroma_dir", "data_dir", mode="after")
    @classmethod
    def resolve_project_paths(cls, value: str) -> str:
        path = Path(value)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return str(path.resolve())

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()