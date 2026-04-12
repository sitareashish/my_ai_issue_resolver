"""Central configuration loaded from .env — imported as `settings` everywhere."""
from __future__ import annotations

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ── LLM ───────────────────────────────────────────────────────────────────
    anthropic_api_key: str
    anthropic_model: str = "claude-sonnet-4-20250514"

    # ── Embeddings ────────────────────────────────────────────────────────────
    openai_api_key: str
    embedding_model: str = "text-embedding-3-large"
    embedding_dimensions: int = 1536

    # ── GitHub ────────────────────────────────────────────────────────────────
    github_token: str
    github_repo: str

    # ── Qdrant ────────────────────────────────────────────────────────────────
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "codebase"
    qdrant_api_key: str = ""

    # ── LangSmith ─────────────────────────────────────────────────────────────
    langchain_tracing_v2: bool = True
    langchain_api_key: str = ""
    langchain_project: str = "github-issue-resolver"

    # ── Pipeline ──────────────────────────────────────────────────────────────
    max_retries: int = 5
    context_top_k: int = 15
    context_rerank_top_k: int = 5
    confidence_threshold: float = 0.75
    max_tokens_per_issue: int = 150_000
    token_warn_threshold: int = 80_000

    # ── Hybrid search ─────────────────────────────────────────────────────────
    bm25_weight: float = 0.3           # weight of BM25 vs dense (0.7)

    # ── Sandbox ───────────────────────────────────────────────────────────────
    sandbox_memory_limit: str = "2g"
    sandbox_cpu_count: int = 2
    sandbox_timeout_seconds: int = 120
    sandbox_pool_size: int = 2         # pre-warmed containers

    # ── API server ────────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8080
    api_workers: int = 1

    # ── Observability ─────────────────────────────────────────────────────────
    metrics_enabled: bool = True
    checkpoint_db: str = "/tmp/resolver_checkpoints.db"

    # ── Validators ────────────────────────────────────────────────────────────
    @field_validator("max_retries")
    @classmethod
    def retries_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("max_retries must be ≥ 1")
        return v

    @field_validator("confidence_threshold")
    @classmethod
    def confidence_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")
        return v

    @model_validator(mode="after")
    def warn_threshold_check(self) -> "Settings":
        if self.token_warn_threshold >= self.max_tokens_per_issue:
            raise ValueError("token_warn_threshold must be less than max_tokens_per_issue")
        return self


settings = Settings()  # type: ignore[call-arg]
