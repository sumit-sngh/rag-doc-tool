"""Environment-driven configuration.

Every knob the brief calls out (API keys, model choice, chunk size/overlap) is an
env var with a sane default, so the app boots with zero configuration and is
tuned without code changes. Values are read once and cached; call
``get_settings.cache_clear()`` in tests to reload.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repo root: .../app/config.py -> .../
BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---- App ------------------------------------------------------------
    app_name: str = "Document Q&A Tool"
    app_env: Literal["dev", "prod", "test"] = "dev"
    log_level: str = "INFO"
    cors_origins: str = "*"

    # Auth is opt-in: leave API_KEY unset and the API is open (fine for local
    # single-user dev). Set it and every route except /health and the frontend
    # requires an `X-API-Key` header.
    api_key: str | None = None

    # ---- Multi-visitor isolation ----------------------------------------
    # Off by default, because a local single-user tool has no reason to partition
    # its own library. Turned on, every document and session is scoped to an
    # opaque id carried in an HttpOnly cookie, so two people hitting the same
    # public URL cannot see, query, or delete each other's uploads.
    #
    # This is per-browser isolation, not authentication. Clearing cookies loses
    # access to your own documents, and anyone who copies the cookie value gets
    # that library. It is the right model for an open demo and the wrong one for
    # anything confidential -- that needs real accounts.
    visitor_isolation: bool = False
    visitor_cookie_name: str = "qa_visitor"
    visitor_cookie_days: int = Field(default=180, ge=1)
    # Owner recorded when isolation is off, and backfilled onto rows that predate
    # it. Existing libraries stay visible in single-user mode rather than
    # vanishing behind a scope they were never tagged with.
    shared_owner: str = "shared"
    # A public deployment accepts uploads from strangers. 0 means unlimited,
    # which is correct locally and reckless on an open URL.
    max_documents_per_visitor: int = Field(default=0, ge=0)

    # ---- Paths ----------------------------------------------------------
    data_dir: Path = BASE_DIR / "data"
    upload_dir: Path = BASE_DIR / "data" / "uploads"
    chroma_dir: Path = BASE_DIR / "data" / "chroma"
    sqlite_path: Path = BASE_DIR / "data" / "metadata.db"

    # ---- Providers ------------------------------------------------------
    # "ollama" for real local inference, "openai" for any hosted service that
    # speaks the OpenAI REST shape (OpenAI, Groq, Together, OpenRouter, vLLM),
    # "fake" for deterministic offline runs (used by the test suite and handy for
    # frontend work with no models pulled).
    #
    # The two are set independently on purpose: Groq is the obvious free choice
    # for generation but serves no embedding model, so LLM_PROVIDER=openai with
    # EMBEDDING_PROVIDER=openai pointed at a different base URL is a normal
    # deployment, as is keeping embeddings local while generation is hosted.
    embedding_provider: Literal["ollama", "fake", "openai"] = "ollama"
    llm_provider: Literal["ollama", "fake", "openai"] = "ollama"

    ollama_base_url: str = "http://localhost:11434"
    embedding_model: str = "nomic-embed-text"
    llm_model: str = "llama3.1"

    # ---- Hosted (OpenAI-compatible) provider ----------------------------
    # EMBEDDING_MODEL and LLM_MODEL above are reused rather than duplicated --
    # the model name belongs to whichever provider is active, so switching to
    # openai means setting LLM_MODEL=gpt-4o-mini (or llama-3.1-8b-instant on
    # Groq) in the same place you'd set llama3.1 for Ollama.
    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"

    # Overrides so embeddings can come from a different vendor than chat. Both
    # fall back to the pair above when unset, which is the single-vendor case.
    openai_embed_base_url: str | None = None
    openai_embed_api_key: str | None = None
    # OpenAI's v3 embedding models can return a shortened vector. Setting this
    # does NOT make an existing index reusable -- see _check_embedding_switch.
    openai_embed_dimensions: int | None = None

    # nomic-embed-text is 768-dim. Only used by the fake provider and as a
    # sanity check against what the real model returns.
    embedding_dim: int = 768

    request_timeout: float = 120.0
    embed_timeout: float = 300.0

    # ---- Chunking -------------------------------------------------------
    chunk_size: int = Field(default=1200, ge=100, le=8000)
    chunk_overlap: int = Field(default=200, ge=0)
    min_chunk_chars: int = Field(default=50, ge=1)

    # ---- Ingestion ------------------------------------------------------
    max_upload_mb: float = Field(default=50.0, gt=0)
    embed_batch_size: int = Field(default=16, ge=1)
    ingest_workers: int = Field(default=2, ge=1, le=16)

    # ---- Retrieval ------------------------------------------------------
    top_k: int = Field(default=5, ge=1, le=50)
    # Wider net before rerank/fusion narrows it back to top_k.
    candidate_k: int = Field(default=20, ge=1, le=200)
    hybrid_search: bool = True
    # Weight of vector vs keyword in reciprocal-rank fusion (1.0 = vector only).
    vector_weight: float = Field(default=0.7, ge=0.0, le=1.0)
    rerank_strategy: Literal["none", "mmr", "cross_encoder"] = "mmr"
    mmr_lambda: float = Field(default=0.7, ge=0.0, le=1.0)
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # Cosine-distance ceiling. Chunks further than this are treated as
    # irrelevant, which is what lets /query honestly answer "I don't know".
    max_distance: float = Field(default=1.15, ge=0.0)

    # ---- Generation -----------------------------------------------------
    llm_temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=1024, ge=1)
    # Hard cap on context assembled from chunks, so a big top_k can't blow the
    # model's context window.
    max_context_chars: int = Field(default=12000, ge=500)

    # ---- OCR (optional) -------------------------------------------------
    ocr_enabled: bool = False
    # A PDF page with fewer extracted chars than this is assumed to be a scan.
    ocr_min_chars_per_page: int = Field(default=100, ge=0)
    ocr_language: str = "eng"
    ocr_dpi: int = Field(default=200, ge=72, le=600)

    # ---- Validators -----------------------------------------------------
    @field_validator("log_level")
    @classmethod
    def _upper(cls, v: str) -> str:
        level = v.upper()
        allowed = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}
        if level not in allowed:
            raise ValueError(f"log_level must be one of {sorted(allowed)}")
        return level

    @model_validator(mode="after")
    def _check_overlap(self) -> "Settings":
        # Overlap >= size would make the chunker loop forever.
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be smaller than "
                f"chunk_size ({self.chunk_size})"
            )
        if self.candidate_k < self.top_k:
            raise ValueError(
                f"candidate_k ({self.candidate_k}) must be >= top_k ({self.top_k})"
            )
        # Fail at boot rather than on the first upload. A hosted provider with no
        # credentials produces a 401 per request forever, and the symptom (every
        # ingestion fails) points nowhere near the cause.
        if self.llm_provider == "openai" and not self.openai_api_key:
            raise ValueError(
                "LLM_PROVIDER=openai requires OPENAI_API_KEY to be set."
            )
        if self.embedding_provider == "openai" and not (
            self.openai_embed_api_key or self.openai_api_key
        ):
            raise ValueError(
                "EMBEDDING_PROVIDER=openai requires OPENAI_EMBED_API_KEY or "
                "OPENAI_API_KEY to be set."
            )
        return self

    # ---- Derived --------------------------------------------------------
    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1024 * 1024)

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def embed_base_url(self) -> str:
        """Base URL for hosted embeddings, falling back to the shared one."""
        return (self.openai_embed_base_url or self.openai_base_url).rstrip("/")

    @property
    def embed_api_key(self) -> str | None:
        """Key for hosted embeddings, falling back to the shared one."""
        return self.openai_embed_api_key or self.openai_api_key

    @property
    def secure_cookies(self) -> bool:
        """Mark the visitor cookie Secure outside dev.

        Tied to app_env rather than a separate flag: a prod deployment is behind
        HTTPS, and a Secure cookie on plain-HTTP localhost is simply never sent,
        which would silently break isolation during local testing.
        """
        return self.app_env == "prod"

    def ensure_dirs(self) -> None:
        for path in (self.data_dir, self.upload_dir, self.chroma_dir):
            path.mkdir(parents=True, exist_ok=True)
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()
