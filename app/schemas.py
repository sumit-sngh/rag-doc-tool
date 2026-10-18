"""Pydantic request/response models.

Kept separate from the domain dataclasses on purpose: the API contract can stay
stable while internals change, and validation lives at the boundary where bad
input actually arrives.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

DocumentStatus = Literal["pending", "processing", "ready", "failed"]


# --------------------------------------------------------------------------- #
# Documents
# --------------------------------------------------------------------------- #
class DocumentSummary(BaseModel):
    id: str
    original_name: str
    extension: str
    content_type: str | None = None
    size_bytes: int
    status: DocumentStatus
    error: str | None = None
    page_count: int = 0
    chunk_count: int = 0
    embedding_model: str | None = None
    created_at: str
    updated_at: str
    # Present only while a worker holds the document.
    progress: dict[str, Any] | None = None


class DocumentList(BaseModel):
    documents: list[DocumentSummary]
    total: int
    limit: int
    offset: int


class UploadResponse(BaseModel):
    document: DocumentSummary
    # True when content-hash matched an existing ready document and ingestion
    # was skipped.
    duplicate: bool = False
    message: str


class ChunkPreview(BaseModel):
    id: str
    chunk_index: int
    page_start: int
    page_end: int
    token_estimate: int
    text: str


class DocumentDetail(DocumentSummary):
    chunks: list[ChunkPreview] = Field(default_factory=list)


class DeleteResponse(BaseModel):
    deleted: str
    message: str


# --------------------------------------------------------------------------- #
# Query
# --------------------------------------------------------------------------- #
class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)

    # Scoping: omit for the whole library, one id for a single document, several
    # for a subset. This is the multi-document support the brief asks for.
    document_ids: list[str] | None = Field(
        default=None,
        description="Restrict retrieval to these documents. Omit to search everything.",
    )

    session_id: str | None = Field(
        default=None,
        description="Groups exchanges into a conversation. Created on first use.",
    )

    top_k: int | None = Field(default=None, ge=1, le=50)
    use_history: bool = True
    stream: bool = False

    @field_validator("question")
    @classmethod
    def _non_blank(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("question cannot be blank")
        return cleaned

    @field_validator("document_ids")
    @classmethod
    def _clean_ids(cls, v: list[str] | None) -> list[str] | None:
        if v is None:
            return None
        # Deduplicate while preserving order; drop blanks. An empty list means
        # "no scope", which is the same as None.
        seen: list[str] = []
        for item in v:
            item = (item or "").strip()
            if item and item not in seen:
                seen.append(item)
        return seen or None


class Citation(BaseModel):
    number: int
    chunk_id: str
    document_id: str
    document_name: str
    chunk_index: int
    page_start: int
    page_end: int
    page_label: str
    text: str
    distance: float | None = None
    similarity: float | None = None
    keyword_score: float | None = None
    rerank_score: float | None = None


class QueryResponse(BaseModel):
    answer: str
    # False when nothing relevant was retrieved, or the model produced no
    # verifiable citation. Clients should surface this.
    grounded: bool
    sources: list[Citation] = Field(default_factory=list)
    retrieved: list[Citation] = Field(default_factory=list)
    session_id: str | None = None
    history_id: str | None = None
    latency_ms: int = 0
    diagnostics: dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Sessions and history
# --------------------------------------------------------------------------- #
class SessionSummary(BaseModel):
    id: str
    title: str | None = None
    created_at: str
    last_active_at: str
    exchange_count: int = 0


class SessionList(BaseModel):
    sessions: list[SessionSummary]


class HistoryEntry(BaseModel):
    id: str
    session_id: str
    question: str
    answer: str
    document_ids: list[str] = Field(default_factory=list)
    sources: list[dict[str, Any]] = Field(default_factory=list)
    grounded: bool = True
    llm_model: str | None = None
    embedding_model: str | None = None
    latency_ms: int | None = None
    created_at: str


class HistoryList(BaseModel):
    history: list[HistoryEntry]
    limit: int
    offset: int


# --------------------------------------------------------------------------- #
# Health
# --------------------------------------------------------------------------- #
class ComponentHealth(BaseModel):
    ok: bool
    detail: str


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    config: dict[str, Any]
    components: dict[str, ComponentHealth]
    stats: dict[str, int]


class ErrorDetail(BaseModel):
    code: str
    message: str
    details: Any = None


class ErrorResponse(BaseModel):
    error: ErrorDetail
