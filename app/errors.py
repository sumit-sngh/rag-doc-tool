"""Typed application errors mapped to HTTP responses.

Every failure the brief calls out (unsupported file type, empty document, no
relevant matches, provider unreachable) gets a distinct class with a stable
``code``, so the frontend can branch on ``error.code`` instead of parsing prose.
"""

from __future__ import annotations

from typing import Any


class AppError(Exception):
    """Base class for expected, client-visible failures."""

    status_code: int = 500
    code: str = "internal_error"

    def __init__(self, message: str, *, details: Any = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.details is not None:
            payload["details"] = self.details
        return {"error": payload}


# ---- 4xx ------------------------------------------------------------------ #
class UnsupportedFileTypeError(AppError):
    status_code = 415
    code = "unsupported_file_type"


class EmptyDocumentError(AppError):
    status_code = 422
    code = "empty_document"


class FileTooLargeError(AppError):
    status_code = 413
    code = "file_too_large"


class DocumentNotFoundError(AppError):
    status_code = 404
    code = "document_not_found"


class SessionNotFoundError(AppError):
    status_code = 404
    code = "session_not_found"


class DocumentNotReadyError(AppError):
    """Queried a document whose ingestion is still pending or has failed."""

    status_code = 409
    code = "document_not_ready"


class NoDocumentsError(AppError):
    """Nothing has been ingested yet, so there is nothing to answer from."""

    status_code = 409
    code = "no_documents"


class InvalidRequestError(AppError):
    status_code = 400
    code = "invalid_request"


class UnauthorizedError(AppError):
    status_code = 401
    code = "unauthorized"


class QuotaExceededError(AppError):
    """The visitor is at their document limit.

    429 rather than 403: the request is well-formed and the credentials are fine,
    the caller has simply used up an allowance. Clients that already back off on
    429 do the right thing without a special case.
    """

    status_code = 429
    code = "quota_exceeded"


# ---- 5xx / upstream ------------------------------------------------------- #
class ProviderError(AppError):
    """An embedding or LLM provider failed or is unreachable."""

    status_code = 502
    code = "provider_error"


class ModelNotFoundError(ProviderError):
    """The configured model is not present on the provider (e.g. not pulled)."""

    status_code = 502
    code = "model_not_found"


class VectorStoreError(AppError):
    status_code = 500
    code = "vector_store_error"


class ExtractionFailedError(AppError):
    status_code = 422
    code = "extraction_failed"
