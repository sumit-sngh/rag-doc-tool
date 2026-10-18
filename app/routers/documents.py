"""Document routes: upload, list, inspect, delete."""

from __future__ import annotations

import hashlib
import logging
import uuid
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, File, Query, UploadFile

from app.deps import AuthDep, OwnerDep, ServicesDep
from app.errors import (
    DocumentNotFoundError,
    FileTooLargeError,
    InvalidRequestError,
    QuotaExceededError,
    UnsupportedFileTypeError,
)
from app.ingestion import extractors
from app.schemas import (
    ChunkPreview,
    DeleteResponse,
    DocumentDetail,
    DocumentList,
    DocumentSummary,
    UploadResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"], dependencies=[AuthDep])

# Read uploads in 1 MiB slices; never load a whole file into memory.
READ_CHUNK = 1024 * 1024


def _summary(doc: dict[str, Any], progress: dict[str, Any] | None = None) -> DocumentSummary:
    return DocumentSummary(
        id=doc["id"],
        original_name=doc["original_name"],
        extension=doc["extension"],
        content_type=doc.get("content_type"),
        size_bytes=doc["size_bytes"],
        status=doc["status"],
        error=doc.get("error"),
        page_count=doc.get("page_count", 0) or 0,
        chunk_count=doc.get("chunk_count", 0) or 0,
        embedding_model=doc.get("embedding_model"),
        created_at=doc["created_at"],
        updated_at=doc["updated_at"],
        progress=progress,
    )


async def _save_upload(upload: UploadFile, dest: Path, max_bytes: int) -> tuple[int, str]:
    """Stream an upload to disk, hashing as we go.

    Returns ``(size, sha256)``. Enforces the size cap mid-stream and deletes the
    partial file if exceeded, so a huge upload can't fill the disk.
    """
    digest = hashlib.sha256()
    size = 0
    try:
        with dest.open("wb") as handle:
            while True:
                block = await upload.read(READ_CHUNK)
                if not block:
                    break
                size += len(block)
                if size > max_bytes:
                    raise FileTooLargeError(
                        f"File exceeds the {max_bytes / (1024 * 1024):.0f} MB limit"
                    )
                digest.update(block)
                handle.write(block)
    except FileTooLargeError:
        dest.unlink(missing_ok=True)
        raise
    except Exception as exc:
        dest.unlink(missing_ok=True)
        raise InvalidRequestError(f"Could not save upload: {exc}") from exc

    if size == 0:
        dest.unlink(missing_ok=True)
        raise InvalidRequestError("Uploaded file is empty")

    return size, digest.hexdigest()


@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=202,
    summary="Upload a document for ingestion",
)
async def upload_document(
    services: ServicesDep,
    owner: OwnerDep,
    file: Annotated[UploadFile, File(description="PDF, DOCX, TXT, or MD file")],
) -> UploadResponse:
    """Accept a file, persist it, and queue it for background ingestion.

    Returns 202: the document is stored and queued, not yet searchable. Poll
    ``GET /documents/{id}`` until ``status`` is ``ready``.
    """
    settings = services.settings
    filename = (file.filename or "").strip()
    if not filename:
        raise InvalidRequestError("No filename supplied")

    # Validate the type before writing anything to disk.
    try:
        extractors.detect_kind(filename)
    except extractors.UnsupportedFileType as exc:
        raise UnsupportedFileTypeError(str(exc)) from exc

    # Checked before the bytes are read, not after: on a public deployment this
    # is the only thing between one visitor and the disk. Zero means no limit,
    # which is the local default.
    if settings.max_documents_per_visitor:
        existing = await services.db.count_documents(owner=owner)
        if existing >= settings.max_documents_per_visitor:
            raise QuotaExceededError(
                f"You have reached the limit of "
                f"{settings.max_documents_per_visitor} documents. Delete one to "
                f"upload another.",
                details={
                    "limit": settings.max_documents_per_visitor,
                    "current": existing,
                },
            )

    doc_id = str(uuid.uuid4())
    extension = Path(filename).suffix.lower()
    # Store under a generated name: the original could collide, or contain path
    # separators and other characters we don't want to trust on the filesystem.
    stored_path = settings.upload_dir / f"{doc_id}{extension}"

    size, sha256 = await _save_upload(file, stored_path, settings.max_upload_bytes)

    # Identical content already ingested? Skip the work and point at the existing
    # document rather than embedding the same text twice. Scoped to this visitor:
    # matching across owners would hand back a reference to someone else's
    # document, which is a leak dressed up as an optimisation.
    duplicate = await services.db.find_duplicate(sha256, owner=owner)
    if duplicate:
        stored_path.unlink(missing_ok=True)
        return UploadResponse(
            document=_summary(duplicate),
            duplicate=True,
            message=(
                f"Identical content was already ingested as "
                f"'{duplicate['original_name']}'. Reusing it."
            ),
        )

    doc = await services.db.create_document(
        doc_id=doc_id,
        owner=owner,
        original_name=filename,
        stored_path=str(stored_path),
        content_type=file.content_type,
        extension=extension,
        size_bytes=size,
        sha256=sha256,
    )

    services.worker.enqueue(doc_id)
    logger.info("Queued %s (%s, %d bytes)", filename, doc_id, size)

    return UploadResponse(
        document=_summary(doc, services.worker.progress_for(doc_id)),
        duplicate=False,
        message="Upload accepted. Processing in the background.",
    )


@router.get("", response_model=DocumentList, summary="List documents")
async def list_documents(
    services: ServicesDep,
    owner: OwnerDep,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
    status: Annotated[str | None, Query(pattern="^(pending|processing|ready|failed)$")] = None,
) -> DocumentList:
    docs, total = await services.db.list_documents(
        owner=owner, limit=limit, offset=offset, status=status
    )
    return DocumentList(
        documents=[_summary(d, services.worker.progress_for(d["id"])) for d in docs],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/{document_id}", response_model=DocumentDetail, summary="Get one document")
async def get_document(
    services: ServicesDep,
    owner: OwnerDep,
    document_id: str,
    include_chunks: Annotated[bool, Query()] = False,
    chunk_limit: Annotated[int, Query(ge=1, le=200)] = 20,
) -> DocumentDetail:
    doc = await services.db.get_document(document_id, owner=owner)
    if doc is None:
        raise DocumentNotFoundError(f"No document with id '{document_id}'")

    chunks: list[ChunkPreview] = []
    if include_chunks:
        rows = await services.db.get_document_chunks(
            document_id, owner=owner, limit=chunk_limit
        )
        chunks = [
            ChunkPreview(
                id=r["id"],
                chunk_index=r["chunk_index"],
                page_start=r["page_start"],
                page_end=r["page_end"],
                token_estimate=r["token_estimate"],
                text=r["text"],
            )
            for r in rows
        ]

    base = _summary(doc, services.worker.progress_for(document_id))
    return DocumentDetail(**base.model_dump(), chunks=chunks)


@router.delete("/{document_id}", response_model=DeleteResponse, summary="Delete a document")
async def delete_document(
    services: ServicesDep, owner: OwnerDep, document_id: str
) -> DeleteResponse:
    """Remove a document, its chunks, its vectors, and its stored file.

    The ownership check is the lookup: an id belonging to someone else reads as
    404 here, so there is no separate "not yours" branch to get wrong.
    """
    doc = await services.db.get_document(document_id, owner=owner)
    if doc is None:
        raise DocumentNotFoundError(f"No document with id '{document_id}'")

    # Vectors first: an orphaned vector would keep surfacing in search results
    # with no row to render, whereas an orphaned row is merely unreachable.
    await services.vector_store.delete_document(document_id, owner=owner)
    await services.db.delete_document(document_id, owner=owner)

    try:
        Path(doc["stored_path"]).unlink(missing_ok=True)
    except OSError as exc:
        logger.warning("Could not delete file for %s: %s", document_id, exc)

    return DeleteResponse(
        deleted=document_id, message=f"Deleted '{doc['original_name']}'"
    )
