"""Document ingestion: extract -> chunk -> embed -> index.

Runs off an in-process asyncio queue rather than inline in the upload request.
``POST /upload`` therefore returns as soon as the bytes are safely on disk, and a
worker picks the document up afterwards -- which is what keeps a 300-page PDF
from holding a HTTP connection open for minutes.

Concurrency shape:

* ``INGEST_WORKERS`` worker tasks consume the queue, so several documents can be
  in flight without an unbounded fan-out onto the embedding provider.
* Text extraction is synchronous and CPU-bound (pypdf, python-docx), so it is
  pushed to a thread; embedding is network-bound and stays on the event loop.
* Embedding runs in batches of ``EMBED_BATCH_SIZE`` -- one request per batch
  rather than per chunk.

Failure policy: a document that fails is marked ``failed`` with the reason
recorded, and the worker moves on. One bad file never takes down ingestion.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anyio.to_thread  # submodule, so import it explicitly (see vector_store.py)

from app.config import Settings
from app.db import Database, make_chunk_id
from app.errors import AppError, EmptyDocumentError, ExtractionFailedError
from app.ingestion import extractors
from app.ingestion.chunker import Chunk, chunk_pages
from app.providers.base import EmbeddingProvider
from app.retrieval.vector_store import OWNER_KEY, VectorStore
from app.schema import SHARED_OWNER

logger = logging.getLogger(__name__)


@dataclass
class IngestionProgress:
    """In-memory progress for a document currently being ingested.

    Deliberately not persisted: it is only meaningful while a worker holds the
    document, and writing per-batch progress to SQLite would add write traffic
    for information that is stale the moment the run ends.
    """

    document_id: str
    stage: str = "queued"  # queued | extracting | chunking | embedding | indexing
    total_chunks: int = 0
    embedded_chunks: int = 0

    def as_dict(self) -> dict[str, Any]:
        pct = (
            round(100 * self.embedded_chunks / self.total_chunks, 1)
            if self.total_chunks
            else 0.0
        )
        return {
            "stage": self.stage,
            "total_chunks": self.total_chunks,
            "embedded_chunks": self.embedded_chunks,
            "percent": pct,
        }


class IngestionPipeline:
    """Turns one uploaded file into indexed, searchable chunks."""

    def __init__(
        self,
        db: Database,
        vector_store: VectorStore,
        embeddings: EmbeddingProvider,
        settings: Settings,
    ) -> None:
        self.db = db
        self.vector_store = vector_store
        self.embeddings = embeddings
        self.settings = settings

    async def process(self, document_id: str, progress: IngestionProgress | None = None) -> None:
        """Ingest a document end to end, recording terminal status in SQLite."""
        progress = progress or IngestionProgress(document_id=document_id)
        # The unscoped read, and the only legitimate one in the app. Ids come off
        # this worker's own queue, not from a request, so there is no visitor to
        # scope to -- the owner is read back off the row instead and carried into
        # the vector metadata below.
        doc = await self.db.get_document_internal(document_id)
        if doc is None:
            logger.warning("Ingestion skipped: document %s no longer exists", document_id)
            return

        await self.db.set_document_status(document_id, "processing")
        path = Path(doc["stored_path"])

        try:
            # ---- extract (sync, CPU-bound -> thread) ----
            progress.stage = "extracting"
            if not path.exists():
                raise ExtractionFailedError(f"Stored file is missing: {path.name}")

            cfg = self.settings
            pages = await anyio.to_thread.run_sync(
                lambda: extractors.extract(
                    path,
                    doc["original_name"],
                    ocr_enabled=cfg.ocr_enabled,
                    ocr_min_chars=cfg.ocr_min_chars_per_page,
                    ocr_language=cfg.ocr_language,
                    ocr_dpi=cfg.ocr_dpi,
                )
            )

            # ---- chunk ----
            progress.stage = "chunking"
            chunks = chunk_pages(
                pages,
                chunk_size=cfg.chunk_size,
                chunk_overlap=cfg.chunk_overlap,
                min_chunk_chars=cfg.min_chunk_chars,
            )
            if not chunks:
                raise EmptyDocumentError(
                    "Document produced no usable text chunks. It may be empty, "
                    "or a scanned PDF needing OCR (set OCR_ENABLED=true)."
                )
            progress.total_chunks = len(chunks)

            # SQLite first: chunk text is the source of truth for citations and
            # keyword search, and it makes the FTS index available even if the
            # vector upsert later fails.
            await self.db.replace_chunks(document_id, chunks)

            # ---- embed + index in batches ----
            progress.stage = "embedding"
            await self._embed_and_index(
                document_id,
                doc["original_name"],
                # Falls back to the shared owner rather than failing: a row written
                # before the migration has no owner, and refusing to ingest it
                # would be a worse outcome than filing it where single-user mode
                # already looks.
                doc.get("owner") or SHARED_OWNER,
                chunks,
                progress,
            )

            progress.stage = "indexing"
            await self.db.mark_document_ready(
                document_id,
                page_count=len(pages),
                chunk_count=len(chunks),
                embedding_model=self.embeddings.model,
            )
            logger.info(
                "Ingested %s: %d page(s), %d chunk(s)",
                doc["original_name"], len(pages), len(chunks),
            )

        except extractors.UnsupportedFileType as exc:
            await self._fail(document_id, f"Unsupported file type: {exc}")
        except extractors.EmptyDocument as exc:
            await self._fail(document_id, str(exc))
        except extractors.ExtractionError as exc:
            await self._fail(document_id, f"Text extraction failed: {exc}")
        except AppError as exc:
            await self._fail(document_id, exc.message)
        except Exception as exc:  # noqa: BLE001 - worker must never die
            logger.exception("Unexpected ingestion failure for %s", document_id)
            await self._fail(document_id, f"Unexpected error: {exc}")

    async def _embed_and_index(
        self,
        document_id: str,
        document_name: str,
        owner: str,
        chunks: list[Chunk],
        progress: IngestionProgress,
    ) -> None:
        batch_size = self.settings.embed_batch_size

        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            vectors = await self.embeddings.embed_documents([c.text for c in batch])

            if len(vectors) != len(batch):
                raise ExtractionFailedError(
                    f"Embedding provider returned {len(vectors)} vectors for "
                    f"{len(batch)} chunks"
                )

            await self.vector_store.upsert_chunks(
                chunk_ids=[make_chunk_id(document_id, c.chunk_index) for c in batch],
                embeddings=vectors,
                texts=[c.text for c in batch],
                # Chroma metadata values must be scalars -- no nested structures.
                # `owner` is written here, at index time, because a metadata filter
                # is the only thing standing between one visitor's question and
                # another visitor's passages.
                metadatas=[
                    {
                        OWNER_KEY: owner,
                        "document_id": document_id,
                        "document_name": document_name,
                        "chunk_index": c.chunk_index,
                        "page_start": c.page_start,
                        "page_end": c.page_end,
                    }
                    for c in batch
                ],
            )
            progress.embedded_chunks += len(batch)

    async def _fail(self, document_id: str, message: str) -> None:
        logger.warning("Ingestion failed for %s: %s", document_id, message)
        await self.db.set_document_status(document_id, "failed", message)


class IngestionWorker:
    """Bounded pool of workers draining an ingestion queue."""

    def __init__(self, pipeline: IngestionPipeline, worker_count: int = 2) -> None:
        self.pipeline = pipeline
        self.worker_count = max(1, worker_count)
        self.queue: asyncio.Queue[str] = asyncio.Queue()
        self._tasks: list[asyncio.Task[None]] = []
        self._progress: dict[str, IngestionProgress] = {}
        self._running = False

    # ---- lifecycle ------------------------------------------------------ #
    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._tasks = [
            asyncio.create_task(self._run(i), name=f"ingest-worker-{i}")
            for i in range(self.worker_count)
        ]
        logger.info("Started %d ingestion worker(s)", self.worker_count)

        # Requeue anything left pending by a previous run (including rows the
        # startup pragma reset from 'processing'), so an app restart resumes
        # ingestion instead of stranding documents forever.
        for doc_id in await self.pipeline.db.pending_document_ids():
            self.enqueue(doc_id)

    async def stop(self) -> None:
        self._running = False
        for task in self._tasks:
            task.cancel()
        # return_exceptions: cancellation is expected, not an error to surface.
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        logger.info("Ingestion workers stopped")

    # ---- queue ---------------------------------------------------------- #
    def enqueue(self, document_id: str) -> None:
        self._progress[document_id] = IngestionProgress(document_id=document_id)
        self.queue.put_nowait(document_id)

    def progress_for(self, document_id: str) -> dict[str, Any] | None:
        entry = self._progress.get(document_id)
        return entry.as_dict() if entry else None

    @property
    def depth(self) -> int:
        return self.queue.qsize()

    async def wait_idle(self, timeout: float = 60.0) -> bool:
        """Block until the queue drains. Test/CLI helper, not used by routes."""
        try:
            await asyncio.wait_for(self.queue.join(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

    # ---- worker loop ---------------------------------------------------- #
    async def _run(self, index: int) -> None:
        while self._running:
            try:
                document_id = await self.queue.get()
            except asyncio.CancelledError:
                break

            try:
                progress = self._progress.setdefault(
                    document_id, IngestionProgress(document_id=document_id)
                )
                await self.pipeline.process(document_id, progress)
            except asyncio.CancelledError:
                # No await here: the task is already cancelled, so any further
                # await re-raises immediately and would never complete. The
                # document stays 'processing' and init()'s RESET_STALE_DOCUMENTS
                # flips it back to 'pending' on the next start, where start()
                # requeues it. Shutdown loses no work.
                self.queue.task_done()
                raise
            except Exception:  # noqa: BLE001
                # pipeline.process already records failure; this is belt-and-braces
                # so an unexpected error can never kill the worker loop.
                logger.exception("Worker %d crashed on %s", index, document_id)
                self.queue.task_done()
                continue

            self.queue.task_done()
            # Progress for a finished document is no longer interesting, and
            # keeping every entry would leak memory over a long-lived process.
            self._progress.pop(document_id, None)
