"""Async SQLite persistence for documents, chunks, sessions, and Q&A history.

Uses a connection-per-operation model rather than a shared long-lived handle.
With WAL enabled, SQLite handles concurrent readers well, and short-lived
connections sidestep an entire class of cross-task cursor and transaction
interference bugs that a shared handle invites -- worth far more than the
microseconds a connection costs at this scale.

All SQL lives in ``app.schema``; this module is the driver-facing layer.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Sequence

import aiosqlite

from app import schema as S
from app.ingestion.chunker import Chunk
from app.retrieval.fusion import RetrievedChunk
from app.retrieval.keyword import build_fts_match_query

logger = logging.getLogger(__name__)


def utcnow() -> str:
    """ISO-8601 UTC timestamp. Lexicographically sortable, so ORDER BY works."""
    return datetime.now(timezone.utc).isoformat()


class Database:
    """Repository over the SQLite metadata store."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    # ---- lifecycle ------------------------------------------------------ #
    @asynccontextmanager
    async def connect(self) -> AsyncIterator[aiosqlite.Connection]:
        """Open a connection with pragmas applied and Row factory set."""
        conn = await aiosqlite.connect(self.path)
        conn.row_factory = aiosqlite.Row
        try:
            for pragma in S.PRAGMAS:
                try:
                    await conn.execute(pragma)
                except sqlite3.DatabaseError as exc:
                    # A missing pragma (e.g. WAL on an exotic filesystem) is not
                    # fatal; the app still works, just with less concurrency.
                    logger.warning("Pragma failed (%s): %s", pragma, exc)
            yield conn
        finally:
            await conn.close()

    async def init(self) -> None:
        """Create tables/indexes if absent and recover interrupted ingestions."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        async with self.connect() as conn:
            # Order is load-bearing. Migrations run first so that an upgraded
            # database has `owner` before SCHEMA_STATEMENTS tries to build an
            # index on it; on a fresh database they no-op and CREATE TABLE
            # supplies the column instead.
            await self._migrate(conn)
            for stmt in S.SCHEMA_STATEMENTS:
                await conn.execute(stmt)
            await conn.execute(S.RESET_STALE_DOCUMENTS, (utcnow(),))
            await conn.commit()
        logger.info("Database ready at %s", self.path)

    async def _migrate(self, conn: aiosqlite.Connection) -> None:
        """Add columns missing from an older database.

        SQLite has no ``ADD COLUMN IF NOT EXISTS``, so each column is guarded by
        ``PRAGMA table_info``. A table that does not exist yet is skipped rather
        than altered -- the CREATE statements that follow already include the
        column, and ALTER on a missing table is an error.
        """
        for table, column, statement in S.ADD_COLUMNS:
            async with conn.execute(S.TABLE_INFO.format(table=table)) as cur:
                rows = await cur.fetchall()
            if not rows:
                continue  # fresh database; CREATE TABLE covers it
            if column in {r[1] for r in rows}:
                continue
            await conn.execute(statement)
            # The NOT NULL DEFAULT backfills every existing row in the same
            # statement, so a library created before isolation existed stays
            # visible under the shared owner instead of vanishing.
            logger.info("Migrated %s: added column %r", table, column)
        await conn.commit()

    # ---- documents ------------------------------------------------------ #
    # `owner` is a required keyword on every method below that reads or writes an
    # owned row. Deliberately not defaulted: a forgotten scope then fails loudly
    # at the call site instead of quietly serving the shared library to a visitor
    # -- or, worse, serving one visitor's documents to another.
    async def create_document(
        self,
        *,
        doc_id: str,
        owner: str,
        original_name: str,
        stored_path: str,
        content_type: str | None,
        extension: str,
        size_bytes: int,
        sha256: str,
    ) -> dict[str, Any]:
        ts = utcnow()
        async with self.connect() as conn:
            await conn.execute(
                S.INSERT_DOCUMENT,
                (
                    doc_id, owner, original_name, stored_path, content_type, extension,
                    size_bytes, sha256, "pending", ts, ts,
                ),
            )
            await conn.commit()
        return {
            "id": doc_id,
            "owner": owner,
            "original_name": original_name,
            "stored_path": stored_path,
            "content_type": content_type,
            "extension": extension,
            "size_bytes": size_bytes,
            "sha256": sha256,
            "status": "pending",
            "error": None,
            "page_count": 0,
            "chunk_count": 0,
            "embedding_model": None,
            "created_at": ts,
            "updated_at": ts,
        }

    async def get_document(self, doc_id: str, *, owner: str) -> dict[str, Any] | None:
        async with self.connect() as conn:
            async with conn.execute(S.SELECT_DOCUMENT, (doc_id, owner)) as cur:
                row = await cur.fetchone()
        return dict(row) if row else None

    async def get_document_internal(self, doc_id: str) -> dict[str, Any] | None:
        """Fetch a document regardless of owner. Not reachable from a request.

        The ingestion worker pulls ids from its own pending queue, so it has no
        visitor context to scope by and no untrusted input to guard against.
        Named to be obvious in a grep: any route calling this is a bug.
        """
        async with self.connect() as conn:
            async with conn.execute(S.SELECT_DOCUMENT_ANY_OWNER, (doc_id,)) as cur:
                row = await cur.fetchone()
        return dict(row) if row else None

    async def find_duplicate(self, sha256: str, *, owner: str) -> dict[str, Any] | None:
        """Find an already-ingested document with identical content, same owner."""
        async with self.connect() as conn:
            async with conn.execute(S.FIND_DUPLICATE, (sha256, owner)) as cur:
                row = await cur.fetchone()
        return dict(row) if row else None

    async def list_documents(
        self, *, owner: str, limit: int = 100, offset: int = 0, status: str | None = None
    ) -> tuple[list[dict[str, Any]], int]:
        async with self.connect() as conn:
            if status:
                query, params = S.LIST_DOCUMENTS_BY_STATUS, (owner, status, limit, offset)
                count_query, count_params = S.COUNT_DOCUMENTS_BY_STATUS, (owner, status)
            else:
                query, params = S.LIST_DOCUMENTS, (owner, limit, offset)
                count_query, count_params = S.COUNT_DOCUMENTS, (owner,)
            async with conn.execute(query, params) as cur:
                rows = await cur.fetchall()
            # Counted with the same filter as the rows: `total` describes the
            # filtered result set, so it's the number a client should paginate
            # against, not the size of the whole table.
            async with conn.execute(count_query, count_params) as cur:
                total_row = await cur.fetchone()
        total = int(total_row["n"]) if total_row else 0
        return [dict(r) for r in rows], total

    async def count_documents(self, *, owner: str) -> int:
        async with self.connect() as conn:
            async with conn.execute(S.COUNT_DOCUMENTS, (owner,)) as cur:
                row = await cur.fetchone()
        return int(row["n"]) if row else 0

    async def set_document_status(
        self, doc_id: str, status: str, error: str | None = None
    ) -> None:
        # Unscoped by design: only the ingestion worker updates status, and it
        # works from its own queue rather than from request input.
        async with self.connect() as conn:
            await conn.execute(S.UPDATE_DOCUMENT_STATUS, (status, error, utcnow(), doc_id))
            await conn.commit()

    async def mark_document_ready(
        self, doc_id: str, page_count: int, chunk_count: int, embedding_model: str
    ) -> None:
        async with self.connect() as conn:
            await conn.execute(
                S.MARK_DOCUMENT_READY,
                (page_count, chunk_count, embedding_model, utcnow(), doc_id),
            )
            await conn.commit()

    async def delete_document(self, doc_id: str, *, owner: str) -> bool:
        """Delete a document and (via FK cascade) its chunks, plus FTS rows.

        Returns whether a row was actually removed, so a caller can tell "gone"
        from "not yours" without a second query.
        """
        async with self.connect() as conn:
            # FTS5 virtual tables don't participate in FK cascades, so the
            # index must be cleaned explicitly or it keeps serving dead chunks.
            # Ordered after the ownership check for the same reason the delete is
            # scoped: an unowned id must not strip another visitor's FTS rows.
            async with conn.execute(S.SELECT_DOCUMENT, (doc_id, owner)) as cur:
                if await cur.fetchone() is None:
                    return False
            await conn.execute(S.DELETE_CHUNKS_FTS_FOR_DOCUMENT, (doc_id,))
            await conn.execute(S.DELETE_DOCUMENT, (doc_id, owner))
            await conn.commit()
        return True

    async def pending_document_ids(self) -> list[str]:
        async with self.connect() as conn:
            async with conn.execute(S.SELECT_PENDING_DOCUMENTS) as cur:
                rows = await cur.fetchall()
        return [r["id"] for r in rows]

    # ---- chunks --------------------------------------------------------- #
    async def replace_chunks(self, doc_id: str, chunks: Sequence[Chunk]) -> None:
        """Replace all chunks for a document, keeping the FTS index in step.

        Idempotent by design: re-ingesting a document cannot leave duplicates
        behind, which matters because a retry after partial failure is normal.
        """
        async with self.connect() as conn:
            await conn.execute(S.DELETE_CHUNKS_FTS_FOR_DOCUMENT, (doc_id,))
            await conn.execute(S.DELETE_CHUNKS_FOR_DOCUMENT, (doc_id,))
            for chunk in chunks:
                chunk_id = make_chunk_id(doc_id, chunk.chunk_index)
                await conn.execute(
                    S.INSERT_CHUNK,
                    (
                        chunk_id, doc_id, chunk.chunk_index,
                        chunk.page_start, chunk.page_end,
                        chunk.char_start, chunk.char_end,
                        chunk.token_estimate, chunk.text,
                    ),
                )
                await conn.execute(S.INSERT_CHUNK_FTS, (chunk.text, chunk_id, doc_id))
            await conn.commit()

    async def get_chunks_by_ids(
        self, chunk_ids: Sequence[str], *, owner: str
    ) -> dict[str, dict[str, Any]]:
        """Fetch chunk rows (with document_name) keyed by chunk id."""
        if not chunk_ids:
            return {}
        placeholders = ",".join("?" * len(chunk_ids))
        query = S.SELECT_CHUNKS_BY_IDS_TEMPLATE.format(placeholders=placeholders)
        async with self.connect() as conn:
            async with conn.execute(query, (owner, *chunk_ids)) as cur:
                rows = await cur.fetchall()
        return {r["id"]: dict(r) for r in rows}

    async def get_document_chunks(
        self, doc_id: str, *, owner: str, limit: int = 100, offset: int = 0
    ) -> list[dict[str, Any]]:
        async with self.connect() as conn:
            async with conn.execute(
                S.SELECT_CHUNKS_FOR_DOCUMENT, (doc_id, owner, limit, offset)
            ) as cur:
                rows = await cur.fetchall()
        return [dict(r) for r in rows]

    async def count_chunks(self, *, owner: str) -> int:
        async with self.connect() as conn:
            async with conn.execute(S.COUNT_CHUNKS, (owner,)) as cur:
                row = await cur.fetchone()
        return int(row["n"]) if row else 0

    # ---- keyword search ------------------------------------------------- #
    async def keyword_search(
        self,
        question: str,
        limit: int,
        *,
        owner: str,
        document_ids: Sequence[str] | None = None,
    ) -> list[RetrievedChunk]:
        """BM25 full-text search over chunks, optionally scoped to documents.

        The owner filter is not optional. This is the keyword half of hybrid
        retrieval, and leaving it unscoped while the vector half is scoped would
        let another visitor's text surface as a cited source.
        """
        match_query = build_fts_match_query(question)
        if not match_query:
            return []

        if document_ids:
            placeholders = ",".join("?" * len(document_ids))
            query = S.KEYWORD_SEARCH_SCOPED_TEMPLATE.format(placeholders=placeholders)
            params: tuple[Any, ...] = (match_query, owner, *document_ids, limit)
        else:
            query, params = S.KEYWORD_SEARCH, (match_query, owner, limit)

        try:
            async with self.connect() as conn:
                async with conn.execute(query, params) as cur:
                    rows = await cur.fetchall()
        except sqlite3.OperationalError as exc:
            # Keyword search is an enhancement, not a requirement. If FTS5 is
            # missing from this SQLite build or a query slips past the sanitizer,
            # degrade to vector-only rather than failing the request.
            logger.warning("Keyword search unavailable, falling back to vector-only: %s", exc)
            return []

        return [
            RetrievedChunk(
                chunk_id=r["chunk_id"],
                document_id=r["document_id"],
                document_name=r["document_name"],
                chunk_index=r["chunk_index"],
                page_start=r["page_start"],
                page_end=r["page_end"],
                text=r["text"],
                keyword_score=float(r["keyword_score"]),
            )
            for r in rows
        ]

    # ---- sessions ------------------------------------------------------- #
    async def create_session(
        self, session_id: str, *, owner: str, title: str | None = None
    ) -> dict[str, Any]:
        ts = utcnow()
        async with self.connect() as conn:
            await conn.execute(S.INSERT_SESSION, (session_id, owner, title, ts, ts))
            await conn.commit()
        return {
            "id": session_id, "title": title,
            "created_at": ts, "last_active_at": ts, "exchange_count": 0,
        }

    async def get_session(self, session_id: str, *, owner: str) -> dict[str, Any] | None:
        async with self.connect() as conn:
            async with conn.execute(S.SELECT_SESSION, (session_id, owner)) as cur:
                row = await cur.fetchone()
        return dict(row) if row else None

    async def ensure_session(self, session_id: str, *, owner: str) -> dict[str, Any]:
        """Get a session, creating it if the client supplied an unknown id.

        Sessions are client-generated so the frontend can start a conversation
        without a round-trip. Auto-creating on first use keeps /query single-call.

        With isolation on, "unknown" includes "exists but belongs to someone
        else". The INSERT then collides on the primary key and the recovery read
        finds nothing for this owner, so it raises rather than silently attaching
        the visitor to another person's conversation.
        """
        existing = await self.get_session(session_id, owner=owner)
        if existing:
            return existing
        try:
            return await self.create_session(session_id, owner=owner)
        except sqlite3.IntegrityError:
            # Either a race with a concurrent request for the same owner, or the
            # id belongs to a different owner. Only the first is recoverable.
            found = await self.get_session(session_id, owner=owner)
            if found:
                return found
            raise

    async def touch_session(
        self, session_id: str, *, owner: str, title: str | None = None
    ) -> None:
        async with self.connect() as conn:
            await conn.execute(S.TOUCH_SESSION, (utcnow(), session_id, owner))
            if title:
                # No-ops if a title is already set (see SET_SESSION_TITLE).
                await conn.execute(S.SET_SESSION_TITLE, (title[:120], session_id, owner))
            await conn.commit()

    async def list_sessions(
        self, *, owner: str, limit: int = 50, offset: int = 0
    ) -> list[dict[str, Any]]:
        async with self.connect() as conn:
            async with conn.execute(S.LIST_SESSIONS, (owner, limit, offset)) as cur:
                rows = await cur.fetchall()
        return [dict(r) for r in rows]

    async def delete_session(self, session_id: str, *, owner: str) -> None:
        async with self.connect() as conn:
            await conn.execute(S.DELETE_SESSION, (session_id, owner))
            await conn.commit()

    # ---- history -------------------------------------------------------- #
    async def add_history(
        self,
        *,
        entry_id: str,
        session_id: str,
        question: str,
        answer: str,
        document_ids: Sequence[str],
        sources: list[dict[str, Any]],
        grounded: bool,
        llm_model: str | None,
        embedding_model: str | None,
        latency_ms: int | None,
    ) -> dict[str, Any]:
        ts = utcnow()
        async with self.connect() as conn:
            await conn.execute(
                S.INSERT_QA,
                (
                    entry_id, session_id, question, answer,
                    json.dumps(list(document_ids)), json.dumps(sources),
                    1 if grounded else 0, llm_model, embedding_model, latency_ms, ts,
                ),
            )
            await conn.commit()
        return {
            "id": entry_id, "session_id": session_id, "question": question,
            "answer": answer, "document_ids": list(document_ids), "sources": sources,
            "grounded": grounded, "llm_model": llm_model,
            "embedding_model": embedding_model, "latency_ms": latency_ms,
            "created_at": ts,
        }

    async def list_history(
        self,
        *,
        owner: str,
        session_id: str | None = None,
        document_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        async with self.connect() as conn:
            if session_id:
                query, params = S.LIST_HISTORY_BY_SESSION, (session_id, owner, limit, offset)
            elif document_id:
                query, params = S.LIST_HISTORY_BY_DOCUMENT, (owner, document_id, limit, offset)
            else:
                query, params = S.LIST_HISTORY, (owner, limit, offset)
            async with conn.execute(query, params) as cur:
                rows = await cur.fetchall()

        results: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["document_ids"] = _load_json(item.get("document_ids"), [])
            item["sources"] = _load_json(item.get("sources"), [])
            item["grounded"] = bool(item.get("grounded", 1))
            results.append(item)
        return results


def make_chunk_id(doc_id: str, chunk_index: int) -> str:
    """Stable, derivable chunk id shared by SQLite and Chroma.

    Deriving rather than randomising means re-ingestion overwrites the same ids
    in both stores, so the two can never drift apart.
    """
    return f"{doc_id}:{chunk_index}"


def _load_json(raw: Any, default: Any) -> Any:
    """Tolerant JSON decode -- a malformed column shouldn't 500 the endpoint."""
    if raw is None:
        return default
    if isinstance(raw, (list, dict)):
        return raw
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        logger.warning("Could not decode JSON column value: %r", raw)
        return default
