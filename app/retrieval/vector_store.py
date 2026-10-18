"""Chroma vector store wrapper.

Chroma runs embedded (``PersistentClient``) so there is no server to operate --
the index is just a directory under ``data/``.

Two things this wrapper is careful about:

* **Never blocking the event loop.** Chroma's client is synchronous, and an HNSW
  query or a multi-thousand-vector upsert is long enough to stall every other
  request in a single-process ASGI app. Every call is offloaded with
  ``anyio.to_thread.run_sync``.

* **Not letting Chroma types leak.** Query results come back as numpy arrays
  depending on version and call shape, so everything is normalised to plain
  Python floats at the boundary.

* **Every read is owner-scoped.** ``owner`` is a required metadata field and a
  required argument, mirroring the ``owner`` column in SQLite. This is the half
  of hybrid retrieval that must not be forgotten: scoping only the keyword side
  would let another visitor's passages come back as a citation.

Embeddings are supplied by us, never computed by Chroma: the provider layer owns
that, so the same vectors can be reused for MMR reranking.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

# `import anyio.to_thread`, not a bare `import anyio`: `to_thread` is a submodule,
# and whether it happens to be imported as a side effect of anyio's __init__ is
# an implementation detail that has changed between releases. Importing it
# explicitly means this can't break on an anyio upgrade.
import anyio.to_thread

from app.errors import VectorStoreError
from app.retrieval.fusion import RetrievedChunk

logger = logging.getLogger(__name__)

# Silenced at import rather than in `configure_logging`, because the tests build
# Services directly and bypass the app lifespan -- so configure_logging never
# runs under pytest, which is where this noise is worst.
#
# chromadb 0.6.3 calls a newer posthog's `capture()` positionally and logs the
# resulting TypeError at ERROR, several times per query: "Failed to send
# telemetry event ClientStartEvent: capture() takes 1 positional argument but 3
# were given". Turning telemetry off does not stop it -- chromadb sets
# `posthog.disabled = True` and then makes the malformed call anyway, so it
# raises before anything could be sent. Nothing leaves the machine; the only
# cost is that it buries real errors in test output.
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)

COLLECTION_NAME = "document_chunks"

# Metadata key holding the visitor scope. Must match the `owner` column in
# SQLite: the two stores are filtered on the same value, and a mismatch would
# show a document in the library list while returning nothing when queried.
OWNER_KEY = "owner"

# Vectors scanned per page during the owner backfill. Ids plus metadata only, so
# a page is small; this is about bounding peak memory on a large collection.
BACKFILL_PAGE = 500


def _to_float_list(value: Any) -> list[float]:
    """Normalise a vector-ish value (numpy array, list, None) to list[float]."""
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    try:
        return [float(x) for x in value]
    except (TypeError, ValueError):
        return []


class VectorStore:
    """Similarity search over chunk embeddings, with metadata scoping."""

    def __init__(self, persist_dir: Path | str) -> None:
        self.persist_dir = Path(persist_dir)
        self._client: Any = None
        self._collection: Any = None

    # ---- lifecycle ------------------------------------------------------ #
    def _init_sync(self) -> None:
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
        except ImportError as exc:  # pragma: no cover
            raise VectorStoreError(
                "chromadb is not installed. Run: pip install -r requirements.txt"
            ) from exc

        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )
        # Cosine distance matches how text embedding models are trained; the
        # default L2 would make the MAX_DISTANCE threshold meaningless.
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )

    async def init(self) -> None:
        try:
            await anyio.to_thread.run_sync(self._init_sync)
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError(f"Could not open vector store: {exc}") from exc
        logger.info("Vector store ready at %s", self.persist_dir)

    @property
    def collection(self) -> Any:
        if self._collection is None:
            raise VectorStoreError("Vector store not initialised; call init() first")
        return self._collection

    async def aclose(self) -> None:
        # PersistentClient flushes on write; nothing to close explicitly.
        self._collection = None
        self._client = None

    # ---- writes --------------------------------------------------------- #
    async def upsert_chunks(
        self,
        *,
        chunk_ids: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        texts: Sequence[str],
        metadatas: Sequence[dict[str, Any]],
    ) -> None:
        """Upsert chunk vectors. Ids are deterministic, so re-ingest overwrites."""
        if not chunk_ids:
            return
        if not (len(chunk_ids) == len(embeddings) == len(texts) == len(metadatas)):
            raise VectorStoreError(
                f"upsert length mismatch: {len(chunk_ids)} ids, {len(embeddings)} "
                f"embeddings, {len(texts)} texts, {len(metadatas)} metadatas"
            )

        def _upsert() -> None:
            self.collection.upsert(
                ids=list(chunk_ids),
                embeddings=[list(map(float, e)) for e in embeddings],
                documents=list(texts),
                metadatas=list(metadatas),
            )

        try:
            await anyio.to_thread.run_sync(_upsert)
        except Exception as exc:
            raise VectorStoreError(f"Failed to write {len(chunk_ids)} vectors: {exc}") from exc

    async def delete_document(self, document_id: str, *, owner: str) -> None:
        """Remove every vector belonging to a document."""
        where = {"$and": [{"document_id": document_id}, {OWNER_KEY: owner}]}

        def _delete() -> None:
            self.collection.delete(where=where)

        try:
            await anyio.to_thread.run_sync(_delete)
        except Exception as exc:
            raise VectorStoreError(f"Failed to delete vectors: {exc}") from exc

    async def count(self) -> int:
        """Total vectors in the index, across all owners.

        Deliberately unscoped: this feeds the operator-facing health payload,
        where the interesting number is the size of the index itself.
        """

        def _count() -> int:
            return int(self.collection.count())

        try:
            return await anyio.to_thread.run_sync(_count)
        except Exception as exc:
            logger.warning("Vector count failed: %s", exc)
            return 0

    async def backfill_owner(self, owner: str) -> int:
        """Stamp ``owner`` onto vectors indexed before the field existed.

        The mirror image of the SQLite ``owner`` migration, and required for the
        same reason. There, ``NOT NULL DEFAULT 'shared'`` backfilled existing rows
        in one pass. Here, a Chroma metadata filter matches only records that
        *have* the key -- a record without ``owner`` is invisible to
        ``{"owner": "shared"}`` rather than defaulting into it. Skip this and an
        upgraded install would list its existing documents but return nothing when
        querying them: the worst kind of failure, silent and plausible.

        Chroma has no way to express "this key is absent" in a ``where`` clause,
        so this pages through the collection and looks. It is a no-op after the
        first run, but it still costs one scan per startup, which is why it fetches
        metadata only -- never embeddings.

        Returns the number of vectors updated.
        """
        def _backfill() -> int:
            updated = 0
            offset = 0
            while True:
                page = self.collection.get(
                    limit=BACKFILL_PAGE, offset=offset, include=["metadatas"]
                )
                ids = list(page.get("ids") or [])
                if not ids:
                    return updated
                metadatas = list(page.get("metadatas") or [])

                stale_ids: list[str] = []
                stale_metas: list[dict[str, Any]] = []
                for i, chunk_id in enumerate(ids):
                    meta = dict(metadatas[i]) if i < len(metadatas) and metadatas[i] else {}
                    if meta.get(OWNER_KEY):
                        continue
                    meta[OWNER_KEY] = owner
                    stale_ids.append(str(chunk_id))
                    # Written back whole rather than as a patch: update() replaces
                    # the metadata for an id, so sending only {"owner": ...} would
                    # drop document_id, page numbers, and the rest of the fields
                    # citations are rendered from.
                    stale_metas.append(meta)

                if stale_ids:
                    self.collection.update(ids=stale_ids, metadatas=stale_metas)
                    updated += len(stale_ids)

                # Advance by the page size, not by the number updated: rows that
                # already had an owner still occupy their slot in the scan.
                offset += len(ids)

        try:
            updated = await anyio.to_thread.run_sync(_backfill)
        except Exception as exc:
            # Non-fatal by choice. A failure here leaves old vectors unscoped and
            # therefore unfindable, which is a degraded search -- refusing to boot
            # would be worse, and the log line says exactly what to do about it.
            logger.warning(
                "Could not backfill vector owner metadata (%s). Documents ingested "
                "before this version may return no results; re-upload them to fix.",
                exc,
            )
            return 0

        if updated:
            logger.info("Backfilled owner=%r onto %d existing vector(s)", owner, updated)
        return updated

    # ---- reads ---------------------------------------------------------- #
    @staticmethod
    def _build_where(owner: str, document_ids: Sequence[str] | None) -> dict[str, Any]:
        """Metadata filter: owner scope, plus optional single/subset document scope.

        The owner clause is unconditional -- there is no code path that queries
        across owners, so there is nothing to forget. Chroma wants ``$and`` for
        more than one condition rather than two sibling keys, so a lone clause is
        returned bare and a pair is wrapped.
        """
        owner_clause: dict[str, Any] = {OWNER_KEY: owner}
        if not document_ids:
            return owner_clause
        if len(document_ids) == 1:
            document_clause: dict[str, Any] = {"document_id": document_ids[0]}
        else:
            document_clause = {"document_id": {"$in": list(document_ids)}}
        return {"$and": [owner_clause, document_clause]}

    async def search(
        self,
        query_embedding: Sequence[float],
        top_k: int,
        document_ids: Sequence[str] | None = None,
        *,
        owner: str,
    ) -> list[RetrievedChunk]:
        """Nearest-neighbour search within one owner's documents."""
        if not query_embedding:
            return []

        where = self._build_where(owner, document_ids)

        def _query() -> dict[str, Any]:
            return self.collection.query(
                query_embeddings=[list(map(float, query_embedding))],
                n_results=max(1, top_k),
                where=where,
                include=["documents", "metadatas", "distances", "embeddings"],
            )

        try:
            raw = await anyio.to_thread.run_sync(_query)
        except Exception as exc:
            raise VectorStoreError(f"Similarity search failed: {exc}") from exc

        return self._parse_results(raw)

    async def fetch_embeddings(self, chunk_ids: Sequence[str]) -> dict[str, list[float]]:
        """Look up stored vectors by chunk id.

        Used to backfill embeddings onto keyword-only hits. Without this, chunks
        found by BM25 but not by vector search would carry no vector, and MMR
        would silently skip them when computing diversity.

        Not owner-filtered, unlike every other read here, and deliberately so: the
        ids come from ``keyword_search``, which is already owner-scoped in SQL, and
        nothing is returned but a vector -- no text, no metadata, nothing quotable.
        Adding a redundant filter would only create a way for MMR to lose its
        inputs silently if the filter and the ids ever disagreed.
        """
        if not chunk_ids:
            return {}

        def _get() -> dict[str, Any]:
            return self.collection.get(ids=list(chunk_ids), include=["embeddings"])

        try:
            raw = await anyio.to_thread.run_sync(_get)
        except Exception as exc:
            # Non-fatal: MMR degrades, retrieval still works.
            logger.warning("Could not fetch embeddings for backfill: %s", exc)
            return {}

        ids = raw.get("ids") or []
        embeddings = raw.get("embeddings")
        if embeddings is None:
            return {}
        if hasattr(embeddings, "tolist"):
            embeddings = embeddings.tolist()

        out: dict[str, list[float]] = {}
        for i, chunk_id in enumerate(ids):
            if i < len(embeddings):
                vector = _to_float_list(embeddings[i])
                if vector:
                    out[str(chunk_id)] = vector
        return out

    @staticmethod
    def _parse_results(raw: dict[str, Any]) -> list[RetrievedChunk]:
        """Flatten Chroma's per-query nested lists into RetrievedChunk objects."""

        def first(key: str) -> list[Any]:
            value = raw.get(key)
            if value is None:
                return []
            # Chroma nests one list per query embedding; we always send exactly one.
            if len(value) > 0 and isinstance(value[0], (list, tuple)):
                return list(value[0])
            if hasattr(value, "tolist"):
                inner = value.tolist()
                return list(inner[0]) if inner and isinstance(inner[0], list) else list(inner)
            return list(value)

        ids = first("ids")
        documents = first("documents")
        metadatas = first("metadatas")
        distances = first("distances")
        embeddings = first("embeddings")

        results: list[RetrievedChunk] = []
        for i, chunk_id in enumerate(ids):
            meta = metadatas[i] if i < len(metadatas) and metadatas[i] else {}
            results.append(
                RetrievedChunk(
                    chunk_id=str(chunk_id),
                    document_id=str(meta.get("document_id", "")),
                    document_name=str(meta.get("document_name", "unknown")),
                    chunk_index=int(meta.get("chunk_index", 0) or 0),
                    page_start=int(meta.get("page_start", 1) or 1),
                    page_end=int(meta.get("page_end", 1) or 1),
                    text=str(documents[i]) if i < len(documents) and documents[i] else "",
                    distance=float(distances[i]) if i < len(distances) else None,
                    embedding=_to_float_list(embeddings[i]) if i < len(embeddings) else None,
                )
            )
        return results
