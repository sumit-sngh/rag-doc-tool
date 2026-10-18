"""Retrieval orchestration: vector + keyword -> fusion -> filter -> rerank.

The pipeline for one question:

1. Embed the question.
2. Vector search for ``candidate_k`` nearest chunks (scoped to the requesting
   visitor, and to the requested documents, via Chroma metadata filtering).
3. BM25 keyword search over the same scope, when hybrid search is on.
4. Fuse both ranked lists with Reciprocal Rank Fusion.
5. Backfill embeddings onto keyword-only hits so reranking sees every candidate.
6. Drop candidates beyond the distance threshold -- this is what makes an honest
   "no relevant matches" possible.
7. Rerank and narrow to ``top_k``.

Retrieving ``candidate_k`` (default 20) before narrowing to ``top_k`` (default 5)
is deliberate: reranking can only reorder what it was given, so the wider net
raises the ceiling on final quality.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

import anyio

from app.config import Settings
from app.db import Database
from app.errors import VectorStoreError
from app.providers.base import EmbeddingProvider
from app.retrieval.fusion import (
    RetrievedChunk,
    filter_by_distance,
    reciprocal_rank_fusion,
)
from app.retrieval.rerank import rerank
from app.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Retrieved chunks plus diagnostics about how they were found."""

    chunks: list[RetrievedChunk]
    query_embedding: list[float] = field(default_factory=list, repr=False)
    vector_hits: int = 0
    keyword_hits: int = 0
    candidates_considered: int = 0
    filtered_out: int = 0
    strategy: str = "vector"

    @property
    def is_empty(self) -> bool:
        return not self.chunks


class Retriever:
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

    async def retrieve(
        self,
        question: str,
        document_ids: Sequence[str] | None = None,
        top_k: int | None = None,
        *,
        owner: str,
    ) -> RetrievalResult:
        """Retrieve for one question, within one owner's documents.

        ``owner`` is keyword-only and has no default on purpose. Both halves of the
        hybrid have to be scoped to the same visitor, and a forgotten argument here
        is the one mistake that would put another person's text in front of the
        model as a citation. A TypeError at the call site is the cheapest possible
        version of that bug.
        """
        cfg = self.settings
        k = top_k or cfg.top_k
        # Never fetch fewer candidates than we intend to return.
        candidate_k = max(cfg.candidate_k, k)

        query_embedding = await self.embeddings.embed_query(question)

        use_hybrid = cfg.hybrid_search and cfg.vector_weight < 1.0
        vector_hits: list[RetrievedChunk] = []
        keyword_hits: list[RetrievedChunk] = []

        if use_hybrid:
            # Independent stores, so run both concurrently -- the request waits
            # on the slower of the two rather than their sum.
            async def _vector() -> None:
                nonlocal vector_hits
                vector_hits = await self.vector_store.search(
                    query_embedding, candidate_k, document_ids, owner=owner
                )

            async def _keyword() -> None:
                nonlocal keyword_hits
                keyword_hits = await self.db.keyword_search(
                    question, candidate_k, owner=owner, document_ids=document_ids
                )

            async with anyio.create_task_group() as tg:
                tg.start_soon(_vector)
                tg.start_soon(_keyword)
        else:
            vector_hits = await self.vector_store.search(
                query_embedding, candidate_k, document_ids, owner=owner
            )

        strategy = "hybrid" if use_hybrid and keyword_hits else "vector"

        candidates = (
            reciprocal_rank_fusion(
                vector_hits, keyword_hits, vector_weight=cfg.vector_weight
            )
            if keyword_hits
            else vector_hits
        )
        considered = len(candidates)

        await self._backfill_embeddings(candidates)

        kept = filter_by_distance(candidates, cfg.max_distance)
        filtered_out = considered - len(kept)

        if not kept:
            logger.info(
                "No candidates within max_distance=%.2f for %r (considered %d)",
                cfg.max_distance, question[:80], considered,
            )
            return RetrievalResult(
                chunks=[], query_embedding=query_embedding,
                vector_hits=len(vector_hits), keyword_hits=len(keyword_hits),
                candidates_considered=considered, filtered_out=filtered_out,
                strategy=strategy,
            )

        final = rerank(
            kept,
            question=question,
            query_embedding=query_embedding,
            top_k=k,
            strategy=cfg.rerank_strategy,
            mmr_lambda=cfg.mmr_lambda,
            cross_encoder_model=cfg.cross_encoder_model,
        )

        return RetrievalResult(
            chunks=final, query_embedding=query_embedding,
            vector_hits=len(vector_hits), keyword_hits=len(keyword_hits),
            candidates_considered=considered, filtered_out=filtered_out,
            strategy=strategy,
        )

    async def _backfill_embeddings(self, candidates: list[RetrievedChunk]) -> None:
        """Attach stored vectors to candidates that arrived without one."""
        missing = [c.chunk_id for c in candidates if not c.embedding]
        if not missing:
            return
        try:
            found = await self.vector_store.fetch_embeddings(missing)
        except VectorStoreError as exc:
            logger.warning("Embedding backfill failed: %s", exc)
            return
        for chunk in candidates:
            if not chunk.embedding:
                vector = found.get(chunk.chunk_id)
                if vector:
                    chunk.embedding = vector
