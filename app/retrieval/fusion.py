"""Candidate fusion for hybrid (vector + keyword) retrieval.

Reciprocal Rank Fusion combines ranked lists without needing their scores to be
on comparable scales -- which matters here because cosine distance from Chroma
and BM25 relevance from SQLite FTS5 are not remotely comparable. RRF only looks
at *rank position*, which sidesteps normalisation entirely.

    score(d) = sum over lists L of  weight(L) / (k + rank_L(d))

Stdlib only, so it is cheap to test.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Standard RRF damping constant from Cormack et al. (2009). Larger values
# flatten the contribution of top ranks; 60 is the widely used default.
RRF_K = 60


@dataclass
class RetrievedChunk:
    """A retrieval candidate, carried through fusion, reranking, and citation."""

    chunk_id: str
    document_id: str
    document_name: str
    chunk_index: int
    page_start: int
    page_end: int
    text: str

    # Raw signals (whichever retriever produced this candidate fills these in)
    distance: float | None = None  # cosine distance from vector search; lower=closer
    keyword_score: float | None = None  # BM25 relevance; higher=better

    # Populated by the pipeline
    fused_score: float = 0.0
    rerank_score: float | None = None
    embedding: list[float] | None = field(default=None, repr=False)

    @property
    def similarity(self) -> float | None:
        """Cosine distance expressed as a 0..1-ish similarity, for display."""
        if self.distance is None:
            return None
        return max(0.0, 1.0 - self.distance / 2.0)

    @property
    def page_label(self) -> str:
        if self.page_start == self.page_end:
            return f"p. {self.page_start}"
        return f"pp. {self.page_start}-{self.page_end}"


def reciprocal_rank_fusion(
    vector_hits: list[RetrievedChunk],
    keyword_hits: list[RetrievedChunk],
    vector_weight: float = 0.7,
    rrf_k: int = RRF_K,
) -> list[RetrievedChunk]:
    """Merge two ranked candidate lists into one, deduplicating by chunk_id.

    Both inputs must already be sorted best-first. `vector_weight` in [0,1]
    splits influence between the two lists; the keyword list gets the remainder.
    A chunk found by both retrievers accumulates both contributions, which is
    exactly the behaviour we want -- agreement between independent signals is
    strong evidence of relevance.
    """
    keyword_weight = 1.0 - vector_weight
    merged: dict[str, RetrievedChunk] = {}
    scores: dict[str, float] = {}

    for hits, weight in ((vector_hits, vector_weight), (keyword_hits, keyword_weight)):
        if weight <= 0:
            continue
        for rank, hit in enumerate(hits, start=1):
            existing = merged.get(hit.chunk_id)
            if existing is None:
                merged[hit.chunk_id] = hit
                scores[hit.chunk_id] = 0.0
            else:
                # Same chunk from the other retriever: keep whichever signals
                # each list carries so downstream filters see both.
                if existing.distance is None and hit.distance is not None:
                    existing.distance = hit.distance
                if existing.keyword_score is None and hit.keyword_score is not None:
                    existing.keyword_score = hit.keyword_score
                if existing.embedding is None and hit.embedding is not None:
                    existing.embedding = hit.embedding
            scores[hit.chunk_id] += weight / (rrf_k + rank)

    results: list[RetrievedChunk] = []
    for chunk_id, chunk in merged.items():
        chunk.fused_score = scores[chunk_id]
        results.append(chunk)

    # Tie-break on chunk_id so ordering is deterministic across runs.
    results.sort(key=lambda c: (-c.fused_score, c.chunk_id))
    return results


def filter_by_distance(
    chunks: list[RetrievedChunk], max_distance: float
) -> list[RetrievedChunk]:
    """Drop chunks the embedding model considers too far from the query.

    This is what allows an honest "no relevant matches" answer instead of the
    model confabulating from whatever happened to rank top-k. Keyword-only hits
    (no distance) are kept -- an exact term match is meaningful evidence even
    when the embedding is lukewarm.
    """
    return [c for c in chunks if c.distance is None or c.distance <= max_distance]
