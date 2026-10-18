"""Reranking of retrieval candidates before they are handed to the LLM.

Three strategies, selected by ``RERANK_STRATEGY``:

``none``
    Keep fusion order. Fastest.

``mmr`` (default)
    Maximal Marginal Relevance -- greedily pick chunks that are relevant to the
    query *and* dissimilar to what's already picked. This matters more than it
    sounds for RAG: overlapping chunks mean near-duplicate neighbours often
    occupy several top-k slots, spending context window on repeated text. MMR
    trades a little relevance for coverage, so the LLM sees more of the document.
    Reuses embeddings already computed during retrieval -- no extra model.
    Takes its relevance signal from the fused score when hybrid retrieval
    supplied one, so the keyword half of the ranking survives this step.

``cross_encoder``
    Genuine query-document relevance scoring with a sentence-transformers
    cross-encoder. Most accurate, but pulls in torch and adds latency.

Pure-python vector math: candidate lists are ~20 items, so numpy would not pay
for itself and this keeps the module trivially testable.
"""

from __future__ import annotations

import logging
import math
from functools import lru_cache

from app.retrieval.fusion import RetrievedChunk

logger = logging.getLogger(__name__)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity in [-1, 1]; 0.0 for zero-magnitude or mismatched input."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


def _relevance_scores(
    candidates: list[RetrievedChunk], query_embedding: list[float]
) -> dict[str, float]:
    """Per-candidate relevance for MMR, in a range comparable to cosine.

    Prefers the fused score when hybrid retrieval produced one. Scoring purely
    on query-chunk cosine here would silently discard the keyword half of
    hybrid search: a chunk that BM25 ranked first but the embedder ranked
    fourth would be judged only on that fourth-place vector similarity, and RRF
    -- which exists precisely to promote it -- would have no effect on the final
    top-k. That is not a hypothetical; it dropped an exact-term match that
    fusion had ranked #1 out of the results entirely.

    RRF scores are deliberately compressed (1/(60+rank) barely separates rank 1
    from rank 5), so they are min-max rescaled onto 0..1. Otherwise relevance
    would be near-constant across candidates and the diversity term would decide
    everything.
    """
    fused = [c.fused_score for c in candidates]
    if any(score > 0.0 for score in fused):
        low, high = min(fused), max(fused)
        if high > low:
            span = high - low
            return {c.chunk_id: (c.fused_score - low) / span for c in candidates}
    # Vector-only retrieval: no fusion ran, so cosine is the only signal there is.
    return {
        c.chunk_id: cosine_similarity(query_embedding, c.embedding or [])
        for c in candidates
    }


def mmr_rerank(
    candidates: list[RetrievedChunk],
    query_embedding: list[float],
    top_k: int,
    lambda_mult: float = 0.7,
) -> list[RetrievedChunk]:
    """Select `top_k` chunks balancing query relevance against redundancy.

    lambda_mult=1.0 is pure relevance (equivalent to no reranking);
    lambda_mult=0.0 is pure diversity. 0.7 leans toward relevance while still
    breaking up near-duplicate runs.

    Relevance comes from `_relevance_scores`, so a fused hybrid ranking survives
    this step instead of being recomputed away.
    """
    if top_k <= 0 or not candidates:
        return []
    if len(candidates) <= 1:
        return candidates[:top_k]

    # MMR needs embeddings for the diversity term. If retrieval didn't supply
    # them (e.g. keyword-only hits), fall back rather than silently misranking.
    usable = [c for c in candidates if c.embedding]
    if len(usable) < 2 or not query_embedding:
        logger.debug("MMR skipped: insufficient embeddings on candidates")
        return candidates[:top_k]

    # Scored over `usable`, not `candidates`, so the 0..1 rescale spans exactly
    # the set MMR can actually choose between.
    relevance = _relevance_scores(usable, query_embedding)
    # Memoise pairwise similarity -- each pair is consulted repeatedly as the
    # selected set grows.
    pair_cache: dict[tuple[str, str], float] = {}

    def pair_sim(x: RetrievedChunk, y: RetrievedChunk) -> float:
        key = (x.chunk_id, y.chunk_id) if x.chunk_id < y.chunk_id else (y.chunk_id, x.chunk_id)
        if key not in pair_cache:
            pair_cache[key] = cosine_similarity(x.embedding or [], y.embedding or [])
        return pair_cache[key]

    remaining = list(usable)
    # Seed with the single most query-relevant chunk.
    first = max(remaining, key=lambda c: relevance[c.chunk_id])
    first.rerank_score = relevance[first.chunk_id]
    selected = [first]
    remaining.remove(first)

    while remaining and len(selected) < top_k:
        best: RetrievedChunk | None = None
        best_score = -math.inf
        for cand in remaining:
            redundancy = max(pair_sim(cand, s) for s in selected)
            score = lambda_mult * relevance[cand.chunk_id] - (1 - lambda_mult) * redundancy
            if score > best_score:
                best_score, best = score, cand
        if best is None:
            break
        best.rerank_score = best_score
        selected.append(best)
        remaining.remove(best)

    # Top up from candidates MMR couldn't consider (no embedding) if short.
    if len(selected) < top_k:
        chosen = {c.chunk_id for c in selected}
        for c in candidates:
            if c.chunk_id not in chosen:
                selected.append(c)
                if len(selected) >= top_k:
                    break

    return selected[:top_k]


@lru_cache(maxsize=2)
def _load_cross_encoder(model_name: str):
    """Load and cache a cross-encoder. Returns None if unavailable."""
    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        logger.warning(
            "RERANK_STRATEGY=cross_encoder but sentence-transformers is not "
            "installed (see requirements-optional.txt). Falling back to MMR."
        )
        return None
    try:
        return CrossEncoder(model_name)
    except Exception as exc:  # pragma: no cover - network/model load failure
        logger.warning("Could not load cross-encoder %s: %s", model_name, exc)
        return None


def cross_encoder_rerank(
    candidates: list[RetrievedChunk],
    question: str,
    top_k: int,
    model_name: str,
) -> list[RetrievedChunk] | None:
    """Rerank with a cross-encoder. Returns None if the model is unavailable."""
    model = _load_cross_encoder(model_name)
    if model is None or not candidates:
        return None
    try:
        pairs = [(question, c.text) for c in candidates]
        scores = model.predict(pairs)
    except Exception as exc:  # pragma: no cover
        logger.warning("Cross-encoder scoring failed: %s", exc)
        return None

    for cand, score in zip(candidates, scores):
        cand.rerank_score = float(score)
    ranked = sorted(candidates, key=lambda c: -(c.rerank_score or 0.0))
    return ranked[:top_k]


def rerank(
    candidates: list[RetrievedChunk],
    question: str,
    query_embedding: list[float],
    top_k: int,
    strategy: str = "mmr",
    mmr_lambda: float = 0.7,
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
) -> list[RetrievedChunk]:
    """Apply the configured rerank strategy, degrading gracefully."""
    if strategy == "none":
        return candidates[:top_k]

    if strategy == "cross_encoder":
        result = cross_encoder_rerank(candidates, question, top_k, cross_encoder_model)
        if result is not None:
            return result
        # fall through to MMR

    return mmr_rerank(candidates, query_embedding, top_k, mmr_lambda)
