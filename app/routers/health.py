"""Health and configuration introspection.

Unauthenticated on purpose so orchestrators can probe it, and because the most
common setup failure -- Ollama not running, or the model not pulled -- needs to be
diagnosable before anything else works. No secrets are echoed: ``api_key`` is
reported only as a boolean.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.deps import OwnerDep, ServicesDep
from app.schemas import ComponentHealth, HealthResponse

router = APIRouter(tags=["health"])

VERSION = "1.0.0"


@router.get("/health", response_model=HealthResponse, summary="Health and config")
async def health(services: ServicesDep, owner: OwnerDep) -> HealthResponse:
    settings = services.settings

    embed_ok, embed_detail = await services.embeddings.health()
    llm_ok, llm_detail = await services.llm.health()

    try:
        vector_count = await services.vector_store.count()
        vector_ok, vector_detail = True, f"{vector_count} vector(s) indexed"
    except Exception as exc:  # noqa: BLE001
        vector_count = 0
        vector_ok, vector_detail = False, str(exc)

    try:
        # Owner-scoped, so the counts match what the caller can actually see. The
        # vector count above is not: it measures the index, which is the number an
        # operator watching this endpoint wants. With isolation on they will
        # disagree, and that is correct rather than a bug.
        docs, total_docs = await services.db.list_documents(owner=owner, limit=1)
        chunk_count = await services.db.count_chunks(owner=owner)
        db_ok, db_detail = True, f"{total_docs} document(s), {chunk_count} chunk(s)"
    except Exception as exc:  # noqa: BLE001
        total_docs, chunk_count = 0, 0
        db_ok, db_detail = False, str(exc)

    components = {
        "database": ComponentHealth(ok=db_ok, detail=db_detail),
        "vector_store": ComponentHealth(ok=vector_ok, detail=vector_detail),
        "embeddings": ComponentHealth(ok=embed_ok, detail=embed_detail),
        "llm": ComponentHealth(ok=llm_ok, detail=llm_detail),
    }

    return HealthResponse(
        status="ok" if all(c.ok for c in components.values()) else "degraded",
        version=VERSION,
        config={
            "embedding_provider": settings.embedding_provider,
            "embedding_model": settings.embedding_model,
            "llm_provider": settings.llm_provider,
            "llm_model": settings.llm_model,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "top_k": settings.top_k,
            "candidate_k": settings.candidate_k,
            "hybrid_search": settings.hybrid_search,
            "vector_weight": settings.vector_weight,
            "rerank_strategy": settings.rerank_strategy,
            "max_distance": settings.max_distance,
            "ocr_enabled": settings.ocr_enabled,
            "max_upload_mb": settings.max_upload_mb,
            "auth_required": bool(settings.api_key),
            # Advertised so the frontend can say "your documents" rather than
            # "all documents" without guessing at the deployment's shape.
            "visitor_isolation": settings.visitor_isolation,
            "max_documents_per_visitor": settings.max_documents_per_visitor,
        },
        components=components,
        stats={
            "documents": total_docs,
            "chunks": chunk_count,
            "vectors": vector_count,
            "queue_depth": services.worker.depth,
        },
    )
