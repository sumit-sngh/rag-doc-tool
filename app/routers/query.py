"""Query routes: grounded question answering, with optional token streaming."""

from __future__ import annotations

import json
import logging
from typing import AsyncIterator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.deps import AuthDep, OwnerDep, ServicesDep
from app.errors import AppError
from app.schemas import QueryRequest, QueryResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["query"], dependencies=[AuthDep])


def _sse(event: dict) -> str:
    """Format one Server-Sent Events frame."""
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


@router.post("/query", response_model=QueryResponse, summary="Ask a grounded question")
async def query(services: ServicesDep, owner: OwnerDep, payload: QueryRequest):
    """Answer a question from the indexed documents.

    Scope with ``document_ids``: omit for the whole library, pass one id for a
    single document, or several for a subset. "Whole library" means the caller's
    own documents -- never anyone else's.

    Set ``stream: true`` for token-by-token Server-Sent Events. The stream emits
    a ``meta`` event with retrieved passages, then ``delta`` events, then a
    ``done`` event carrying the final answer with resolved citations. Citations
    can only be resolved after the full text arrives, which is why the final
    answer is re-sent rather than assembled purely from deltas.
    """
    if payload.stream:
        return StreamingResponse(
            # The owner is resolved here, in the request scope, and captured by
            # the generator. Reading it off the request inside the generator would
            # be too late: the response has already been handed to Starlette.
            _stream_events(services, payload, owner),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                # Stops nginx buffering the stream into a single flush.
                "X-Accel-Buffering": "no",
            },
        )

    result = await services.qa.answer(
        question=payload.question,
        document_ids=payload.document_ids,
        session_id=payload.session_id,
        top_k=payload.top_k,
        use_history=payload.use_history,
        owner=owner,
    )
    return QueryResponse(**result.as_dict())


async def _stream_events(
    services, payload: QueryRequest, owner: str
) -> AsyncIterator[str]:
    """Bridge the engine's event stream to SSE frames.

    Errors are emitted as an ``error`` event rather than raised: the HTTP status
    is already committed by the time streaming starts, so the only way to inform
    the client is in-band.
    """
    try:
        async for event in services.qa.stream(
            question=payload.question,
            document_ids=payload.document_ids,
            session_id=payload.session_id,
            top_k=payload.top_k,
            use_history=payload.use_history,
            owner=owner,
        ):
            yield _sse(event)
    except AppError as exc:
        logger.info("Stream failed: %s", exc.message)
        yield _sse({"type": "error", **exc.to_dict()["error"]})
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected streaming failure")
        yield _sse(
            {
                "type": "error",
                "code": "internal_error",
                "message": f"Unexpected error: {exc}",
            }
        )
    finally:
        # Sentinel so clients can distinguish a clean end from a dropped socket.
        yield "data: [DONE]\n\n"
