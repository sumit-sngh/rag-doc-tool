"""Session and Q&A history routes."""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Query

from app.deps import AuthDep, OwnerDep, ServicesDep
from app.errors import InvalidRequestError, SessionNotFoundError
from app.schemas import (
    DeleteResponse,
    HistoryEntry,
    HistoryList,
    SessionList,
    SessionSummary,
)

router = APIRouter(tags=["history"], dependencies=[AuthDep])


@router.get("/history", response_model=HistoryList, summary="List past Q&A exchanges")
async def list_history(
    services: ServicesDep,
    owner: OwnerDep,
    session_id: Annotated[str | None, Query(description="Filter to one session")] = None,
    document_id: Annotated[
        str | None, Query(description="Filter to exchanges scoped to this document")
    ] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> HistoryList:
    """Past exchanges, newest first, optionally filtered by session or document."""
    if session_id and document_id:
        raise InvalidRequestError(
            "Filter by session_id or document_id, not both"
        )

    rows = await services.db.list_history(
        owner=owner,
        session_id=session_id,
        document_id=document_id,
        limit=limit,
        offset=offset,
    )
    return HistoryList(
        history=[HistoryEntry(**r) for r in rows], limit=limit, offset=offset
    )


@router.post("/sessions", response_model=SessionSummary, status_code=201,
             summary="Create a session")
async def create_session(services: ServicesDep, owner: OwnerDep) -> SessionSummary:
    """Create a conversation session.

    Optional: passing an unseen ``session_id`` to ``/query`` creates one
    implicitly. This exists for clients that prefer an explicit handle.
    """
    session = await services.db.create_session(str(uuid.uuid4()), owner=owner)
    return SessionSummary(**session)


@router.get("/sessions", response_model=SessionList, summary="List sessions")
async def list_sessions(
    services: ServicesDep,
    owner: OwnerDep,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> SessionList:
    rows = await services.db.list_sessions(owner=owner, limit=limit, offset=offset)
    return SessionList(sessions=[SessionSummary(**r) for r in rows])


@router.get("/sessions/{session_id}", response_model=HistoryList,
            summary="Get a session's exchanges")
async def get_session(
    services: ServicesDep,
    owner: OwnerDep,
    session_id: str,
    limit: Annotated[int, Query(ge=1, le=200)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> HistoryList:
    session = await services.db.get_session(session_id, owner=owner)
    if session is None:
        raise SessionNotFoundError(f"No session with id '{session_id}'")

    rows = await services.db.list_history(
        owner=owner, session_id=session_id, limit=limit, offset=offset
    )
    return HistoryList(
        history=[HistoryEntry(**r) for r in rows], limit=limit, offset=offset
    )


@router.delete("/sessions/{session_id}", response_model=DeleteResponse,
               summary="Delete a session")
async def delete_session(
    services: ServicesDep, owner: OwnerDep, session_id: str
) -> DeleteResponse:
    """Delete a session and, by FK cascade, all of its exchanges."""
    session = await services.db.get_session(session_id, owner=owner)
    if session is None:
        raise SessionNotFoundError(f"No session with id '{session_id}'")

    await services.db.delete_session(session_id, owner=owner)
    return DeleteResponse(deleted=session_id, message="Session deleted")
