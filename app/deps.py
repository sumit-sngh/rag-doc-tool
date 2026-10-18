"""Service composition and FastAPI dependencies.

A single ``Services`` container is built once during app startup and hung on
``app.state``. Routers reach it through the ``ServicesDep`` dependency, so no
module-level singletons exist and tests can construct a container pointed at a
temp directory with fake providers.
"""

from __future__ import annotations

import logging
import re
import secrets
from dataclasses import dataclass
from typing import Annotated, Awaitable, Callable

from fastapi import Depends, Header, Request, Response

from app.config import Settings, get_settings
from app.db import Database
from app.errors import UnauthorizedError
from app.ingestion.pipeline import IngestionPipeline, IngestionWorker
from app.providers.base import EmbeddingProvider, LLMProvider
from app.qa.engine import QAEngine
from app.retrieval.retriever import Retriever
from app.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)


def build_embedding_provider(settings: Settings) -> EmbeddingProvider:
    if settings.embedding_provider == "fake":
        from app.providers.fake import FakeEmbeddingProvider

        return FakeEmbeddingProvider(dim=settings.embedding_dim)

    if settings.embedding_provider == "openai":
        from app.providers.openai_compat import OpenAIEmbeddingProvider

        return OpenAIEmbeddingProvider(
            base_url=settings.embed_base_url,
            api_key=settings.embed_api_key,
            model=settings.embedding_model,
            timeout=settings.embed_timeout,
            dimensions=settings.openai_embed_dimensions,
        )

    from app.providers.ollama import OllamaEmbeddingProvider

    return OllamaEmbeddingProvider(
        base_url=settings.ollama_base_url,
        model=settings.embedding_model,
        timeout=settings.embed_timeout,
    )


def build_llm_provider(settings: Settings) -> LLMProvider:
    if settings.llm_provider == "fake":
        from app.providers.fake import FakeLLMProvider

        return FakeLLMProvider()

    if settings.llm_provider == "openai":
        from app.providers.openai_compat import OpenAILLMProvider

        return OpenAILLMProvider(
            base_url=settings.openai_base_url,
            api_key=settings.openai_api_key,
            model=settings.llm_model,
            timeout=settings.request_timeout,
        )

    from app.providers.ollama import OllamaLLMProvider

    return OllamaLLMProvider(
        base_url=settings.ollama_base_url,
        model=settings.llm_model,
        timeout=settings.request_timeout,
    )


@dataclass
class Services:
    """Everything the routes need, wired together."""

    settings: Settings
    db: Database
    vector_store: VectorStore
    embeddings: EmbeddingProvider
    llm: LLMProvider
    retriever: Retriever
    qa: QAEngine
    pipeline: IngestionPipeline
    worker: IngestionWorker

    @classmethod
    def build(cls, settings: Settings | None = None) -> "Services":
        settings = settings or get_settings()
        settings.ensure_dirs()

        db = Database(settings.sqlite_path)
        vector_store = VectorStore(settings.chroma_dir)
        embeddings = build_embedding_provider(settings)
        llm = build_llm_provider(settings)

        retriever = Retriever(db, vector_store, embeddings, settings)
        qa = QAEngine(db, retriever, llm, settings)
        pipeline = IngestionPipeline(db, vector_store, embeddings, settings)
        worker = IngestionWorker(pipeline, worker_count=settings.ingest_workers)

        return cls(
            settings=settings, db=db, vector_store=vector_store,
            embeddings=embeddings, llm=llm, retriever=retriever, qa=qa,
            pipeline=pipeline, worker=worker,
        )

    async def startup(self) -> None:
        await self.db.init()
        await self.vector_store.init()
        # Paired with the SQLite `owner` migration in Database.init(). Both stores
        # are filtered on the same value, so both have to be backfilled or an
        # upgraded install would list its old documents and then find nothing in
        # them. Runs after init() because it needs the collection open.
        await self.vector_store.backfill_owner(self.settings.shared_owner)
        await self.worker.start()

    async def shutdown(self) -> None:
        await self.worker.stop()
        await self.vector_store.aclose()
        await self.embeddings.aclose()
        await self.llm.aclose()


def get_services(request: Request) -> Services:
    services = getattr(request.app.state, "services", None)
    if services is None:  # pragma: no cover - startup always sets this
        raise RuntimeError("Services not initialised; app startup did not run")
    return services


ServicesDep = Annotated[Services, Depends(get_services)]


async def require_api_key(
    request: Request,
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> None:
    """Enforce the API key when one is configured.

    Auth is opt-in: with ``API_KEY`` unset the app is open, which is the right
    default for a local single-user tool. Setting it turns on enforcement for
    every route that declares this dependency.
    """
    services = get_services(request)
    expected = services.settings.api_key
    if not expected:
        return

    supplied = x_api_key
    if not supplied:
        # Also accept "Authorization: Bearer <key>" for convenience with
        # standard HTTP clients.
        auth = request.headers.get("Authorization", "")
        if auth.lower().startswith("bearer "):
            supplied = auth[7:].strip()

    if not supplied or not secrets.compare_digest(supplied, expected):
        raise UnauthorizedError(
            "Missing or invalid API key. Send it as the 'X-API-Key' header."
        )


AuthDep = Depends(require_api_key)


# --------------------------------------------------------------------------- #
# Visitor identity
# --------------------------------------------------------------------------- #
# 18 bytes of urandom -> 24 url-safe characters. Long enough that guessing
# another visitor's id is not a realistic attack; short enough to read in a
# cookie jar while debugging.
VISITOR_ID_BYTES = 18

# Cookies arrive from the client, so the value is untrusted. Parameterised SQL
# already makes injection impossible -- this exists so a hostile or corrupted
# cookie can't write megabytes of junk into the owner column, and so every id in
# the database has the same shape as the ones we mint.
VISITOR_ID_RE = re.compile(r"^[A-Za-z0-9_-]{16,64}$")


def new_visitor_id() -> str:
    return secrets.token_urlsafe(VISITOR_ID_BYTES)


def read_visitor_cookie(request: Request, settings: Settings) -> str | None:
    raw = request.cookies.get(settings.visitor_cookie_name)
    if raw and VISITOR_ID_RE.match(raw):
        return raw
    return None


async def visitor_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Resolve the visitor for this request and persist the cookie.

    Middleware rather than a dependency, for a specific reason: FastAPI only
    merges headers from the injected ``Response`` when the endpoint returns a
    serialisable body. A route that returns a ``Response`` subclass directly --
    which the SSE streaming endpoint does -- replaces it wholesale and drops
    those headers. A dependency would therefore set the cookie on every route
    except the streaming one, and a visitor whose first request was a stream
    would get a fresh identity on their next call.

    Setting it here also means the frontend needs no changes at all: the browser
    carries the cookie automatically, and nothing has to be kept in
    localStorage, which is consistent with how the API key is already handled.
    """
    settings: Settings = request.app.state.settings

    if not settings.visitor_isolation:
        request.state.owner = settings.shared_owner
        return await call_next(request)

    owner = read_visitor_cookie(request, settings) or new_visitor_id()
    request.state.owner = owner

    response = await call_next(request)
    # Re-issued on every response so an active visitor's expiry slides forward,
    # and so a newly minted id is persisted even when the response is a stream.
    response.set_cookie(
        key=settings.visitor_cookie_name,
        value=owner,
        max_age=settings.visitor_cookie_days * 86400,
        path="/",
        httponly=True,  # JS never reads this; keeps it out of XSS reach
        samesite="lax",
        secure=settings.secure_cookies,
    )
    return response


def get_owner(request: Request) -> str:
    """The owner scope for this request.

    Reads what the middleware resolved. If isolation is on and that value is
    missing, the middleware is not installed -- which would mean every visitor
    silently sharing one library. That is the failure this whole feature exists
    to prevent, so it raises rather than guessing.
    """
    owner = getattr(request.state, "owner", None)
    if owner:
        return str(owner)

    settings = get_services(request).settings
    if settings.visitor_isolation:
        raise RuntimeError(
            "VISITOR_ISOLATION is on but no visitor was resolved for this "
            "request. visitor_middleware is not installed on the app."
        )
    return settings.shared_owner


OwnerDep = Annotated[str, Depends(get_owner)]
