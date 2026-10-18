"""FastAPI application entry point.

    uvicorn app.main:app --reload

Wires the service container into the app lifespan, registers exception handlers
that render every failure in one consistent envelope, mounts the routers, and
serves the frontend as static files from the same origin (so no CORS setup is
needed for normal use).
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import BASE_DIR, Settings, get_settings
from app.deps import Services, visitor_middleware
from app.errors import AppError
from app.routers import documents, health, history, query

logger = logging.getLogger(__name__)

FRONTEND_DIR = BASE_DIR / "frontend"
VERSION = health.VERSION

DESCRIPTION = """
Ask natural-language questions about your own documents and get answers grounded
in retrieved passages, with citations you can verify.

**Flow**

1. `POST /documents/upload` — send a PDF, DOCX, TXT, or MD file. Returns `202`
   immediately; text extraction and embedding happen in the background.
2. `GET /documents` — poll until `status` is `ready`.
3. `POST /query` — ask a question. Scope it with `document_ids`, or omit that to
   search the whole library. Set `stream: true` for token-by-token SSE.
4. `GET /history` — review past exchanges.

Answers carry a `grounded` flag. When it is `false`, nothing relevant was
retrieved (or the model produced no verifiable citation) and the answer text says
so rather than guessing.
"""


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )
    # Chroma and httpx are chatty at INFO and drown out our own logs. Chroma's
    # broken telemetry logger is silenced in app.retrieval.vector_store instead,
    # at import time, because the tests bypass this lifespan entirely.
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = getattr(app.state, "settings", None) or get_settings()
    configure_logging(settings.log_level)

    services = Services.build(settings)
    app.state.services = services
    await services.startup()

    logger.info(
        "%s ready | embeddings=%s:%s llm=%s:%s | chunk=%d/%d top_k=%d",
        settings.app_name,
        settings.embedding_provider, settings.embedding_model,
        settings.llm_provider, settings.llm_model,
        settings.chunk_size, settings.chunk_overlap, settings.top_k,
    )
    try:
        yield
    finally:
        await services.shutdown()


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the application.

    A factory rather than a module-level singleton so tests can stand up an
    isolated app against a temp directory, and so a deployment can inject
    settings without going through the env-var cache.
    """
    settings = settings or get_settings()

    app = FastAPI(
        title="Document Q&A Tool",
        description=DESCRIPTION,
        version=VERSION,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )
    app.state.settings = settings

    origins = settings.cors_origin_list
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        # Browsers reject `Allow-Credentials: true` alongside `Allow-Origin: *`,
        # and the bundled frontend is same-origin anyway, so only enable it when
        # the operator has named explicit origins.
        allow_credentials="*" not in origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Added after CORS, so it sits outside it and runs first. Every request gets
    # request.state.owner before any route sees it, including the SSE stream --
    # see visitor_middleware for why this cannot be a dependency.
    app.middleware("http")(visitor_middleware)

    register_handlers(app)

    app.include_router(health.router)
    app.include_router(documents.router)
    app.include_router(query.router)
    app.include_router(history.router)

    # response_model=None because FastAPI otherwise treats the return annotation
    # as a response model to validate against, and a Starlette Response subclass
    # is not a Pydantic type -- app creation fails outright with "Invalid args
    # for response field". Returning a Response instance bypasses serialization
    # anyway, so there is nothing to model here.
    @app.get("/", include_in_schema=False, response_model=None)
    async def index() -> FileResponse | JSONResponse:
        """Serve the frontend, or point at the docs if it isn't present."""
        index_file = FRONTEND_DIR / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        return JSONResponse(
            {"message": "Document Q&A Tool API. See /docs for the OpenAPI UI."}
        )

    # Mounted last so it can't shadow the API routes above.
    if FRONTEND_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    return app


# --------------------------------------------------------------------------- #
# Error handling -- one envelope shape for every failure:
#   {"error": {"code": "...", "message": "...", "details": ...}}
# so the frontend can branch on error.code instead of parsing prose.
# --------------------------------------------------------------------------- #
def register_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        if exc.status_code >= 500:
            logger.error("%s on %s: %s", exc.code, request.url.path, exc.message)
        else:
            logger.info("%s on %s: %s", exc.code, request.url.path, exc.message)
        return JSONResponse(status_code=exc.status_code, content=exc.to_dict())

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """Reshape FastAPI's 422 body into the same envelope as everything else."""
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "validation_error",
                    "message": "Request validation failed",
                    "details": [
                        {
                            "field": ".".join(str(p) for p in err.get("loc", [])),
                            "message": err.get("msg", ""),
                        }
                        for err in exc.errors()
                    ],
                }
            },
        )

    @app.exception_handler(Exception)
    async def handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error on %s", request.url.path)
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "internal_error",
                    # Deliberately generic: internals shouldn't leak to clients.
                    "message": "An unexpected error occurred. Check the server logs.",
                }
            },
        )


app = create_app()
