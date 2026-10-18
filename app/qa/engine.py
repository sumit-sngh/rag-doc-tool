"""Question answering: retrieve, ground, generate, cite, record.

Order of operations matters here, and two choices are worth calling out:

* **The LLM is not called when retrieval finds nothing.** If no chunk clears the
  distance threshold, the endpoint returns ``grounded: false`` with an honest
  message. Handing a model an empty context and hoping it declines to answer is
  how RAG systems end up confidently inventing things.

* **Citations resolve against the chunks that fit in the context window**, not
  everything retrieved. ``build_messages`` returns the included subset, and that
  subset is what source numbers refer to. Using the pre-truncation list would
  misattribute every citation after the cut.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Sequence

from app.config import Settings
from app.db import Database
from app.errors import DocumentNotFoundError, DocumentNotReadyError, NoDocumentsError
from app.providers.base import LLMProvider
from app.qa.citations import resolve_citations
from app.qa.prompts import INSUFFICIENT_CONTEXT, build_messages
from app.retrieval.fusion import RetrievedChunk
from app.retrieval.retriever import RetrievalResult, Retriever

logger = logging.getLogger(__name__)

NO_CONTEXT_MESSAGE = (
    "I could not find anything in your documents that answers this question. "
    "Try rephrasing it, or check that the relevant document has finished processing."
)

UNGROUNDED_MESSAGE = (
    "The retrieved passages do not contain enough information to answer this "
    "question. The closest matches are listed below so you can judge for yourself."
)

# How many prior turns of a session to replay for follow-up questions. Enough for
# "what about the other one?" to resolve, without crowding out retrieved context.
HISTORY_TURNS = 3


def citation_dict(index: int, chunk: RetrievedChunk) -> dict[str, Any]:
    """Serialise a chunk as a user-facing citation."""
    return {
        "number": index,
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "document_name": chunk.document_name,
        "chunk_index": chunk.chunk_index,
        "page_start": chunk.page_start,
        "page_end": chunk.page_end,
        "page_label": chunk.page_label,
        "text": chunk.text,
        "distance": chunk.distance,
        "similarity": chunk.similarity,
        "keyword_score": chunk.keyword_score,
        "rerank_score": chunk.rerank_score,
    }


@dataclass
class QAResult:
    answer: str
    grounded: bool
    sources: list[dict[str, Any]] = field(default_factory=list)
    retrieved: list[dict[str, Any]] = field(default_factory=list)
    session_id: str | None = None
    history_id: str | None = None
    latency_ms: int = 0
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "grounded": self.grounded,
            "sources": self.sources,
            "retrieved": self.retrieved,
            "session_id": self.session_id,
            "history_id": self.history_id,
            "latency_ms": self.latency_ms,
            "diagnostics": self.diagnostics,
        }


class QAEngine:
    def __init__(
        self,
        db: Database,
        retriever: Retriever,
        llm: LLMProvider,
        settings: Settings,
    ) -> None:
        self.db = db
        self.retriever = retriever
        self.llm = llm
        self.settings = settings

    # ---- scope validation ----------------------------------------------- #
    async def _validate_scope(
        self, document_ids: Sequence[str] | None, owner: str
    ) -> None:
        """Fail fast with a specific error rather than returning empty results.

        Doubles as the authorisation check on ``document_ids``. The lookup is
        owner-scoped, so a visitor naming someone else's document id gets the same
        "no document with id" they would get for a nonexistent one -- which is the
        right answer, and reveals nothing about whether it exists.
        """
        if document_ids:
            for doc_id in document_ids:
                doc = await self.db.get_document(doc_id, owner=owner)
                if doc is None:
                    raise DocumentNotFoundError(f"No document with id '{doc_id}'")
                if doc["status"] != "ready":
                    raise DocumentNotReadyError(
                        f"'{doc['original_name']}' is not ready to query "
                        f"(status: {doc['status']}"
                        + (f" - {doc['error']}" if doc.get("error") else "")
                        + ")",
                        details={"document_id": doc_id, "status": doc["status"]},
                    )
            return

        # Whole-library scope: make sure something is actually queryable.
        ready, _ = await self.db.list_documents(owner=owner, limit=1, status="ready")
        if not ready:
            raise NoDocumentsError(
                "No documents are ready to query yet. Upload one, or wait for "
                "an in-progress upload to finish processing."
            )

    async def _history_turns(
        self, session_id: str | None, owner: str
    ) -> list[tuple[str, str]]:
        """Recent (question, answer) pairs, oldest first, for follow-up context."""
        if not session_id:
            return []
        rows = await self.db.list_history(
            owner=owner, session_id=session_id, limit=HISTORY_TURNS
        )
        # Stored newest-first; the prompt needs chronological order.
        return [
            (r["question"], r["answer"])
            for r in reversed(rows)
            if r.get("grounded", True)
        ]

    # ---- non-streaming -------------------------------------------------- #
    async def answer(
        self,
        question: str,
        document_ids: Sequence[str] | None = None,
        session_id: str | None = None,
        top_k: int | None = None,
        use_history: bool = True,
        *,
        owner: str,
    ) -> QAResult:
        started = time.perf_counter()
        await self._validate_scope(document_ids, owner)

        retrieval = await self.retriever.retrieve(
            question, document_ids, top_k, owner=owner
        )

        if retrieval.is_empty:
            return await self._finish(
                question=question, answer=NO_CONTEXT_MESSAGE, grounded=False,
                cited=[], included=[], retrieval=retrieval,
                document_ids=document_ids, session_id=session_id, owner=owner,
                started=started,
            )

        history = await self._history_turns(session_id, owner) if use_history else []
        messages, included = build_messages(
            question, retrieval.chunks, self.settings.max_context_chars, history
        )

        raw = await self.llm.generate(
            messages,
            temperature=self.settings.llm_temperature,
            max_tokens=self.settings.llm_max_tokens,
        )

        answer_text, cited, grounded = self._interpret(raw, included)
        return await self._finish(
            question=question, answer=answer_text, grounded=grounded,
            cited=cited, included=included, retrieval=retrieval,
            document_ids=document_ids, session_id=session_id, owner=owner,
            started=started,
        )

    # ---- streaming ------------------------------------------------------ #
    async def stream(
        self,
        question: str,
        document_ids: Sequence[str] | None = None,
        session_id: str | None = None,
        top_k: int | None = None,
        use_history: bool = True,
        *,
        owner: str,
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield SSE-shaped events: ``meta`` -> ``delta``* -> ``done``.

        Citations can only be resolved once the full text has arrived, so deltas
        stream raw and the ``done`` event carries the final, renumbered answer
        plus resolved sources. Clients render deltas for responsiveness and
        replace with the ``done`` payload at the end.
        """
        started = time.perf_counter()
        await self._validate_scope(document_ids, owner)

        retrieval = await self.retriever.retrieve(
            question, document_ids, top_k, owner=owner
        )

        if retrieval.is_empty:
            result = await self._finish(
                question=question, answer=NO_CONTEXT_MESSAGE, grounded=False,
                cited=[], included=[], retrieval=retrieval,
                document_ids=document_ids, session_id=session_id, owner=owner,
                started=started,
            )
            yield {"type": "meta", "retrieved": [], "diagnostics": result.diagnostics}
            yield {"type": "delta", "text": NO_CONTEXT_MESSAGE}
            yield {"type": "done", **result.as_dict()}
            return

        history = await self._history_turns(session_id, owner) if use_history else []
        messages, included = build_messages(
            question, retrieval.chunks, self.settings.max_context_chars, history
        )

        yield {
            "type": "meta",
            "retrieved": [citation_dict(i, c) for i, c in enumerate(included, 1)],
            "diagnostics": self._diagnostics(retrieval, included),
        }

        parts: list[str] = []
        async for delta in self.llm.stream(
            messages,
            temperature=self.settings.llm_temperature,
            max_tokens=self.settings.llm_max_tokens,
        ):
            parts.append(delta)
            yield {"type": "delta", "text": delta}

        answer_text, cited, grounded = self._interpret("".join(parts), included)
        result = await self._finish(
            question=question, answer=answer_text, grounded=grounded,
            cited=cited, included=included, retrieval=retrieval,
            document_ids=document_ids, session_id=session_id, owner=owner,
            started=started,
        )
        yield {"type": "done", **result.as_dict()}

    # ---- shared helpers ------------------------------------------------- #
    def _interpret(
        self, raw: str, included: list[RetrievedChunk]
    ) -> tuple[str, list[RetrievedChunk], bool]:
        """Turn a raw completion into (answer, cited chunks, grounded flag)."""
        text = (raw or "").strip()

        # The model was told to emit this exact token when the context is
        # insufficient. Substring rather than equality: models often wrap it in
        # a sentence despite instructions.
        if not text or INSUFFICIENT_CONTEXT in text:
            return UNGROUNDED_MESSAGE, [], False

        answer, cited, invalid = resolve_citations(text, included)
        if invalid:
            logger.info("Model cited out-of-range sources %s; dropped", invalid)

        # An answer with no citations at all isn't verifiable. Report it as
        # ungrounded but keep the text, and surface the retrieved passages so the
        # user can check it themselves.
        if not cited:
            return answer, [], False
        return answer, cited, True

    def _diagnostics(
        self, retrieval: RetrievalResult, included: list[RetrievedChunk]
    ) -> dict[str, Any]:
        return {
            "strategy": retrieval.strategy,
            "vector_hits": retrieval.vector_hits,
            "keyword_hits": retrieval.keyword_hits,
            "candidates_considered": retrieval.candidates_considered,
            "filtered_out_by_distance": retrieval.filtered_out,
            "chunks_in_context": len(included),
            "rerank_strategy": self.settings.rerank_strategy,
            "llm_model": self.llm.model,
            "embedding_model": self.retriever.embeddings.model,
        }

    async def _finish(
        self,
        *,
        question: str,
        answer: str,
        grounded: bool,
        cited: list[RetrievedChunk],
        included: list[RetrievedChunk],
        retrieval: RetrievalResult,
        document_ids: Sequence[str] | None,
        session_id: str | None,
        owner: str,
        started: float,
    ) -> QAResult:
        """Assemble the result and persist it to history."""
        latency_ms = int((time.perf_counter() - started) * 1000)
        sources = [citation_dict(i, c) for i, c in enumerate(cited, 1)]
        retrieved = [citation_dict(i, c) for i, c in enumerate(included, 1)]
        diagnostics = self._diagnostics(retrieval, included)

        history_id: str | None = None
        if session_id:
            try:
                # Session ids come from the client, so this is also where a
                # borrowed id is rejected: ensure_session raises rather than
                # attaching this exchange to another visitor's conversation.
                await self.db.ensure_session(session_id, owner=owner)
                history_id = str(uuid.uuid4())
                await self.db.add_history(
                    entry_id=history_id,
                    session_id=session_id,
                    question=question,
                    answer=answer,
                    document_ids=list(document_ids or []),
                    sources=sources,
                    grounded=grounded,
                    llm_model=self.llm.model,
                    embedding_model=self.retriever.embeddings.model,
                    latency_ms=latency_ms,
                )
                # First question doubles as the session title.
                await self.db.touch_session(session_id, owner=owner, title=question)
            except Exception:  # noqa: BLE001
                # History is a convenience; losing it must not fail the answer
                # the user is waiting on.
                logger.exception("Failed to record history for session %s", session_id)
                history_id = None

        return QAResult(
            answer=answer,
            grounded=grounded,
            sources=sources,
            retrieved=retrieved,
            session_id=session_id,
            history_id=history_id,
            latency_ms=latency_ms,
            diagnostics=diagnostics,
        )
