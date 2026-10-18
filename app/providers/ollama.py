"""Ollama-backed embedding and chat providers.

Talks to a local Ollama daemon over HTTP. Two details worth knowing:

* **Embedding endpoint drift.** Ollama moved from ``POST /api/embeddings``
  (single ``prompt``, one vector) to ``POST /api/embed`` (batched ``input``, many
  vectors). Both are in the wild depending on version, so the client probes the
  batch endpoint once and remembers the answer, falling back to per-text calls on
  the legacy route. Batching matters: it is the difference between one request
  and one-per-chunk when ingesting a large PDF.

* **Failure translation.** A missing model is the single most common setup
  problem (``ollama pull`` not run). It surfaces as a distinct
  ``ModelNotFoundError`` carrying the exact command to fix it, rather than a
  generic 502.
"""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.errors import ModelNotFoundError, ProviderError
from app.providers.base import ChatMessage

logger = logging.getLogger(__name__)

# Transient conditions only. A 404 for a missing model is permanent -- retrying
# it just delays a clear error message.
RETRYABLE = (httpx.TimeoutException, httpx.ConnectError, httpx.RemoteProtocolError)


class OllamaClientMixin:
    """Shared HTTP plumbing and error translation."""

    def __init__(self, base_url: str, model: str, timeout: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    @property
    def client(self) -> httpx.AsyncClient:
        # Lazily created so constructing a provider never needs a running loop.
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url, timeout=self.timeout
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
        self._client = None

    def _translate(self, exc: Exception) -> ProviderError:
        if isinstance(exc, httpx.HTTPStatusError):
            body = ""
            try:
                body = exc.response.text[:400]
            except Exception:  # pragma: no cover
                pass
            if exc.response.status_code == 404 or "not found" in body.lower():
                return ModelNotFoundError(
                    f"Ollama has no model named '{self.model}'. "
                    f"Pull it first:  ollama pull {self.model}",
                    details={"model": self.model, "response": body},
                )
            return ProviderError(
                f"Ollama returned {exc.response.status_code}: {body}",
                details={"status": exc.response.status_code},
            )
        if isinstance(exc, httpx.ConnectError):
            return ProviderError(
                f"Cannot reach Ollama at {self.base_url}. Is it running? "
                "Start it with:  ollama serve",
                details={"base_url": self.base_url},
            )
        if isinstance(exc, httpx.TimeoutException):
            return ProviderError(
                f"Ollama timed out after {self.timeout}s. Large documents or a "
                "cold model can exceed this; raise EMBED_TIMEOUT/REQUEST_TIMEOUT.",
            )
        return ProviderError(f"Ollama request failed: {exc}")

    async def _health(self) -> tuple[bool, str]:
        """Check the daemon is up and the configured model is present."""
        try:
            response = await self.client.get("/api/tags")
            response.raise_for_status()
            payload = response.json()
            names = [m.get("name", "") for m in payload.get("models", [])]
            # Ollama reports "llama3.1:latest" for a model pulled as "llama3.1".
            base = self.model.split(":")[0]
            if any(n == self.model or n.split(":")[0] == base for n in names):
                return True, f"model '{self.model}' available"
            return False, (
                f"model '{self.model}' not pulled (available: {', '.join(names) or 'none'})"
            )
        except Exception as exc:
            return False, str(self._translate(exc))


class OllamaEmbeddingProvider(OllamaClientMixin):
    """Embeddings via Ollama, batched where the daemon supports it."""

    name = "ollama"

    def __init__(self, base_url: str, model: str, timeout: float = 300.0) -> None:
        super().__init__(base_url, model, timeout)
        # None = not yet probed; True/False = use /api/embed or the legacy route.
        self._supports_batch: bool | None = None

    @retry(
        retry=retry_if_exception_type(RETRYABLE),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
        reraise=True,
    )
    async def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        response = await self.client.post(path, json=payload)
        response.raise_for_status()
        return response.json()

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Batch route: POST /api/embed with an `input` list."""
        data = await self._post("/api/embed", {"model": self.model, "input": texts})
        vectors = data.get("embeddings")
        if not isinstance(vectors, list) or len(vectors) != len(texts):
            raise ProviderError(
                f"Ollama /api/embed returned {len(vectors) if isinstance(vectors, list) else 'no'} "
                f"embeddings for {len(texts)} inputs"
            )
        return [[float(x) for x in vec] for vec in vectors]

    async def _embed_legacy(self, texts: list[str]) -> list[list[float]]:
        """Legacy route: one request per text against /api/embeddings."""
        vectors: list[list[float]] = []
        for text in texts:
            data = await self._post(
                "/api/embeddings", {"model": self.model, "prompt": text}
            )
            vec = data.get("embedding")
            if not isinstance(vec, list) or not vec:
                raise ProviderError("Ollama /api/embeddings returned no embedding")
            vectors.append([float(x) for x in vec])
        return vectors

    async def _embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            if self._supports_batch is None:
                # Probe once with the real payload; on 404 the daemon predates
                # /api/embed and we pin to the legacy route for this process.
                try:
                    vectors = await self._embed_batch(texts)
                    self._supports_batch = True
                    return vectors
                except httpx.HTTPStatusError as exc:
                    if exc.response.status_code != 404:
                        raise
                    logger.info(
                        "Ollama lacks /api/embed; using legacy /api/embeddings"
                    )
                    self._supports_batch = False

            if self._supports_batch:
                return await self._embed_batch(texts)
            return await self._embed_legacy(texts)
        except ProviderError:
            raise
        except Exception as exc:
            raise self._translate(exc) from exc

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self._embed(texts)

    async def embed_query(self, text: str) -> list[float]:
        vectors = await self._embed([text])
        if not vectors:
            raise ProviderError("Embedding provider returned no vector for query")
        return vectors[0]

    async def health(self) -> tuple[bool, str]:
        return await self._health()


class OllamaLLMProvider(OllamaClientMixin):
    """Chat completions via Ollama, with token streaming."""

    name = "ollama"

    def __init__(self, base_url: str, model: str, timeout: float = 120.0) -> None:
        super().__init__(base_url, model, timeout)

    def _payload(
        self, messages: list[ChatMessage], temperature: float, max_tokens: int, stream: bool
    ) -> dict[str, Any]:
        return {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": stream,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }

    async def generate(
        self,
        messages: list[ChatMessage],
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> str:
        try:
            response = await self.client.post(
                "/api/chat", json=self._payload(messages, temperature, max_tokens, False)
            )
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            raise self._translate(exc) from exc

        content = (data.get("message") or {}).get("content", "")
        if not content:
            raise ProviderError("Ollama returned an empty response")
        return content

    async def stream(
        self,
        messages: list[ChatMessage],
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> AsyncIterator[str]:
        """Yield content deltas from Ollama's NDJSON stream."""
        payload = self._payload(messages, temperature, max_tokens, True)
        try:
            async with self.client.stream("POST", "/api/chat", json=payload) as response:
                if response.is_error:
                    # Pull the body in before raising. _translate() wants it for
                    # the message, but on an unread streaming response `.text`
                    # raises ResponseNotRead -- it catches that and carries on
                    # with an empty body, so without this the most common
                    # failure (model not pulled) loses the text naming it.
                    await response.aread()
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        # A partial line mid-stream is not worth aborting over.
                        logger.debug("Skipping non-JSON stream line: %r", line[:120])
                        continue
                    if event.get("error"):
                        raise ProviderError(f"Ollama stream error: {event['error']}")
                    delta = (event.get("message") or {}).get("content", "")
                    if delta:
                        yield delta
                    if event.get("done"):
                        break
        except ProviderError:
            raise
        except Exception as exc:
            raise self._translate(exc) from exc

    async def health(self) -> tuple[bool, str]:
        return await self._health()
