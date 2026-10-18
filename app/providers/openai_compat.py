"""OpenAI-compatible embedding and chat providers.

One client for every vendor that speaks the OpenAI REST shape -- OpenAI itself,
Groq, Together, OpenRouter, DeepInfra, or a self-hosted vLLM. This covers the
case the Ollama providers structurally cannot: a deployed server has no local
daemon to talk to, and the models are too large to ship in the image.

Four details here are load-bearing rather than incidental:

* **Embedding order is not guaranteed by the API.** ``POST /v1/embeddings``
  returns objects each carrying an ``index``, and the spec never promises they
  arrive in input order. ``EmbeddingProvider`` *does* promise it, and the
  ingestion pipeline zips vectors against chunks positionally. So the response is
  sorted by ``index`` rather than trusted. Getting this wrong would attach the
  wrong vector to every chunk and corrupt retrieval and citations without ever
  raising -- the worst class of bug this file could contain.

* **Chat and embeddings are often different vendors.** Groq is the obvious
  free-tier choice for generation and serves no embedding model at all. The
  embedding client therefore takes its own base URL and key, falling back to the
  shared pair when they aren't set.

* **The max-tokens field drifted.** Reasoning-era models (o1, o3, gpt-5) reject
  ``max_tokens`` and require ``max_completion_tokens``; everything older accepts
  only the former. Which one applies is probed once from the 400 the server
  returns, then remembered -- the same approach ``OllamaEmbeddingProvider`` takes
  to the ``/api/embed`` vs ``/api/embeddings`` split.

* **Rate limits are retryable; missing models are not.** A 429 or 5xx means try
  again with backoff. A 404 naming the model means the name is wrong, and
  retrying only delays the message that says so.

Azure OpenAI is deliberately not supported: it authenticates with an ``api-key``
header and puts the deployment name in the path, so it is a different client
rather than a different base URL.
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

# Transport-level failures worth another attempt. HTTP status codes are handled
# separately in _should_retry_status, because httpx raises those as a different
# exception type carrying the code.
RETRYABLE = (httpx.TimeoutException, httpx.ConnectError, httpx.RemoteProtocolError)

# 429 is a rate limit and 5xx is the vendor having a bad minute. Both pass with
# backoff; a 4xx that isn't 429 will not improve on retry.
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


class RetryableStatus(Exception):
    """Wraps a retryable HTTP status so tenacity can see it as retryable.

    ``httpx.HTTPStatusError`` covers both a 429 (retry) and a 404 (never retry),
    so retrying on the exception type would be wrong. This narrows it.
    """

    def __init__(self, original: httpx.HTTPStatusError) -> None:
        super().__init__(str(original))
        self.original = original


class _TokenFieldRetry(Exception):
    """Internal signal: restart the stream with the newer max-tokens field."""


class OpenAICompatMixin:
    """Shared HTTP plumbing, auth, and error translation."""

    # Overridden per subclass so error messages name the right env var.
    key_env = "OPENAI_API_KEY"

    def __init__(
        self, base_url: str, api_key: str | None, model: str, timeout: float
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    @property
    def client(self) -> httpx.AsyncClient:
        # Lazily created so constructing a provider never requires a running loop
        # -- Services.build() runs before the event loop in some test paths.
        if self._client is None or self._client.is_closed:
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            self._client = httpx.AsyncClient(
                base_url=self.base_url, timeout=self.timeout, headers=headers
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
        self._client = None

    # ---- errors --------------------------------------------------------- #
    @staticmethod
    def _body_of(response: httpx.Response) -> str:
        try:
            return response.text[:400]
        except Exception:  # pragma: no cover - unread streaming response
            return ""

    def _translate(self, exc: Exception) -> ProviderError:
        if isinstance(exc, RetryableStatus):
            exc = exc.original

        if isinstance(exc, httpx.HTTPStatusError):
            status = exc.response.status_code
            body = self._body_of(exc.response)
            low = body.lower()

            if status in (401, 403):
                return ProviderError(
                    f"{self.base_url} rejected the credentials (HTTP {status}). "
                    f"Check {self.key_env} is set and still valid.",
                    details={"status": status, "response": body},
                )
            if status == 404:
                # Two very different causes share this code. A body that mentions
                # the model is a wrong model name; anything else is far more
                # likely a base URL missing its version segment, which is the
                # mistake people actually make (…/openai instead of …/openai/v1).
                if "model" in low:
                    return ModelNotFoundError(
                        f"'{self.model}' is not a model at {self.base_url}. "
                        "Check the name matches the provider's catalogue.",
                        details={"model": self.model, "response": body},
                    )
                return ProviderError(
                    f"404 from {self.base_url}. The base URL must include the "
                    "version path -- e.g. https://api.groq.com/openai/v1, "
                    "not https://api.groq.com/openai.",
                    details={"status": status, "response": body},
                )
            if status == 429:
                return ProviderError(
                    "Rate limited or out of quota after retries. Free tiers cap "
                    "requests per minute; a large document can hit that during "
                    "ingestion. Lower EMBED_BATCH_SIZE or wait and retry.",
                    details={"status": status, "response": body},
                )
            if status == 400 and "context" in low:
                return ProviderError(
                    f"Request exceeded the model's context window: {body}. "
                    "Lower MAX_CONTEXT_CHARS or TOP_K.",
                    details={"status": status},
                )
            return ProviderError(
                f"{self.base_url} returned {status}: {body}",
                details={"status": status},
            )

        if isinstance(exc, httpx.ConnectError):
            return ProviderError(
                f"Cannot reach {self.base_url}. Check the URL and that the "
                "server has outbound network access.",
                details={"base_url": self.base_url},
            )
        if isinstance(exc, httpx.TimeoutException):
            return ProviderError(
                f"{self.base_url} timed out after {self.timeout}s. Raise "
                "REQUEST_TIMEOUT (generation) or EMBED_TIMEOUT (ingestion).",
            )
        return ProviderError(f"Request to {self.base_url} failed: {exc}")

    def _raise_for_status(self, response: httpx.Response) -> None:
        """Raise, tagging retryable codes so tenacity can distinguish them."""
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            if response.status_code in RETRYABLE_STATUS:
                raise RetryableStatus(exc) from exc
            raise

    def _missing_key_error(self) -> ProviderError:
        return ProviderError(
            f"{self.key_env} is not set, so the '{self.name}' provider has no "
            "credentials. Set it, or switch back to EMBEDDING_PROVIDER=ollama "
            "/ LLM_PROVIDER=ollama for local inference.",
            details={"env": self.key_env},
        )

    # ---- health --------------------------------------------------------- #
    async def _health(self) -> tuple[bool, str]:
        """Confirm the endpoint answers and, where it can, that the model exists."""
        if not self.api_key:
            return False, f"{self.key_env} not set"
        try:
            response = await self.client.get("/models")
            if response.status_code == 404:
                # Plenty of gateways implement chat but not the catalogue. That
                # is not a failure, but it does mean the model name is unverified
                # and the report should not imply otherwise.
                return True, f"endpoint reachable; /models unsupported, '{self.model}' unverified"
            self._raise_for_status(response)
            payload = response.json()
            names = [
                m.get("id", "")
                for m in (payload.get("data") or [])
                if isinstance(m, dict)
            ]
            if not names:
                return True, f"endpoint reachable; empty catalogue, '{self.model}' unverified"
            if self.model in names:
                return True, f"model '{self.model}' available"
            return False, (
                f"model '{self.model}' not offered by {self.base_url} "
                f"({len(names)} models available)"
            )
        except Exception as exc:
            return False, str(self._translate(exc))


class OpenAIEmbeddingProvider(OpenAICompatMixin):
    """Embeddings via an OpenAI-compatible ``/embeddings`` endpoint."""

    name = "openai"
    key_env = "OPENAI_EMBED_API_KEY (or OPENAI_API_KEY)"

    def __init__(
        self,
        base_url: str,
        api_key: str | None,
        model: str,
        timeout: float = 300.0,
        dimensions: int | None = None,
    ) -> None:
        super().__init__(base_url, api_key, model, timeout)
        # OpenAI's v3 embedding models can return a shortened vector. Useful for
        # matching an existing Chroma collection's width, but see the warning in
        # the README: same width does not make vectors from two different models
        # comparable, and re-ingestion is still required.
        self.dimensions = dimensions

    @retry(
        retry=retry_if_exception_type(RETRYABLE + (RetryableStatus,)),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=1, max=12),
        reraise=True,
    )
    async def _post_embeddings(self, texts: list[str]) -> dict[str, Any]:
        payload: dict[str, Any] = {"model": self.model, "input": texts}
        if self.dimensions:
            payload["dimensions"] = self.dimensions
        response = await self.client.post("/embeddings", json=payload)
        self._raise_for_status(response)
        return response.json()

    async def _embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if not self.api_key:
            raise self._missing_key_error()

        try:
            data = await self._post_embeddings(texts)
        except ProviderError:
            raise
        except Exception as exc:
            raise self._translate(exc) from exc

        items = data.get("data")
        if not isinstance(items, list) or len(items) != len(texts):
            got = len(items) if isinstance(items, list) else "no"
            raise ProviderError(
                f"{self.base_url} returned {got} embeddings for {len(texts)} inputs",
                details={"model": self.model},
            )

        # Sort by the server-supplied index instead of relying on arrival order.
        # A missing index falls back to position so a spec-loose gateway still
        # works, but a present one always wins.
        try:
            ordered = sorted(
                enumerate(items),
                key=lambda pair: int(pair[1].get("index", pair[0])),
            )
        except (AttributeError, TypeError, ValueError) as exc:
            raise ProviderError(
                f"Malformed embedding response from {self.base_url}: {exc}"
            ) from exc

        vectors: list[list[float]] = []
        for _, item in ordered:
            vec = item.get("embedding")
            if not isinstance(vec, list) or not vec:
                raise ProviderError(
                    f"{self.base_url} returned an empty embedding",
                    details={"model": self.model},
                )
            vectors.append([float(x) for x in vec])
        return vectors

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self._embed(texts)

    async def embed_query(self, text: str) -> list[float]:
        vectors = await self._embed([text])
        if not vectors:
            raise ProviderError("Embedding provider returned no vector for query")
        return vectors[0]

    async def health(self) -> tuple[bool, str]:
        return await self._health()


class OpenAILLMProvider(OpenAICompatMixin):
    """Chat completions via an OpenAI-compatible ``/chat/completions`` endpoint."""

    name = "openai"
    key_env = "OPENAI_API_KEY"

    def __init__(
        self,
        base_url: str,
        api_key: str | None,
        model: str,
        timeout: float = 120.0,
    ) -> None:
        super().__init__(base_url, api_key, model, timeout)
        # "max_tokens" until a 400 proves this model wants the newer field.
        self._token_field = "max_tokens"

    def _payload(
        self,
        messages: list[ChatMessage],
        temperature: float,
        max_tokens: int,
        stream: bool,
    ) -> dict[str, Any]:
        return {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": stream,
            "temperature": temperature,
            self._token_field: max_tokens,
        }

    def _is_token_field_error(self, response: httpx.Response) -> bool:
        """True if a 400 is the max_tokens/max_completion_tokens rename."""
        if response.status_code != 400:
            return False
        if self._token_field != "max_tokens":
            return False  # already switched; a second 400 is something else
        body = self._body_of(response).lower()
        return "max_completion_tokens" in body or "max_tokens" in body

    def _switch_token_field(self) -> None:
        logger.info(
            "%s rejects max_tokens; using max_completion_tokens for '%s'",
            self.base_url,
            self.model,
        )
        self._token_field = "max_completion_tokens"

    async def generate(
        self,
        messages: list[ChatMessage],
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> str:
        if not self.api_key:
            raise self._missing_key_error()

        try:
            response = await self.client.post(
                "/chat/completions",
                json=self._payload(messages, temperature, max_tokens, False),
            )
            if self._is_token_field_error(response):
                self._switch_token_field()
                response = await self.client.post(
                    "/chat/completions",
                    json=self._payload(messages, temperature, max_tokens, False),
                )
            self._raise_for_status(response)
            data = response.json()
        except ProviderError:
            raise
        except Exception as exc:
            raise self._translate(exc) from exc

        choices = data.get("choices") or []
        content = ""
        if choices and isinstance(choices[0], dict):
            content = (choices[0].get("message") or {}).get("content") or ""
        if not content.strip():
            finish = choices[0].get("finish_reason") if choices else None
            raise ProviderError(
                f"{self.base_url} returned an empty response"
                + (f" (finish_reason={finish})" if finish else ""),
                details={"model": self.model},
            )
        return content

    async def stream(
        self,
        messages: list[ChatMessage],
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> AsyncIterator[str]:
        """Yield content deltas from the server-sent-events stream."""
        if not self.api_key:
            raise self._missing_key_error()

        try:
            async for delta in self._stream_once(messages, temperature, max_tokens):
                yield delta
        except _TokenFieldRetry:
            # Raised before any delta was yielded, so restarting the stream
            # cannot duplicate output the caller has already seen.
            self._switch_token_field()
            async for delta in self._stream_once(messages, temperature, max_tokens):
                yield delta

    async def _stream_once(
        self, messages: list[ChatMessage], temperature: float, max_tokens: int
    ) -> AsyncIterator[str]:
        payload = self._payload(messages, temperature, max_tokens, True)
        try:
            async with self.client.stream(
                "POST", "/chat/completions", json=payload
            ) as response:
                if response.is_error:
                    # Read the body before raising: on an unread streaming
                    # response `.text` raises ResponseNotRead, and _translate
                    # would lose the text naming the actual problem.
                    await response.aread()
                    if self._is_token_field_error(response):
                        raise _TokenFieldRetry()
                self._raise_for_status(response)

                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line or line.startswith(":"):
                        continue  # blank separator or SSE keep-alive comment
                    if line.startswith("event:"):
                        continue
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        event = json.loads(data)
                    except json.JSONDecodeError:
                        logger.debug("Skipping non-JSON SSE payload: %r", data[:120])
                        continue
                    if event.get("error"):
                        detail = event["error"]
                        message = (
                            detail.get("message")
                            if isinstance(detail, dict)
                            else str(detail)
                        )
                        raise ProviderError(f"Stream error: {message}")
                    for choice in event.get("choices") or []:
                        if not isinstance(choice, dict):
                            continue
                        delta = (choice.get("delta") or {}).get("content")
                        if delta:
                            yield delta
        except (ProviderError, _TokenFieldRetry):
            raise
        except Exception as exc:
            raise self._translate(exc) from exc

    async def health(self) -> tuple[bool, str]:
        return await self._health()
