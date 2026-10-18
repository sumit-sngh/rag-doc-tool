"""Provider interfaces for embeddings and text generation.

Protocols rather than base classes: an implementation only needs the right method
signatures, which keeps the fake provider honest (it can't inherit behaviour it
doesn't implement) and makes swapping providers a config change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterator, Protocol, runtime_checkable


@dataclass(frozen=True)
class ChatMessage:
    role: str  # "system" | "user" | "assistant"
    content: str


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Turns text into vectors."""

    name: str
    model: str

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of passages. Order of results must match input order."""
        ...

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single search query.

        Separate from embed_documents because some models use asymmetric
        prefixes for queries vs passages (nomic-embed-text among them).
        """
        ...

    async def health(self) -> tuple[bool, str]:
        """Return (ok, human-readable detail) without raising."""
        ...

    async def aclose(self) -> None:
        ...


@runtime_checkable
class LLMProvider(Protocol):
    """Generates an answer from a prompt."""

    name: str
    model: str

    async def generate(
        self,
        messages: list[ChatMessage],
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> str:
        ...

    def stream(
        self,
        messages: list[ChatMessage],
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> AsyncIterator[str]:
        """Yield text deltas as they arrive.

        Not an ``async def`` -- it returns the async iterator directly so callers
        can ``async for`` over it without an extra await.
        """
        ...

    async def health(self) -> tuple[bool, str]:
        ...

    async def aclose(self) -> None:
        ...
