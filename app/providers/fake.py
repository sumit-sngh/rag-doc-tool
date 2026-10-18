"""Deterministic offline providers -- no daemon, no network, no model download.

These exist for three reasons:

* **Testable retrieval.** ``FakeEmbeddingProvider`` is not random. It builds a
  hashed bag-of-words vector, so texts sharing vocabulary land close together in
  cosine space. Retrieval, fusion, and MMR can therefore be asserted on real
  ranking behaviour rather than mocked away.
* **A runnable app with zero setup.** Set ``EMBEDDING_PROVIDER=fake`` and
  ``LLM_PROVIDER=fake`` and the whole pipeline works end to end -- useful for
  frontend work, CI, and demos on a machine with no Ollama.
* **Exercising the citation path.** ``FakeLLMProvider`` is extractive: it parses
  the numbered sources back out of the prompt, picks the best-overlapping one,
  and answers with a real inline citation. So citation resolution, renumbering,
  and the INSUFFICIENT_CONTEXT path all get genuinely tested.

Not a stand-in for a real model's fluency -- only for its interface and shape.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import AsyncIterator

from app.providers.base import ChatMessage
from app.qa.prompts import INSUFFICIENT_CONTEXT
from app.retrieval.keyword import STOPWORDS, tokenize

# Each token is hashed into this many dimensions. More than one reduces the
# damage from a single hash collision landing two unrelated terms together.
HASHES_PER_TOKEN = 3


class FakeEmbeddingProvider:
    """Hashed bag-of-words embeddings: deterministic and lexically meaningful."""

    name = "fake"

    def __init__(self, model: str = "fake-embed", dim: int = 768) -> None:
        self.model = model
        self.dim = dim

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        tokens = [t for t in tokenize(text) if len(t) > 1 and t not in STOPWORDS]
        if not tokens:
            # Distinct-but-stable vector for empty/stopword-only text, so it
            # never collides with real content and never divides by zero.
            vec[0] = 1.0
            return vec

        counts: dict[str, int] = {}
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1

        for token, count in counts.items():
            # Sublinear term frequency, as in classic TF-IDF weighting.
            weight = 1.0 + math.log(count)
            for h in range(HASHES_PER_TOKEN):
                digest = hashlib.blake2b(
                    f"{token}:{h}".encode("utf-8"), digest_size=8
                ).digest()
                idx = int.from_bytes(digest, "big") % self.dim
                # Signed contribution keeps unrelated tokens from all pushing
                # vectors into the same positive corner of the space.
                sign = 1.0 if digest[0] & 1 else -1.0
                vec[idx] += sign * weight

        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    async def health(self) -> tuple[bool, str]:
        return True, f"fake embedding provider (dim={self.dim})"

    async def aclose(self) -> None:
        return None


# Matches the source blocks emitted by app.qa.prompts.format_source:
#   [1] filename — p. 3
#   <chunk text, possibly multiple lines>
SOURCE_BLOCK = re.compile(
    r"^\[(\d+)\]\s+(.+?)\s+—\s+(pp?\.\s*[\d\-]+)\s*\n(.*?)(?=\n\[\d+\]\s|\Z)",
    re.MULTILINE | re.DOTALL,
)
QUESTION_LINE = re.compile(r"^Question:\s*(.+)$", re.MULTILINE)
SOURCES_SECTION = re.compile(r"<sources>\s*(.*?)\s*</sources>", re.DOTALL)


class FakeLLMProvider:
    """Extractive stand-in LLM that produces genuinely cited answers."""

    name = "fake"

    def __init__(self, model: str = "fake-llm") -> None:
        self.model = model

    @staticmethod
    def _parse(prompt: str) -> tuple[str, list[tuple[int, str]]]:
        """Recover (question, [(source_number, text), ...]) from the user prompt."""
        question_match = QUESTION_LINE.search(prompt)
        question = question_match.group(1).strip() if question_match else ""

        # Isolate the sources section first. The final source block's text is
        # matched non-greedily up to end-of-string, so without this it would
        # swallow "</sources>" and the trailing "Question:" line -- and the
        # question would then look like the best-matching passage.
        section = SOURCES_SECTION.search(prompt)
        body = section.group(1) if section else ""

        sources = [
            (int(m.group(1)), m.group(4).strip()) for m in SOURCE_BLOCK.finditer(body)
        ]
        return question, sources

    def _answer(self, messages: list[ChatMessage]) -> str:
        user_messages = [m for m in messages if m.role == "user"]
        if not user_messages:
            return INSUFFICIENT_CONTEXT

        prompt = user_messages[-1].content

        # Title-generation prompt has no <sources> block; handle it separately.
        if "<sources>" not in prompt:
            words = [w for w in prompt.split() if w][:6]
            return " ".join(words).rstrip("?.,") or "Untitled"

        question, sources = self._parse(prompt)
        if not sources:
            return INSUFFICIENT_CONTEXT

        q_terms = {t for t in tokenize(question) if len(t) > 1 and t not in STOPWORDS}
        if not q_terms:
            return INSUFFICIENT_CONTEXT

        # Score each source by how much of the question's vocabulary it covers.
        scored: list[tuple[float, int, str]] = []
        for number, text in sources:
            terms = {t for t in tokenize(text) if len(t) > 1 and t not in STOPWORDS}
            overlap = len(q_terms & terms)
            if overlap:
                scored.append((overlap / len(q_terms), number, text))
        if not scored:
            # Faithfully reproduces what a well-behaved model does when the
            # retrieved context genuinely doesn't address the question.
            return INSUFFICIENT_CONTEXT

        scored.sort(key=lambda s: (-s[0], s[1]))
        _, best_number, best_text = scored[0]

        # Quote the sentence with the most question-term overlap.
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", best_text) if s.strip()]
        best_sentence = best_text.strip()
        best_hits = -1
        for sentence in sentences:
            hits = len(q_terms & set(tokenize(sentence)))
            if hits > best_hits:
                best_hits, best_sentence = hits, sentence

        answer = f"{best_sentence.rstrip('.')} [{best_number}]."

        # Mention a corroborating second source when one exists, so multi-source
        # citation handling is exercised too.
        if len(scored) > 1 and scored[1][0] >= scored[0][0] * 0.5:
            answer += f" This is also supported by source [{scored[1][1]}]."
        return answer

    async def generate(
        self,
        messages: list[ChatMessage],
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> str:
        return self._answer(messages)

    async def stream(
        self,
        messages: list[ChatMessage],
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> AsyncIterator[str]:
        """Yield the answer in word-sized deltas to mimic token streaming."""
        answer = self._answer(messages)
        for i, token in enumerate(answer.split(" ")):
            yield token if i == 0 else f" {token}"

    async def health(self) -> tuple[bool, str]:
        return True, "fake LLM provider (extractive)"

    async def aclose(self) -> None:
        return None
