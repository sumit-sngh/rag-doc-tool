"""Prompt construction for grounded question answering.

Three things this module is responsible for:

1. **Grounding.** The system prompt forbids outside knowledge and requires inline
   citations. Without an explicit instruction, models happily answer from
   pre-training and the citations become decorative.

2. **An explicit "I don't know" channel.** The model is told to reply with the
   exact sentinel ``INSUFFICIENT_CONTEXT`` when the sources don't cover the
   question. A sentinel is far more reliable to detect than trying to pattern-match
   hedging prose, and it lets the API return a truthful
   ``no_relevant_context`` result instead of a confabulation.

3. **Context budgeting.** Chunks are packed in relevance order up to a character
   budget, so a generous ``TOP_K`` can never overflow the model's context window.
   Truncation drops whole low-ranked chunks rather than cutting one mid-sentence.
"""

from __future__ import annotations

from app.providers.base import ChatMessage
from app.retrieval.fusion import RetrievedChunk

INSUFFICIENT_CONTEXT = "INSUFFICIENT_CONTEXT"

SYSTEM_PROMPT = f"""\
You are a careful document analyst. You answer questions using ONLY the numbered \
sources supplied by the user.

Rules:
1. Ground every factual claim in the sources and cite it inline with square \
brackets, e.g. "The warranty runs 25 years [1]." Cite multiple sources as [1, 3].
2. Use ONLY the sources. Do not add outside knowledge, and do not guess.
3. If the sources do not contain enough information to answer, reply with exactly \
this token and nothing else: {INSUFFICIENT_CONTEXT}
4. If the sources conflict, say so and cite each conflicting source.
5. Reproduce numbers, dates, names, and identifiers exactly as they appear.
6. Be concise and factual. Do not mention "the sources" or "the context" as \
objects; just answer the question and cite.
"""

# Slightly relaxed variant: same grounding discipline, conversational continuity.
FOLLOWUP_NOTE = """\
Earlier turns of this conversation are provided for reference only. Facts must \
still come from the numbered sources below.
"""


def format_source(index: int, chunk: RetrievedChunk) -> str:
    """Render one numbered source block.

    The header carries filename and page so the model can attribute naturally,
    and so a human reading the raw prompt can audit what was retrieved.
    """
    return f"[{index}] {chunk.document_name} — {chunk.page_label}\n{chunk.text}"


def build_context(
    chunks: list[RetrievedChunk], max_chars: int = 12000
) -> tuple[str, list[RetrievedChunk]]:
    """Pack chunks into a numbered context block within a character budget.

    Returns ``(context_text, included_chunks)``. The returned list is what the
    citation numbers refer to, so callers must use it -- not the input list --
    when resolving citations. Getting this wrong would misattribute every source.
    """
    included: list[RetrievedChunk] = []
    blocks: list[str] = []
    used = 0

    for chunk in chunks:
        block = format_source(len(included) + 1, chunk)
        # +2 for the blank line between blocks.
        cost = len(block) + 2
        if included and used + cost > max_chars:
            # Keep scanning: a later, shorter chunk may still fit.
            continue
        included.append(chunk)
        blocks.append(block)
        used += cost

    return "\n\n".join(blocks), included


def build_messages(
    question: str,
    chunks: list[RetrievedChunk],
    max_context_chars: int = 12000,
    history: list[tuple[str, str]] | None = None,
) -> tuple[list[ChatMessage], list[RetrievedChunk]]:
    """Assemble the chat messages and report which chunks made it into context.

    ``history`` is an optional list of prior ``(question, answer)`` pairs, oldest
    first, included so follow-ups like "and what about the second one?" resolve.
    """
    context, included = build_context(chunks, max_context_chars)

    system = SYSTEM_PROMPT
    if history:
        system = f"{SYSTEM_PROMPT}\n{FOLLOWUP_NOTE}"

    messages: list[ChatMessage] = [ChatMessage(role="system", content=system)]

    for prev_q, prev_a in history or []:
        messages.append(ChatMessage(role="user", content=prev_q))
        messages.append(ChatMessage(role="assistant", content=prev_a))

    messages.append(
        ChatMessage(
            role="user",
            content=(
                f"<sources>\n{context}\n</sources>\n\n"
                f"Question: {question.strip()}"
            ),
        )
    )
    return messages, included


def build_title_prompt(question: str) -> list[ChatMessage]:
    """Prompt for naming a session from its first question."""
    return [
        ChatMessage(
            role="system",
            content=(
                "Summarise the user's question as a title of at most 6 words. "
                "Reply with the title only -- no quotes, no trailing period."
            ),
        ),
        ChatMessage(role="user", content=question.strip()),
    ]
