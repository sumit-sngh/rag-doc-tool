"""Text chunking with overlap, page attribution, and boundary awareness.

Deliberately dependency-free (stdlib only) so it is fast to unit-test and can be
reasoned about in isolation. All tuning parameters are passed in rather than read
from config, so the caller owns policy.

The chunker works on the *concatenated* document text rather than per-page, so a
sentence spanning a page break stays in one chunk. A page-offset map is kept
alongside so every chunk can still report the page(s) it came from -- which is
what makes citations precise.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Break candidates, highest priority first. The chunker prefers to cut at a
# paragraph break, then a line break, then a sentence end, and only falls back
# to a bare space if nothing better exists in the search window.
SEPARATORS: tuple[str, ...] = (
    "\n\n",
    "\n",
    ". ",
    "? ",
    "! ",
    "; ",
    ": ",
    ", ",
    " ",
)

# Only look for a break point in the last (1 - RATIO) of the window, so we never
# emit a chunk that is drastically shorter than chunk_size just because there
# happened to be an early newline.
BREAK_SEARCH_RATIO = 0.6

PAGE_JOINER = "\n\n"


@dataclass(frozen=True)
class ExtractedPage:
    """One page (PDF) or logical block (DOCX/TXT) of extracted text."""

    page_number: int  # 1-indexed
    text: str


@dataclass(frozen=True)
class Chunk:
    """A retrievable unit of text with enough metadata to cite it."""

    chunk_index: int
    text: str
    char_start: int
    char_end: int
    page_start: int
    page_end: int
    token_estimate: int = 0

    @property
    def page_label(self) -> str:
        """Human-readable page reference, e.g. 'p. 4' or 'pp. 4-5'."""
        if self.page_start == self.page_end:
            return f"p. {self.page_start}"
        return f"pp. {self.page_start}-{self.page_end}"


@dataclass
class DocumentText:
    """Concatenated document text plus the page map needed to attribute chunks."""

    text: str
    # (page_number, start_offset, end_offset) into `text`, end exclusive
    page_spans: list[tuple[int, int, int]] = field(default_factory=list)

    @property
    def page_count(self) -> int:
        return len(self.page_spans)


def estimate_tokens(text: str) -> int:
    """Rough token count without pulling in a tokenizer.

    ~4 characters per token holds well enough for English prose to size context
    budgets. It is an estimate, not a billing figure.
    """
    return max(1, len(text) // 4) if text else 0


def assemble_document(pages: list[ExtractedPage]) -> DocumentText:
    """Join pages into one string while recording where each page landed."""
    parts: list[str] = []
    spans: list[tuple[int, int, int]] = []
    cursor = 0

    for i, page in enumerate(pages):
        text = page.text or ""
        start = cursor
        end = start + len(text)
        spans.append((page.page_number, start, end))
        parts.append(text)
        cursor = end
        if i < len(pages) - 1:
            parts.append(PAGE_JOINER)
            cursor += len(PAGE_JOINER)

    return DocumentText(text="".join(parts), page_spans=spans)


def _pages_for_span(
    page_spans: list[tuple[int, int, int]], start: int, end: int
) -> tuple[int, int]:
    """Return (first_page, last_page) overlapping the character span."""
    touched = [
        page_no
        for page_no, p_start, p_end in page_spans
        # Strict overlap; a zero-length page can't be touched.
        if p_start < end and p_end > start
    ]
    if touched:
        return min(touched), max(touched)

    # Span fell entirely inside a page joiner (or the doc has no pages).
    # Attribute it to the nearest preceding page so citations never show page 0.
    preceding = [p for p, p_start, _ in page_spans if p_start <= start]
    fallback = max(preceding) if preceding else 1
    return fallback, fallback


def _find_break(text: str, start: int, end: int) -> int:
    """Best cut point at or before `end`, or -1 if no good candidate exists."""
    window_start = start + int((end - start) * BREAK_SEARCH_RATIO)
    for sep in SEPARATORS:
        idx = text.rfind(sep, window_start, end)
        if idx != -1:
            # Cut *after* the separator so it stays with the preceding chunk.
            return idx + len(sep)
    return -1


def split_spans(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[tuple[int, int]]:
    """Compute (start, end) character spans for a sliding window over `text`.

    Guarantees forward progress even under pathological parameters, so this can
    never hang on adversarial input.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be non-negative")
    # Defensive: config validates this too, but a runaway overlap here would
    # mean an infinite loop, so clamp rather than trust the caller.
    if chunk_overlap >= chunk_size:
        chunk_overlap = chunk_size // 2

    n = len(text)
    if n == 0:
        return []

    spans: list[tuple[int, int]] = []
    start = 0
    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            candidate = _find_break(text, start, end)
            if candidate > start:
                end = candidate
        spans.append((start, end))

        if end >= n:
            break

        next_start = end - chunk_overlap
        if next_start <= start:
            # Overlap would move us backwards or stall -> step to the boundary.
            next_start = end
        start = next_start

    return spans


def chunk_pages(
    pages: list[ExtractedPage],
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
    min_chunk_chars: int = 50,
) -> list[Chunk]:
    """Turn extracted pages into overlapping, page-attributed chunks.

    Chunks whose trimmed length falls below `min_chunk_chars` are dropped -- they
    are almost always headers, page numbers, or OCR noise, and they pollute
    retrieval. The final chunk is exempt only if it is the *only* chunk, so a
    genuinely tiny document still produces one searchable unit.
    """
    doc = assemble_document(pages)
    if not doc.text.strip():
        return []

    spans = split_spans(doc.text, chunk_size, chunk_overlap)

    chunks: list[Chunk] = []
    for start, end in spans:
        raw = doc.text[start:end]
        stripped = raw.strip()
        if len(stripped) < min_chunk_chars:
            continue

        # Re-anchor offsets onto the trimmed text so citations point at real
        # content rather than leading whitespace.
        lead = len(raw) - len(raw.lstrip())
        true_start = start + lead
        true_end = true_start + len(stripped)

        page_start, page_end = _pages_for_span(doc.page_spans, true_start, true_end)
        chunks.append(
            Chunk(
                chunk_index=len(chunks),
                text=stripped,
                char_start=true_start,
                char_end=true_end,
                page_start=page_start,
                page_end=page_end,
                token_estimate=estimate_tokens(stripped),
            )
        )

    # Salvage case: every window was below the floor, but the document does have
    # content. Emit it as a single chunk rather than silently indexing nothing.
    if not chunks:
        stripped = doc.text.strip()
        lead = len(doc.text) - len(doc.text.lstrip())
        page_start, page_end = _pages_for_span(
            doc.page_spans, lead, lead + len(stripped)
        )
        chunks.append(
            Chunk(
                chunk_index=0,
                text=stripped,
                char_start=lead,
                char_end=lead + len(stripped),
                page_start=page_start,
                page_end=page_end,
                token_estimate=estimate_tokens(stripped),
            )
        )

    return chunks
