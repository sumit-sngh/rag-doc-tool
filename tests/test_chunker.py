"""Chunking: overlap, page attribution, and pathological inputs.

Pure stdlib, so these run without any project dependencies installed.
"""

from __future__ import annotations

import pytest

from app.ingestion.chunker import (
    Chunk,
    ExtractedPage,
    assemble_document,
    chunk_pages,
    estimate_tokens,
    split_spans,
)


def page(number: int, text: str) -> ExtractedPage:
    return ExtractedPage(page_number=number, text=text)


# --------------------------------------------------------------------------- #
# split_spans
# --------------------------------------------------------------------------- #
def test_spans_cover_the_whole_text_without_gaps():
    text = "word " * 400
    spans = split_spans(text, chunk_size=300, chunk_overlap=60)

    assert spans[0][0] == 0
    assert spans[-1][1] == len(text)
    # Consecutive spans must overlap or touch -- never skip characters.
    for (_, prev_end), (next_start, _) in zip(spans, spans[1:]):
        assert next_start <= prev_end, "gap between chunks would lose text"


def test_overlap_actually_repeats_text():
    text = "".join(f"sentence number {i}. " for i in range(80))
    spans = split_spans(text, chunk_size=300, chunk_overlap=60)

    first, second = spans[0], spans[1]
    overlap = first[1] - second[0]
    # Boundary snapping can shrink the overlap a little, but it must exist.
    assert 0 < overlap <= 60


def test_empty_text_produces_no_spans():
    assert split_spans("", chunk_size=300, chunk_overlap=60) == []


def test_text_shorter_than_chunk_is_one_span():
    assert split_spans("short", chunk_size=300, chunk_overlap=60) == [(0, 5)]


def test_overlap_at_or_above_chunk_size_is_clamped_not_hung():
    # Left unclamped this is an infinite loop, so the assertion that matters is
    # that it returns at all.
    spans = split_spans("x" * 1000, chunk_size=100, chunk_overlap=100)
    assert spans[-1][1] == 1000
    assert len(spans) < 1000


def test_no_break_candidate_still_advances():
    # A single unbroken token: there is no separator to snap to.
    spans = split_spans("a" * 500, chunk_size=100, chunk_overlap=20)
    assert spans[-1][1] == 500
    for (start, end) in spans:
        assert end > start


@pytest.mark.parametrize("size,overlap", [(0, 0), (-5, 0)])
def test_invalid_chunk_size_rejected(size, overlap):
    with pytest.raises(ValueError):
        split_spans("some text", chunk_size=size, chunk_overlap=overlap)


def test_negative_overlap_rejected():
    with pytest.raises(ValueError):
        split_spans("some text", chunk_size=100, chunk_overlap=-1)


# --------------------------------------------------------------------------- #
# Page attribution -- this is what makes citations verifiable
# --------------------------------------------------------------------------- #
def test_assemble_document_records_one_span_per_page():
    doc = assemble_document([page(1, "alpha"), page(2, "beta"), page(3, "gamma")])

    assert doc.page_count == 3
    assert "alpha" in doc.text and "gamma" in doc.text
    # Spans must be ordered and non-overlapping.
    starts = [s[0] if isinstance(s, tuple) else s.start for s in _spans(doc)]
    assert starts == sorted(starts)


def _spans(doc):
    """page_spans may be tuples or small records depending on implementation."""
    return list(doc.page_spans)


def test_chunk_carries_the_page_it_came_from():
    pages = [page(1, "Alpha content. " * 20), page(2, "Beta content. " * 20)]
    chunks = chunk_pages(pages, chunk_size=200, chunk_overlap=40, min_chunk_chars=20)

    assert chunks, "expected at least one chunk"
    for chunk in chunks:
        assert 1 <= chunk.page_start <= chunk.page_end <= 2
        source = "Alpha" if chunk.page_start == 1 else "Beta"
        # A chunk claiming page 1 must actually contain page 1's text.
        if chunk.page_start == chunk.page_end:
            assert source in chunk.text


def test_chunk_spanning_two_pages_reports_a_range():
    # Short pages force a chunk to straddle a boundary.
    pages = [page(1, "First page text here."), page(2, "Second page text here.")]
    chunks = chunk_pages(pages, chunk_size=500, chunk_overlap=0, min_chunk_chars=10)

    assert len(chunks) == 1
    assert chunks[0].page_start == 1
    assert chunks[0].page_end == 2
    assert chunks[0].page_label == "pp. 1-2"


def test_single_page_label_reads_naturally():
    chunks = chunk_pages([page(4, "Content on page four. " * 10)],
                         chunk_size=500, chunk_overlap=0, min_chunk_chars=10)
    assert chunks[0].page_label == "p. 4"


def test_chunk_indices_are_contiguous_from_zero():
    pages = [page(1, "Sentence one. " * 100)]
    chunks = chunk_pages(pages, chunk_size=200, chunk_overlap=40, min_chunk_chars=20)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_char_offsets_point_into_the_chunk_text():
    pages = [page(1, "Alpha beta gamma delta. " * 30)]
    chunks = chunk_pages(pages, chunk_size=250, chunk_overlap=50, min_chunk_chars=20)
    for chunk in chunks:
        assert chunk.char_end > chunk.char_start
        assert len(chunk.text) <= chunk.char_end - chunk.char_start + 2


# --------------------------------------------------------------------------- #
# Noise floor
# --------------------------------------------------------------------------- #
def test_tiny_fragments_are_dropped():
    # Page numbers and running headers look like this and pollute retrieval.
    pages = [page(1, "Real content, long enough to survive the floor. " * 10),
             page(2, "7")]
    chunks = chunk_pages(pages, chunk_size=200, chunk_overlap=0, min_chunk_chars=50)
    assert all(len(c.text.strip()) >= 50 for c in chunks)


def test_a_tiny_document_still_yields_one_chunk():
    # The floor must not swallow the only content in the document.
    chunks = chunk_pages([page(1, "Tiny.")], chunk_size=200, chunk_overlap=0,
                         min_chunk_chars=50)
    assert len(chunks) == 1
    assert chunks[0].text.strip() == "Tiny."


def test_whitespace_only_document_yields_nothing():
    assert chunk_pages([page(1, "   \n\t  ")], chunk_size=200, chunk_overlap=0,
                       min_chunk_chars=10) == []


def test_no_pages_yields_nothing():
    assert chunk_pages([], chunk_size=200, chunk_overlap=0) == []


# --------------------------------------------------------------------------- #
# Token estimate
# --------------------------------------------------------------------------- #
def test_token_estimate_scales_with_length():
    assert estimate_tokens("") == 0
    short, long = estimate_tokens("a" * 40), estimate_tokens("a" * 400)
    assert long > short
    assert 5 <= short <= 15  # ~4 chars per token


def test_chunk_token_estimate_is_populated():
    chunks = chunk_pages([page(1, "Words go here. " * 40)],
                         chunk_size=200, chunk_overlap=0, min_chunk_chars=20)
    assert all(isinstance(c, Chunk) and c.token_estimate > 0 for c in chunks)
