"""Citation resolution: the machinery that makes answers verifiable.

This is where a plausible-looking answer becomes a checkable one, so the failure
modes matter: a marker pointing at the wrong chunk misattributes a fact, and a
hallucinated marker points the user at a source that does not exist.
"""

from __future__ import annotations

import pytest

from app.qa.citations import (
    _tidy,
    extract_citation_numbers,
    resolve_citations,
)
from app.retrieval.fusion import RetrievedChunk


def chunk(cid: str, text: str = "body") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=cid, document_id="d1", document_name="notes.txt", chunk_index=0,
        page_start=1, page_end=1, text=text,
    )


def context(n: int) -> list[RetrievedChunk]:
    return [chunk(f"c{i}", f"source {i} body") for i in range(1, n + 1)]


# --------------------------------------------------------------------------- #
# Marker extraction
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("answer,expected", [
    ("Panels last 25 years [1].", [1]),
    ("Two facts [1] and another [2].", [1, 2]),
    ("Combined marker [1, 3].", [1, 3]),
    ("No spaces [2,4].", [2, 4]),
    ("Adjacent form [1][2].", [1, 2]),
    ("Double digits [12].", [12]),
    ("Ragged spacing [ 1 , 2 ].", [1, 2]),
])
def test_marker_forms_are_all_recognised(answer, expected):
    assert extract_citation_numbers(answer) == expected


def test_order_of_first_appearance_is_preserved():
    assert extract_citation_numbers("Later [3] then earlier [1].") == [3, 1]


def test_repeated_markers_are_deduplicated():
    assert extract_citation_numbers("Here [1] and again [1] and [1].") == [1]


@pytest.mark.parametrize("answer", [
    "No citations at all.",
    "",
    "Brackets with words [see above].",
    "Empty brackets [].",
    "Ranges are not our format [1-3].",
    "Roman numerals [iv].",
])
def test_non_markers_are_ignored(answer):
    assert extract_citation_numbers(answer) == []


def test_a_bracketed_range_does_not_silently_become_one_citation():
    # "[1-3]" must not be read as [1]; the model wasn't asked for ranges and
    # guessing at intent here would fabricate an attribution.
    assert extract_citation_numbers("Spanning [1-3].") == []


# --------------------------------------------------------------------------- #
# Resolution and renumbering
# --------------------------------------------------------------------------- #
def test_cited_chunks_come_back_in_citation_order():
    chunks = context(3)
    answer, cited, invalid = resolve_citations("Fact A [2]. Fact B [1].", chunks)

    assert [c.chunk_id for c in cited] == ["c2", "c1"]
    assert invalid == []


def test_sparse_citations_are_renumbered_contiguously():
    # The model cited sources 2 and 5 of 5. The user should see [1] and [2]
    # against a two-item list -- not [2] and [5] with gaps.
    chunks = context(5)
    answer, cited, _ = resolve_citations("Alpha [2]. Beta [5].", chunks)

    assert "[1]" in answer and "[2]" in answer
    assert "[5]" not in answer
    assert [c.chunk_id for c in cited] == ["c2", "c5"]


def test_renumbering_follows_first_mention_not_numeric_order():
    chunks = context(4)
    answer, cited, _ = resolve_citations("Later source [4]. Earlier one [2].", chunks)

    assert answer.index("[1]") < answer.index("[2]")
    assert cited[0].chunk_id == "c4"  # [4] became [1]
    assert cited[1].chunk_id == "c2"


def test_repeated_citation_maps_consistently():
    chunks = context(3)
    answer, cited, _ = resolve_citations("First [3]. Again [3]. Other [1].", chunks)

    assert answer.count("[1]") == 2  # both mentions of source 3
    assert len(cited) == 2


def test_combined_markers_are_renumbered_together():
    chunks = context(5)
    answer, cited, _ = resolve_citations("Both agree [2, 4].", chunks)

    assert "[1, 2]" in answer
    assert [c.chunk_id for c in cited] == ["c2", "c4"]


def test_duplicate_numbers_inside_one_marker_collapse():
    chunks = context(3)
    answer, _, _ = resolve_citations("Odd but possible [1, 1].", chunks)
    assert "[1]" in answer and "[1, 1]" not in answer


def test_uncited_answer_is_returned_untouched():
    original = "Panels degrade over time."
    answer, cited, invalid = resolve_citations(original, context(3))

    assert answer == original
    assert cited == [] and invalid == []


def test_resolution_against_empty_context_reports_everything_invalid():
    answer, cited, invalid = resolve_citations("Claim [1].", [])
    assert cited == [] and invalid == [1]
    assert "[1]" not in answer


# --------------------------------------------------------------------------- #
# Hallucinated markers
# --------------------------------------------------------------------------- #
def test_out_of_range_marker_is_reported_and_stripped():
    # A [9] against 3 sources is a hallucination. Leaving it in the answer would
    # show the user a reference they cannot follow.
    chunks = context(3)
    answer, cited, invalid = resolve_citations("Real [1]. Invented [9].", chunks)

    assert invalid == [9]
    assert "[9]" not in answer
    assert [c.chunk_id for c in cited] == ["c1"]


def test_zero_is_out_of_range():
    _, cited, invalid = resolve_citations("Off by one [0].", context(3))
    assert cited == [] and invalid == [0]


def test_an_entirely_hallucinated_citation_set_strips_all_markers():
    answer, cited, invalid = resolve_citations("Nonsense [7] and [8].", context(2))

    assert cited == []
    assert invalid == [7, 8]
    assert "[" not in answer


def test_mixed_valid_and_invalid_inside_one_marker():
    chunks = context(2)
    answer, cited, invalid = resolve_citations("Mixed [1, 9].", chunks)

    assert invalid == [9]
    assert "[1]" in answer and "9" not in answer
    assert [c.chunk_id for c in cited] == ["c1"]


# --------------------------------------------------------------------------- #
# Whitespace repair after marker removal
# --------------------------------------------------------------------------- #
def test_punctuation_reattaches_after_a_marker_is_removed():
    answer, _, _ = resolve_citations("The warranty runs 25 years [9].", context(1))
    assert answer == "The warranty runs 25 years."


def test_citation_only_parenthetical_is_removed_entirely():
    answer, _, _ = resolve_citations("Cleaning is twice yearly (see [9]).", context(1))
    assert answer == "Cleaning is twice yearly."


@pytest.mark.parametrize("raw,expected", [
    ("word  and   spaces", "word and spaces"),
    ("trailing space .", "trailing space."),
    ("comma , here", "comma, here"),
    ("open ( paren", "open (paren"),
    ("empty ()", "empty"),
    ("ref (see )", "ref"),
    ("refs (sources: )", "refs"),
    ("cf (cf. )", "cf"),
    ("line  \nnext", "line\nnext"),
])
def test_tidy_repairs(raw, expected):
    assert _tidy(raw) == expected


def test_tidy_leaves_meaningful_parentheses_alone():
    text = "Output drops (up to 15 percent) in dry climates."
    assert _tidy(text) == text


def test_tidy_preserves_paragraph_breaks():
    # Collapsing these would run the model's formatted answer together.
    assert "\n\n" in _tidy("First para.\n\nSecond para.")


def test_valid_citations_do_not_disturb_surrounding_text():
    chunks = context(2)
    original = "Inverters last 12 years [1], microinverters 25 [2]."
    answer, _, _ = resolve_citations(original, chunks)
    assert answer == original


# --------------------------------------------------------------------------- #
# Realistic end-to-end shapes
# --------------------------------------------------------------------------- #
def test_multi_paragraph_answer_with_sparse_citations():
    chunks = context(6)
    raw = (
        "String inverters have a design life of about 12 years [3].\n\n"
        "Microinverters are warranted for 25 years and are usually replaced only "
        "on failure [3, 6]. Budget around 10 percent of system cost [6]."
    )
    answer, cited, invalid = resolve_citations(raw, chunks)

    assert invalid == []
    assert [c.chunk_id for c in cited] == ["c3", "c6"]
    assert "[3]" not in answer and "[6]" not in answer
    assert "[1]" in answer and "[1, 2]" in answer and "[2]" in answer
    assert "\n\n" in answer


def test_insufficient_context_sentinel_passes_through_unharmed():
    # The engine detects this token, so citation handling must not mangle it.
    answer, cited, invalid = resolve_citations("INSUFFICIENT_CONTEXT", context(3))
    assert answer == "INSUFFICIENT_CONTEXT"
    assert cited == [] and invalid == []
