"""Retrieval mechanics: fusion, distance filtering, MMR, and FTS query building.

These are the parts that decide *which* passages the model gets to see, so they
carry most of the answer-quality risk. All pure stdlib -- no vector database, no
embedding model, no network.
"""

from __future__ import annotations

import pytest

from app.qa.prompts import INSUFFICIENT_CONTEXT, build_context, build_messages
from app.retrieval.fusion import (
    RRF_K,
    RetrievedChunk,
    filter_by_distance,
    reciprocal_rank_fusion,
)
from app.retrieval.keyword import build_fts_match_query, tokenize
from app.retrieval.rerank import cosine_similarity, mmr_rerank, rerank


def chunk(cid: str, *, doc: str = "d1", name: str = "notes.txt", index: int = 0,
          page: int = 1, text: str = "text", distance: float | None = None,
          keyword_score: float | None = None,
          embedding: list[float] | None = None) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=cid, document_id=doc, document_name=name, chunk_index=index,
        page_start=page, page_end=page, text=text, distance=distance,
        keyword_score=keyword_score, embedding=embedding,
    )


# --------------------------------------------------------------------------- #
# Reciprocal rank fusion
# --------------------------------------------------------------------------- #
def test_fusion_deduplicates_by_chunk_id():
    vector = [chunk("a", distance=0.1), chunk("b", distance=0.2)]
    keyword = [chunk("b", keyword_score=-2.0), chunk("c", keyword_score=-1.0)]

    fused = reciprocal_rank_fusion(vector, keyword)
    assert sorted(c.chunk_id for c in fused) == ["a", "b", "c"]


def test_agreement_between_retrievers_wins():
    # "b" is ranked second by both retrievers; "a" is first by vector only.
    # Two independent signals agreeing should beat one signal leading.
    vector = [chunk("a", distance=0.1), chunk("b", distance=0.2)]
    keyword = [chunk("c", keyword_score=-3.0), chunk("b", keyword_score=-2.0)]

    fused = reciprocal_rank_fusion(vector, keyword, vector_weight=0.5)
    assert fused[0].chunk_id == "b"


def test_fusion_preserves_signals_from_both_lists():
    # Downstream distance filtering needs the distance even when the chunk was
    # first seen via keyword search, and MMR needs the embedding.
    vector = [chunk("x", distance=0.15, embedding=[1.0, 0.0])]
    keyword = [chunk("x", keyword_score=-4.2)]

    merged = reciprocal_rank_fusion(vector, keyword)[0]
    assert merged.distance == 0.15
    assert merged.keyword_score == -4.2
    assert merged.embedding == [1.0, 0.0]


def test_signals_merge_regardless_of_which_list_was_seen_first():
    vector = [chunk("x", distance=0.15)]
    keyword = [chunk("x", keyword_score=-4.2)]

    # Weighting keyword higher does not change which list is iterated first, so
    # assert the reverse-discovery path explicitly.
    merged = reciprocal_rank_fusion(keyword, vector, vector_weight=0.5)[0]
    assert merged.keyword_score == -4.2
    assert merged.distance == 0.15


def test_fused_score_matches_the_rrf_formula():
    vector = [chunk("a"), chunk("b")]
    fused = {c.chunk_id: c.fused_score
             for c in reciprocal_rank_fusion(vector, [], vector_weight=1.0)}

    assert fused["a"] == pytest.approx(1.0 / (RRF_K + 1))
    assert fused["b"] == pytest.approx(1.0 / (RRF_K + 2))


def test_vector_weight_one_ignores_keyword_list_entirely():
    vector = [chunk("a", distance=0.3)]
    keyword = [chunk("z", keyword_score=-9.0)]

    fused = reciprocal_rank_fusion(vector, keyword, vector_weight=1.0)
    assert [c.chunk_id for c in fused] == ["a"]


def test_vector_weight_zero_is_keyword_only():
    fused = reciprocal_rank_fusion(
        [chunk("a", distance=0.3)], [chunk("z", keyword_score=-9.0)],
        vector_weight=0.0,
    )
    assert [c.chunk_id for c in fused] == ["z"]


def test_fusion_of_two_empty_lists_is_empty():
    assert reciprocal_rank_fusion([], []) == []


def test_ordering_is_deterministic_on_ties():
    # Equal scores must not depend on dict iteration luck, or the same question
    # returns differently ordered citations on different runs.
    a = reciprocal_rank_fusion([chunk("zz"), chunk("aa")], [], vector_weight=1.0)
    assert [c.chunk_id for c in a] == ["zz", "aa"]  # rank order, not id order

    tie = reciprocal_rank_fusion([], [chunk("zz"), chunk("aa")], vector_weight=0.5)
    assert [c.chunk_id for c in tie] == ["zz", "aa"]


# --------------------------------------------------------------------------- #
# Distance filtering -- the basis for an honest "I don't know"
# --------------------------------------------------------------------------- #
def test_distant_chunks_are_dropped():
    chunks = [chunk("near", distance=0.4), chunk("far", distance=1.8)]
    kept = filter_by_distance(chunks, max_distance=0.9)
    assert [c.chunk_id for c in kept] == ["near"]


def test_boundary_distance_is_inclusive():
    kept = filter_by_distance([chunk("edge", distance=0.9)], max_distance=0.9)
    assert len(kept) == 1


def test_keyword_only_hits_survive_the_distance_filter():
    # An exact term match is real evidence even when no embedding scored it.
    kept = filter_by_distance([chunk("kw", keyword_score=-3.0)], max_distance=0.5)
    assert len(kept) == 1


def test_filtering_everything_out_is_allowed():
    # This empty list is what triggers the no_relevant_context path.
    assert filter_by_distance([chunk("far", distance=1.9)], max_distance=0.8) == []


# --------------------------------------------------------------------------- #
# Cosine similarity
# --------------------------------------------------------------------------- #
def test_identical_vectors_are_similarity_one():
    assert cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_orthogonal_vectors_are_zero():
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_opposite_vectors_are_negative_one():
    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


def test_magnitude_does_not_affect_similarity():
    assert cosine_similarity([1.0, 1.0], [5.0, 5.0]) == pytest.approx(1.0)


@pytest.mark.parametrize("a,b", [
    ([], [1.0]),
    ([1.0], []),
    ([1.0, 2.0], [1.0]),        # dimension mismatch
    ([0.0, 0.0], [1.0, 1.0]),   # zero magnitude would divide by zero
])
def test_degenerate_similarity_inputs_return_zero_not_raise(a, b):
    assert cosine_similarity(a, b) == 0.0


# --------------------------------------------------------------------------- #
# MMR reranking
# --------------------------------------------------------------------------- #
def test_mmr_breaks_up_near_duplicate_chunks():
    # Chunk overlap means adjacent chunks are near-identical. Without MMR they
    # occupy consecutive top-k slots and waste the context window.
    #
    # The candidates sit off the query axis deliberately. If the top candidate
    # were identical to the query then sim(x, seed) == relevance(x) for every x,
    # the MMR objective collapses to (2*lambda - 1) * relevance, and the test
    # would prove nothing about diversity.
    query = [1.0, 0.0, 0.0]
    candidates = [
        chunk("dup1", embedding=[0.80, 0.60, 0.00]),   # relevance 0.80
        chunk("dup2", embedding=[0.79, 0.61, 0.02]),   # 0.79, and ~identical to dup1
        chunk("other", embedding=[0.62, 0.00, 0.78]),  # 0.62, but genuinely distinct
    ]

    balanced = [c.chunk_id for c in mmr_rerank(candidates, query, top_k=2,
                                               lambda_mult=0.5)]
    assert balanced == ["dup1", "other"], "MMR should spend slot two on coverage"

    # Same candidates, diversity term switched off: now the near-duplicate wins,
    # which is what confirms the reordering above came from MMR and not from the
    # relevance ranking alone.
    relevance_only = [c.chunk_id for c in mmr_rerank(candidates, query, top_k=2,
                                                     lambda_mult=1.0)]
    assert relevance_only == ["dup1", "dup2"]


def test_lambda_one_is_pure_relevance():
    query = [1.0, 0.0]
    candidates = [
        chunk("mid", embedding=[0.7, 0.7]),
        chunk("best", embedding=[1.0, 0.0]),
    ]
    picked = [c.chunk_id for c in mmr_rerank(candidates, query, top_k=2,
                                            lambda_mult=1.0)]
    assert picked[0] == "best"


def test_mmr_seeds_with_the_most_relevant_chunk():
    query = [0.0, 1.0]
    candidates = [
        chunk("off", embedding=[1.0, 0.0]),
        chunk("on", embedding=[0.0, 1.0]),
    ]
    assert mmr_rerank(candidates, query, top_k=1)[0].chunk_id == "on"


def test_mmr_preserves_the_fused_ranking():
    # The bug this guards against: MMR recomputed relevance as raw query-chunk
    # cosine, which discarded the fusion result entirely. A chunk that keyword
    # search ranked first but the embedder ranked last was then judged on that
    # weak vector score alone -- so RRF, whose entire job is to promote it, had
    # no effect on the final top-k. It was not merely demoted, it was dropped.
    query = [1.0, 0.0, 0.0]
    # "exact" is the exact-term match: BM25's top hit, and the embedder's worst.
    vector = [
        chunk("near", embedding=[0.99, 0.14, 0.00]),
        chunk("mid", embedding=[0.90, 0.00, 0.44]),
        chunk("exact", embedding=[0.55, 0.00, 0.84]),
    ]
    keyword = [chunk("exact", keyword_score=-1.0)]

    fused = reciprocal_rank_fusion(vector, keyword, vector_weight=0.5)
    assert fused[0].chunk_id == "exact", "precondition: fusion promotes the term match"

    picked = [c.chunk_id for c in mmr_rerank(fused, query, top_k=2)]
    assert picked[0] == "exact", "the top fused candidate should seed MMR"


def test_mmr_falls_back_to_cosine_without_fusion_scores():
    # Vector-only retrieval never runs RRF, so every fused_score is 0.0 and
    # there is no ranking to preserve -- cosine is the only signal available.
    query = [1.0, 0.0]
    candidates = [
        chunk("far", embedding=[0.0, 1.0]),
        chunk("close", embedding=[1.0, 0.0]),
    ]
    assert all(c.fused_score == 0.0 for c in candidates)
    assert mmr_rerank(candidates, query, top_k=1)[0].chunk_id == "close"


def test_mmr_without_embeddings_falls_back_to_fusion_order():
    # Keyword-only candidates have no embedding; misranking them silently would
    # be worse than leaving the order alone.
    candidates = [chunk("a"), chunk("b"), chunk("c")]
    picked = [c.chunk_id for c in mmr_rerank(candidates, [1.0, 0.0], top_k=2)]
    assert picked == ["a", "b"]


def test_mmr_without_a_query_embedding_falls_back():
    candidates = [chunk("a", embedding=[1.0, 0.0]), chunk("b", embedding=[0.0, 1.0])]
    picked = [c.chunk_id for c in mmr_rerank(candidates, [], top_k=1)]
    assert picked == ["a"]


def test_mmr_tops_up_from_unembedded_candidates_when_short():
    # Only one candidate has an embedding, but top_k=2 was asked for; returning
    # one result would silently shrink the context.
    candidates = [chunk("emb", embedding=[1.0, 0.0]), chunk("kw")]
    picked = [c.chunk_id for c in mmr_rerank(candidates, [1.0, 0.0], top_k=2)]
    assert len(picked) == 2


def test_mmr_never_returns_more_than_top_k():
    candidates = [chunk(str(i), embedding=[1.0, i / 10]) for i in range(10)]
    assert len(mmr_rerank(candidates, [1.0, 0.0], top_k=3)) == 3


def test_mmr_never_repeats_a_chunk():
    candidates = [chunk(str(i), embedding=[1.0, i / 10]) for i in range(6)]
    picked = [c.chunk_id for c in mmr_rerank(candidates, [1.0, 0.0], top_k=4)]
    assert len(picked) == len(set(picked))


@pytest.mark.parametrize("top_k", [0, -1])
def test_mmr_with_nonpositive_top_k_is_empty(top_k):
    assert mmr_rerank([chunk("a", embedding=[1.0])], [1.0], top_k=top_k) == []


def test_mmr_on_empty_candidates_is_empty():
    assert mmr_rerank([], [1.0, 0.0], top_k=3) == []


def test_rerank_strategy_none_truncates_without_reordering():
    candidates = [chunk("a", embedding=[0.1, 1.0]), chunk("b", embedding=[1.0, 0.0])]
    picked = rerank(candidates, "q", [1.0, 0.0], top_k=1, strategy="none")
    assert [c.chunk_id for c in picked] == ["a"]


def test_unavailable_cross_encoder_degrades_to_mmr():
    # sentence-transformers is an optional dependency; asking for it when it
    # isn't installed must not break querying.
    candidates = [chunk("off", embedding=[1.0, 0.0]), chunk("on", embedding=[0.0, 1.0])]
    picked = rerank(candidates, "q", [0.0, 1.0], top_k=1, strategy="cross_encoder")
    assert len(picked) == 1


# --------------------------------------------------------------------------- #
# FTS5 query building -- injection surface, since questions are free text
# --------------------------------------------------------------------------- #
def test_content_words_become_an_or_query():
    query = build_fts_match_query("What is the inverter warranty period?")
    assert query is not None
    assert '"inverter"' in query and '"warranty"' in query
    assert " OR " in query


def test_stopwords_are_dropped():
    query = build_fts_match_query("What is the state of the inverter?")
    assert '"what"' not in query and '"the"' not in query
    assert '"inverter"' in query and '"state"' in query


def test_every_term_is_quoted_so_operators_cannot_be_injected():
    # Bare AND/OR/NOT/NEAR would be parsed as FTS5 operators.
    for hostile in ["panels AND NOT inverters", "cost NEAR budget", 'quote " inside',
                    "prefix*", "^anchor", "-negated", "col:filter", "(group)"]:
        query = build_fts_match_query(hostile)
        if query is None:
            continue
        for term in query.split(" OR "):
            assert term.startswith('"') and term.endswith('"'), term
            # Punctuation is stripped at tokenization, so no quote can escape.
            assert '"' not in term[1:-1]


def test_terms_are_deduplicated():
    query = build_fts_match_query("inverter inverter inverter cost")
    assert query.count('"inverter"') == 1


def test_term_count_is_capped():
    query = build_fts_match_query(" ".join(f"term{i}" for i in range(200)))
    assert len(query.split(" OR ")) <= 24


def test_custom_term_cap_is_honoured():
    query = build_fts_match_query("alpha beta gamma delta epsilon", max_terms=2)
    assert len(query.split(" OR ")) == 2


def test_all_stopword_question_still_produces_a_query():
    # "What is it?" is a real follow-up. Returning None would silently disable
    # half of hybrid search.
    query = build_fts_match_query("What is it?")
    assert query is not None and query.strip() != ""


@pytest.mark.parametrize("question", ["", "   ", "!!! ??? ...", "--- +++"])
def test_questions_with_nothing_searchable_return_none(question):
    # None is the signal to skip keyword search rather than match everything.
    assert build_fts_match_query(question) is None


def test_single_characters_are_not_searched_alone():
    query = build_fts_match_query("a b inverter")
    assert query is not None
    assert '"a"' not in query and '"b"' not in query


def test_numbers_are_searchable():
    # "What does IEC 62446 require?" -- the number is the whole question.
    query = build_fts_match_query("What does IEC 62446 require?")
    assert '"62446"' in query


def test_tokenizer_lowercases_and_splits_on_punctuation():
    assert tokenize("Inverter's mid-life COST!") == ["inverter", "s", "mid", "life", "cost"]


def test_unicode_words_survive_tokenization():
    assert tokenize("café naïve") == ["café", "naïve"]


# --------------------------------------------------------------------------- #
# Context packing -- citation numbers refer to this list, so it must be exact
# --------------------------------------------------------------------------- #
def test_sources_are_numbered_from_one_in_order():
    chunks = [chunk("a", text="Alpha body"), chunk("b", text="Beta body")]
    context, included = build_context(chunks)

    assert "[1] notes.txt" in context and "[2] notes.txt" in context
    assert context.index("Alpha") < context.index("Beta")
    assert included == chunks


def test_source_headers_carry_filename_and_page():
    c = chunk("a", name="handbook.pdf", page=7, text="Body")
    context, _ = build_context([c])
    assert "handbook.pdf" in context and "p. 7" in context


def test_page_range_is_labelled_as_a_range():
    c = chunk("a", text="Body")
    c.page_end = 3
    context, _ = build_context([c])
    assert "pp. 1-3" in context


def test_budget_drops_whole_chunks_not_partial_sentences():
    chunks = [chunk(str(i), text="x" * 400) for i in range(10)]
    context, included = build_context(chunks, max_chars=1000)

    assert len(included) < len(chunks)
    # Every included chunk's text must be present in full.
    for c in included:
        assert c.text in context


def test_included_list_is_what_citation_numbers_refer_to():
    # If build_context returned the input list, dropped chunks would shift every
    # citation number and misattribute the whole answer.
    chunks = [chunk(str(i), text="y" * 500) for i in range(6)]
    context, included = build_context(chunks, max_chars=1200)

    for position, c in enumerate(included, start=1):
        assert f"[{position}]" in context
    assert f"[{len(included) + 1}]" not in context


def test_first_chunk_is_kept_even_if_it_blows_the_budget():
    # A single oversized chunk must not produce an empty context -- that would
    # turn a good retrieval into "no relevant matches".
    context, included = build_context([chunk("big", text="z" * 5000)], max_chars=100)
    assert len(included) == 1 and context


def test_a_later_smaller_chunk_can_still_fit():
    chunks = [chunk("a", text="a" * 100), chunk("huge", text="b" * 5000),
              chunk("c", text="c" * 100)]
    _, included = build_context(chunks, max_chars=400)
    assert [c.chunk_id for c in included] == ["a", "c"]


def test_empty_context_packs_to_empty():
    context, included = build_context([])
    assert context == "" and included == []


# --------------------------------------------------------------------------- #
# Message assembly
# --------------------------------------------------------------------------- #
def test_messages_start_with_the_grounding_system_prompt():
    messages, _ = build_messages("How long do inverters last?", [chunk("a")])
    assert messages[0].role == "system"
    assert INSUFFICIENT_CONTEXT in messages[0].content
    assert "ONLY" in messages[0].content


def test_question_and_sources_are_in_the_final_user_message():
    messages, _ = build_messages("How long?", [chunk("a", text="Twelve years.")])
    last = messages[-1]
    assert last.role == "user"
    assert "How long?" in last.content
    assert "Twelve years." in last.content
    assert "<sources>" in last.content


def test_history_is_interleaved_oldest_first():
    messages, _ = build_messages(
        "And the second?", [chunk("a")],
        history=[("First question", "First answer"), ("Second question", "Second answer")],
    )
    roles = [m.role for m in messages]
    assert roles == ["system", "user", "assistant", "user", "assistant", "user"]
    assert messages[1].content == "First question"
    assert messages[-1].content.endswith("Question: And the second?")


def test_history_adds_the_followup_grounding_note():
    plain, _ = build_messages("Q", [chunk("a")])
    with_history, _ = build_messages("Q", [chunk("a")], history=[("a", "b")])
    assert len(with_history[0].content) > len(plain[0].content)
    assert "reference only" in with_history[0].content


def test_build_messages_reports_the_chunks_actually_included():
    chunks = [chunk(str(i), text="w" * 600) for i in range(8)]
    _, included = build_messages("Q", chunks, max_context_chars=1500)
    assert 0 < len(included) < len(chunks)
