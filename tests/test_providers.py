"""The fake providers themselves.

Every other test in this suite trusts these, so their properties are worth
asserting directly. Two matter most: the embedder must be *lexically meaningful*
(texts sharing vocabulary land closer in cosine space) rather than merely
deterministic, and the LLM must emit real citations against the numbered sources
it was actually given. If either were fake in the shallow sense, the retrieval
and citation tests would pass without testing anything.
"""

from __future__ import annotations

import asyncio

import pytest

from app.providers.base import ChatMessage
from app.providers.fake import FakeEmbeddingProvider, FakeLLMProvider
from app.qa.prompts import INSUFFICIENT_CONTEXT, build_messages
from app.retrieval.fusion import RetrievedChunk
from app.retrieval.rerank import cosine_similarity


def run(coro):
    """Drive a coroutine without pytest-asyncio, so this module stays portable."""
    return asyncio.run(coro)


def chunk(cid: str, text: str, name: str = "notes.txt", page: int = 1) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=cid, document_id="d1", document_name=name, chunk_index=0,
        page_start=page, page_end=page, text=text,
    )


# --------------------------------------------------------------------------- #
# Embeddings
# --------------------------------------------------------------------------- #
def test_embeddings_are_deterministic():
    provider = FakeEmbeddingProvider(dim=64)
    first = run(provider.embed_query("inverter replacement cost"))
    second = run(provider.embed_query("inverter replacement cost"))
    assert first == second


def test_embedding_dimension_is_honoured():
    for dim in (8, 64, 768):
        assert len(run(FakeEmbeddingProvider(dim=dim).embed_query("text"))) == dim


def test_embeddings_are_unit_length():
    # Chroma is configured for cosine space; unnormalised vectors would make the
    # distance thresholds in config meaningless.
    vec = run(FakeEmbeddingProvider(dim=64).embed_query("panel efficiency"))
    assert sum(x * x for x in vec) == pytest.approx(1.0)


def test_shared_vocabulary_means_higher_similarity():
    # This is the property the retrieval tests depend on.
    provider = FakeEmbeddingProvider(dim=512)
    query = run(provider.embed_query("How long do string inverters last?"))
    relevant, irrelevant = run(provider.embed_documents([
        "String inverters carry a typical design life of 12 years.",
        "Dust and pollen reduce panel output by up to 15 percent.",
    ]))

    assert cosine_similarity(query, relevant) > cosine_similarity(query, irrelevant)


def test_unrelated_texts_are_not_near_duplicates():
    provider = FakeEmbeddingProvider(dim=512)
    a, b = run(provider.embed_documents([
        "Inverter replacement is the largest mid-life maintenance cost.",
        "Harness anchor points must be certified before rooftop work.",
    ]))
    assert cosine_similarity(a, b) < 0.5


def test_identical_texts_are_maximally_similar():
    provider = FakeEmbeddingProvider(dim=256)
    a, b = run(provider.embed_documents(["same text here", "same text here"]))
    assert cosine_similarity(a, b) == pytest.approx(1.0)


def test_query_and_document_paths_agree():
    # Asymmetry between the two entry points would quietly wreck retrieval.
    provider = FakeEmbeddingProvider(dim=128)
    text = "Soiling losses in arid regions"
    doc_vec = run(provider.embed_documents([text]))[0]
    assert run(provider.embed_query(text)) == doc_vec


@pytest.mark.parametrize("text", ["", "   ", "the is of and a", "!!!"])
def test_contentless_text_still_yields_a_usable_vector(text):
    # A zero vector would divide by zero in cosine distance inside Chroma.
    vec = run(FakeEmbeddingProvider(dim=32).embed_query(text))
    assert len(vec) == 32
    assert any(x != 0.0 for x in vec)


def test_batch_embedding_matches_one_at_a_time():
    provider = FakeEmbeddingProvider(dim=64)
    texts = ["alpha content", "beta content", "gamma content"]
    batched = run(provider.embed_documents(texts))
    assert batched == [run(provider.embed_query(t)) for t in texts]


def test_repeated_terms_are_weighted_sublinearly():
    # Ten mentions of a word should not make a chunk ten times more similar.
    provider = FakeEmbeddingProvider(dim=256)
    once, many = run(provider.embed_documents(["inverter cost", "inverter " * 10 + "cost"]))
    assert cosine_similarity(once, many) > 0.5


def test_embedding_health_reports_ready():
    ok, detail = run(FakeEmbeddingProvider(dim=64).health())
    assert ok and "fake" in detail.lower()


# --------------------------------------------------------------------------- #
# LLM: grounding
# --------------------------------------------------------------------------- #
def make_prompt(question: str, chunks: list[RetrievedChunk], **kwargs):
    messages, included = build_messages(question, chunks, **kwargs)
    return messages, included


def test_answer_quotes_the_relevant_source_and_cites_it():
    chunks = [
        chunk("a", "Dust and pollen reduce output by up to 15 percent in dry climates."),
        chunk("b", "String inverters carry a typical design life of 12 years."),
    ]
    messages, _ = make_prompt("What is the design life of string inverters?", chunks)
    answer = run(FakeLLMProvider().generate(messages))

    assert "12 years" in answer
    assert "[2]" in answer  # the inverter passage is source 2


def test_citation_number_matches_the_source_position():
    # Off-by-one here would make every citation in the suite point one chunk away
    # while still looking plausible.
    chunks = [
        chunk("pad1", "Unrelated filler about mounting rails and flashing."),
        chunk("pad2", "More filler concerning conduit routing and labels."),
        chunk("hit", "Microinverters are warranted for 25 years."),
    ]
    messages, _ = make_prompt("How long are microinverters warranted?", chunks)
    answer = run(FakeLLMProvider().generate(messages))

    assert "[3]" in answer and "[1]" not in answer


def test_uncovered_question_returns_the_sentinel():
    # The honest-refusal path. A model that answered anyway would make the
    # no_relevant_context branch untestable.
    chunks = [chunk("a", "Torque the module clamp to 16 newton metres.")]
    messages, _ = make_prompt("What is the tax treatment of depreciation?", chunks)
    assert run(FakeLLMProvider().generate(messages)) == INSUFFICIENT_CONTEXT


def test_no_sources_returns_the_sentinel():
    messages, _ = make_prompt("Anything at all?", [])
    assert run(FakeLLMProvider().generate(messages)) == INSUFFICIENT_CONTEXT


def test_all_stopword_question_returns_the_sentinel():
    chunks = [chunk("a", "Panel efficiency sits between 19 and 22 percent.")]
    messages, _ = make_prompt("What is it?", chunks)
    assert run(FakeLLMProvider().generate(messages)) == INSUFFICIENT_CONTEXT


def test_a_second_corroborating_source_is_cited_too():
    # Exercises the multi-citation resolution path.
    chunks = [
        chunk("a", "Panels should be cleaned twice per year."),
        chunk("b", "Cleaning panels quarterly suits arid regions."),
    ]
    messages, _ = make_prompt("How often should panels be cleaned?", chunks)
    answer = run(FakeLLMProvider().generate(messages))

    assert "[1]" in answer and "[2]" in answer


def test_the_question_itself_is_never_quoted_as_a_source():
    # The sources block is parsed non-greedily, so without the <sources> boundary
    # the trailing "Question:" line becomes the best-matching passage.
    chunks = [chunk("a", "Inverter replacement is the largest mid-life cost.")]
    question = "What is the largest mid-life cost?"
    messages, _ = make_prompt(question, chunks)
    answer = run(FakeLLMProvider().generate(messages))

    assert "Question:" not in answer
    assert question.rstrip("?") not in answer


def test_multiline_source_text_is_parsed_whole():
    body = ("Inverter replacement is the largest mid-life maintenance cost.\n"
            "Budget roughly 10 percent of the original system cost.")
    messages, _ = make_prompt("What percent should be budgeted for inverters?",
                              [chunk("a", body)])
    answer = run(FakeLLMProvider().generate(messages))
    assert "10 percent" in answer


def test_page_range_headers_do_not_break_source_parsing():
    # format_source emits "pp. 1-2" for straddling chunks; the block regex has to
    # accept both label forms or those sources become invisible.
    spanning = chunk("a", "Soiling reduces output by up to 15 percent.")
    spanning.page_end = 2
    messages, _ = make_prompt("How much does soiling reduce output?", [spanning])
    answer = run(FakeLLMProvider().generate(messages))
    assert "15 percent" in answer


def test_em_dash_in_the_filename_does_not_break_parsing():
    # The source header itself is "name — p. N", so a dash inside the filename is
    # a genuine ambiguity for the block regex.
    named = chunk("a", "Panel efficiency reaches 22 percent.", name="q3 — final.pdf")
    messages, _ = make_prompt("What efficiency do panels reach?", [named])
    assert "22 percent" in run(FakeLLMProvider().generate(messages))


# --------------------------------------------------------------------------- #
# LLM: streaming and titles
# --------------------------------------------------------------------------- #
def test_stream_reassembles_into_the_same_answer():
    chunks = [chunk("a", "String inverters last about 12 years.")]
    messages, _ = make_prompt("How long do string inverters last?", chunks)
    provider = FakeLLMProvider()

    async def collect():
        return "".join([delta async for delta in provider.stream(messages)])

    assert run(collect()) == run(provider.generate(messages))


def test_stream_emits_more_than_one_delta():
    # A single-chunk "stream" would let a broken SSE frame parser pass.
    chunks = [chunk("a", "Panels should be cleaned twice per year in temperate areas.")]
    messages, _ = make_prompt("How often should panels be cleaned?", chunks)

    async def collect():
        return [d async for d in FakeLLMProvider().stream(messages)]

    assert len(run(collect())) > 1


def test_stream_deltas_carry_their_own_spacing():
    # Deltas are concatenated verbatim by the client, so spacing has to live in
    # the deltas rather than be re-inserted on join.
    chunks = [chunk("a", "Microinverters are warranted for 25 years.")]
    messages, _ = make_prompt("How long are microinverters warranted?", chunks)

    async def collect():
        return [d async for d in FakeLLMProvider().stream(messages)]

    deltas = run(collect())
    assert not deltas[0].startswith(" ")
    assert "".join(deltas).count("  ") == 0


def test_title_prompt_produces_a_short_title():
    from app.qa.prompts import build_title_prompt

    title = run(FakeLLMProvider().generate(
        build_title_prompt("How long do string inverters last in practice?")
    ))
    assert title and len(title.split()) <= 6
    assert not title.endswith("?")


def test_title_generation_does_not_return_the_sentinel():
    from app.qa.prompts import build_title_prompt

    title = run(FakeLLMProvider().generate(build_title_prompt("Cleaning schedule?")))
    assert title != INSUFFICIENT_CONTEXT


def test_llm_health_reports_ready():
    ok, detail = run(FakeLLMProvider().health())
    assert ok and detail


def test_generate_with_no_user_message_is_the_sentinel():
    messages = [ChatMessage(role="system", content="You are a document analyst.")]
    assert run(FakeLLMProvider().generate(messages)) == INSUFFICIENT_CONTEXT
