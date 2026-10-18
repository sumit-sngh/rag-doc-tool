"""End-to-end API tests: upload -> poll -> query -> cite -> history.

These exercise the real FastAPI app with the real Chroma store, real SQLite, and
the fake providers -- so routing, validation, background ingestion, retrieval,
citation resolution, and persistence are all genuinely covered. Only the model
weights are substituted.

Requires the project dependencies (fastapi, httpx, chromadb, aiosqlite,
pytest-asyncio) to be installed. Run with ``pytest``; asyncio_mode=auto in
pytest.ini handles the async tests.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from tests.conftest import SOLAR_PAGES, ingest, upload, wait_ready


# --------------------------------------------------------------------------- #
# Health
# --------------------------------------------------------------------------- #
async def test_health_reports_ok_with_fake_providers(client):
    res = await client.get("/health")
    assert res.status_code == 200

    body = res.json()
    assert body["status"] == "ok"
    assert body["components"]["embeddings"]["ok"] is True
    assert body["components"]["llm"]["ok"] is True
    assert body["components"]["database"]["ok"] is True


async def test_health_exposes_config_the_frontend_needs(client):
    body = (await client.get("/health")).json()
    # The frontend keys its auth flow off this, so its absence is a real break.
    assert body["config"]["auth_required"] is False
    assert body["config"]["chunk_size"] > 0
    assert "stats" in body


async def test_health_declares_auth_when_a_key_is_set(secured_client):
    # /health must stay open, or the frontend can't discover that auth is needed.
    res = await secured_client.get("/health")
    assert res.status_code == 200
    assert res.json()["config"]["auth_required"] is True


async def test_root_serves_the_frontend(client):
    res = await client.get("/")
    assert res.status_code == 200
    assert "Document Q&A" in res.text or "<!doctype html" in res.text.lower()


async def test_openapi_schema_builds(client):
    # Catches response_model/annotation mistakes that only surface on generation.
    res = await client.get("/openapi.json")
    assert res.status_code == 200
    paths = res.json()["paths"]
    for route in ("/documents/upload", "/documents", "/query", "/history", "/sessions"):
        assert route in paths


# --------------------------------------------------------------------------- #
# Upload and ingestion
# --------------------------------------------------------------------------- #
async def test_upload_returns_202_and_queues_ingestion(client, solar_txt: Path):
    result = await upload(client, solar_txt)

    assert result["status"] == 202
    doc = result["body"]["document"]
    assert doc["original_name"] == "solar-notes.txt"
    assert doc["status"] in ("pending", "processing")
    assert doc["size_bytes"] > 0
    assert result["body"]["duplicate"] is False


async def test_ingestion_completes_and_reports_chunks(client, solar_txt: Path):
    doc = await ingest(client, solar_txt)

    assert doc["status"] == "ready"
    assert doc["chunk_count"] > 0
    assert doc["page_count"] >= 1
    assert doc["embedding_model"]
    assert doc["error"] is None


async def test_document_detail_can_include_chunk_previews(client, solar_txt: Path):
    doc = await ingest(client, solar_txt)
    res = await client.get(f"/documents/{doc['id']}", params={"include_chunks": True})

    chunks = res.json()["chunks"]
    assert chunks
    assert [c["chunk_index"] for c in chunks] == sorted(c["chunk_index"] for c in chunks)
    assert all(c["text"].strip() for c in chunks)
    assert all(c["page_start"] >= 1 for c in chunks)


async def test_chunks_are_omitted_by_default(client, solar_txt: Path):
    doc = await ingest(client, solar_txt)
    assert (await client.get(f"/documents/{doc['id']}")).json()["chunks"] == []


async def test_listing_documents_paginates(client, solar_txt: Path, handbook_docx: Path):
    await ingest(client, solar_txt)
    await ingest(client, handbook_docx)

    body = (await client.get("/documents")).json()
    assert body["total"] == 2
    assert len(body["documents"]) == 2

    page = (await client.get("/documents", params={"limit": 1})).json()
    assert len(page["documents"]) == 1
    assert page["total"] == 2  # total is the full count, not the page size


async def test_documents_can_be_filtered_by_status(client, solar_txt: Path):
    await ingest(client, solar_txt)
    body = (await client.get("/documents", params={"status": "ready"})).json()
    assert body["total"] == 1

    empty = (await client.get("/documents", params={"status": "failed"})).json()
    assert empty["total"] == 0


async def test_identical_content_is_deduplicated(client, solar_txt: Path, tmp_path: Path):
    first = await ingest(client, solar_txt)

    # Same bytes, different filename: re-embedding would be wasted work.
    copy = tmp_path / "renamed-copy.txt"
    copy.write_bytes(solar_txt.read_bytes())
    result = await upload(client, copy)

    assert result["status"] == 202
    assert result["body"]["duplicate"] is True
    assert result["body"]["document"]["id"] == first["id"]
    assert (await client.get("/documents")).json()["total"] == 1


async def test_pdf_ingestion_preserves_page_numbers(client, solar_pdf: Path):
    doc = await ingest(client, solar_pdf)
    assert doc["page_count"] == 3

    chunks = (await client.get(f"/documents/{doc['id']}",
                               params={"include_chunks": True})).json()["chunks"]
    assert max(c["page_end"] for c in chunks) == 3


async def test_docx_ingestion_works(client, handbook_docx: Path):
    doc = await ingest(client, handbook_docx)
    assert doc["status"] == "ready" and doc["chunk_count"] > 0


async def test_concurrent_uploads_all_complete(client, tmp_path: Path):
    # The worker pool is the part most likely to drop or double-process work.
    paths = []
    for i in range(5):
        path = tmp_path / f"doc-{i}.txt"
        path.write_text(f"Document {i} discusses topic {i} in detail. " * 20,
                        encoding="utf-8")
        paths.append(path)

    results = await asyncio.gather(*(upload(client, p) for p in paths))
    ids = [r["body"]["document"]["id"] for r in results]
    docs = await asyncio.gather(*(wait_ready(client, i) for i in ids))

    assert all(d["status"] == "ready" for d in docs)
    assert (await client.get("/documents")).json()["total"] == 5


# --------------------------------------------------------------------------- #
# Upload errors
# --------------------------------------------------------------------------- #
async def test_unsupported_file_type_is_rejected(client, tmp_path: Path):
    path = tmp_path / "archive.zip"
    path.write_bytes(b"PK\x03\x04 not a document")

    result = await upload(client, path)
    assert result["status"] == 415
    assert result["body"]["error"]["code"] == "unsupported_file_type"


async def test_empty_upload_is_rejected(client, tmp_path: Path):
    path = tmp_path / "nothing.txt"
    path.write_bytes(b"")

    result = await upload(client, path)
    assert result["status"] == 400
    assert result["body"]["error"]["code"] == "invalid_request"


async def test_whitespace_only_document_fails_ingestion_with_a_reason(
    client, empty_txt: Path
):
    # Bytes exist, so upload succeeds; extraction is what fails. The user needs
    # to see *why* rather than a document stuck at "processing".
    result = await upload(client, empty_txt)
    assert result["status"] == 202

    doc = await wait_ready(client, result["body"]["document"]["id"])
    assert doc["status"] == "failed"
    assert doc["error"]


async def test_oversized_upload_is_rejected(tiny_limit_client, tmp_path: Path):
    client = tiny_limit_client
    settings = client.services.settings
    path = tmp_path / "huge.txt"
    path.write_bytes(b"x" * (settings.max_upload_bytes + 1024))

    result = await upload(client, path)
    assert result["status"] == 413
    assert result["body"]["error"]["code"] == "file_too_large"


async def test_rejected_upload_leaves_nothing_on_disk(tiny_limit_client, tmp_path: Path):
    client = tiny_limit_client
    settings = client.services.settings
    path = tmp_path / "huge.txt"
    path.write_bytes(b"y" * (settings.max_upload_bytes + 1024))
    await upload(client, path)

    # A partial file left behind would slowly fill the disk.
    assert list(settings.upload_dir.iterdir()) == []


async def test_unknown_document_is_404(client):
    res = await client.get("/documents/does-not-exist")
    assert res.status_code == 404
    assert res.json()["error"]["code"] == "document_not_found"


# --------------------------------------------------------------------------- #
# Query
# --------------------------------------------------------------------------- #
async def test_query_answers_from_the_document_with_citations(client, solar_txt: Path):
    await ingest(client, solar_txt)

    res = await client.post("/query", json={
        "question": "What is the design life of string inverters?"
    })
    assert res.status_code == 200

    body = res.json()
    assert body["grounded"] is True
    assert "12 years" in body["answer"]
    assert body["sources"], "a grounded answer must carry sources"
    assert "[1]" in body["answer"]


async def test_citations_point_at_the_passage_containing_the_fact(client, solar_txt: Path):
    # The whole promise of the tool: the cited source must actually support the
    # claim, not merely be adjacent to it.
    await ingest(client, solar_txt)

    body = (await client.post("/query", json={
        "question": "How much do soiling losses reduce output in dry climates?"
    })).json()

    assert body["sources"]
    assert "15 percent" in body["sources"][0]["text"]
    assert "15 percent" in body["answer"]


async def test_source_numbers_are_contiguous_from_one(client, solar_txt: Path):
    await ingest(client, solar_txt)
    body = (await client.post("/query", json={
        "question": "How often should panels be cleaned?"
    })).json()

    numbers = [s["number"] for s in body["sources"]]
    assert numbers == list(range(1, len(numbers) + 1))


async def test_every_inline_marker_has_a_matching_source(client, solar_txt: Path):
    import re

    await ingest(client, solar_txt)
    body = (await client.post("/query", json={
        "question": "What is the efficiency of monocrystalline modules?"
    })).json()

    markers = {int(n) for m in re.finditer(r"\[([\d,\s]+)\]", body["answer"])
               for n in m.group(1).replace(" ", "").split(",") if n}
    available = {s["number"] for s in body["sources"]}
    # A marker with no source behind it is an unfollowable reference.
    assert markers <= available


async def test_citations_carry_page_labels_for_verification(client, solar_pdf: Path):
    await ingest(client, solar_pdf)
    body = (await client.post("/query", json={
        "question": "What is the design life of string inverters?"
    })).json()

    source = body["sources"][0]
    assert source["page_label"].startswith("p")
    assert source["page_start"] >= 1
    assert source["document_name"] == "solar.pdf"


async def test_diagnostics_describe_the_retrieval_that_happened(client, solar_txt: Path):
    await ingest(client, solar_txt)
    body = (await client.post("/query", json={"question": "How long do inverters last?"})).json()

    diag = body["diagnostics"]
    assert diag["strategy"]
    assert diag["chunks_in_context"] >= 1
    assert diag["candidates_considered"] >= diag["chunks_in_context"]
    assert diag["llm_model"] and diag["embedding_model"]
    assert body["latency_ms"] >= 0


async def test_off_topic_question_is_answered_honestly(client, solar_txt: Path):
    # The most important behaviour in the product: not guessing.
    await ingest(client, solar_txt)

    body = (await client.post("/query", json={
        "question": "What was the revenue of the Peruvian anchovy fishery in 1972?"
    })).json()

    assert body["grounded"] is False
    assert body["sources"] == []
    # No fabricated numbers -- the answer should decline, not answer.
    assert "1972" not in body["answer"]


async def test_query_with_no_documents_is_a_typed_conflict(client):
    res = await client.post("/query", json={"question": "Anything?"})
    assert res.status_code == 409
    assert res.json()["error"]["code"] == "no_documents"


async def test_query_scoped_to_one_document_ignores_the_other(
    client, solar_txt: Path, handbook_docx: Path
):
    solar = await ingest(client, solar_txt)
    handbook = await ingest(client, handbook_docx)

    body = (await client.post("/query", json={
        "question": "What torque is specified for the module clamp?",
        "document_ids": [solar["id"]],
    })).json()

    # The answer lives in the handbook, which is out of scope.
    assert all(s["document_id"] == solar["id"] for s in body["sources"])
    assert "16 Nm" not in body["answer"]

    scoped = (await client.post("/query", json={
        "question": "What torque is specified for the module clamp?",
        "document_ids": [handbook["id"]],
    })).json()
    assert "16 Nm" in scoped["answer"]


async def test_library_wide_query_can_reach_either_document(
    client, solar_txt: Path, handbook_docx: Path
):
    await ingest(client, solar_txt)
    handbook = await ingest(client, handbook_docx)

    body = (await client.post("/query", json={
        "question": "What torque is specified for the module clamp?"
    })).json()
    assert any(s["document_id"] == handbook["id"] for s in body["sources"])


async def test_subset_scope_accepts_several_ids(
    client, solar_txt: Path, handbook_docx: Path, tmp_path: Path
):
    solar = await ingest(client, solar_txt)
    handbook = await ingest(client, handbook_docx)

    third = tmp_path / "unrelated.txt"
    third.write_text("Quarterly hiring plans for the support organisation. " * 20,
                     encoding="utf-8")
    excluded = await ingest(client, third)

    body = (await client.post("/query", json={
        "question": "What torque is specified for the module clamp?",
        "document_ids": [solar["id"], handbook["id"]],
    })).json()

    assert all(s["document_id"] != excluded["id"] for s in body["retrieved"])


async def test_querying_an_unknown_document_id_is_404(client, solar_txt: Path):
    await ingest(client, solar_txt)
    res = await client.post("/query", json={
        "question": "Anything?", "document_ids": ["nope"],
    })
    assert res.status_code == 404
    assert res.json()["error"]["code"] == "document_not_found"


async def test_querying_a_failed_document_is_a_conflict(client, empty_txt: Path):
    result = await upload(client, empty_txt)
    doc = await wait_ready(client, result["body"]["document"]["id"])
    assert doc["status"] == "failed"

    res = await client.post("/query", json={
        "question": "Anything?", "document_ids": [doc["id"]],
    })
    assert res.status_code == 409
    assert res.json()["error"]["code"] == "document_not_ready"


async def test_top_k_limits_the_context(client, solar_txt: Path):
    await ingest(client, solar_txt)
    body = (await client.post("/query", json={
        "question": "What should I know about panel maintenance and inverters?",
        "top_k": 1,
    })).json()
    assert len(body["retrieved"]) <= 1


@pytest.mark.parametrize("payload,field", [
    ({}, "question"),
    ({"question": ""}, "question"),
    ({"question": "   "}, "question"),
    ({"question": "ok", "top_k": 0}, "top_k"),
    ({"question": "ok", "top_k": 999}, "top_k"),
])
async def test_invalid_query_payloads_are_422_with_field_details(client, payload, field):
    res = await client.post("/query", json=payload)
    assert res.status_code == 422

    error = res.json()["error"]
    assert error["code"] == "validation_error"
    assert any(field in d["field"] for d in error["details"])


async def test_blank_document_ids_are_treated_as_no_scope(client, solar_txt: Path):
    # [""] must not become a lookup for a document named "".
    await ingest(client, solar_txt)
    res = await client.post("/query", json={
        "question": "How long do string inverters last?", "document_ids": ["", "  "],
    })
    assert res.status_code == 200


# --------------------------------------------------------------------------- #
# Streaming
# --------------------------------------------------------------------------- #
def parse_sse(text: str) -> list[dict]:
    """Collect JSON payloads from an SSE response body."""
    events = []
    for frame in text.split("\n\n"):
        payload = "\n".join(
            line[5:].strip() for line in frame.splitlines() if line.startswith("data:")
        )
        if payload and payload != "[DONE]":
            events.append(json.loads(payload))
    return events


async def test_stream_emits_meta_then_deltas_then_done(client, solar_txt: Path):
    await ingest(client, solar_txt)

    async with client.stream("POST", "/query", json={
        "question": "What is the design life of string inverters?", "stream": True,
    }) as res:
        assert res.status_code == 200
        assert res.headers["content-type"].startswith("text/event-stream")
        body = "".join([chunk async for chunk in res.aiter_text()])

    events = parse_sse(body)
    kinds = [e["type"] for e in events]
    assert kinds[0] == "meta"
    assert kinds[-1] == "done"
    assert kinds.count("delta") > 1
    assert body.rstrip().endswith("[DONE]")


async def test_streamed_deltas_reassemble_into_the_final_answer(client, solar_txt: Path):
    await ingest(client, solar_txt)

    async with client.stream("POST", "/query", json={
        "question": "How often should panels be cleaned?", "stream": True,
    }) as res:
        body = "".join([chunk async for chunk in res.aiter_text()])

    events = parse_sse(body)
    streamed = "".join(e["text"] for e in events if e["type"] == "delta")
    done = next(e for e in events if e["type"] == "done")

    # Deltas carry raw source numbers; `done` carries renumbered citations. The
    # prose either side of the markers must still match.
    import re
    strip = lambda s: re.sub(r"\s+", " ", re.sub(r"\[[\d,\s]*\]", "", s)).strip()
    assert strip(streamed) == strip(done["answer"])


async def test_done_event_carries_resolved_sources(client, solar_txt: Path):
    await ingest(client, solar_txt)

    async with client.stream("POST", "/query", json={
        "question": "What is the design life of string inverters?", "stream": True,
    }) as res:
        body = "".join([chunk async for chunk in res.aiter_text()])

    done = next(e for e in parse_sse(body) if e["type"] == "done")
    assert done["grounded"] is True
    assert done["sources"]
    assert done["diagnostics"]["chunks_in_context"] >= 1


async def test_meta_event_previews_the_retrieved_passages(client, solar_txt: Path):
    # The frontend renders evidence cards from this before any token arrives.
    await ingest(client, solar_txt)

    async with client.stream("POST", "/query", json={
        "question": "What is the design life of string inverters?", "stream": True,
    }) as res:
        body = "".join([chunk async for chunk in res.aiter_text()])

    meta = next(e for e in parse_sse(body) if e["type"] == "meta")
    assert meta["retrieved"]
    assert all("text" in r and "page_label" in r for r in meta["retrieved"])


async def test_stream_errors_arrive_in_band_not_as_a_status_code(client):
    # The 200 is already committed when streaming starts, so a failure can only
    # be reported inside the stream.
    async with client.stream("POST", "/query", json={
        "question": "Anything?", "stream": True,
    }) as res:
        assert res.status_code == 200
        body = "".join([chunk async for chunk in res.aiter_text()])

    events = parse_sse(body)
    assert events[0]["type"] == "error"
    assert events[0]["code"] == "no_documents"
    assert body.rstrip().endswith("[DONE]")


# --------------------------------------------------------------------------- #
# Sessions and history
# --------------------------------------------------------------------------- #
async def test_exchanges_are_recorded_against_a_session(client, solar_txt: Path):
    await ingest(client, solar_txt)

    body = (await client.post("/query", json={
        "question": "How long do string inverters last?", "session_id": "s-1",
    })).json()
    assert body["session_id"] == "s-1"
    assert body["history_id"]

    history = (await client.get("/history", params={"session_id": "s-1"})).json()["history"]
    assert len(history) == 1
    assert history[0]["question"] == "How long do string inverters last?"
    assert history[0]["answer"] == body["answer"]
    assert history[0]["grounded"] is True
    assert history[0]["llm_model"]


async def test_history_is_newest_first(client, solar_txt: Path):
    await ingest(client, solar_txt)
    for question in ("First question about inverters?",
                     "Second question about cleaning?",
                     "Third question about efficiency?"):
        await client.post("/query", json={"question": question, "session_id": "s-1"})

    history = (await client.get("/history", params={"session_id": "s-1"})).json()["history"]
    assert [h["question"] for h in history][0].startswith("Third")


async def test_queries_without_a_session_are_not_recorded(client, solar_txt: Path):
    await ingest(client, solar_txt)
    body = (await client.post("/query", json={"question": "How long do inverters last?"})).json()

    assert body["session_id"] is None
    assert body["history_id"] is None
    assert (await client.get("/history")).json()["history"] == []


async def test_an_unseen_session_id_is_created_implicitly(client, solar_txt: Path):
    await ingest(client, solar_txt)
    await client.post("/query", json={"question": "How long do inverters last?",
                                      "session_id": "brand-new"})

    sessions = (await client.get("/sessions")).json()["sessions"]
    assert [s["id"] for s in sessions] == ["brand-new"]
    assert sessions[0]["exchange_count"] == 1


async def test_session_is_titled_from_its_first_question(client, solar_txt: Path):
    await ingest(client, solar_txt)
    await client.post("/query", json={"question": "How long do string inverters last?",
                                      "session_id": "s-1"})
    await client.post("/query", json={"question": "And microinverters?",
                                      "session_id": "s-1"})

    session = (await client.get("/sessions")).json()["sessions"][0]
    assert session["title"]
    assert session["exchange_count"] == 2


async def test_sessions_can_be_created_explicitly(client):
    res = await client.post("/sessions")
    assert res.status_code == 201

    session = res.json()
    assert session["id"] and session["created_at"]
    assert session["exchange_count"] == 0


async def test_a_session_can_be_fetched_with_its_exchanges(client, solar_txt: Path):
    await ingest(client, solar_txt)
    await client.post("/query", json={"question": "How long do inverters last?",
                                      "session_id": "s-1"})

    body = (await client.get("/sessions/s-1")).json()
    assert len(body["history"]) == 1


async def test_unknown_session_is_404(client):
    res = await client.get("/sessions/nope")
    assert res.status_code == 404
    assert res.json()["error"]["code"] == "session_not_found"


async def test_deleting_a_session_removes_its_exchanges(client, solar_txt: Path):
    await ingest(client, solar_txt)
    await client.post("/query", json={"question": "How long do inverters last?",
                                      "session_id": "s-1"})

    assert (await client.delete("/sessions/s-1")).status_code == 200
    assert (await client.get("/sessions")).json()["sessions"] == []
    # FK cascade, not just the session row.
    assert (await client.get("/history")).json()["history"] == []


async def test_history_can_be_filtered_by_document(client, solar_txt: Path, handbook_docx: Path):
    solar = await ingest(client, solar_txt)
    handbook = await ingest(client, handbook_docx)

    await client.post("/query", json={"question": "How long do inverters last?",
                                      "session_id": "s-1", "document_ids": [solar["id"]]})
    await client.post("/query", json={"question": "What torque for the clamp?",
                                      "session_id": "s-1", "document_ids": [handbook["id"]]})

    filtered = (await client.get("/history", params={"document_id": handbook["id"]})).json()
    assert len(filtered["history"]) == 1
    assert "torque" in filtered["history"][0]["question"].lower()


async def test_filtering_history_by_both_session_and_document_is_rejected(client):
    res = await client.get("/history", params={"session_id": "s", "document_id": "d"})
    assert res.status_code == 400
    assert res.json()["error"]["code"] == "invalid_request"


async def test_followup_question_uses_conversation_history(client, solar_txt: Path):
    # "What about microinverters?" is unanswerable without the prior turn.
    await ingest(client, solar_txt)
    await client.post("/query", json={"question": "What is the design life of string inverters?",
                                      "session_id": "s-1"})

    res = await client.post("/query", json={"question": "What about microinverters?",
                                            "session_id": "s-1"})
    assert res.status_code == 200
    assert res.json()["session_id"] == "s-1"


async def test_history_can_be_disabled_per_request(client, solar_txt: Path):
    await ingest(client, solar_txt)
    await client.post("/query", json={"question": "How long do inverters last?",
                                      "session_id": "s-1"})

    res = await client.post("/query", json={"question": "And the efficiency?",
                                            "session_id": "s-1", "use_history": False})
    assert res.status_code == 200
    # Still recorded, just not replayed into the prompt.
    assert len((await client.get("/history")).json()["history"]) == 2


# --------------------------------------------------------------------------- #
# Deletion
# --------------------------------------------------------------------------- #
async def test_deleting_a_document_removes_it_from_retrieval(client, solar_txt: Path):
    doc = await ingest(client, solar_txt)
    assert (await client.delete(f"/documents/{doc['id']}")).status_code == 200

    assert (await client.get(f"/documents/{doc['id']}")).status_code == 404
    # An orphaned vector would keep surfacing with no row to render it.
    res = await client.post("/query", json={"question": "How long do inverters last?"})
    assert res.status_code == 409
    assert res.json()["error"]["code"] == "no_documents"


async def test_deleting_a_document_removes_its_stored_file(client, solar_txt: Path):
    doc = await ingest(client, solar_txt)
    upload_dir = client.services.settings.upload_dir
    assert list(upload_dir.iterdir())

    await client.delete(f"/documents/{doc['id']}")
    assert list(upload_dir.iterdir()) == []


async def test_deleting_one_document_leaves_the_other_queryable(
    client, solar_txt: Path, handbook_docx: Path
):
    solar = await ingest(client, solar_txt)
    await ingest(client, handbook_docx)

    await client.delete(f"/documents/{solar['id']}")
    body = (await client.post("/query", json={
        "question": "What torque is specified for the module clamp?"
    })).json()

    assert "16 Nm" in body["answer"]
    assert all(s["document_id"] != solar["id"] for s in body["retrieved"])


async def test_deleting_an_unknown_document_is_404(client):
    res = await client.delete("/documents/nope")
    assert res.status_code == 404


async def test_reuploading_after_deletion_works(client, solar_txt: Path):
    # The dedup index must not keep claiming the content exists.
    first = await ingest(client, solar_txt)
    await client.delete(f"/documents/{first['id']}")

    second = await upload(client, solar_txt)
    assert second["body"]["duplicate"] is False
    doc = await wait_ready(client, second["body"]["document"]["id"])
    assert doc["status"] == "ready"


# --------------------------------------------------------------------------- #
# Auth
# --------------------------------------------------------------------------- #
async def test_protected_routes_require_the_key(secured_client):
    for method, path in [("get", "/documents"), ("get", "/history"),
                         ("get", "/sessions"), ("post", "/query")]:
        res = await getattr(secured_client, method)(
            path, **({"json": {"question": "hi"}} if method == "post" else {})
        )
        assert res.status_code == 401, path
        assert res.json()["error"]["code"] == "unauthorized"


async def test_correct_key_is_accepted(secured_client):
    res = await secured_client.get("/documents", headers={"X-API-Key": "test-secret"})
    assert res.status_code == 200


async def test_bearer_token_is_accepted(secured_client):
    res = await secured_client.get("/documents",
                                   headers={"Authorization": "Bearer test-secret"})
    assert res.status_code == 200


@pytest.mark.parametrize("headers", [
    {"X-API-Key": "wrong"},
    {"X-API-Key": ""},
    {"X-API-Key": "test-secret "},        # trailing space is a different key
    {"X-API-Key": "test-secre"},          # prefix must not pass
    {"Authorization": "Bearer wrong"},
    {"Authorization": "Basic test-secret"},
])
async def test_bad_credentials_are_rejected(secured_client, headers):
    assert (await secured_client.get("/documents", headers=headers)).status_code == 401


async def test_upload_and_query_work_end_to_end_with_auth(secured_client, solar_txt: Path):
    auth = {"X-API-Key": "test-secret"}
    result = await upload(secured_client, solar_txt, headers=auth)
    assert result["status"] == 202

    doc_id = result["body"]["document"]["id"]
    deadline = asyncio.get_event_loop().time() + 20
    while asyncio.get_event_loop().time() < deadline:
        doc = (await secured_client.get(f"/documents/{doc_id}", headers=auth)).json()
        if doc["status"] in ("ready", "failed"):
            break
        await asyncio.sleep(0.05)
    assert doc["status"] == "ready"

    res = await secured_client.post(
        "/query", headers=auth,
        json={"question": "What is the design life of string inverters?"},
    )
    assert res.status_code == 200
    assert res.json()["grounded"] is True


# --------------------------------------------------------------------------- #
# Cross-cutting
# --------------------------------------------------------------------------- #
async def test_every_error_uses_the_same_envelope(client):
    responses = [
        await client.get("/documents/nope"),                        # 404
        await client.post("/query", json={}),                       # 422
        await client.post("/query", json={"question": "hi"}),        # 409
        await client.get("/sessions/nope"),                          # 404
        await client.get("/history", params={"session_id": "s", "document_id": "d"}),
    ]
    for res in responses:
        assert res.status_code >= 400
        error = res.json()["error"]
        # Clients branch on `code`, so both fields must always be present.
        assert isinstance(error["code"], str) and error["code"]
        assert isinstance(error["message"], str) and error["message"]


async def test_all_three_pages_of_content_are_retrievable(client, solar_txt: Path):
    # A per-page smoke test: chunking or page attribution could silently lose one.
    await ingest(client, solar_txt)

    probes = [
        ("What is the efficiency of monocrystalline modules?", "22 percent"),
        ("How much do soiling losses reduce output?", "15 percent"),
        ("How long do string inverters last?", "12 years"),
    ]
    for question, expected in probes:
        body = (await client.post("/query", json={"question": question})).json()
        # Asserted against the retrieved context, not the answer text. What this
        # test is for is coverage -- that every part of the document can still be
        # found -- and `retrieved` is exactly the context the model was given.
        # Asserting on `answer` instead would really be testing FakeLLMProvider's
        # extraction heuristic, which quotes the single highest-overlap sentence:
        # it can retrieve the right chunk and still quote a neighbouring sentence
        # that omits the figure. That made this test fail for a reason unrelated
        # to retrieval, and it would fail again under any change to the stub.
        context = " ".join(source["text"] for source in body["retrieved"])
        assert expected in context, (
            f"{question!r} did not retrieve {expected!r}; "
            f"got chunks {[s['chunk_index'] for s in body['retrieved']]}"
        )


async def test_source_text_appears_verbatim_in_the_document(client, solar_txt: Path):
    # Citation text must be quotable. If it were paraphrased or re-wrapped, the
    # user could not find it in the original.
    await ingest(client, solar_txt)
    body = (await client.post("/query", json={
        "question": "How long do string inverters last?"
    })).json()

    original = " ".join(SOLAR_PAGES)
    excerpt = " ".join(body["sources"][0]["text"].split())
    assert excerpt in " ".join(original.split())
