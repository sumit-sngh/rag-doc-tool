# Document Q&A Tool

Upload your documents, ask questions in plain language, get answers grounded in
the passages that actually support them — with citations you can click through to
verify.

Runs entirely on your machine. No API keys, no data leaving the box: local
embeddings and a local LLM via [Ollama](https://ollama.com), an embedded Chroma
vector store, and SQLite for metadata.

```
PDF/DOCX/TXT/MD  ──▶  extract  ──▶  chunk  ──▶  embed  ──▶  Chroma
                                                              │
question  ──▶  embed  ──▶  vector search  ─┐                  │
          └──▶  FTS5 keyword search  ──────┴─▶ RRF ─▶ MMR ─▶ top-k
                                                              │
                                            prompt + passages ─▶  LLM  ──▶  answer [1][2]
```

---

## Quick start

Two paths. Pick the second if you just want to see the thing work.

**Python 3.10–3.12.** `requirements.txt` is exactly pinned, and some of those
packages ship compiled wheels per CPython version. On a newer interpreter pip
finds no wheel, tries to compile from source, and fails on a missing Rust
toolchain — see [Troubleshooting](#troubleshooting). `make venv` picks a
supported interpreter for you and refuses with instructions if it can't find
one; Docker pins 3.12 regardless.

Either way, `python3 scripts/preflight.py` (or `make check`) is the fastest way to
find out what's missing. It checks the Python version, every dependency, your
config, whether the data directory is writable, whether Ollama is up and the
models are pulled, and whether the app imports — then tells you which one is
broken. Uses only the standard library, so it works before anything is installed.

### With real models (Ollama)

```bash
# 1. Ollama, plus the two models
#    macOS/Linux: https://ollama.com/download
ollama serve &                # skip if the desktop app is already running
ollama pull llama3.1          # generation (~4.7 GB)
ollama pull nomic-embed-text  # embeddings (~275 MB)

# 2. Python deps
make venv && source .venv/bin/activate

# 3. Config (optional — the defaults already match the models above)
cp .env.example .env

# 4. Check, then run
python3 scripts/preflight.py
uvicorn app.main:app --reload
```

Open <http://localhost:8000>. Upload a PDF, wait for the status dot to turn
green, ask it something.

### Without downloading any models

Set both providers to `fake` and everything runs offline, instantly:

```bash
make venv && source .venv/bin/activate
make run-fake     # == EMBEDDING_PROVIDER=fake LLM_PROVIDER=fake uvicorn app.main:app --reload
```

The `fake` providers are deterministic and *lexically real* — the embedder is a
hashed bag-of-words, so similar text genuinely lands nearby, and the LLM does
extractive answering with real citation markers. Retrieval, ranking, citations,
scoping, streaming and history all behave correctly. Only the prose quality is
missing. This is what the test suite runs against, and it's the fastest way to
work on the frontend.

### Docker

```bash
make up       # API in a container, Ollama on your host
make up-all   # both containerised, models pulled for you on first boot
```

Or without make:

```bash
docker compose up
docker compose -f docker-compose.yml -f docker-compose.ollama.yml up
```

Prefer host Ollama on a laptop: a containerised Ollama gets no GPU access
without extra setup, and CPU inference of llama3.1 is slow enough to feel
broken. `make down` stops everything; the named volumes (ingested data, pulled
models) survive.

`make help` lists the rest.

### Online, for other people

See **[DEPLOY.md](DEPLOY.md)** for the full path. The essentials, because they
are easy to get wrong and fail quietly:

- Inference moves to a hosted OpenAI-compatible API. There is no Ollama and no
  GPU in the container, so an API key exists in the deployment — supply it as a
  secret, never in `fly.toml`.
- Set `VISITOR_ISOLATION=true`. Without it, everyone on a public link shares one
  library and can read and delete each other's uploads.
- The deployed instance starts empty on purpose. Vectors embedded by
  `nomic-embed-text` (768-dim) mean nothing to `text-embedding-3-small`
  (1536-dim); mixing them raises no error and quietly ruins retrieval, so
  documents get re-uploaded rather than migrated.
- Exactly one machine, and a persistent volume mounted at `DATA_DIR`. The
  ingestion queue is in-process and both stores hold file locks, so a second
  replica means uploads land where queries don't look. No volume means every
  restart begins with an empty library.

```bash
fly volumes create qa_data --region iad --size 3
fly secrets set OPENAI_API_KEY=sk-your-key-here
fly deploy
```

---

## What it does

**Ingestion.** PDF (per-page, so citations can name a page), DOCX (headings
become Markdown levels, tables become pipe rows), TXT and MD. Text is split on
paragraph boundaries first, sentence boundaries second, and only mid-sentence as
a last resort, with a configurable character overlap so a fact never gets
severed from the sentence that qualifies it. Uploads return `202` immediately and
embed in a background worker — a 200-page PDF doesn't block the request.

**Retrieval.** Vector search over Chroma (cosine) and BM25 keyword search over
SQLite FTS5, fused with reciprocal rank fusion, then reranked with MMR to stop
overlapping near-duplicate chunks from eating the whole context window. Hybrid
search matters more than it sounds: pure vector search reliably misses exact
identifiers — part numbers, section references, surnames — because embeddings
smooth precisely the tokens you were searching for.

**Grounding.** Retrieved chunks past `MAX_DISTANCE` are discarded before the
prompt is built, so when nothing relevant exists the model is told nothing
relevant exists. It replies `INSUFFICIENT_CONTEXT`, which the API turns into an
honest "I don't know" with `grounded: false` — rather than a confident answer
assembled from whatever happened to rank fifth.

**Citations.** The model cites `[1]`, `[2]` against numbered sources. Those
markers are parsed, renumbered in order of first appearance, and resolved back to
document name, page range and the verbatim chunk text. Markers pointing at
sources that don't exist are stripped and reported in `diagnostics`. If an answer
ends up with no valid citation at all, `grounded` is `false` — the flag is
earned, not assumed.

**Scoping.** `document_ids` takes one id, several, or nothing at all (whole
library).

**Sessions.** Exchanges are grouped by `session_id`, with prior turns fed back as
context so follow-ups like "what about the second one?" resolve.

Also included: SSE token streaming, content-hash upload dedup, optional
cross-encoder reranking, optional OCR for scanned PDFs, and optional API-key
auth.

---

## API

Interactive docs at `/docs`, schema at `/openapi.json`.

| Method   | Path                    | Purpose                                        |
| -------- | ----------------------- | ---------------------------------------------- |
| `POST`   | `/documents/upload`     | Upload a file. `202`, ingestion is async       |
| `GET`    | `/documents`            | List documents. `?status=`, `?limit=`, `?offset=` |
| `GET`    | `/documents/{id}`       | One document. `?include_chunks=true` to inspect chunking |
| `DELETE` | `/documents/{id}`       | Remove document, chunks, vectors and file      |
| `POST`   | `/query`                | Ask a question. `stream: true` for SSE         |
| `GET`    | `/history`              | Past exchanges. `?session_id=` or `?document_id=` |
| `POST`   | `/sessions`             | Create a session explicitly                    |
| `GET`    | `/sessions`             | List sessions                                  |
| `GET`    | `/sessions/{id}`        | One session's exchanges                        |
| `DELETE` | `/sessions/{id}`        | Delete a session and its exchanges             |
| `GET`    | `/health`               | Component health, effective config, stats      |

### Upload

```bash
curl -F "file=@handbook.pdf" http://localhost:8000/documents/upload
```

```json
{
  "document": { "id": "3f2a…", "original_name": "handbook.pdf", "status": "pending", "chunk_count": 0 },
  "duplicate": false,
  "message": "Upload accepted. Processing in the background."
}
```

`202`, not `200`: the file is stored and queued, not yet searchable. Poll until
ready:

```bash
curl "http://localhost:8000/documents/3f2a…" | python3 -m json.tool
```

Re-uploading identical content returns `duplicate: true` and points at the
existing document instead of embedding it twice.

### Ask

```bash
curl -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"question": "How often should panels be cleaned?"}'
```

```json
{
  "answer": "Panels should be cleaned twice per year, and quarterly in arid or agricultural regions [1].",
  "grounded": true,
  "sources": [
    {
      "number": 1,
      "document_name": "handbook.pdf",
      "page_label": "p. 2",
      "text": "Soiling losses are the single most underestimated factor…",
      "similarity": 0.71
    }
  ],
  "retrieved": [ "…every passage that went into the prompt, cited or not…" ],
  "session_id": "8c1d…",
  "latency_ms": 2843,
  "diagnostics": { "strategy": "hybrid", "vector_hits": 12, "keyword_hits": 8, "filtered_out_by_distance": 3 }
}
```

`sources` is what the answer actually cited; `retrieved` is everything the model
was shown. The difference is the useful part — it's how you tell "the passage
wasn't retrieved" from "it was retrieved and the model ignored it," which are
opposite fixes (tune retrieval vs. tune the prompt).

Scope and continue a conversation:

```bash
curl -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{
        "question": "And the inverter budget?",
        "document_ids": ["3f2a…"],
        "session_id": "8c1d…",
        "top_k": 8
      }'
```

### Streaming

```bash
curl -N -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"question": "Summarise the maintenance schedule", "stream": true}'
```

```
data: {"type":"meta","retrieved":[…],"diagnostics":{…}}

data: {"type":"delta","text":"Panels"}

data: {"type":"delta","text":" should"}

data: {"type":"done","answer":"…","grounded":true,"sources":[…],"latency_ms":2843}

data: [DONE]
```

`meta` arrives before the first token, so the UI can render sources while the
answer is still being written. The `done` event repeats the full answer because
citations can only be resolved once the final text exists — don't try to
reconstruct `sources` from the deltas.

Errors mid-stream arrive as `{"type": "error", "code": …, "message": …}`, not as
an HTTP status: by the time streaming begins the `200` is already sent. Clients
must handle in-band errors. `[DONE]` always terminates the stream, which is how
you distinguish a clean finish from a dropped socket.

### Errors

Every failure — including FastAPI's own validation errors — uses one envelope:

```json
{ "error": { "code": "unsupported_file_type", "message": "Unsupported file type '.xlsx'. Supported: .docx, .markdown, .md, .pdf, .txt" } }
```

`details` is present only when there's something structured to say — validation
errors carry a per-field list. Branch on `code`, never on the message text:

| Code                    | HTTP | Means                                            |
| ----------------------- | ---- | ------------------------------------------------ |
| `unsupported_file_type` | 415  | Not PDF/DOCX/TXT/MD                              |
| `file_too_large`        | 413  | Over `MAX_UPLOAD_MB`                             |
| `empty_document`        | 422  | No extractable text (a scan? try `OCR_ENABLED`)  |
| `extraction_failed`     | 422  | Corrupt or password-protected file               |
| `validation_error`      | 422  | Malformed request body                           |
| `invalid_request`       | 400  | Blank question, empty upload, conflicting filters |
| `unauthorized`          | 401  | Missing or wrong `X-API-Key`                     |
| `document_not_found`    | 404  | Unknown document id                              |
| `session_not_found`     | 404  | Unknown session id                               |
| `document_not_ready`    | 409  | Scoped to a document still ingesting or failed   |
| `no_documents`          | 409  | Library is empty                                 |
| `model_not_found`       | 502  | Model not pulled in Ollama                       |
| `provider_error`        | 502  | Ollama unreachable or erroring                   |
| `vector_store_error`    | 500  | Chroma failure                                   |
| `internal_error`        | 500  | Unhandled bug. Message is deliberately generic — the detail is in the server log |

Note the two "no answer" cases are deliberately different: `no_documents` /
`document_not_ready` are `409`s because the *request* can't be served, while "I
searched and found nothing relevant" is a `200` with `grounded: false`. The
second is a legitimate answer, not an error, and treating it as one would make
every honest refusal look like a bug.

---

## Configuration

Everything is env-driven; `.env.example` documents every value with its default
and is worth reading in full. The most useful knobs:

| Variable                                  | Default            | Notes                                       |
| ----------------------------------------- | ------------------ | ------------------------------------------- |
| `EMBEDDING_PROVIDER` / `LLM_PROVIDER`     | `ollama`           | `ollama` \| `fake`                          |
| `OLLAMA_BASE_URL`                         | `http://localhost:11434` |                                       |
| `EMBEDDING_MODEL` / `LLM_MODEL`           | `nomic-embed-text` / `llama3.1` |                            |
| `EMBEDDING_DIM`                           | `768`              | Must match the model; changing it means re-ingesting |
| `CHUNK_SIZE` / `CHUNK_OVERLAP`            | `1200` / `200`     | Characters, ~4 per token                    |
| `TOP_K` / `CANDIDATE_K`                   | `5` / `20`         | Prompt slots / pre-rerank net               |
| `HYBRID_SEARCH` / `VECTOR_WEIGHT`         | `true` / `0.7`     | `1.0` = vector only, `0.0` = keyword only   |
| `RERANK_STRATEGY` / `MMR_LAMBDA`          | `mmr` / `0.7`      | `none` \| `mmr` \| `cross_encoder`          |
| `MAX_DISTANCE`                            | `1.15`             | Relevance cutoff — see tuning below         |
| `LLM_TEMPERATURE`                         | `0.1`              | Low on purpose: extraction, not prose       |
| `MAX_CONTEXT_CHARS`                       | `12000`            | Hard ceiling on assembled context           |
| `MAX_UPLOAD_MB`                           | `50`               |                                             |
| `API_KEY`                                 | unset              | Set to require `X-API-Key` on every route except `/health` |
| `OCR_ENABLED`                             | `false`            | Needs `requirements-optional.txt` + tesseract |

### Tuning retrieval

Read `diagnostics` on a `/query` response first — it reports vector hits, keyword
hits, and how many candidates the distance filter rejected. Then:

- **Good questions get "I don't know."** `MAX_DISTANCE` is too tight, or the
  answer spans more text than one chunk holds. Raise it toward `1.3`, or raise
  `TOP_K`.
- **Irrelevant passages get cited.** Lower `MAX_DISTANCE` toward `1.0`. A tighter
  cutoff produces more refusals — which is the correct trade for anything where a
  wrong answer costs more than no answer.
- **Exact terms (part numbers, section refs) aren't found.** Keep
  `HYBRID_SEARCH=true` and lower `VECTOR_WEIGHT` to `0.5`. This is the one failure
  mode pure vector search is structurally bad at.
- **Sources are five near-identical passages.** Lower `MMR_LAMBDA` to `0.5` to
  buy coverage with the later slots.
- **Answers cite the right document but the wrong part.** Reduce `CHUNK_SIZE` to
  `600–800` and re-ingest. Smaller chunks localise better; they also cut more
  facts away from their qualifiers, so it's a real trade.

Changing `CHUNK_SIZE`, `CHUNK_OVERLAP`, `EMBEDDING_MODEL` or `EMBEDDING_DIM`
only affects documents ingested afterwards. To apply them everywhere, delete and
re-upload — or wipe `data/` for a clean slate.

---

## Layout

```
app/
  config.py         env-driven Settings, validated at startup
  main.py           app factory, error handlers, static frontend
  deps.py           service container + auth dependency
  db.py             aiosqlite: documents, chunks, sessions, history, FTS5
  schema.py         SQL DDL           schemas.py  Pydantic contracts
  errors.py         typed errors → HTTP codes
  ingestion/        extractors.py · chunker.py · pipeline.py (bg worker)
  providers/        base.py (protocol) · ollama.py · fake.py
  retrieval/        vector_store.py · keyword.py · fusion.py · rerank.py · retriever.py
  qa/               prompts.py · engine.py · citations.py
  routers/          documents · query · history · health
frontend/index.html single file, no build step
tests/              pytest suite + a dependency-free node test for the frontend
```

Two decisions worth knowing before you change things:

*Providers sit behind a protocol* (`app/providers/base.py`), so adding OpenAI or
Anthropic means one new module and one enum value — nothing in retrieval or the
QA engine knows which provider it's talking to.

*The app is built by a factory* (`create_app(settings)`) rather than a module
singleton, so tests stand up a fully isolated instance against a temp directory.
That's what makes the API suite fast and parallel-safe.

---

## Tests

```bash
pytest                      # ~200 assertions across chunking, extraction,
                            # retrieval, citations, providers, and the full API
node tests/frontend.test.js # SSE parsing, citation rendering, escaping —
                            # no npm install required
```

The suite runs against the `fake` providers, so it needs no Ollama and no
network. `pip install -r requirements-optional.txt` adds `reportlab`, which
unlocks the PDF fixtures — without it the PDF tests skip, and nothing verifies
page-number attribution, which is exactly what citations point at.

`tests/test_providers.py` deserves a note: it exists because everything else
trusts the fakes. If the fake embedder weren't lexically meaningful, or the fake
LLM didn't emit real citations against the sources it was given, the retrieval
and citation suites would pass while testing nothing.

---

## Troubleshooting

**`Failed building wheel for pydantic-core` during `pip install`.** Your Python is
newer than the pins. `pydantic-core` is written in Rust and published as
prebuilt wheels per CPython version; when none matches your interpreter, pip
falls back to compiling it, which needs a Rust toolchain. Fix by naming a
supported interpreter:

```bash
rm -rf .venv
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

No `python3.12`? `brew install python@3.12` on macOS, or `make up` to let Docker
supply it. Note that pip installs atomically, so *one* failed build leaves the
whole venv empty — preflight then reports every package as missing. That's one
problem, not thirteen.

**`provider_error` / everything 502s.** Ollama isn't running.
`curl http://localhost:11434/api/tags` should return JSON; start it with
`ollama serve`. `Connection refused` (errno 61 on macOS) at preflight means the
same thing. To skip models entirely while you check the rest works, use
`make run-fake`. In Docker, `localhost` means the container — use
`host.docker.internal`.

**`model_not_found`.** The model isn't pulled: `ollama pull llama3.1` and
`ollama pull nomic-embed-text`. `ollama list` shows what you have. Names must
match `LLM_MODEL` / `EMBEDDING_MODEL` exactly, tag included.

**Upload stays `pending` forever.** Embedding is slow on CPU — a long PDF can
take minutes. `/health` reports `queue_depth`; the server log reports per-batch
progress. If `status` is `failed`, `error` on the document says why.

**`empty_document` on a PDF.** It's a scan with no text layer. Install the OCR
extras plus tesseract and poppler, then set `OCR_ENABLED=true`.

**Answers are always "I don't know."** Confirm the document is `ready` with a
non-zero `chunk_count`, then check `diagnostics.filtered_out_by_distance` — if
it's high, `MAX_DISTANCE` is the culprit.

**Everything answers "I don't know" after switching from `fake` to Ollama.** The
vectors are stale. The `fake` embedder is also 768-dimensional, so switching to
`nomic-embed-text` raises no dimension error — Chroma just compares real query
vectors against hashed-bag-of-words document vectors and returns noise, which the
distance filter then discards. Wipe and re-ingest when you change provider:

```bash
make clean    # deletes data/, documents must be re-uploaded
```

`GET /documents/{id}` reports the `embedding_model` each document was indexed
with, which is how you confirm this is what happened.

**Dimension mismatch after switching embedding models.** Vectors of different
widths can't be compared. Delete `data/chroma` and re-ingest.

---

## Known limits

Single-process by design: the ingestion queue is in-process asyncio, so running
more than one uvicorn worker gives each its own queue and its own view of what's
pending. Multi-worker means moving that queue to Redis first.

Chroma runs embedded rather than as a server, which is right for a local tool and
wrong for concurrent writers.

`API_KEY` is a single shared secret — enough to stop a stray request on your LAN,
not a multi-tenant auth system. Per-user isolation would need real accounts and
document ownership.
