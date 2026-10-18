"""Database schema and SQL statements as plain strings.

Kept free of any driver import so the schema can be applied and exercised with
stdlib ``sqlite3`` in tests, while the app uses ``aiosqlite`` at runtime.

Design notes:

* Chunk text lives in SQLite, not only in Chroma. Chroma is the vector index;
  SQLite is the source of truth for text and metadata. That keeps citation
  rendering, keyword search, and re-indexing independent of the vector store.
* ``chunks_fts`` is an FTS5 index maintained by the repository rather than by
  triggers -- every write goes through one code path, and explicit inserts are
  easier to reason about than trigger cascades.
* Timestamps are ISO-8601 UTC strings. They sort lexicographically, which means
  ORDER BY works without a date function.
* ``documents.owner`` and ``sessions.owner`` scope rows to one visitor. There is
  deliberately no unscoped variant of any query that reads them: with isolation
  off every row is owned by ``shared`` and the predicate is a no-op, so a single
  code path serves both modes. Two variants would mean a query could leak by
  omission, and "we forgot to scope that one" is the whole bug class here.
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# Connection setup
# --------------------------------------------------------------------------- #
# WAL lets the ingestion worker write while queries read. NORMAL sync is the
# right trade for a local tool: durable across app crashes, fast enough to
# batch-insert thousands of chunks.
PRAGMAS: tuple[str, ...] = (
    "PRAGMA journal_mode = WAL",
    "PRAGMA synchronous = NORMAL",
    "PRAGMA foreign_keys = ON",
    "PRAGMA busy_timeout = 5000",
)

DOCUMENT_STATUSES = ("pending", "processing", "ready", "failed")

# Owner recorded when visitor isolation is off, and backfilled onto rows written
# before the column existed. Must match Settings.shared_owner; the default lives
# in the DDL too so a stray INSERT can never produce a NULL owner.
SHARED_OWNER = "shared"

# --------------------------------------------------------------------------- #
# DDL
# --------------------------------------------------------------------------- #
# `owner` sits second here for readability, but ALTER TABLE appends it, so an
# upgraded database has it last. Nothing reads columns positionally, so the two
# layouts are interchangeable.
CREATE_DOCUMENTS = """
CREATE TABLE IF NOT EXISTS documents (
    id              TEXT PRIMARY KEY,
    owner           TEXT    NOT NULL DEFAULT 'shared',
    original_name   TEXT    NOT NULL,
    stored_path     TEXT    NOT NULL,
    content_type    TEXT,
    extension       TEXT    NOT NULL,
    size_bytes      INTEGER NOT NULL,
    sha256          TEXT    NOT NULL,
    status          TEXT    NOT NULL DEFAULT 'pending'
                    CHECK (status IN ('pending','processing','ready','failed')),
    error           TEXT,
    page_count      INTEGER NOT NULL DEFAULT 0,
    chunk_count     INTEGER NOT NULL DEFAULT 0,
    embedding_model TEXT,
    created_at      TEXT    NOT NULL,
    updated_at      TEXT    NOT NULL
)
"""

# Not UNIQUE: a failed or deleted upload must be re-uploadable. Dedup is a
# deliberate application-level decision (see DocumentRepository.find_duplicate).
CREATE_DOCUMENTS_SHA_INDEX = (
    "CREATE INDEX IF NOT EXISTS idx_documents_sha256 ON documents(sha256)"
)
CREATE_DOCUMENTS_STATUS_INDEX = (
    "CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status, created_at DESC)"
)
# Every document list is owner-scoped and ordered by recency, so this is the
# index that actually gets used once more than one visitor exists.
CREATE_DOCUMENTS_OWNER_INDEX = (
    "CREATE INDEX IF NOT EXISTS idx_documents_owner "
    "ON documents(owner, created_at DESC)"
)

CREATE_CHUNKS = """
CREATE TABLE IF NOT EXISTS chunks (
    id             TEXT PRIMARY KEY,
    document_id    TEXT    NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index    INTEGER NOT NULL,
    page_start     INTEGER NOT NULL,
    page_end       INTEGER NOT NULL,
    char_start     INTEGER NOT NULL,
    char_end       INTEGER NOT NULL,
    token_estimate INTEGER NOT NULL DEFAULT 0,
    text           TEXT    NOT NULL,
    UNIQUE (document_id, chunk_index)
)
"""

CREATE_CHUNKS_DOC_INDEX = (
    "CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id, chunk_index)"
)

# Porter stemming means "cleaning" matches "clean"; unicode61 handles accents.
# chunk_id/document_id are stored but UNINDEXED -- retrievable and usable in a
# plain WHERE clause without polluting the full-text index.
CREATE_CHUNKS_FTS = """
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text,
    chunk_id    UNINDEXED,
    document_id UNINDEXED,
    tokenize = 'porter unicode61'
)
"""

CREATE_SESSIONS = """
CREATE TABLE IF NOT EXISTS sessions (
    id             TEXT PRIMARY KEY,
    owner          TEXT NOT NULL DEFAULT 'shared',
    title          TEXT,
    created_at     TEXT NOT NULL,
    last_active_at TEXT NOT NULL
)
"""

CREATE_SESSIONS_OWNER_INDEX = (
    "CREATE INDEX IF NOT EXISTS idx_sessions_owner "
    "ON sessions(owner, last_active_at DESC)"
)

CREATE_QA_HISTORY = """
CREATE TABLE IF NOT EXISTS qa_history (
    id              TEXT PRIMARY KEY,
    session_id      TEXT    NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    question        TEXT    NOT NULL,
    answer          TEXT    NOT NULL,
    document_ids    TEXT    NOT NULL DEFAULT '[]',
    sources         TEXT    NOT NULL DEFAULT '[]',
    grounded        INTEGER NOT NULL DEFAULT 1,
    llm_model       TEXT,
    embedding_model TEXT,
    latency_ms      INTEGER,
    created_at      TEXT    NOT NULL
)
"""

CREATE_QA_HISTORY_INDEX = """
CREATE INDEX IF NOT EXISTS idx_qa_history_session
    ON qa_history(session_id, created_at DESC)
"""

SCHEMA_STATEMENTS: tuple[str, ...] = (
    CREATE_DOCUMENTS,
    CREATE_DOCUMENTS_SHA_INDEX,
    CREATE_DOCUMENTS_STATUS_INDEX,
    CREATE_DOCUMENTS_OWNER_INDEX,
    CREATE_CHUNKS,
    CREATE_CHUNKS_DOC_INDEX,
    CREATE_CHUNKS_FTS,
    CREATE_SESSIONS,
    CREATE_SESSIONS_OWNER_INDEX,
    CREATE_QA_HISTORY,
    CREATE_QA_HISTORY_INDEX,
)

# --------------------------------------------------------------------------- #
# Migrations
# --------------------------------------------------------------------------- #
# CREATE TABLE IF NOT EXISTS is a no-op against a database that already has the
# table, so a new column never appears on an existing install. These run after
# the DDL above, guarded by PRAGMA table_info -- SQLite has no
# ADD COLUMN IF NOT EXISTS.
#
# Each entry is (table, column, ALTER statement). The NOT NULL DEFAULT does the
# backfill in one pass: every pre-existing row becomes owned by 'shared', which
# is exactly what single-user mode reads. Upgrading therefore cannot make an
# existing library disappear.
ADD_COLUMNS: tuple[tuple[str, str, str], ...] = (
    (
        "documents",
        "owner",
        "ALTER TABLE documents ADD COLUMN owner TEXT NOT NULL DEFAULT 'shared'",
    ),
    (
        "sessions",
        "owner",
        "ALTER TABLE sessions ADD COLUMN owner TEXT NOT NULL DEFAULT 'shared'",
    ),
)

TABLE_INFO = "PRAGMA table_info({table})"

# --------------------------------------------------------------------------- #
# Documents
# --------------------------------------------------------------------------- #
INSERT_DOCUMENT = """
INSERT INTO documents (
    id, owner, original_name, stored_path, content_type, extension,
    size_bytes, sha256, status, created_at, updated_at
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

UPDATE_DOCUMENT_STATUS = """
UPDATE documents
   SET status = ?, error = ?, updated_at = ?
 WHERE id = ?
"""

MARK_DOCUMENT_READY = """
UPDATE documents
   SET status = 'ready', error = NULL, page_count = ?, chunk_count = ?,
       embedding_model = ?, updated_at = ?
 WHERE id = ?
"""

# Owner-scoped, so an id guessed or copied from someone else's session reads as
# "not found" rather than returning their document. This is the single lookup
# every document route funnels through, which is why it is the one that matters.
SELECT_DOCUMENT = "SELECT * FROM documents WHERE id = ? AND owner = ?"

# Unscoped: only the ingestion worker uses this, and it operates on ids it pulled
# from the pending queue rather than from a request. Never reachable from a
# route -- see Database.get_document_internal.
SELECT_DOCUMENT_ANY_OWNER = "SELECT * FROM documents WHERE id = ?"

LIST_DOCUMENTS = """
SELECT * FROM documents
 WHERE owner = ?
 ORDER BY created_at DESC
 LIMIT ? OFFSET ?
"""

LIST_DOCUMENTS_BY_STATUS = """
SELECT * FROM documents
 WHERE owner = ? AND status = ?
 ORDER BY created_at DESC
 LIMIT ? OFFSET ?
"""

COUNT_DOCUMENTS = "SELECT COUNT(*) AS n FROM documents WHERE owner = ?"

# Must mirror LIST_DOCUMENTS_BY_STATUS' WHERE clause. A filtered list paired with
# an unfiltered total tells clients to page through results that aren't there.
COUNT_DOCUMENTS_BY_STATUS = (
    "SELECT COUNT(*) AS n FROM documents WHERE owner = ? AND status = ?"
)

# Owner-scoped for a security reason, not a tidiness one: unscoped, two visitors
# uploading the same file would have the second handed a reference to the first
# visitor's document, which is a direct cross-visitor leak through the dedup
# path. Each owner gets their own copy of identical content.
FIND_DUPLICATE = """
SELECT * FROM documents
 WHERE sha256 = ? AND owner = ? AND status = 'ready'
 ORDER BY created_at DESC
 LIMIT 1
"""

DELETE_DOCUMENT = "DELETE FROM documents WHERE id = ? AND owner = ?"

# Interrupted runs leave rows stuck mid-flight; reset them at startup so the
# worker can pick them up again instead of them hanging as 'processing' forever.
RESET_STALE_DOCUMENTS = """
UPDATE documents
   SET status = 'pending', updated_at = ?
 WHERE status = 'processing'
"""

SELECT_PENDING_DOCUMENTS = """
SELECT id FROM documents
 WHERE status = 'pending'
 ORDER BY created_at ASC
"""

# --------------------------------------------------------------------------- #
# Chunks
# --------------------------------------------------------------------------- #
INSERT_CHUNK = """
INSERT OR REPLACE INTO chunks (
    id, document_id, chunk_index, page_start, page_end,
    char_start, char_end, token_estimate, text
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

INSERT_CHUNK_FTS = """
INSERT INTO chunks_fts (text, chunk_id, document_id) VALUES (?, ?, ?)
"""

DELETE_CHUNKS_FOR_DOCUMENT = "DELETE FROM chunks WHERE document_id = ?"
DELETE_CHUNKS_FTS_FOR_DOCUMENT = "DELETE FROM chunks_fts WHERE document_id = ?"

# Owner-scoped as defence in depth. Retrieval is already scoped on both the
# vector and keyword sides, so a foreign chunk id should never reach here -- but
# this is the step that turns an id into quotable text, so it is the right place
# for a second check rather than a trusting one.
SELECT_CHUNKS_BY_IDS_TEMPLATE = """
SELECT c.*, d.original_name AS document_name
  FROM chunks c
  JOIN documents d ON d.id = c.document_id
 WHERE d.owner = ?
   AND c.id IN ({placeholders})
"""

SELECT_CHUNKS_FOR_DOCUMENT = """
SELECT c.*, d.original_name AS document_name
  FROM chunks c
  JOIN documents d ON d.id = c.document_id
 WHERE c.document_id = ?
   AND d.owner = ?
 ORDER BY c.chunk_index
 LIMIT ? OFFSET ?
"""

COUNT_CHUNKS = """
SELECT COUNT(*) AS n
  FROM chunks c
  JOIN documents d ON d.id = c.document_id
 WHERE d.owner = ?
"""

# bm25() returns a negative score where more-negative is more relevant, so it is
# negated to give the "higher is better" convention used everywhere else.
# Only 'ready' documents are searchable -- a half-ingested doc must not leak in.
#
# The owner predicate here is the one most easily forgotten and the most damaging
# to omit. Hybrid retrieval fuses this BM25 half with the vector half; scoping
# only the vector store would let another visitor's chunks arrive through keyword
# search and be quoted verbatim as a citation.
KEYWORD_SEARCH = """
SELECT f.chunk_id            AS chunk_id,
       f.document_id         AS document_id,
       -bm25(chunks_fts)     AS keyword_score,
       c.chunk_index         AS chunk_index,
       c.page_start          AS page_start,
       c.page_end            AS page_end,
       c.text                AS text,
       d.original_name       AS document_name
  FROM chunks_fts f
  JOIN chunks    c ON c.id = f.chunk_id
  JOIN documents d ON d.id = f.document_id
 WHERE chunks_fts MATCH ?
   AND d.status = 'ready'
   AND d.owner = ?
 ORDER BY bm25(chunks_fts)
 LIMIT ?
"""

KEYWORD_SEARCH_SCOPED_TEMPLATE = """
SELECT f.chunk_id            AS chunk_id,
       f.document_id         AS document_id,
       -bm25(chunks_fts)     AS keyword_score,
       c.chunk_index         AS chunk_index,
       c.page_start          AS page_start,
       c.page_end            AS page_end,
       c.text                AS text,
       d.original_name       AS document_name
  FROM chunks_fts f
  JOIN chunks    c ON c.id = f.chunk_id
  JOIN documents d ON d.id = f.document_id
 WHERE chunks_fts MATCH ?
   AND d.status = 'ready'
   AND d.owner = ?
   AND f.document_id IN ({placeholders})
 ORDER BY bm25(chunks_fts)
 LIMIT ?
"""

# --------------------------------------------------------------------------- #
# Sessions and history
# --------------------------------------------------------------------------- #
INSERT_SESSION = """
INSERT INTO sessions (id, owner, title, created_at, last_active_at)
VALUES (?, ?, ?, ?, ?)
"""

# Session ids are client-generated, so this is what stops one visitor from
# resuming another's conversation by pasting their id.
SELECT_SESSION = "SELECT * FROM sessions WHERE id = ? AND owner = ?"

TOUCH_SESSION = "UPDATE sessions SET last_active_at = ? WHERE id = ? AND owner = ?"

SET_SESSION_TITLE = (
    "UPDATE sessions SET title = ? WHERE id = ? AND owner = ? AND title IS NULL"
)

LIST_SESSIONS = """
SELECT s.*,
       (SELECT COUNT(*) FROM qa_history h WHERE h.session_id = s.id) AS exchange_count
  FROM sessions s
 WHERE s.owner = ?
 ORDER BY s.last_active_at DESC
 LIMIT ? OFFSET ?
"""

DELETE_SESSION = "DELETE FROM sessions WHERE id = ? AND owner = ?"

INSERT_QA = """
INSERT INTO qa_history (
    id, session_id, question, answer, document_ids, sources,
    grounded, llm_model, embedding_model, latency_ms, created_at
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

# qa_history carries no owner of its own -- it inherits one through its session,
# so every read joins rather than duplicating the column. One less place for the
# two copies to disagree.
LIST_HISTORY_BY_SESSION = """
SELECT h.* FROM qa_history h
  JOIN sessions s ON s.id = h.session_id
 WHERE h.session_id = ? AND s.owner = ?
 ORDER BY h.created_at DESC
 LIMIT ? OFFSET ?
"""

LIST_HISTORY = """
SELECT h.* FROM qa_history h
  JOIN sessions s ON s.id = h.session_id
 WHERE s.owner = ?
 ORDER BY h.created_at DESC
 LIMIT ? OFFSET ?
"""

# json_each is used so a document filter matches history rows whose scoped
# document_ids JSON array contains the id.
LIST_HISTORY_BY_DOCUMENT = """
SELECT h.* FROM qa_history h
  JOIN sessions s ON s.id = h.session_id
 WHERE s.owner = ?
   AND EXISTS (
        SELECT 1 FROM json_each(h.document_ids) je WHERE je.value = ?
       )
 ORDER BY h.created_at DESC
 LIMIT ? OFFSET ?
"""

COUNT_HISTORY_BY_SESSION = "SELECT COUNT(*) AS n FROM qa_history WHERE session_id = ?"
