"""Shared fixtures.

Every test runs against fake providers (``EMBEDDING_PROVIDER=fake``,
``LLM_PROVIDER=fake``) so the suite needs no Ollama, no network, and no model
downloads, and still exercises the real retrieval and citation code paths. The
fake embedder is deterministic hashed bag-of-words and the fake LLM is
extractive, which means retrieval quality assertions are meaningful: if the
right chunk isn't retrieved, the fake LLM can't cite it.

Each ``client`` gets its own SQLite file and Chroma directory, so tests are
order-independent. The app's lifespan is bypassed in favour of building
``Services`` explicitly -- httpx's ASGI transport doesn't run lifespan events,
and doing it by hand keeps setup visible.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import AsyncIterator

import httpx
import pytest
import pytest_asyncio

from app.config import Settings
from app.deps import Services
from app.main import create_app

# reportlab isn't a project dependency, so PDF fixtures are opt-in.
try:  # pragma: no cover - availability varies by environment
    from reportlab.lib.pagesizes import LETTER
    from reportlab.pdfgen import canvas as _canvas

    HAS_REPORTLAB = True
except ImportError:  # pragma: no cover
    HAS_REPORTLAB = False


SOLAR_PAGES = [
    # Distinct, checkable facts per page so citations can be asserted by page.
    "Rooftop photovoltaic systems convert sunlight into direct current "
    "electricity. A residential array in a temperate climate typically "
    "produces between 3 and 6 kilowatt-hours per installed kilowatt per day. "
    "Panel efficiency for commercially available monocrystalline modules sits "
    "between 19 and 22 percent as of this writing.",
    "Soiling losses are the single most underestimated factor in rooftop "
    "system performance. Dust, pollen and bird droppings reduce output by up "
    "to 15 percent in dry climates. Panels should be cleaned twice per year, "
    "and quarterly in arid or agricultural regions where airborne particulate "
    "settles faster.",
    "Inverter replacement is the largest mid-life maintenance cost. String "
    "inverters carry a typical design life of 12 years, while microinverters "
    "are warranted for 25 years and are usually replaced only on failure. "
    "Budget roughly 10 percent of the original system cost for a single "
    "inverter replacement.",
]


def make_settings(root: Path, **overrides) -> Settings:
    """Settings pointed entirely at a temp directory.

    ``_env_file=None`` plus explicit kwargs: init arguments outrank environment
    variables in pydantic-settings, so a developer's real ``.env`` or exported
    ``OLLAMA_*`` vars can't bleed into the suite.
    """
    defaults = dict(
        app_env="test",
        log_level="WARNING",
        embedding_provider="fake",
        llm_provider="fake",
        embedding_dim=64,
        data_dir=root / "data",
        upload_dir=root / "data" / "uploads",
        chroma_dir=root / "data" / "chroma",
        sqlite_path=root / "data" / "metadata.db",
        chunk_size=300,
        chunk_overlap=60,
        min_chunk_chars=40,
        top_k=3,
        candidate_k=10,
        ingest_workers=1,
        api_key=None,
    )
    defaults.update(overrides)
    return Settings(_env_file=None, **defaults)


@pytest_asyncio.fixture
async def client(tmp_path: Path) -> AsyncIterator[httpx.AsyncClient]:
    """An HTTP client wired to an isolated app instance.

    ``ac.services`` exposes the service container, for the occasional assertion
    that has to look past the HTTP layer (vector counts, queue depth) or seed
    state directly instead of going through six requests to arrange it.
    """
    settings = make_settings(tmp_path)
    app = create_app(settings)
    container = Services.build(settings)
    app.state.services = container
    await container.startup()

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        ac.app = app  # handy for tests that need the container directly
        ac.services = container
        try:
            yield ac
        finally:
            await container.shutdown()


@pytest_asyncio.fixture
async def secured_client(tmp_path: Path) -> AsyncIterator[httpx.AsyncClient]:
    """Same app, with API-key auth switched on."""
    settings = make_settings(tmp_path, api_key="test-secret")
    app = create_app(settings)
    container = Services.build(settings)
    app.state.services = container
    await container.startup()

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        try:
            yield ac
        finally:
            await container.shutdown()


@pytest_asyncio.fixture
async def tiny_limit_client(tmp_path: Path) -> AsyncIterator[httpx.AsyncClient]:
    """Same app with a 1 MB upload cap.

    The limit is enforced mid-stream, so testing it against the inherited 50 MB
    default would mean generating and streaming 50 MB per test. A small cap
    exercises exactly the same branch.
    """
    settings = make_settings(tmp_path, max_upload_mb=1)
    app = create_app(settings)
    container = Services.build(settings)
    app.state.services = container
    await container.startup()

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        ac.services = container
        try:
            yield ac
        finally:
            await container.shutdown()


# --------------------------------------------------------------------------- #
# Sample files
# --------------------------------------------------------------------------- #
@pytest.fixture
def solar_txt(tmp_path: Path) -> Path:
    path = tmp_path / "solar-notes.txt"
    path.write_text("\n\n".join(SOLAR_PAGES), encoding="utf-8")
    return path


@pytest.fixture
def solar_pdf(tmp_path: Path) -> Path:
    if not HAS_REPORTLAB:  # pragma: no cover
        pytest.skip("reportlab not installed; PDF fixture unavailable")
    path = tmp_path / "solar.pdf"
    pdf = _canvas.Canvas(str(path), pagesize=LETTER)
    for body in SOLAR_PAGES:
        y = 720
        # Wrap by hand: reportlab's canvas has no paragraph flow.
        words, line = body.split(), ""
        for word in words:
            if len(line) + len(word) + 1 > 78:
                pdf.drawString(72, y, line)
                y -= 16
                line = word
            else:
                line = f"{line} {word}".strip()
        pdf.drawString(72, y, line)
        pdf.showPage()
    pdf.save()
    return path


@pytest.fixture
def handbook_docx(tmp_path: Path) -> Path:
    docx = pytest.importorskip("docx")
    path = tmp_path / "handbook.docx"
    doc = docx.Document()
    doc.add_heading("Field Handbook", 0)
    doc.add_heading("Safety", 1)
    doc.add_paragraph(
        "Always isolate the array at the DC disconnect before servicing. "
        "Rooftop work requires a harness anchored to a certified point."
    )
    doc.add_heading("Torque values", 1)
    table = doc.add_table(rows=3, cols=2)
    rows = [("Fastener", "Torque"), ("Rail splice bolt", "22 Nm"), ("Module clamp", "16 Nm")]
    for row, (a, b) in zip(table.rows, rows):
        row.cells[0].text = a
        row.cells[1].text = b
    doc.save(path)
    return path


@pytest.fixture
def empty_txt(tmp_path: Path) -> Path:
    path = tmp_path / "blank.txt"
    path.write_text("   \n\n  \t\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
async def upload(client: httpx.AsyncClient, path: Path, **kwargs) -> dict:
    """POST a file to /documents/upload and return the JSON body."""
    with path.open("rb") as handle:
        res = await client.post(
            "/documents/upload",
            files={"file": (path.name, handle, "application/octet-stream")},
            **kwargs,
        )
    return {"status": res.status_code, "body": res.json()}


async def wait_ready(client: httpx.AsyncClient, doc_id: str, timeout: float = 20.0) -> dict:
    """Poll until ingestion settles. Returns the final document body."""
    # The running loop's clock rather than get_event_loop(), which is deprecated
    # inside a coroutine, and rather than time.monotonic(), so a test that ever
    # runs on a virtual-time loop still sees a consistent clock.
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        res = await client.get(f"/documents/{doc_id}")
        body = res.json()
        if body["status"] in ("ready", "failed"):
            return body
        await asyncio.sleep(0.05)
    raise AssertionError(f"document {doc_id} never settled within {timeout}s")


async def ingest(client: httpx.AsyncClient, path: Path) -> dict:
    """Upload and block until the document is queryable."""
    result = await upload(client, path)
    assert result["status"] == 202, result["body"]
    doc = await wait_ready(client, result["body"]["document"]["id"])
    assert doc["status"] == "ready", doc.get("error")
    return doc
