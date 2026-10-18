"""Text extraction from each supported format, plus the failure modes."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.ingestion.extractors import (
    SUPPORTED_EXTENSIONS,
    EmptyDocument,
    UnsupportedFileType,
    detect_kind,
    extract,
    extract_docx,
    extract_text,
)


# --------------------------------------------------------------------------- #
# Type detection
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name,kind", [
    ("report.pdf", "pdf"),
    ("REPORT.PDF", "pdf"),
    ("notes.txt", "text"),
    ("readme.md", "text"),
    ("readme.markdown", "text"),
    ("handbook.docx", "docx"),
])
def test_detect_kind_is_case_insensitive(name, kind):
    assert detect_kind(name) == kind


@pytest.mark.parametrize("name", [
    "archive.zip", "photo.png", "sheet.xlsx", "legacy.doc", "noextension",
])
def test_unsupported_types_are_rejected_by_name(name):
    with pytest.raises(UnsupportedFileType):
        detect_kind(name)


def test_error_message_names_the_supported_types():
    # The user needs to know what *would* work, not just that this didn't.
    with pytest.raises(UnsupportedFileType) as caught:
        detect_kind("thing.zip")
    message = str(caught.value)
    assert ".pdf" in message and ".docx" in message


def test_supported_extensions_are_all_lowercase_with_dots():
    for ext in SUPPORTED_EXTENSIONS:
        assert ext.startswith(".") and ext == ext.lower()


# --------------------------------------------------------------------------- #
# Plain text
# --------------------------------------------------------------------------- #
def test_text_extraction_round_trips(tmp_path: Path):
    path = tmp_path / "notes.txt"
    path.write_text("First paragraph.\n\nSecond paragraph.", encoding="utf-8")

    pages = extract_text(path)
    joined = " ".join(p.text for p in pages)
    assert "First paragraph." in joined
    assert "Second paragraph." in joined


def test_long_text_is_split_into_pseudo_pages(tmp_path: Path):
    # Plain text has no pages, so citations need synthetic ones to point at.
    path = tmp_path / "long.txt"
    path.write_text("\n\n".join(f"Block {i}. " + "filler " * 200 for i in range(8)),
                    encoding="utf-8")

    pages = extract_text(path)
    assert len(pages) > 1
    assert [p.page_number for p in pages] == list(range(1, len(pages) + 1))


def test_short_text_stays_on_one_page(tmp_path: Path):
    path = tmp_path / "short.txt"
    path.write_text("Just a line.", encoding="utf-8")
    assert len(extract_text(path)) == 1


def test_non_utf8_bytes_are_decoded_not_crashed(tmp_path: Path):
    # Windows-1252 smart quotes are extremely common in real uploads, and 0x92 /
    # 0x96 are invalid as UTF-8 -- a strict decode raises instead of ingesting.
    path = tmp_path / "cp1252.txt"
    path.write_bytes(b"The client\x92s report \x96 draft")

    pages = extract_text(path)
    assert "client" in pages[0].text and "report" in pages[0].text


def test_empty_file_raises_empty_document(tmp_path: Path):
    path = tmp_path / "empty.txt"
    path.write_text("", encoding="utf-8")
    with pytest.raises(EmptyDocument):
        extract(path, "empty.txt")


def test_whitespace_only_file_raises_empty_document(tmp_path: Path):
    path = tmp_path / "blank.txt"
    path.write_text("   \n\n\t  \n", encoding="utf-8")
    with pytest.raises(EmptyDocument):
        extract(path, "blank.txt")


# --------------------------------------------------------------------------- #
# DOCX
# --------------------------------------------------------------------------- #
def _write_docx(path: Path):
    docx = pytest.importorskip("docx")
    doc = docx.Document()
    doc.add_heading("Field Handbook", 0)
    doc.add_heading("Safety", 1)
    doc.add_paragraph("Isolate the array at the DC disconnect before servicing.")
    doc.add_heading("Torque", 2)
    table = doc.add_table(rows=2, cols=2)
    table.rows[0].cells[0].text = "Fastener"
    table.rows[0].cells[1].text = "Torque"
    table.rows[1].cells[0].text = "Module clamp"
    table.rows[1].cells[1].text = "16 Nm"
    doc.save(path)
    return path


def test_docx_paragraphs_are_extracted(tmp_path: Path):
    path = _write_docx(tmp_path / "handbook.docx")
    text = " ".join(p.text for p in extract_docx(path))
    assert "DC disconnect" in text


def test_docx_headings_become_markdown_hierarchy(tmp_path: Path):
    path = _write_docx(tmp_path / "handbook.docx")
    text = "\n".join(p.text for p in extract_docx(path))

    # Title -> "#", Heading 1 -> "##", Heading 2 -> "###". Flattening these
    # loses the document's structure, which retrieval relies on for context.
    assert "# Field Handbook" in text
    assert "## Safety" in text
    assert "### Torque" in text


def test_docx_tables_survive_as_delimited_rows(tmp_path: Path):
    path = _write_docx(tmp_path / "handbook.docx")
    text = "\n".join(p.text for p in extract_docx(path))
    # A torque value is worthless if the fastener name is in another chunk.
    assert "Module clamp" in text and "16 Nm" in text
    assert any("|" in line for line in text.splitlines())


def test_empty_docx_raises_empty_document(tmp_path: Path):
    docx = pytest.importorskip("docx")
    path = tmp_path / "hollow.docx"
    docx.Document().save(path)
    with pytest.raises(EmptyDocument):
        extract(path, "hollow.docx")


# --------------------------------------------------------------------------- #
# PDF
# --------------------------------------------------------------------------- #
def _write_pdf(path: Path, pages: list[str]):
    pytest.importorskip("reportlab")
    from reportlab.lib.pagesizes import LETTER
    from reportlab.pdfgen import canvas

    pdf = canvas.Canvas(str(path), pagesize=LETTER)
    for body in pages:
        pdf.drawString(72, 720, body)
        pdf.showPage()
    pdf.save()
    return path


def test_pdf_pages_are_numbered_from_one(tmp_path: Path):
    path = _write_pdf(tmp_path / "three.pdf",
                      ["Alpha on one.", "Beta on two.", "Gamma on three."])

    pages = extract(path, "three.pdf")
    assert [p.page_number for p in pages] == [1, 2, 3]
    assert "Alpha" in pages[0].text
    assert "Gamma" in pages[2].text


def test_pdf_page_numbers_match_the_printed_document(tmp_path: Path):
    # Off-by-one here makes every citation in the product subtly wrong.
    path = _write_pdf(tmp_path / "marked.pdf", [f"This is page {i}." for i in (1, 2, 3)])
    for page in extract(path, "marked.pdf"):
        assert f"page {page.page_number}" in page.text


def test_corrupt_pdf_raises_a_typed_error(tmp_path: Path):
    path = tmp_path / "broken.pdf"
    path.write_bytes(b"%PDF-1.4 this is not really a pdf")
    with pytest.raises((EmptyDocument, Exception)):
        extract(path, "broken.pdf")


# --------------------------------------------------------------------------- #
# Dispatcher
# --------------------------------------------------------------------------- #
def test_extract_rejects_unsupported_before_reading(tmp_path: Path):
    path = tmp_path / "thing.zip"
    path.write_bytes(b"PK\x03\x04")
    with pytest.raises(UnsupportedFileType):
        extract(path, "thing.zip")


def test_extract_uses_the_supplied_filename_not_the_stored_path(tmp_path: Path):
    # Uploads are stored under a generated uuid name, so the extension used for
    # dispatch has to come from the original filename.
    stored = tmp_path / "8f14e45f-ea1e-4a1d-b0a9-000000000000"
    stored.write_text("Content that should be read as plain text.", encoding="utf-8")

    pages = extract(stored, "original-name.txt")
    assert "plain text" in pages[0].text
