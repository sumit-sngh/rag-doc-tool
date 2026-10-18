"""Text extraction for PDF, DOCX, TXT/MD.

Only depends on pypdf / python-docx / charset-normalizer -- no web framework, no
config object -- so extraction can be tested and reused standalone.

Each extractor returns ``list[ExtractedPage]``. PDFs map 1:1 to real pages. DOCX
and TXT have no intrinsic pagination, so they are split into synthetic "pages"
of roughly ``PSEUDO_PAGE_CHARS`` characters at paragraph boundaries. That keeps
citations usefully granular ("p. 3 of 7") instead of always saying "p. 1".
"""

from __future__ import annotations

import logging
from pathlib import Path

from app.ingestion.chunker import ExtractedPage

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS: dict[str, str] = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".txt": "text",
    ".md": "text",
    ".markdown": "text",
}

# Roughly one screen of prose; used to paginate formats without real pages.
PSEUDO_PAGE_CHARS = 3000


class ExtractionError(Exception):
    """Raised when a file cannot be parsed into usable text."""


class UnsupportedFileType(ExtractionError):
    """Raised for extensions outside SUPPORTED_EXTENSIONS."""


class EmptyDocument(ExtractionError):
    """Raised when a file parses cleanly but yields no usable text."""


def detect_kind(filename: str) -> str:
    """Map a filename to an extractor kind, or raise UnsupportedFileType."""
    ext = Path(filename).suffix.lower()
    kind = SUPPORTED_EXTENSIONS.get(ext)
    if kind is None:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise UnsupportedFileType(
            f"Unsupported file type '{ext or filename}'. Supported: {supported}"
        )
    return kind


def _paginate(blocks: list[str], target_chars: int = PSEUDO_PAGE_CHARS) -> list[ExtractedPage]:
    """Group paragraph blocks into synthetic pages of ~target_chars."""
    pages: list[ExtractedPage] = []
    buffer: list[str] = []
    size = 0

    for block in blocks:
        buffer.append(block)
        size += len(block) + 2
        if size >= target_chars:
            pages.append(ExtractedPage(len(pages) + 1, "\n\n".join(buffer)))
            buffer, size = [], 0

    if buffer:
        pages.append(ExtractedPage(len(pages) + 1, "\n\n".join(buffer)))
    return pages


# --------------------------------------------------------------------------- #
# PDF
# --------------------------------------------------------------------------- #
def extract_pdf(
    path: Path,
    ocr_enabled: bool = False,
    ocr_min_chars: int = 100,
    ocr_language: str = "eng",
    ocr_dpi: int = 200,
) -> list[ExtractedPage]:
    """Extract per-page text from a PDF, optionally OCR-ing image-only pages."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover
        raise ExtractionError("pypdf is not installed") from exc

    try:
        reader = PdfReader(str(path))
    except Exception as exc:
        raise ExtractionError(f"Could not open PDF: {exc}") from exc

    if reader.is_encrypted:
        # Many PDFs are "encrypted" with an empty owner password; try that first.
        try:
            reader.decrypt("")
        except Exception as exc:
            raise ExtractionError(
                "PDF is password-protected and cannot be read"
            ) from exc

    pages: list[ExtractedPage] = []
    needs_ocr: list[int] = []

    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception as exc:
            # One malformed page shouldn't sink the whole document.
            logger.warning("PDF page %d extraction failed: %s", i + 1, exc)
            text = ""

        if ocr_enabled and len(text.strip()) < ocr_min_chars:
            needs_ocr.append(i)
        pages.append(ExtractedPage(i + 1, text))

    if needs_ocr:
        logger.info("OCR-ing %d low-text page(s) in %s", len(needs_ocr), path.name)
        ocr_results = _ocr_pdf_pages(path, needs_ocr, ocr_language, ocr_dpi)
        for idx, ocr_text in ocr_results.items():
            # Keep whichever source produced more text.
            if len(ocr_text.strip()) > len(pages[idx].text.strip()):
                pages[idx] = ExtractedPage(idx + 1, ocr_text)

    if not any(p.text.strip() for p in pages):
        raise EmptyDocument(
            "No extractable text found. If this is a scanned PDF, enable OCR "
            "by setting OCR_ENABLED=true and installing requirements-optional.txt."
        )
    return pages


def _ocr_pdf_pages(
    path: Path, page_indices: list[int], language: str, dpi: int
) -> dict[int, str]:
    """OCR specific pages. Returns {page_index: text}; empty dict if unavailable."""
    try:
        import pytesseract
        from pdf2image import convert_from_path
    except ImportError:
        logger.warning(
            "OCR requested but pytesseract/pdf2image missing; "
            "install requirements-optional.txt. Skipping OCR."
        )
        return {}

    results: dict[int, str] = {}
    for idx in page_indices:
        try:
            # pdf2image is 1-indexed and inclusive on both ends.
            images = convert_from_path(
                str(path), dpi=dpi, first_page=idx + 1, last_page=idx + 1
            )
            if images:
                results[idx] = pytesseract.image_to_string(images[0], lang=language)
        except Exception as exc:
            logger.warning("OCR failed on page %d: %s", idx + 1, exc)
    return results


# --------------------------------------------------------------------------- #
# DOCX
# --------------------------------------------------------------------------- #
def extract_docx(path: Path) -> list[ExtractedPage]:
    """Extract paragraphs and tables from a DOCX into synthetic pages."""
    try:
        import docx
    except ImportError as exc:  # pragma: no cover
        raise ExtractionError("python-docx is not installed") from exc

    try:
        document = docx.Document(str(path))
    except Exception as exc:
        raise ExtractionError(f"Could not open DOCX: {exc}") from exc

    blocks: list[str] = []

    for para in document.paragraphs:
        text = (para.text or "").strip()
        if not text:
            continue
        style = (para.style.name or "").lower() if para.style is not None else ""
        # Markdown-ish heading markers survive chunking and give the LLM
        # structural cues about where it is in the document.
        if style.startswith("heading"):
            # "Heading 1" -> "##", so it nests under a Title-styled "#".
            digits = "".join(c for c in style if c.isdigit())
            level = int(digits) + 1 if digits else 2
            blocks.append(f"{'#' * min(level, 6)} {text}")
        elif style.startswith("title"):
            blocks.append(f"# {text}")
        else:
            blocks.append(text)

    # Tables carry a lot of the answerable facts in real documents; flatten them
    # to pipe-delimited rows so they stay searchable.
    for table in document.tables:
        rows: list[str] = []
        for row in table.rows:
            cells = [(c.text or "").strip().replace("\n", " ") for c in row.cells]
            if any(cells):
                rows.append(" | ".join(cells))
        if rows:
            blocks.append("\n".join(rows))

    if not blocks:
        raise EmptyDocument("DOCX contains no readable text")
    return _paginate(blocks)


# --------------------------------------------------------------------------- #
# Plain text / Markdown
# --------------------------------------------------------------------------- #
def extract_text(path: Path) -> list[ExtractedPage]:
    """Read a text file, detecting encoding rather than assuming UTF-8."""
    raw = path.read_bytes()
    if not raw.strip():
        raise EmptyDocument("File is empty")

    content: str | None = None
    try:
        content = raw.decode("utf-8")
    except UnicodeDecodeError:
        try:
            from charset_normalizer import from_bytes

            best = from_bytes(raw).best()
            if best is not None:
                content = str(best)
        except ImportError:
            pass
        if content is None:
            # Last resort: never fail an upload over a stray byte.
            content = raw.decode("utf-8", errors="replace")

    if not content.strip():
        raise EmptyDocument("File contains no readable text")

    blocks = [b.strip() for b in content.split("\n\n") if b.strip()]
    if not blocks:
        blocks = [content.strip()]
    return _paginate(blocks)


# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #
def extract(
    path: Path,
    filename: str | None = None,
    *,
    ocr_enabled: bool = False,
    ocr_min_chars: int = 100,
    ocr_language: str = "eng",
    ocr_dpi: int = 200,
) -> list[ExtractedPage]:
    """Extract text from a document, dispatching on the original filename.

    `filename` is passed separately because uploads are stored under generated
    names; the user's original extension is what tells us the format.
    """
    kind = detect_kind(filename or path.name)

    if kind == "pdf":
        return extract_pdf(
            path,
            ocr_enabled=ocr_enabled,
            ocr_min_chars=ocr_min_chars,
            ocr_language=ocr_language,
            ocr_dpi=ocr_dpi,
        )
    if kind == "docx":
        return extract_docx(path)
    return extract_text(path)
