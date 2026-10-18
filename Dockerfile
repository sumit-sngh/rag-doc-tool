# Document Q&A Tool -- container image.
#
# Only the API runs here. Ollama stays on the host (or in its own container, see
# docker-compose.yml): the models are multi-gigabyte and want GPU access, so
# baking them into this image would make it enormous and non-portable.

FROM python:3.12-slim AS base

# PYTHONUNBUFFERED so logs appear immediately rather than on buffer flush --
# without it, `docker logs` looks frozen during a slow ingestion.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# --------------------------------------------------------------------------- #
# Dependencies
# --------------------------------------------------------------------------- #
# Copied and installed before the source so edits to app code don't invalidate
# the (slow) dependency layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# For OCR of scanned PDFs, uncomment both lines below and set OCR_ENABLED=true.
# Left out by default because tesseract and poppler add ~150 MB to the image.
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     tesseract-ocr poppler-utils && rm -rf /var/lib/apt/lists/*
# COPY requirements-optional.txt .
# RUN pip install --no-cache-dir pytesseract pdf2image Pillow

# --------------------------------------------------------------------------- #
# Application
# --------------------------------------------------------------------------- #
COPY app/ ./app/
COPY frontend/ ./frontend/

# Runtime state lives here. Mount a volume over it or every restart starts from
# an empty library -- uploads, the Chroma index, and the SQLite metadata are all
# under this path.
RUN mkdir -p /app/data/uploads /app/data/chroma

# Non-root: nothing here needs privileges, and a container escape should not
# land on a root shell.
RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app/data
USER appuser

# Inside a container, localhost is the container itself. Reach the host's Ollama
# via host.docker.internal (Docker Desktop) or set OLLAMA_BASE_URL explicitly.
ENV OLLAMA_BASE_URL=http://host.docker.internal:11434 \
    DATA_DIR=/app/data \
    UPLOAD_DIR=/app/data/uploads \
    CHROMA_DIR=/app/data/chroma \
    SQLITE_PATH=/app/data/metadata.db

EXPOSE 8000

# /health reports degraded (not unhealthy) when Ollama is unreachable, so probe
# for a served response rather than for status=="ok" -- otherwise the container
# is marked unhealthy for a dependency it does not own.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

# No --reload and a single worker: ingestion state lives in an in-process asyncio
# queue, so a second worker process would have its own queue and its own view of
# what is pending. Scaling out means moving that queue to Redis first.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
