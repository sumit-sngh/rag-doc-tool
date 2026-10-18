#!/usr/bin/env python3
"""Preflight check -- run this before the first `uvicorn` to localise failures.

Starting the app is a poor diagnostic: a missing dependency, a stopped Ollama and
an unpulled model all surface as a wall of traceback or a 502 at query time, long
after the thing that was actually wrong. This checks each layer in order and says
which one is broken.

Uses only the standard library for the checks that must work *before* the
dependencies are installed, so it never fails for the reason you are debugging.

    python3 scripts/preflight.py

Exit code is 0 when the app should start, 1 when something blocking is wrong.
"""

from __future__ import annotations

import json
import os
import socket
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"
if not sys.stdout.isatty() or os.environ.get("NO_COLOR"):
    GREEN = RED = YELLOW = DIM = RESET = ""

blocking: list[str] = []
warnings: list[str] = []


def ok(msg: str, detail: str = "") -> None:
    print(f"  {GREEN}✓{RESET} {msg}" + (f" {DIM}{detail}{RESET}" if detail else ""))


def bad(msg: str, fix: str) -> None:
    print(f"  {RED}✗{RESET} {msg}")
    print(f"      {DIM}fix:{RESET} {fix}")
    blocking.append(msg)


def warn(msg: str, fix: str) -> None:
    print(f"  {YELLOW}!{RESET} {msg}")
    print(f"      {DIM}{fix}{RESET}")
    warnings.append(msg)


def section(title: str) -> None:
    print(f"\n{title}")


# --------------------------------------------------------------------------- #
def read_env_file() -> dict[str, str]:
    """Parse .env well enough to report effective config. Not a full parser."""
    values: dict[str, str] = {}
    path = ROOT / ".env"
    if not path.exists():
        return values
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if "=" in line and not line.startswith("export "):
            key, _, val = line.partition("=")
            values[key.strip()] = val.strip().strip("'\"")
    return values


def setting(name: str, default: str, env_file: dict[str, str]) -> str:
    """Environment wins over .env, matching pydantic-settings' precedence."""
    return os.environ.get(name) or env_file.get(name) or default


env_file = read_env_file()

emb_provider = setting("EMBEDDING_PROVIDER", "ollama", env_file).lower()
llm_provider = setting("LLM_PROVIDER", "ollama", env_file).lower()
base_url = setting("OLLAMA_BASE_URL", "http://localhost:11434", env_file).rstrip("/")
emb_model = setting("EMBEDDING_MODEL", "nomic-embed-text", env_file)
llm_model = setting("LLM_MODEL", "llama3.1", env_file)

print("Document Q&A Tool -- preflight")
print(f"{DIM}project: {ROOT}{RESET}")

# --------------------------------------------------------------------------- #
section("Python")
# --------------------------------------------------------------------------- #
major, minor = sys.version_info[:2]
# Upper bound as well as lower. requirements.txt is exactly pinned, and several
# of those packages ship compiled wheels per CPython version (pydantic-core and
# chromadb's dependencies are Rust/C). On an interpreter newer than the pins,
# pip finds no matching wheel, falls back to building from source, and dies on a
# missing Rust toolchain -- an error that says "Failed building wheel for
# pydantic-core" and gives no hint that the Python version is the cause.
TESTED_MAX = (3, 12)
if (major, minor) < (3, 10):
    bad(
        f"Python {major}.{minor} is too old (need 3.10+)",
        "the code uses `X | None` annotations, which pydantic evaluates at "
        "runtime. Use python3.10 or newer; the Docker image uses 3.12.",
    )
elif (major, minor) > TESTED_MAX:
    warn(
        f"Python {major}.{minor} is newer than the pinned dependencies "
        f"(tested on 3.10-{TESTED_MAX[0]}.{TESTED_MAX[1]})",
        "if `pip install -r requirements.txt` failed to build a wheel "
        "(pydantic-core is the usual one), this is why. Easiest fix is a "
        f"{TESTED_MAX[0]}.{TESTED_MAX[1]} interpreter:\n"
        f"        python{TESTED_MAX[0]}.{TESTED_MAX[1]} -m venv .venv "
        "&& source .venv/bin/activate && pip install -r requirements.txt\n"
        "      Or use Docker, which pins 3.12 for you: make up",
    )
else:
    ok(f"Python {major}.{minor}", sys.executable)

if sys.prefix == sys.base_prefix:
    warn(
        "not running inside a virtualenv",
        "fine, but `python3 -m venv .venv && source .venv/bin/activate` keeps "
        "this project's pins off your system Python.",
    )
else:
    ok("virtualenv active", sys.prefix)

# --------------------------------------------------------------------------- #
section("Dependencies")
# --------------------------------------------------------------------------- #
REQUIRED = [
    ("fastapi", "FastAPI"),
    ("uvicorn", "ASGI server"),
    ("pydantic", "validation"),
    ("pydantic_settings", "env config"),
    ("multipart", "file uploads"),
    ("httpx", "Ollama client"),
    ("aiosqlite", "metadata store"),
    ("chromadb", "vector store"),
    ("pypdf", "PDF extraction"),
    ("docx", "DOCX extraction"),
    ("charset_normalizer", "encoding detection"),
    ("anyio", "thread offloading"),
    ("tenacity", "provider retries"),
]
missing = []
for module, purpose in REQUIRED:
    try:
        __import__(module)
    except ImportError:
        missing.append((module, purpose))

if missing:
    for module, purpose in missing:
        print(f"  {RED}✗{RESET} {module} {DIM}({purpose}){RESET}")
    bad(
        f"{len(missing)} required package(s) missing",
        "pip install -r requirements.txt",
    )
else:
    ok(f"all {len(REQUIRED)} required packages importable")

OPTIONAL = [
    ("pytest", "test suite", "pip install -r requirements.txt"),
    ("reportlab", "PDF test fixtures", "pip install -r requirements-optional.txt"),
    ("pytesseract", "OCR for scanned PDFs", "pip install -r requirements-optional.txt"),
    ("sentence_transformers", "cross-encoder reranking", "pip install -r requirements-optional.txt"),
]
for module, purpose, fix in OPTIONAL:
    try:
        __import__(module)
        ok(f"{module} {DIM}({purpose}){RESET}")
    except ImportError:
        print(f"  {DIM}- {module} not installed ({purpose}) -- {fix}{RESET}")

# --------------------------------------------------------------------------- #
section("Configuration")
# --------------------------------------------------------------------------- #
if (ROOT / ".env").exists():
    ok(".env found", f"{len(env_file)} value(s) set")
else:
    print(f"  {DIM}- no .env; using defaults (cp .env.example .env to change them){RESET}")

ok(f"embeddings: {emb_provider}:{emb_model}")
ok(f"llm: {llm_provider}:{llm_model}")

using_fake = emb_provider == "fake" and llm_provider == "fake"
if using_fake:
    print(
        f"  {DIM}- fake providers: fully offline, no Ollama needed. Retrieval and "
        f"citations are real; prose quality is not.{RESET}"
    )

if (ROOT / ".env").exists():
    try:
        chunk_size = int(setting("CHUNK_SIZE", "1200", env_file))
        overlap = int(setting("CHUNK_OVERLAP", "200", env_file))
        if overlap >= chunk_size:
            bad(
                f"CHUNK_OVERLAP ({overlap}) >= CHUNK_SIZE ({chunk_size})",
                "the app refuses to start: overlap that large makes chunking "
                "non-terminating. Lower CHUNK_OVERLAP.",
            )
        top_k = int(setting("TOP_K", "5", env_file))
        candidate_k = int(setting("CANDIDATE_K", "20", env_file))
        if candidate_k < top_k:
            bad(
                f"CANDIDATE_K ({candidate_k}) < TOP_K ({top_k})",
                "you cannot rerank down to more chunks than you retrieved. "
                "Raise CANDIDATE_K.",
            )
    except ValueError as exc:
        warn(f"could not parse a numeric setting: {exc}", "check .env for typos")

# --------------------------------------------------------------------------- #
section("Storage")
# --------------------------------------------------------------------------- #
data_dir = Path(setting("DATA_DIR", str(ROOT / "data"), env_file))
if not data_dir.is_absolute():
    data_dir = (ROOT / data_dir).resolve()
probe = data_dir / ".preflight"
try:
    (data_dir / "uploads").mkdir(parents=True, exist_ok=True)
    probe.write_text("ok")
    ok("data directory writable", str(data_dir))
except OSError as exc:
    bad(f"cannot write to {data_dir}: {exc}", "check permissions or set DATA_DIR")
finally:
    # Cleanup is not part of the check. On a filesystem that permits create but
    # not unlink, failing here would report the directory as unwritable when the
    # write plainly succeeded -- and would leave this file behind either way.
    try:
        probe.unlink(missing_ok=True)
    except OSError:
        warn(
            f"could not remove the probe file {probe}",
            "harmless, but the directory allows creating files and not deleting "
            "them, which will bite when you delete a document. Delete it by hand.",
        )

chroma_dir = data_dir / "chroma"
db_path = Path(setting("SQLITE_PATH", str(data_dir / "metadata.db"), env_file))
existing = []
if chroma_dir.exists() and any(chroma_dir.iterdir()):
    existing.append("vector index")
if db_path.exists():
    existing.append("metadata db")
if existing:
    ok(f"existing data found ({', '.join(existing)})", "documents are already ingested")
else:
    print(f"  {DIM}- no data yet; the library starts empty{RESET}")

# FTS5 is compiled into most builds but not all, and its absence only shows up
# as a failed CREATE VIRTUAL TABLE at init.
try:
    import sqlite3
    con = sqlite3.connect(":memory:")
    con.execute("CREATE VIRTUAL TABLE t USING fts5(x)")
    con.close()
    ok("SQLite FTS5 available", "hybrid keyword search will work")
except Exception:
    warn(
        "SQLite was built without FTS5",
        "set HYBRID_SEARCH=false; vector-only search still works, but exact "
        "identifiers (part numbers, section refs) will retrieve worse.",
    )

# --------------------------------------------------------------------------- #
section("Ollama")
# --------------------------------------------------------------------------- #
if using_fake:
    print(f"  {DIM}- skipped: both providers are 'fake'{RESET}")
else:
    tags = None
    try:
        with urllib.request.urlopen(f"{base_url}/api/tags", timeout=5) as res:
            tags = json.loads(res.read())
        ok("Ollama reachable", base_url)
    except urllib.error.URLError as exc:
        bad(
            f"cannot reach Ollama at {base_url} ({exc.reason})",
            "start it with `ollama serve` (or the desktop app). Or skip models "
            "entirely -- `make run-fake` runs offline, and `make check-fake` "
            "checks everything else without this test.",
        )
    except Exception as exc:
        bad(f"unexpected response from {base_url}: {exc}", "is that really Ollama?")

    if tags is not None:
        installed = {m.get("name", "") for m in tags.get("models", [])}
        # Ollama reports "llama3.1:latest"; a bare "llama3.1" is the same model.
        def present(want: str) -> bool:
            return want in installed or f"{want}:latest" in installed

        for label, model in (("embedding", emb_model), ("llm", llm_model)):
            if present(model):
                ok(f"{label} model '{model}' pulled")
            else:
                bad(
                    f"{label} model '{model}' is not pulled",
                    f"ollama pull {model}",
                )
        if installed:
            print(f"  {DIM}  installed: {', '.join(sorted(installed))}{RESET}")

# --------------------------------------------------------------------------- #
section("Port")
# --------------------------------------------------------------------------- #
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.settimeout(1)
    if sock.connect_ex(("127.0.0.1", 8000)) == 0:
        warn(
            "something is already listening on :8000",
            "either the app is already running, or use --port 8001",
        )
    else:
        ok("port 8000 free")

# --------------------------------------------------------------------------- #
section("App import")
# --------------------------------------------------------------------------- #
if missing:
    print(f"  {DIM}- skipped: install dependencies first{RESET}")
else:
    sys.path.insert(0, str(ROOT))
    try:
        from app.config import get_settings

        settings = get_settings()
        ok("settings validated", f"chunk={settings.chunk_size}/{settings.chunk_overlap} top_k={settings.top_k}")
    except Exception as exc:
        bad(f"settings failed to load: {type(exc).__name__}: {exc}", "check .env values")

    try:
        from app.main import create_app

        application = create_app()
        routes = [r for r in application.routes if getattr(r, "methods", None)]
        ok("app builds", f"{len(routes)} routes registered")
    except Exception as exc:
        bad(
            f"app failed to build: {type(exc).__name__}: {exc}",
            "this is a real bug, not a config problem -- send this traceback",
        )
        import traceback

        print(DIM + "".join("      " + l for l in traceback.format_exc().splitlines(True)) + RESET)

# --------------------------------------------------------------------------- #
print()
if blocking:
    print(f"{RED}{len(blocking)} blocking problem(s).{RESET} The app will not start until they are fixed:")
    for item in blocking:
        print(f"  - {item}")
    sys.exit(1)

msg = f"{GREEN}Ready.{RESET}"
if warnings:
    msg += f" {YELLOW}({len(warnings)} warning(s) above.){RESET}"
print(msg)
print()
print("  Start it with:")
print("    uvicorn app.main:app --reload")
if not using_fake:
    print(f"  {DIM}Or fully offline, with no models pulled:{RESET}")
    print("    make run-fake")
print()
print("  Then open http://localhost:8000")
