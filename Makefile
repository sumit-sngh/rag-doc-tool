# Shortcuts. Everything here is a plain command you can also run by hand --
# `make -n <target>` prints it without executing.

.PHONY: help check check-fake venv install install-optional run run-fake test test-py test-js \
        docker up up-all down clean

# The pins in requirements.txt include compiled wheels published per CPython
# version, so the interpreter is not a free choice -- see `venv` below. First
# match wins, and a bare `python3` is deliberately last.
PYTHON ?= $(shell for p in python3.12 python3.11 python3.10 python3; do \
                    command -v $$p >/dev/null 2>&1 && { echo $$p; break; }; \
                  done)

help:  ## Show this help
	@grep -E '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[1m%-18s\033[0m %s\n", $$1, $$2}'

check:  ## Diagnose the environment before starting (run this first)
	python3 scripts/preflight.py

check-fake:  ## Same, but for the offline providers (skips the Ollama checks)
	EMBEDDING_PROVIDER=fake LLM_PROVIDER=fake python3 scripts/preflight.py

venv:  ## Create .venv with a supported interpreter (3.10-3.12)
	@v=$$($(PYTHON) -c 'import sys; print("%d.%d" % sys.version_info[:2])'); \
	case $$v in \
	  3.10|3.11|3.12) echo "Using $(PYTHON) ($$v)";; \
	  *) echo "$(PYTHON) is $$v; this project needs 3.10-3.12."; \
	     echo "pydantic-core ships prebuilt wheels per CPython version, and on a"; \
	     echo "newer interpreter pip compiles it from source and needs Rust."; \
	     echo "  macOS: brew install python@3.12 && make venv"; \
	     echo "  or skip the venv entirely: make up"; \
	     exit 1;; \
	esac
	$(PYTHON) -m venv .venv
	./.venv/bin/pip install --upgrade pip
	./.venv/bin/pip install -r requirements.txt
	@echo
	@echo "Done. Activate it, then check:"
	@echo "  source .venv/bin/activate && make check"

install:  ## Install runtime + test dependencies into the active environment
	pip install -r requirements.txt

install-optional:  ## Add OCR, cross-encoder reranking, and PDF test fixtures
	pip install -r requirements-optional.txt

run:  ## Serve on :8000 with autoreload (needs Ollama)
	uvicorn app.main:app --reload

run-fake:  ## Same, with offline deterministic providers (no models needed)
	EMBEDDING_PROVIDER=fake LLM_PROVIDER=fake uvicorn app.main:app --reload

test: test-py test-js  ## Run every test

test-py:  ## Python suite
	pytest

test-js:  ## Frontend suite (no npm install required)
	node tests/frontend.test.js

docker:  ## Build the image
	docker build -t document-qa .

up:  ## Run in Docker against Ollama on the host
	docker compose up

up-all:  ## Run in Docker with Ollama containerised and models pulled
	docker compose -f docker-compose.yml -f docker-compose.ollama.yml up

down:  ## Stop containers (named volumes survive)
	docker compose -f docker-compose.yml -f docker-compose.ollama.yml down

clean:  ## Delete all ingested data. Documents must be re-uploaded afterwards.
	rm -rf data/
