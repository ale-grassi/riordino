.PHONY: install install-dev venv hooks check lint format typecheck test ci dist clean docker

PYTHON  ?= python3.14
VENV    := .venv
BIN     := $(VENV)/bin
UV      := $(shell command -v uv 2>/dev/null)

# ── Setup ─────────────────────────────────────────────────────────────

venv:
	$(PYTHON) -m venv $(VENV)

install: venv
ifdef UV
	$(UV) pip install . --python $(BIN)/python
else
	$(BIN)/pip install .
endif

install-dev: venv hooks
ifdef UV
	$(UV) pip install -e '.[dev]' --python $(BIN)/python
else
	$(BIN)/pip install -e '.[dev]'
endif

hooks:
	git config core.hooksPath .githooks

# ── Dependency checks ─────────────────────────────────────────────────

check:
	@echo "Python:"
	@$(BIN)/python --version
	@echo ""
	@echo "Tesseract:"
	@tesseract --version 2>&1 | head -1
	@echo ""
	@echo "Tesseract languages:"
	@tesseract --list-langs 2>&1
	@echo ""
	@echo "GOOGLE_API_KEY: $(if $(GOOGLE_API_KEY),set,NOT SET)"

# ── Quality ───────────────────────────────────────────────────────────

lint:
	$(BIN)/ruff check .
	$(BIN)/ruff format --check .

format:
	$(BIN)/ruff check --fix .
	$(BIN)/ruff format .

typecheck:
	$(BIN)/mypy riordino.py

test:
	$(BIN)/pytest --tb=short -q || [ $$? -eq 5 ]

ci: lint typecheck test

# ── Run ───────────────────────────────────────────────────────────────

run:
	$(BIN)/python riordino.py $(ARGS)

# ── Build & Publish ───────────────────────────────────────────────────

dist: clean
	$(BIN)/python -m build

# ── Docker ────────────────────────────────────────────────────────────

docker:
	docker build -t riordino $(if $(LANGS),--build-arg LANGS="$(LANGS)",) .

# ── Cleanup ───────────────────────────────────────────────────────────

clean:
	rm -rf $(VENV) __pycache__ *.egg-info .mypy_cache .ruff_cache .pytest_cache
