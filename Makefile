.PHONY: install dev test test-unit lint format docker-up docker-down run index evaluate clean

install:
	pip install -e .

dev:
	pip install -e ".[dev]"
	pre-commit install

# ── Infrastructure ─────────────────────────────────────────────────────────────
docker-up:
	docker compose -f docker/docker-compose.yml up -d qdrant
	@echo "Qdrant → http://localhost:6333"

docker-down:
	docker compose -f docker/docker-compose.yml down

docker-full:
	docker compose -f docker/docker-compose.yml up -d

# ── Quality ────────────────────────────────────────────────────────────────────
lint:
	ruff check src/ scripts/
	black --check src/ scripts/
	mypy src/ --ignore-missing-imports

format:
	black src/ scripts/
	ruff check --fix src/ scripts/

# ── Tests ──────────────────────────────────────────────────────────────────────
test:
	pytest src/tests/ -v --cov=src --cov-report=term-missing

test-unit:
	pytest src/tests/test_all.py -v -k "not TestAPI"

test-api:
	pytest src/tests/test_all.py -v -k "TestAPI"

test-cov:
	pytest src/tests/ --cov=src --cov-report=html
	@echo "Open htmlcov/index.html"

# ── Pipeline ───────────────────────────────────────────────────────────────────
index:
	python -m src.indexer.cli --repo-path $(REPO_PATH)

index-github:
	python -m src.indexer.cli --github-repo $(GITHUB_REPO)

run:
	python -m src.main --issue $(ISSUE)

run-fast:
	python -m src.main --issue $(ISSUE) --skip-indexing

api:
	python -m src.api.server

# ── Evaluation ─────────────────────────────────────────────────────────────────
evaluate:
	python -m scripts.evaluate --issues-file scripts/benchmark_issues.json

evaluate-fast:
	python -m scripts.evaluate --issues-file scripts/benchmark_issues.json --skip-indexing

# ── Cleanup ────────────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	rm -rf .mypy_cache .ruff_cache dist build *.egg-info htmlcov
	rm -f /tmp/resolver_checkpoints.db evaluation_report.json coverage.xml
