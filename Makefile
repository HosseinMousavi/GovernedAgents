.PHONY: install test lint seed ingest eval up down trace

install:
	pip install -e ".[dev]"

test:
	pytest tests/unit

test-all:
	pytest

lint:
	ruff check src tests && mypy

seed:
	python -m store_support.data.seed

ingest:
	python -m store_support.knowledge.ingest data/policies

eval:
	python -m store_support.evals.runner

up:
	docker compose up -d --build

down:
	docker compose down -v

drift:
	python -m store_support.evals.drift
