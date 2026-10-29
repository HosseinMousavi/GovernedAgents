.PHONY: install test lint seed ingest eval up down trace

install:
	pip install -e ".[dev]"

install-pii:
	pip install -e ".[dev,pii]" && python -m spacy download en_core_web_lg

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

eval-deepeval:   # LLM-judged tier (DeepEval, Claude as judge). Needs ANTHROPIC_API_KEY. DEEPEVAL_MODE=online for the live stack.
	@set -a; [ -f .env ] && . ./.env; set +a; DATABASE_URL=$(COMPOSE_PG_URL) deepeval test run tests/evals -q

tokens:   # signs with TOKEN_SIGNING_KEY from .env so tokens match the running containers
	@set -a; [ -f .env ] && . ./.env; set +a; python scripts/tokens.py

demo:
	./scripts/demo.sh

demo-offline:
	python scripts/demo_offline.py

reset-demo:   # put order 1042 back to "delivered 2 days ago, not refunded" so `make demo` can run again
	docker compose exec -T postgres psql -U ssa -d ssa -c "UPDATE orders SET ordered_on = CURRENT_DATE - 2, status='delivered', refunded_cents=0 WHERE order_id=1042; DELETE FROM refunds WHERE order_id=1042;"

# Host-side scripts talk to the compose Postgres, whose host port is whatever Docker picked.
COMPOSE_PG_URL = postgresql://ssa:ssa@localhost:$$(docker compose port postgres 5432 | sed 's/.*://')/ssa

compare-routing:   # live stack: all_default vs routed vs all_cheap on the 30 questions -> cost/quality table
	@set -a; [ -f .env ] && . ./.env; set +a; DATABASE_URL=$(COMPOSE_PG_URL) python scripts/compare_routing.py

eval-online:       # scorecard against the live stack (pgvector + Claude)
	@set -a; [ -f .env ] && . ./.env; set +a; DATABASE_URL=$(COMPOSE_PG_URL) python -m store_support.evals.runner --online
