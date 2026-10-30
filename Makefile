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

# ---------- LoRA: local cheap tier ----------
lora-dataset:      # Haiku labels grade/filter on synthetic questions -> data/lora/{train,val}.jsonl (~$0.50)
	@set -a; [ -f .env ] && . ./.env; set +a; python -m store_support.lora.build_dataset

lora-train:        # PEFT LoRA on Qwen2.5-1.5B-Instruct (MPS/CUDA/CPU) -> models/lora/ssa-grader/{adapter,merged}
	python -m store_support.lora.train $(ARGS)

lora-train-small:  # 0.5B student, 1 epoch: fits a 16 GB Mac in ~30 min
	python -m store_support.lora.train --base Qwen/Qwen2.5-0.5B-Instruct --epochs 1 $(ARGS)

LORA_DIR = models/lora/ssa-grader

lora-serve:        # merged safetensors -> GGUF (llama.cpp converter) -> Ollama `ssa-grader`
	@[ -d models/llama.cpp ] || git clone -q --depth 1 https://github.com/ggml-org/llama.cpp models/llama.cpp
	pip install -q gguf sentencepiece protobuf
	python models/llama.cpp/convert_hf_to_gguf.py $(LORA_DIR)/merged --outfile $(LORA_DIR)/ssa-grader.gguf --outtype f16
	docker compose exec -T ollama mkdir -p /models
	docker compose cp $(LORA_DIR)/ssa-grader.gguf ollama:/models/ssa-grader.gguf
	docker compose exec -T ollama sh -c 'printf "FROM /models/ssa-grader.gguf\nPARAMETER temperature 0\nPARAMETER num_ctx 4096\n" > /tmp/Modelfile && ollama create ssa-grader -f /tmp/Modelfile'
	docker compose exec -T ollama ollama show ssa-grader --modelfile | head -3

LORA_BASE_TAG ?= qwen2.5:0.5b-instruct   # untuned base for the before/after; qwen2.5:1.5b-instruct for `make lora-train`

lora-eval:         # student vs teacher agreement on the held-out split (tuned vs untuned base)
	@set -a; [ -f .env ] && . ./.env; set +a; python -m store_support.lora.eval_adapter --model ssa-grader
	docker compose exec -T ollama ollama pull $(LORA_BASE_TAG)
	@set -a; [ -f .env ] && . ./.env; set +a; python -m store_support.lora.eval_adapter --model $(LORA_BASE_TAG) || true

compare-lora:      # routing table with the local tier as `cheap` (CHEAP_MODEL=ollama/ssa-grader)
	@set -a; [ -f .env ] && . ./.env; set +a; DATABASE_URL=$(COMPOSE_PG_URL) CHEAP_MODEL=ollama/ssa-grader python scripts/compare_routing.py --policies routed,all_cheap
