# Lead-owned. Targets wrap owner scripts; recipes marked TODO are filled when owners deliver them.
PY      ?= .venv/bin/python
PIP     ?= .venv/bin/pip
PROFILE ?= gpu
COMPOSE := docker compose -f deploy/compose.yaml
# D-030: avoid Triton JIT (needs python3.12-dev headers, absent on this host)
export TORCH_DISABLE_NATIVE_JIT ?= 1
OFFLINE_MARKERS := not gpu and not real_model and not docker and not slow

.PHONY: help venv install lint format typecheck test test-all ci up down gpu-check stack-smoke api-host

help:
	@grep -E '^[a-z-]+:' Makefile | cut -d: -f1 | sort | tr '\n' ' '; echo

venv:  ## create .venv with CUDA 13 torch (aarch64 GB10)
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install --index-url https://download.pytorch.org/whl/cu130 -c constraints/torch-cu130.txt torch

install:  ## install project + extras without touching torch
	$(PIP) install -c constraints/torch-cu130.txt --extra-index-url https://download.pytorch.org/whl/cu130 \
		-e ".[ml,retrieval,api,ui,tracking,eval,dev]"

lint:
	.venv/bin/ruff check src tests
	.venv/bin/ruff format --check src tests

format:
	.venv/bin/ruff format src tests

typecheck:
	.venv/bin/mypy src/medquad_qa

test:
	.venv/bin/pytest -m "$(OFFLINE_MARKERS)" -q

test-all:
	.venv/bin/pytest -q

ci: lint typecheck test
	$(COMPOSE) --profile cpu config -q

# --- local stack (recipes owned by service-platform-engineer scripts) ---
up:
	$(COMPOSE) --profile $(PROFILE) up -d --build

down:
	$(COMPOSE) --profile gpu --profile cpu --profile monitoring down

gpu-check:
	bash scripts/ops/validate_gpu_container.sh

stack-smoke:
	bash scripts/ops/stack_up_smoke.sh

api-host:
	bash scripts/service/run_api_host.sh
