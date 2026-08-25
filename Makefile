.PHONY: help install test test-cov lint format gpu-test build study-install study-test study-lint study-docker-pull study-sirf-test study-docker-python study-docker-run

help:
	@echo "cil-krl development commands"
	@echo "============================"
	@echo "  make install           - install package editable with dev tools (uv)"
	@echo "  make test              - run CPU test suite"
	@echo "  make gpu-test          - run GPU test suite (needs CUDA; opt-in)"
	@echo "  make lint              - ruff check"
	@echo "  make format            - ruff autofix"
	@echo "  make build             - build sdist and wheel"
	@echo ""
	@echo "  make study-install     - install studies package editable with dev tools (uv)"
	@echo "  make study-test        - run study tests (native)"
	@echo "  make study-lint        - ruff check studies/"
	@echo "  make study-docker-pull - pull pinned SIRF image"
	@echo "  make study-sirf-test   - run SIRF tests in container"
	@echo "  make study-docker-python - open python shell in SIRF container"
	@echo "  make study-docker-run  - run a study scenario in container (SCENARIO=path ARGS=... required)"

install:
	uv pip install -e ".[dev]"

test:
	python -m pytest tests/

gpu-test:
	KRL_RUN_GPU_TESTS=1 python -m pytest tests/test_gpu_kernel_operator.py tests/test_cuda_fallback.py

lint:
	ruff check src/ tests/

format:
	ruff check src/ tests/ --fix

build:
	python -m build

study-install:
	uv pip install -e "./studies[dev]"

study-test:
	python -m pytest studies/tests

study-lint:
	ruff check src/ tests/ studies/

study-docker-pull:
	docker pull synerbi/sirf@sha256:643c7955717ac08c6f44c6d3fe2ef064ebb54167f1da68771ed3e6dc07caf58d

study-sirf-test:
	docker compose -f studies/docker-compose.yaml run --rm sirf

study-docker-python:
	docker compose -f studies/docker-compose.yaml run --rm sirf bash

study-docker-run:
	test -n "$(SCENARIO)"
	docker compose -f studies/docker-compose.yaml run --rm study --scenario "$(SCENARIO)" $(ARGS)