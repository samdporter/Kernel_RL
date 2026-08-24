.PHONY: help install test test-cov lint format gpu-test build study-install study-test study-lint study-docker-pull study-sirf-test study-docker-python

help:
	@echo "cil-krl development commands"
	@echo "============================"
	@echo "  make install   - install package editable with dev tools (uv)"
	@echo "  make test      - run CPU test suite"
	@echo "  make gpu-test  - run GPU test suite (needs CUDA; opt-in)"
	@echo "  make lint      - ruff check"
	@echo "  make format    - ruff autofix"
	@echo "  make build     - build sdist and wheel"

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
	docker pull synerbi/sirf:latest

study-sirf-test:
	docker compose -f studies/docker-compose.yaml run --rm sirf

study-docker-python:
	docker compose -f studies/docker-compose.yaml run --rm sirf bash
