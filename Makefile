.PHONY: help install test test-cov lint format gpu-test build

# Interpreter that has CIL installed, e.g. the python of a micromamba env:
#   make test PYTHON=$(micromamba run -n krl which python)
PYTHON ?= python

help:
	@echo "cil-krl development commands"
	@echo "============================"
	@echo "  make install   - install package editable with dev tools (uv)"
	@echo "  make test      - run CPU test suite"
	@echo "  make gpu-test  - run GPU test suite (needs CUDA; opt-in)"
	@echo "  make lint      - ruff check"
	@echo "  make format    - ruff autofix"
	@echo "  make build     - build sdist and wheel"
	@echo ""
	@echo "Set PYTHON to the interpreter that has CIL, e.g.:"
	@echo "  make install PYTHON=\$$(micromamba run -n krl which python)"

install:
	uv pip install --python "$(PYTHON)" -e ".[dev]"

test:
	$(PYTHON) -m pytest tests/

gpu-test:
	KRL_RUN_GPU_TESTS=1 $(PYTHON) -m pytest tests/test_gpu_kernel_operator.py tests/test_cuda_fallback.py

lint:
	$(PYTHON) -m ruff check src/ tests/

format:
	$(PYTHON) -m ruff check src/ tests/ --fix

build:
	uv build
