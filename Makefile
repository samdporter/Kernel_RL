.PHONY: help build start stop restart shell clean logs test test-cov format lint install

# Default target
help:
	@echo "KRL Docker Development Commands"
	@echo "================================"
	@echo ""
	@echo "Setup:"
	@echo "  make build        - Build the Docker image"
	@echo "  make install      - Install the package in editable mode (inside container)"
	@echo ""
	@echo "Container Management:"
	@echo "  make start        - Start the container"
	@echo "  make stop         - Stop the container"
	@echo "  make restart      - Restart the container"
	@echo "  make shell        - Open interactive shell in container"
	@echo "  make logs         - Show container logs"
	@echo "  make clean        - Remove containers and volumes"
	@echo ""
	@echo "Development:"
	@echo "  make test         - Run tests"
	@echo "  make test-cov     - Run tests with coverage"
	@echo "  make format       - Format code with black"
	@echo "  make lint         - Run linting (flake8)"
	@echo ""
	@echo "For more information, see DOCKER.md"

# Build the Docker image
build:
	docker-compose build

# Start the container
start:
	@if docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q '"nvidia"'; then \
		echo "Detected NVIDIA runtime; starting container with GPU support."; \
		docker-compose -f docker-compose.yml -f docker-compose.gpu.yml up -d; \
	else \
		echo "No NVIDIA runtime detected; starting container in CPU-only mode."; \
		docker-compose up -d; \
	fi
	@echo "Container started. Use 'make shell' to access it."

# Stop the container
stop:
	docker-compose stop

# Restart the container
restart:
	docker-compose restart

# Open interactive shell
shell:
	docker-compose exec krl bash

# Show logs
logs:
	docker-compose logs -f krl

# Clean up everything
clean:
	docker-compose down -v

# Run tests
test:
	docker-compose exec krl pytest

# Run tests with coverage
test-cov:
	docker-compose exec krl pytest --cov=krl --cov-report=html --cov-report=term

# Format code with black
format:
	docker-compose exec krl black src/ tests/

# Run linting
lint:
	docker-compose exec krl flake8 src/ tests/

# Install package in editable mode
install:
	docker-compose exec krl pip install -e '.[dev]'

# Build and start in one command
up: build start

# Run the main deconvolution script
run-deconv:
	docker-compose exec krl python run_deconv.py --help
