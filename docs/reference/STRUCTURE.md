# KRL Project Structure

Clean, organized directory layout for the KRL project.

## Directory Overview

```
KRL/
├── 📦 src/krl/              # Main Python package
├── 🧪 tests/                # Test suite
├── 📚 docs/                 # All documentation
├── 🐳 docker/               # Docker configuration
├── 🛠️  scripts/              # Development scripts
├── 📁 data/                 # Your data files (gitignored)
└── 📁 results/              # Output files (gitignored)
```

## File Guide

### Root Directory (Keep It Clean!)

| File | Purpose |
|------|---------|
| `README.md` | **START HERE** - Project overview and quick start |
| `pyproject.toml` | Package configuration (pip install) |
| `requirements*.txt` | Python dependencies |
| `Makefile` | Quick commands (`make build`, `make test`, etc.) |
| `docker-compose.yml` | Docker setup (uses `docker/`) |
| `docker-run.sh` | Docker helper script |
| `run_deconv.py` | Main CLI script (legacy, for compatibility) |
| `dependencies.py` | Check installed dependencies |

### Documentation (`docs/`)

| File/Folder | Purpose |
|-------------|---------|
| `docs/README.md` | Documentation index |
| `docs/GETTING-STARTED.md` | Installation and first steps |
| `docs/reference/` | Historical notes (SIRF comparisons, refactoring notes) |

### Docker (`docker/`)

| File | Purpose |
|------|---------|
| `docker/README.md` | **Docker guide** - All commands and troubleshooting |
| `docker/Dockerfile` | Image definition |
| `docker/docker-compose.yml` | Container configuration |
| `docker/verify.sh` | Environment verification script |

### Source Code (`src/`)

| Path | Purpose |
|------|---------|
| `src/krl/` | Main package (properly structured) |
| `src/krl/operators/` | Kernel, blurring, gradient operators |
| `src/krl/algorithms/` | MAPRL algorithm |
| `src/krl/cli/` | Command-line interface |
| `src/*.py` | Legacy modules (to be migrated to `krl/`) |

### Tests (`tests/`)

All test files use pytest conventions:
- `conftest.py` - Shared fixtures
- `test_*.py` - Test modules

### Scripts (`scripts/`)

Development and experimental scripts:
- Parameter sweeps
- Manual testing
- Analysis tools

## Quick Reference

### First Time Setup

```bash
# Docker (easiest)
make build && make start && make shell

# Or Conda
conda install -c conda-forge -c ccpi cil
pip install -e '.[dev]'
```

### Find Documentation

1. **Getting started?** → [`docs/GETTING-STARTED.md`](docs/GETTING-STARTED.md)
2. **Docker help?** → [`docker/README.md`](docker/README.md)
3. **Overview?** → [`README.md`](README.md)
4. **All docs?** → [`docs/README.md`](docs/README.md)

### Common Tasks

```bash
# Development
make shell          # Enter container
make test           # Run tests
make format         # Format code
make lint           # Check code

# Docker management
make build          # Build image
make start          # Start container
make stop           # Stop container
make clean          # Remove all
```

## Design Principles

### ✅ Good Practices

- **Root is clean** - Only essential config files
- **Docs are organized** - Clear navigation in `docs/`
- **Docker is separate** - All Docker files in `docker/`
- **Code is packaged** - Main code in `src/krl/`
- **Tests are isolated** - All tests in `tests/`

### ❌ Avoid

- Putting docs in root (use `docs/`)
- Scattering Docker files everywhere (use `docker/`)
- Mixing source and tests
- Leaving large data files in repo (use `data/` - gitignored)

## Migration Notes

### Old → New

| Old Location | New Location | Status |
|--------------|--------------|--------|
| `.devcontainer/Dockerfile` | `docker/Dockerfile` | ✅ Moved |
| `.devcontainer/docker-compose.yml` | `docker/docker-compose.yml` | ✅ Moved |
| `.devcontainer/verify-setup.sh` | `docker/verify.sh` | ✅ Moved |
| `DOCKER.md` | `docker/README.md` | ✅ Replaced |
| `QUICKSTART.md` | `docs/GETTING-STARTED.md` | ✅ Rewritten |
| `SIRF_VS_CIL.md` | `docs/reference/` | ✅ Archived |
| `NO_SIRF_NEEDED.md` | `docs/reference/` | ✅ Archived |
| `REFACTORING_SUMMARY.md` | `docs/reference/` | ✅ Archived |

### Docker Files Location Explained

**Why some Docker files are at root:**

```
Root level (required by tools):
├── docker-compose.yml    # Docker Compose expects this here
├── .dockerignore         # Docker build context requires this here
├── Makefile              # Standard location for make commands
└── docker-run.sh         # Quick access convenience script

docker/ directory (organized storage):
├── Dockerfile            # The actual image definition
├── README.md             # All Docker documentation
└── verify.sh             # Docker-specific utility
```

This split is intentional:
- Tools like `docker-compose` and VS Code DevContainers look for `docker-compose.yml` at root
- Docker build requires `.dockerignore` at the build context root
- But detailed config (Dockerfile) and docs live in `docker/` for organization

### `.devcontainer/`

Minimal - only contains:
- `devcontainer.json` - VS Code config pointing to root `docker-compose.yml`
- Uses shared `docker/Dockerfile` and root `docker-compose.yml`

## VS Code Dev Container

Works seamlessly with new structure:
1. Open project in VS Code
2. Press `F1` → "Dev Containers: Reopen in Container"
3. Uses `docker-compose.yml` and `docker/Dockerfile` automatically

## Benefits of New Structure

✨ **Cleaner root** - Easy to find important files
✨ **Better organization** - Everything has a place
✨ **Easier navigation** - Clear directory purpose
✨ **Less duplication** - Single source of truth for docs
✨ **Maintainable** - Easy to add new features
✨ **Professional** - Follows Python best practices

## Questions?

- Check [`docs/README.md`](docs/README.md) for all documentation
- See [`docker/README.md`](docker/README.md) for Docker help
- Open an issue on GitHub for bugs/features
