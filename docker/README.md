# Docker Guide

## Quick Commands

```bash
make build     # Build image (~10 min first time)
make start     # Start container
make shell     # Access bash
make stop      # Stop
make clean     # Remove everything
```

## All Commands

```bash
# Setup
make build             # Build Docker image
make start             # Start in background

# Daily use
make shell             # Interactive bash
make test              # Run tests
make format            # Format code (black)
make lint              # Check code (flake8)

# Management
make stop              # Stop container
make restart           # Restart
make logs              # View logs
make clean             # Remove all
```

## Without Make

```bash
# Using docker-compose
docker-compose up -d              # Start
docker-compose exec krl bash     # Shell
docker-compose stop               # Stop
docker-compose down               # Remove
```

## VS Code

1. Install "Dev Containers" extension
2. Press `F1` → "Dev Containers: Reopen in Container"
3. Done!

## What's Inside

- Python 3.10 + Conda
- CIL from conda-forge
- All dependencies (numpy, scipy, matplotlib, nibabel)
- Optional: numba, torch
- Dev tools: pytest, black, flake8

## Your Files

| Host | Container | Purpose |
|------|-----------|---------|
| `./` | `/workspace` | Your code (live sync) |
| `./data/` | `/workspace/data` | Data files |
| `./results/` | `/workspace/results` | Outputs |

Changes sync instantly - no restart needed!

## GPU Support

1. Install [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)
2. Edit `docker-compose.yml` - uncomment GPU section
3. Rebuild: `make build`

## Troubleshooting

**Container won't start:**
```bash
docker ps  # Check Docker is running
make logs  # See errors
```

**Out of disk space:**
```bash
docker system prune -a
```

**Permission errors:**
```bash
sudo chown -R $USER:$USER .
```

**Package not found:**
```bash
make shell
pip install -e .
```

**Slow build:**
- First build takes ~10 min (CIL install)
- Rebuilds are fast (cached layers)

## Verify Setup

```bash
make shell
./docker/verify.sh
```
