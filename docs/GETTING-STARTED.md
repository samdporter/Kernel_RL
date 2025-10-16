# Getting Started

## Install

### Docker (Recommended)

```bash
git clone https://github.com/KCL-BMEIS/KRL.git && cd KRL
make build    # First time: ~10 minutes
make start
make shell
```

### Conda

```bash
git clone https://github.com/KCL-BMEIS/KRL.git && cd KRL
conda install -c conda-forge -c ccpi cil
pip install -e .
```

### VS Code

1. Open folder in VS Code
2. Press `F1` → "Dev Containers: Reopen in Container"
3. Wait for build

## Test It Works

```bash
# Check dependencies
python -c "import numpy, scipy, matplotlib, cil; print('OK')"

# Check GPU availability (optional accelerators)
python src/gpu_utils.py

# Run tests
pytest

# Try the CLI
python -m krl.cli.run_deconv --help
```

## Run Deconvolution

```bash
# Basic deconvolution with KRL
python -m krl.cli.run_deconv \
  --data-path data/spheres \
  --emission-file phant_pet.nii \
  --guidance-file phant_mri.nii \
  --enable-krl

# Run hyperparameter sweeps (multiple configurations)
python scripts/run_deconv_sweeps.py \
  --datasets spheres \
  --pipelines krl hkrl

# Preview sweep configurations without running
python scripts/run_deconv_sweeps.py --dry-run
```

Results saved to `results/`

## GPU Acceleration

The Docker container includes GPU support for significant performance improvements:

**Automatic GPU detection:**
- GPU acceleration is automatically used when available
- Falls back to CPU if no GPU is detected
- Supports: PyTorch (CUDA), CuPy, and Numba

**Check GPU status:**
```bash
python src/gpu_utils.py
```

**Requirements for GPU:**
- NVIDIA GPU with CUDA support
- NVIDIA Docker runtime installed: https://github.com/NVIDIA/nvidia-docker
- Docker Compose V2 (for GPU deploy syntax)

**Install NVIDIA Docker (if needed):**
```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

## Common Issues

**Docker won't start:**
```bash
sudo systemctl start docker  # Linux
# Or start Docker Desktop (Mac/Windows)
```

**GPU not detected:**
```bash
# Test GPU outside container
nvidia-smi

# If nvidia-smi fails, install/update NVIDIA drivers
# Rebuild container after installing NVIDIA Docker
make build
```

**Out of space:**
```bash
docker system prune -a
```

**Import errors:**
```bash
pip install -e .
```

## Next Steps

- See [docker/README.md](../docker/README.md) for more Docker commands
- Check `python -m krl.cli.run_deconv --help` for all options
- Run sweeps with `python scripts/run_deconv_sweeps.py --help`
- Example reconstruction: `python scripts/example_spheres_reconstruction.py`
