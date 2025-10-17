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

The Docker container can use NVIDIA GPUs for significant performance improvements.

**Automatic detection and fallback:**
- `make start` checks for the `nvidia` Docker runtime.
- GPU acceleration is enabled automatically when the runtime is available.
- If GPUs are unavailable or the runtime is missing, the container starts in CPU-only mode with no extra steps required.

**Check GPU status inside the container:**
```bash
python src/gpu_utils.py
```

**Host requirements for GPU support:**
- NVIDIA GPU with CUDA support.
- Recent NVIDIA driver (verify on the host with `nvidia-smi`).
- Docker Engine with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

**Install NVIDIA Container Toolkit (Ubuntu 22.04 example):**
```bash
sudo apt-get update
sudo apt-get install -y curl ca-certificates gnupg
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo sed -i 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit.gpg] https://#' \
  /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

After installing, run `docker info | grep -i nvidia` to confirm that the `nvidia` runtime is listed before starting the container.

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
