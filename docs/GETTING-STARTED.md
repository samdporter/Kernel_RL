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

KRL supports GPU acceleration for both blurring operators and kernel operators, enabling processing of large volumes (up to 256³) on consumer GPUs.

### Kernel Operator GPU Support (New!)

The kernel operator now supports PyTorch CUDA backend with automatic GPU detection:

```python
from krl.operators import get_kernel_operator

# Automatic backend selection (prefers GPU)
op = get_kernel_operator(geometry, backend='auto')

# Force GPU with float32 for large volumes (256³)
op = get_kernel_operator(geometry, backend='torch', dtype='float32',
                         num_neighbours=5, mask_k=20)

# Force CPU (Numba)
op = get_kernel_operator(geometry, backend='numba')
```

**Memory Requirements for 256³ Volumes:**
- Small config (n=5, k=20): **3.6 GB** → Fits RTX 3060 (12 GB)
- Medium config (n=7, k=48): **8.8 GB** → Fits RTX 4090 (24 GB)
- Use `dtype='float32'` to halve memory usage

**Check GPU status:**
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

### Docker GPU Support

The Docker container can use NVIDIA GPUs for blurring operations.

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
