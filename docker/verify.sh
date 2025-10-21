#!/bin/bash
# Verification script to check if the Docker environment is set up correctly

echo "============================================"
echo "KRL Docker Environment Verification"
echo "============================================"
echo ""

# Check Python version
echo "1. Checking Python version..."
python --version
echo ""

# Check if CIL is installed
echo "2. Checking CIL installation..."
python -c "import cil; print(f'CIL version: {cil.__version__}')" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✓ CIL is installed"
else
    echo "✗ CIL is NOT installed"
fi
echo ""

# Check core dependencies
echo "3. Checking core dependencies..."
for pkg in numpy scipy matplotlib nibabel; do
    python -c "import $pkg; print(f'  {\"$pkg\":.<20} ✓ (version: {$pkg.__version__})')" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo "  $pkg ✗ NOT installed"
    fi
done
echo ""

# Check optional dependencies
echo "4. Checking optional dependencies..."
for pkg in numba torch; do
    python -c "import $pkg; print(f'  {\"$pkg\":.<20} ✓ (version: {$pkg.__version__})')" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo "  $pkg ........................ (not installed - optional)"
    fi
done
echo ""

# Check dev dependencies
echo "5. Checking development tools..."
for pkg in pytest black flake8; do
    if command -v $pkg &> /dev/null; then
        version=$($pkg --version 2>&1 | head -n1)
        echo "  $pkg ✓ ($version)"
    else
        echo "  $pkg ✗ NOT installed"
    fi
done
echo ""

# Check if KRL package is installed
echo "6. Checking KRL package..."
python -c "import sys; sys.path.insert(0, '/workspace/src'); import krl; print('  KRL package ✓ (importable)')" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✓ KRL package is importable"
else
    echo "  (Run 'pip install -e .' to install in editable mode)"
fi
echo ""

# Check directory structure
echo "7. Checking directory structure..."
for dir in src tests data results; do
    if [ -d "/workspace/$dir" ] || [ "$dir" = "results" ]; then
        echo "  /workspace/$dir ✓"
    else
        echo "  /workspace/$dir ✗"
    fi
done
echo ""

# Check if data directory has files
echo "8. Checking data directory..."
if [ -d "/workspace/data" ] && [ "$(ls -A /workspace/data 2>/dev/null)" ]; then
    echo "  Data directory has files:"
    ls -la /workspace/data | head -n 10
else
    echo "  Data directory is empty (this is normal for a fresh setup)"
fi
echo ""

# Check GPU availability
echo "9. Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    echo "  NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    python -c "import torch; print(f'  PyTorch CUDA available: {torch.cuda.is_available()}')" 2>/dev/null
else
    echo "  No GPU detected (this is fine for CPU-only usage)"
fi
echo ""

echo "============================================"
echo "Verification Complete!"
echo "============================================"
echo ""
echo "To run tests: pytest"
echo "To run the main script: python run_deconv.py --help"
echo ""
