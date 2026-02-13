#!/bin/bash
git submodule init
git submodule update
uv venv .venv --seed --python 3.12
source .venv/bin/activate
uv pip install --upgrade pip setuptools wheel

# 3. Step one: pin PyTorch 2.5.1 and related components
# Explicitly use CUDA 12.4 index to prevent drift
echo "Installing PyTorch core components..."
uv pip install "torch==2.5.1" "torchvision==0.20.1" "xformers>=0.0.28" --index-url https://download.pytorch.org/whl/cu124

# 4. Step two: build and install vLLM
# This step is slow; wait for build to finish
echo "Building vLLM (compatible with PyTorch 2.5.1)..."
VLLM_USE_PRECOMPILED=1 uv pip install --editable . -vv

# 5. Step three: install all REAP dependencies
# Force reinstall scikit-learn, numpy, autoawq, etc., to ensure correct linkage
echo "Installing toolchain dependencies..."
uv pip install \
    "numpy>=2.0.0" \
    "scikit-learn" \
    "regex" \
    "pillow" \
    "autoawq>=0.1.8" \
    "transformers" \
    "evalplus" \
    "livecodebench"

# 6. Final validation
echo "Running final system check..."
python -c "import torch; import vllm; import sklearn; import awq; print('✅ Done! The system is rebuilt and you can run REAP scripts now!')"