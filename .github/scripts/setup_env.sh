#!/bin/bash
# Common environment setup for CI jobs
# Usage: source setup_env.sh [--with-cmake] [--cuda-version <version>] [--torch-version <version>] <torch-channel>
#   --with-cmake: Install cmake and ninja-build
#   --cuda-version: CUDA version (e.g., "12.8") - required for nightly builds
#   --torch-version: Exact torch version to install (e.g., "2.6.0.dev20250101")
#   torch-channel: "stable" or "nightly"

set -ex

INSTALL_CMAKE=false
TORCH_CHANNEL=""
CUDA_VERSION=""
TORCH_VERSION=""

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --with-cmake)
      INSTALL_CMAKE=true
      shift
      ;;
    --cuda-version)
      CUDA_VERSION="$2"
      shift 2
      ;;
    --torch-version)
      TORCH_VERSION="$2"
      shift 2
      ;;
    *)
      TORCH_CHANNEL="$1"
      shift
      ;;
  esac
done

if [ -z "$TORCH_CHANNEL" ]; then
  echo "Error: torch-channel argument required (stable or nightly)"
  exit 1
fi

# Set up conda environment
conda config --set solver libmamba
conda create -n venv python=3.12 -c conda-forge -y
conda activate venv
python -m pip install --upgrade pip

if [ "$INSTALL_CMAKE" = true ]; then
  conda install -y conda-forge::cmake conda-forge::ninja
fi

# Nuke conda libstd++ to avoid conflicts with system toolset
rm -f "$CONDA_PREFIX/lib/libstdc"* || true

# Install torch (nightly or stable)
if [ -n "$CUDA_VERSION" ]; then
  # Convert CUDA version (e.g., "12.8") to PyTorch format (e.g., "cu128")
  CUDA_TAG="cu$(echo "$CUDA_VERSION" | tr -d '.')"

  if [ "$TORCH_CHANNEL" = "nightly" ]; then
    INDEX_URL="https://download.pytorch.org/whl/nightly/${CUDA_TAG}"
    if [ -n "$TORCH_VERSION" ]; then
      pip install --pre torch=="${TORCH_VERSION}" --index-url "$INDEX_URL"
    else
      pip install --pre torch --index-url "$INDEX_URL"
    fi
  else
    # Stable with CUDA - use PyTorch wheel index
    INDEX_URL="https://download.pytorch.org/whl/${CUDA_TAG}"
    if [ -n "$TORCH_VERSION" ]; then
      pip install torch=="${TORCH_VERSION}" --index-url "$INDEX_URL"
    else
      pip install torch --index-url "$INDEX_URL"
    fi
  fi
else
  # No CUDA version - install from PyPI (stable only)
  if [ -n "$TORCH_VERSION" ]; then
    pip install torch=="${TORCH_VERSION}"
  fi
fi

pip install -r requirements.txt
