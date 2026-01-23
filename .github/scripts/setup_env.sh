#!/bin/bash
# Common environment setup for CI jobs
# Usage: source setup_env.sh [--with-cmake] [--cuda-version <version>] <torch-version>
#   --with-cmake: Install cmake and ninja-build
#   --cuda-version: CUDA version (e.g., "12.8") - required for nightly builds
#   torch-version: "stable" or "nightly"

set -ex

INSTALL_CMAKE=false
TORCH_VERSION=""
CUDA_VERSION=""

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
    *)
      TORCH_VERSION="$1"
      shift
      ;;
  esac
done

if [ -z "$TORCH_VERSION" ]; then
  echo "Error: torch-version argument required (stable or nightly)"
  exit 1
fi

# Install system packages
dnf config-manager --set-enabled powertools
dnf install -y almalinux-release-devel

if [ "$INSTALL_CMAKE" = true ]; then
  dnf install -y ninja-build cmake
  # Remove old cmake/ninja from conda/local
  rm -f "/opt/conda/bin/ninja" || true
  rm -f "/opt/conda/bin/cmake" || true
  rm -f "/usr/local/bin/cmake" || true
fi

# Set up conda environment
conda config --set solver libmamba
conda create -n venv python=3.12 -y
conda activate venv
python -m pip install --upgrade pip

# Nuke conda libstd++ to avoid conflicts with system toolset
rm -f "$CONDA_PREFIX/lib/libstdc"* || true

# Install torch (nightly or stable via requirements.txt)
if [ "$TORCH_VERSION" = "nightly" ]; then
  if [ -z "$CUDA_VERSION" ]; then
    echo "Error: --cuda-version required for nightly builds"
    exit 1
  fi
  # Convert CUDA version (e.g., "12.8") to PyTorch format (e.g., "cu128")
  CUDA_TAG="cu$(echo "$CUDA_VERSION" | tr -d '.')"
  pip install --pre torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/nightly/${CUDA_TAG}"
fi

pip install -r requirements.txt
