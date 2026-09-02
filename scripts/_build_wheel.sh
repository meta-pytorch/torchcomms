#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.

# DO NOT DELETE
# This script runs within the docker container invoked by docker_build_wheel.sh .
# Run that script instead.

set -ex

dnf config-manager --set-enabled powertools
dnf install -y almalinux-release-devel
dnf install -y ninja-build cmake

# Nuke conda cmake, ninja and libstdc++ we want to install to use system libraries.
rm -f "$CONDA_PREFIX/lib/libstdc"* || true
conda remove -y cmake ninja || true
rm -f "$CONDA_PREFIX/bin/ninja" || true
rm -f "$CONDA_PREFIX/bin/cmake" || true
rm -f "/opt/conda/bin/ninja" || true
rm -f "/opt/conda/bin/cmake" || true

python --version
which python

pip install -r requirements.txt
# pyyaml is a build-time-only dep (used by extractcvars.py codegen, run from
# CMake); it is intentionally not in requirements.txt/install_requires so the
# runtime wheel resolves from the PyTorch index alone.
pip install pyyaml

export NCCL_SKIP_CONDA_INSTALL=1
export CLEAN_BUILD=1
# Match the NCCLX feedstock and TorchComms iter build. NCCLX headers require
# C++20, and fmt needs its NVCC C++20 compatibility patch applied after the
# environment's dependency installation.
export CXXSTD="-std=c++20"
export NCCL_PATCH_FMT_NVCC_CXX20=1
# NCCLX device compilation can exhaust memory at full host parallelism.
# Lower to 8 to avoid OOM on g5.12xlarge self-hosted runners for old CUDA
# builders (cu126/cu130) that have been hitting "lost communication" during
# device kernel compilation.
export NCCL_BUILD_JOBS=8

python setup.py bdist_wheel
