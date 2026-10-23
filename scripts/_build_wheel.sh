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

# NCCLX is deprecated and excluded from wheels. Transport needs its folly
# build, so it is excluded too.
export USE_NCCLX=0
export USE_TRANSPORT=0
# CUDA device compilation can exhaust memory at full host parallelism.
export CMAKE_BUILD_PARALLEL_LEVEL=8

python setup.py bdist_wheel
