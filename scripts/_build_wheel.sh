#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.

# DO NOT DELETE
# This script runs within the docker container invoked by docker_build_wheel.sh .
# Run that script instead.

set -ex

dnf config-manager --set-enabled powertools
dnf install -y almalinux-release-devel
dnf install -y openssl-static ninja-build cmake

# Install older openssl which is compatible with manylinux2.28 system openssl.
conda install -y openssl==3.0.18

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

export NCCL_SKIP_CONDA_INSTALL=1
CLEAN_BUILD=1 ./build_ncclx.sh

python setup.py bdist_wheel
