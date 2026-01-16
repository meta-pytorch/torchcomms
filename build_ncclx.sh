#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This script installs conda dependencies required for building torchcomms
# when USE_SYSTEM_LIBS is set. The actual build is handled by CMake.
#
# Usage:
#   USE_SYSTEM_LIBS=1 ./build_ncclx.sh
#
# To skip conda install (e.g., if deps are already installed):
#   USE_SYSTEM_LIBS=1 NCCL_SKIP_CONDA_INSTALL=1 ./build_ncclx.sh

set -e

# Only install conda dependencies when USE_SYSTEM_LIBS is set
if [[ -n "${USE_SYSTEM_LIBS}" ]]; then
    if [[ -z "${NCCL_SKIP_CONDA_INSTALL}" ]]; then
        echo "Installing conda dependencies for USE_SYSTEM_LIBS build..."
        DEPS=(
            cmake
            ninja
            jemalloc
            gtest
            boost
            double-conversion
            libevent
            conda-forge::libsodium
            libunwind
            snappy
            conda-forge::fast_float
            libdwarf-dev
            gflags
            glog==0.4.0
            xxhash
            zstd
            conda-forge::zlib
            conda-forge::libopenssl-static
            fmt
        )
        conda install "${DEPS[@]}" --yes
        echo "Conda dependencies installed successfully."
    else
        echo "NCCL_SKIP_CONDA_INSTALL is set, skipping conda install."
    fi
else
    echo "USE_SYSTEM_LIBS is not set. Conda dependencies are not needed."
    echo "Third-party dependencies will be built from source by CMake."
fi
