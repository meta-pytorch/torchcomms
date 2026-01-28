#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.

set -x # print commands before running them
set -euo pipefail # exit script on errors

python -m pip install --upgrade pip
conda install conda-forge::libopenssl-static conda-forge::rsync -y
conda install conda-forge::glog=0.4.0 conda-forge::gflags conda-forge::fmt -y
ITER_BUILD_DIR=/tmp/torchcomms_build2/fbcode
mkdir -p "$ITER_BUILD_DIR"
pushd "$ITER_BUILD_DIR" || exit
FBCODE_DIR=$HOME/fbsource/fbcode
rsync -av --exclude 'analyzer/ground_truth_csvs' --exclude 'analyzer/integration_test_data' "$FBCODE_DIR"/comms . -q
cp "$FBCODE_DIR"/comms/github/*.sh .
cp "$FBCODE_DIR"/comms/github/setup.py .
cp "$FBCODE_DIR"/comms/github/CMakeLists.txt .
cp "$FBCODE_DIR"/comms/github/version.txt .
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/rocm6.4
export BUILD_RCCL_ONLY=1
export BUILDDIR=${PWD}/build/rcclx
export ROCM_HOME=/opt/rocm
export RCCLX_INCLUDE=${BUILDDIR}/include/rccl
export RCCLX_LIB=${BUILDDIR}/lib

./build_rcclx.sh
pip install numpy

USE_TRANSPORT=OFF USE_NCCL=0 USE_NCCLX=0 USE_GLOO=0 USE_RCCL=0 USE_RCCLX=1 pip install --no-build-isolation -v '.[dev]'
