#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.

# DO NOT DELETE
# This script runs the Python integration tests.
# This is used as part of the GitHub CI.

set -ex

cd "$(dirname "$0")/../tests/integration/py"

run_tests () {
    for file in *Test.py; do
        torchrun --nnodes 1 --nproc_per_node 4 "$file" --verbose
    done
}

# NCCL
export TEST_BACKEND=nccl
run_tests

# NCCLX
export TEST_BACKEND=ncclx
run_tests

# Gloo
export TEST_BACKEND=gloo
export TEST_DEVICE=cpu
export CUDA_VISIBLE_DEVICES=""
run_tests
unset TEST_DEVICE
unset CUDA_VISIBLE_DEVICES
