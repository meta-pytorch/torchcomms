#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.

# DO NOT DELETE
# This script runs the Python integration tests.
# This is used as part of the GitHub CI.

set -ex

cd "$(dirname "$0")/../tests/integration/py"

NUM_GPUS=$(python3 -c "import torch; print(max(1, torch.cuda.device_count()))")

run_tests () {
    for file in *Test.py; do
        # TORCHCOMM_TEST_SINGLE_PARENT marker means the test manages its own
        # subprocesses and must be launched as a single process.
        if grep -q "TORCHCOMM_TEST_SINGLE_PARENT" "$file"; then
            nproc=1
        else
            nproc="$NUM_GPUS"
        fi
        torchrun --nnodes 1 --nproc_per_node "$nproc" "$file" --verbose
    done
}

# NCCL
export TEST_BACKEND=nccl
run_tests

# NCCLX
export TEST_BACKEND=ncclx
run_tests

# Gloo with CPU
export TEST_BACKEND=gloo
export TEST_DEVICE=cpu
export CUDA_VISIBLE_DEVICES=""
run_tests
unset TEST_DEVICE
unset CUDA_VISIBLE_DEVICES

# Gloo with CUDA
export TEST_BACKEND=gloo
export TEST_DEVICE=cuda
run_tests
unset TEST_DEVICE
