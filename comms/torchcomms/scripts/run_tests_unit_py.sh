#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.

set -ex

run_test () {
    pytest -v comms/torchcomms/tests/unit/py
}

# NCCL
export TEST_BACKEND=nccl
run_test

# NCCLX
export TEST_BACKEND=ncclx
run_test

# Gloo with CPU
export TEST_BACKEND=gloo
export TEST_DEVICE=cpu
export CUDA_VISIBLE_DEVICES=""
run_test
unset TEST_DEVICE
unset CUDA_VISIBLE_DEVICES

# Gloo with CUDA
export TEST_BACKEND=gloo
export TEST_DEVICE=cuda
run_test
unset TEST_DEVICE
