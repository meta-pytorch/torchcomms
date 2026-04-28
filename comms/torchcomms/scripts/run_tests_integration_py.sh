#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.

# DO NOT DELETE
# This script runs the Python integration tests.
# This is used as part of the GitHub CI.

set -ex

# If no InfiniBand devices are present, disable the IB backend to avoid
# CtranIbSingleton init failures ("Operation not permitted") and cascading
# CUDA graph registration errors.
if [ ! -d /sys/class/infiniband ] || [ -z "$(ls /sys/class/infiniband 2>/dev/null)" ]; then
    echo "No InfiniBand devices found, disabling IB backend"
    export NCCL_CTRAN_BACKENDS="socket"
fi

TORCHCOMMS_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

INTEGRATION_TEST_DIRS=$(find "$TORCHCOMMS_ROOT" -path '*/tests/integration/py' -type d \
    ! -path '*/ncclx/*' \
    ! -path '*/rccl/*' \
    ! -path '*/rcclx/*' \
    ! -path '*/fb/*' | sort -u)

NCCLX_INTEGRATION_TEST_DIRS=$(find "$TORCHCOMMS_ROOT" -path '*/ncclx/tests/integration/py' -type d \
    ! -path '*/fb/*' | sort -u)

run_tests () {
    local dirs="${1:-$INTEGRATION_TEST_DIRS}"
    for dir in $dirs; do
        for file in "$dir"/*Test.py; do
            [ -f "$file" ] || continue
            torchrun --nnodes 1 --nproc_per_node gpu --no-python pytest -s -v "$file"
        done
    done
}

# NCCL
export TEST_BACKEND=nccl
run_tests

# NCCLX (skip if built with USE_NCCLX=0)
if [ "${USE_NCCLX}" != "0" ] && [ "${USE_NCCLX}" != "OFF" ]; then
    export TEST_BACKEND=ncclx
    run_tests
    # TODO(d4l3k): reenable once NCCLX tests are passing
    # Failed to initialize NCCL communicator: internal error
    #run_tests "$NCCLX_INTEGRATION_TEST_DIRS"
else
    echo "Skipping ncclx tests (USE_NCCLX=${USE_NCCLX})"
fi

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
