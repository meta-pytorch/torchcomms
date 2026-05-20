#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.

set -ex

TORCHCOMMS_ROOT="comms/torchcomms"

collect_unit_test_dirs () {
    find "$TORCHCOMMS_ROOT" -path '*/tests/py' -type d \
        ! -path '*/tests/integration/*' \
        ! -path '*/rccl/*' \
        ! -path '*/rcclx/*' \
        ! -path '*/fb/*'
    find "$TORCHCOMMS_ROOT" -path '*/tests/unit/py' -type d \
        ! -path '*/rccl/*' \
        ! -path '*/rcclx/*' \
        ! -path '*/fb/*'
}

UNIT_TEST_DIRS=$(collect_unit_test_dirs | sort -u)

run_tests () {
    for dir in $UNIT_TEST_DIRS; do
        if [[ "$dir" == *transport* ]] && { [ "${USE_TRANSPORT}" = "0" ] || [ "${USE_TRANSPORT}" = "OFF" ]; }; then
            echo "Skipping $dir (USE_TRANSPORT=${USE_TRANSPORT})"
            continue
        fi
        if find "$dir" -maxdepth 1 -name 'test_*.py' -print -quit | read -r; then
            pytest -v "$dir"
        fi
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
