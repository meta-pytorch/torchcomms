#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# DO NOT DELETE — used in GitHub CI.
#
# Runs the torchcomms Python integration tests against one or more backends.
#
# Set TEST_BACKENDS to a comma-separated list of backends to exercise.
# Default: nccl,gloo (the core wheel). Use TEST_BACKENDS=ncclx after
# installing the standalone torchcomms-ncclx wheel.

set -ex

TEST_BACKENDS=${TEST_BACKENDS:-nccl,gloo}

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
            torchrun --nnodes 1 --nproc_per_node 4 -m pytest -v -s "$file"
        done
    done
}

run_for_backend () {
    local backend="$1"
    case "$backend" in
        nccl)
            export TEST_BACKEND=nccl
            run_tests
            unset TEST_BACKEND
            ;;
        ncclx)
            python -c "import torchcomms, torchcomms_ncclx; assert torchcomms.is_backend_built('ncclx'), 'ncclx backend not registered'"
            export TEST_BACKEND=ncclx
            run_tests
            # TODO(d4l3k): reenable ncclx-specific integration tests once they
            # are passing. Failed to initialize NCCL communicator: internal error
            # run_tests "$NCCLX_INTEGRATION_TEST_DIRS"
            unset TEST_BACKEND
            ;;
        gloo)
            export TEST_BACKEND=gloo
            export TEST_DEVICE=cpu
            export CUDA_VISIBLE_DEVICES=""
            run_tests
            unset TEST_DEVICE CUDA_VISIBLE_DEVICES

            export TEST_DEVICE=cuda
            run_tests
            unset TEST_BACKEND TEST_DEVICE
            ;;
        *)
            echo "Unknown backend in TEST_BACKENDS: $backend" >&2
            exit 1
            ;;
    esac
}

IFS=',' read -ra _backends <<< "$TEST_BACKENDS"
for backend in "${_backends[@]}"; do
    run_for_backend "$backend"
done
