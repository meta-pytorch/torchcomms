#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Runs the torchcomms Python unit tests against one or more backends.
#
# Set TEST_BACKENDS to a comma-separated list of backends to exercise.
# Default: nccl,gloo (the core wheel). Use TEST_BACKENDS=ncclx after
# installing the standalone torchcomms-ncclx wheel.

set -ex

TEST_BACKENDS=${TEST_BACKENDS:-nccl,gloo}

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
            unset TEST_BACKEND
            ;;
        gloo)
            # CPU then CUDA
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
