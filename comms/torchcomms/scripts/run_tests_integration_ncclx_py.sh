#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Runs the Python integration tests against the standalone torchcomms-ncclx
# wheel. Requires both torchcomms and torchcomms-ncclx to be installed.

set -ex

# Sanity check that the ncclx backend was installed.
python -c "import torchcomms_ncclx; import torchcomms; assert torchcomms.is_backend_built('ncclx'), 'ncclx backend not registered'"

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

export TEST_BACKEND=ncclx
run_tests
# TODO(d4l3k): reenable once NCCLX-specific integration tests are passing.
# Failed to initialize NCCL communicator: internal error
#run_tests "$NCCLX_INTEGRATION_TEST_DIRS"
