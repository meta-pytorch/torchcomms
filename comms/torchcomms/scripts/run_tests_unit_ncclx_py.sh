#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Runs the Python unit tests against the standalone torchcomms-ncclx wheel.
# Requires both torchcomms and torchcomms-ncclx to be installed.

set -ex

# Sanity check that the ncclx backend was installed.
python -c "import torchcomms_ncclx; import torchcomms; assert torchcomms.is_backend_built('ncclx'), 'ncclx backend not registered'"

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

export TEST_BACKEND=ncclx
run_tests
