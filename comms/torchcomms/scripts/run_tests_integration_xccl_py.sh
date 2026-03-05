#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.

# DO NOT DELETE
# This script runs the Python integration tests.
# This is used as part of the GitHub CI.

set -ex

cd "$(dirname "$0")/../tests/integration/py"

NPROC_PER_NODE=${NPROC_PER_NODE:-4}

# Set ZE_AFFINITY_MASK to limit visible devices to match nproc_per_node
# e.g. NPROC_PER_NODE=4 → ZE_AFFINITY_MASK="0,1,2,3"
if [[ -z "${ZE_AFFINITY_MASK}" ]]; then
    MASK=$(seq -s ',' 0 $((NPROC_PER_NODE - 1)))
    export ZE_AFFINITY_MASK="${MASK}"
    echo "Setting ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK}"
fi

run_tests () {
    local tests=(
        AllGatherSingleTest.py
        AllGatherTest.py
        AllGatherVTest.py
        AllReduceTest.py
        AllToAllSingleTest.py
        AllToAllTest.py
        AllToAllvSingleTest.py
        BarrierTest.py
        BatchSendRecvTest.py
        BroadcastTest.py
        DDPCommTest.py
        DeviceMeshTest.py
        FSDPCommTest.py
        GatherTest.py
        MultiCommTest.py
        ObjColTest.py
        OptionsTest.py
        ReduceScatterSingleTest.py
        ReduceScatterTest.py
        ReduceScatterVTest.py
        ReduceTest.py
        SendRecvTest.py
        ScatterTest.py
        SplitTest.py
        TPCommTest.py
    )

    for test_file in "${tests[@]}"; do
        torchrun --nnodes 1 --nproc_per_node "${NPROC_PER_NODE}" "$test_file" --verbose
    done
}

# XCCL
export TEST_BACKEND=xccl
export TEST_DEVICE="xpu"
run_tests
