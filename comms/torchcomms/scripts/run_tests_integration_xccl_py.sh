#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.

# DO NOT DELETE
# This script runs the Python integration tests.
# This is used as part of the GitHub CI.

set -ex

cd "$(dirname "$0")/../tests/integration/py"

NPROC_PER_NODE=${NPROC_PER_NODE:-4}

run_tests () {
    local tests=(
        AllGatherSingleTest.py
        AllGatherTest.py
        AllGatherVTest.py
        AllReduceTest.py
        AllToAllSingleTest.py
        AllToAllTest.py
        AllToAllvSingleTest.py
        BackendWrapperAllGatherAliasedTest.py
        BackendWrapperCoalescingTest.py
        BackendWrapperShutdownTest.py
        BarrierTest.py
        BatchSendRecvTest.py
        BroadcastTest.py
        C10dBatchIsendIrecvTest.py
        C10dTorchCommTest.py
        DDPCommTest.py
        DeviceMeshTest.py
        DPTPCommTest.py
        FinalizeWarningTest.py
        FSDPCommTest.py
        FullgraphCompileAutogradTest.py
        FullgraphCompileTest.py
        GatherTest.py
        MemPoolTest.py
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
        NodeRankLayoutTest.py
        TPCommTest.py
        WaitBlockingTest.py
    )

    for test_file in "${tests[@]}"; do
        torchrun --nnodes 1 --nproc_per_node "${NPROC_PER_NODE}" -m pytest -v -s "$test_file"
    done

    cd -
    cd "$(dirname "$0")/../hooks/fr/tests/py"
    torchrun --nnodes 1 --nproc_per_node "${NPROC_PER_NODE}" -m pytest -v -s FlightRecorderFinalizeOpIdTest.py
    torchrun --nnodes 1 --nproc_per_node "${NPROC_PER_NODE}" -m pytest -v -s FlightRecorderTest.py
}

# XCCL
export TEST_BACKEND="xccl"
export TEST_DEVICE="xpu"
run_tests
