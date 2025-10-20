#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.

# DO NOT DELETE
# This script runs the Python integration tests.
# This is used as part of the GitHub CI.

set -ex

cd "$(dirname "$0")/../tests/integration/py"

run_tests () {
    HSA_NO_SCRATCH_RECLAIM=1 torchrun --nnodes 1 --nproc_per_node 4 AllGatherTest.py --verbose
    HSA_NO_SCRATCH_RECLAIM=1 torchrun --nnodes 1 --nproc_per_node 4 AllGatherSingleTest.py --verbose
    HSA_NO_SCRATCH_RECLAIM=1 torchrun --nnodes 1 --nproc_per_node 4 AllToAllTest.py --verbose
    HSA_NO_SCRATCH_RECLAIM=1 torchrun --nnodes 1 --nproc_per_node 4 AllReduceTest.py --verbose
    HSA_NO_SCRATCH_RECLAIM=1 torchrun --nnodes 1 --nproc_per_node 4 BarrierTest.py --verbose
    HSA_NO_SCRATCH_RECLAIM=1 torchrun --nnodes 1 --nproc_per_node 4 BroadcastTest.py --verbose
    HSA_NO_SCRATCH_RECLAIM=1 torchrun --nnodes 1 --nproc_per_node 4 BatchSendRecvTest.py --verbose
    HSA_NO_SCRATCH_RECLAIM=1 torchrun --nnodes 1 --nproc_per_node 4 ReduceScatterTest.py --verbose
    HSA_NO_SCRATCH_RECLAIM=1 torchrun --nnodes 1 --nproc_per_node 4 ReduceTest.py --verbose
    HSA_NO_SCRATCH_RECLAIM=1 torchrun --nnodes 1 --nproc_per_node 4 SendRecvTest.py --verbose
    HSA_NO_SCRATCH_RECLAIM=1 torchrun --nnodes 1 --nproc_per_node 4 ScatterTest.py --verbose
    HSA_NO_SCRATCH_RECLAIM=1 torchrun --nnodes 1 --nproc_per_node 4 SplitTest.py --verbose
}

# RCCL
export TEST_BACKEND=rccl
run_tests
