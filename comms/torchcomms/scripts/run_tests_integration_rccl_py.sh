#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.

# DO NOT DELETE
# This script runs the Python integration tests.
# This is used as part of the GitHub CI.

set -ex

cd "$(dirname "$0")/../tests/integration/py"

run_tests () {
    HSA_NO_SCRATCH_RECLAIM=1 torchrun --nnodes 1 --nproc_per_node 4 -m pytest -v -s AllGatherTest.py
    HSA_NO_SCRATCH_RECLAIM=1 torchrun --nnodes 1 --nproc_per_node 4 -m pytest -v -s AllGatherSingleTest.py
    HSA_NO_SCRATCH_RECLAIM=1 torchrun --nnodes 1 --nproc_per_node 4 -m pytest -v -s AllToAllTest.py
    HSA_NO_SCRATCH_RECLAIM=1 torchrun --nnodes 1 --nproc_per_node 4 -m pytest -v -s AllReduceTest.py
    HSA_NO_SCRATCH_RECLAIM=1 torchrun --nnodes 1 --nproc_per_node 4 -m pytest -v -s BarrierTest.py
    HSA_NO_SCRATCH_RECLAIM=1 torchrun --nnodes 1 --nproc_per_node 4 -m pytest -v -s BroadcastTest.py
    HSA_NO_SCRATCH_RECLAIM=1 torchrun --nnodes 1 --nproc_per_node 4 -m pytest -v -s BatchSendRecvTest.py
    HSA_NO_SCRATCH_RECLAIM=1 torchrun --nnodes 1 --nproc_per_node 4 -m pytest -v -s ReduceScatterTest.py
    HSA_NO_SCRATCH_RECLAIM=1 torchrun --nnodes 1 --nproc_per_node 4 -m pytest -v -s ReduceTest.py
    HSA_NO_SCRATCH_RECLAIM=1 torchrun --nnodes 1 --nproc_per_node 4 -m pytest -v -s SendRecvTest.py
    HSA_NO_SCRATCH_RECLAIM=1 torchrun --nnodes 1 --nproc_per_node 4 -m pytest -v -s ScatterTest.py
    HSA_NO_SCRATCH_RECLAIM=1 torchrun --nnodes 1 --nproc_per_node 4 -m pytest -v -s SplitTest.py
}

# RCCL
export TEST_BACKEND=rccl
run_tests
