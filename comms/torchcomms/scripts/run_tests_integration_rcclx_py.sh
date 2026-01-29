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
    HSA_NO_SCRATCH_RECLAIM=1 torchrun --nnodes 1 --nproc_per_node 4 AllReduceTest.py --verbose
    HSA_NO_SCRATCH_RECLAIM=1 torchrun --nnodes 1 --nproc_per_node 4 BarrierTest.py --verbose
    HSA_NO_SCRATCH_RECLAIM=1 torchrun --nnodes 1 --nproc_per_node 4 BroadcastTest.py --verbose
    HSA_NO_SCRATCH_RECLAIM=1 torchrun --nnodes 1 --nproc_per_node 4 BatchSendRecvTest.py --verbose
    HSA_NO_SCRATCH_RECLAIM=1 torchrun --nnodes 1 --nproc_per_node 4 ReduceScatterTest.py --verbose
    HSA_NO_SCRATCH_RECLAIM=1 torchrun --nnodes 1 --nproc_per_node 4 ReduceTest.py --verbose
    HSA_NO_SCRATCH_RECLAIM=1 torchrun --nnodes 1 --nproc_per_node 4 SendRecvTest.py --verbose
    HSA_NO_SCRATCH_RECLAIM=1 torchrun --nnodes 1 --nproc_per_node 4 SplitTest.py --verbose
}

# RCCL
# Use the locally built librccl.so instead of the PyTorch bundled one
# Set LD_LIBRARY_PATH first so torch can import successfully
export LD_LIBRARY_PATH=/tmp/torchcomms_build2/fbcode/comms/rcclx/develop/build/release/build/lib:/opt/rocm/lib:${LD_LIBRARY_PATH:-}

# Rename PyTorch's bundled librccl.so so LD_LIBRARY_PATH can find ours
TORCH_LIB_DIR=$(python -c "import torch; print(torch.__path__[0])")/lib
TORCH_RCCL_LIB="${TORCH_LIB_DIR}/librccl.so"
if [ -f "${TORCH_RCCL_LIB}" ]; then
    mv "${TORCH_RCCL_LIB}" "${TORCH_RCCL_LIB}.bak"
    echo "Renamed PyTorch's librccl.so to librccl.so.bak"
fi

export TEST_BACKEND=rcclx
run_tests
