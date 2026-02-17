#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Flight Recorder Verification Script

This script demonstrates how to use the FlightRecorderHook with torchcomms
to record collective operations and dump traces for debugging.

The traces can be analyzed using the PyTorch flight_recorder trace analyzer:
    python -m torch.distributed.flight_recorder.fr_trace <trace_dir>

Usage:
    # Set the dump directory (traces will be written as <prefix><rank>)
    export TORCHCOMM_FR_DUMP_TEMP_FILE=/tmp/flight_recorder_traces/rank_

    torchrun --nproc_per_node=2 verify_flight_recorder.py

    Or with a specific backend:
    TEST_BACKEND=nccl torchrun --nproc_per_node=2 verify_flight_recorder.py
"""

import os
from datetime import timedelta

import torch
from torchcomms import new_comm, ReduceOp
from torchcomms.hooks import FlightRecorderHook


def main() -> None:
    # Get backend from environment or default to gloo
    backend = os.environ.get("TEST_BACKEND", "gloo")
    device = torch.device(os.environ.get("TEST_DEVICE", "cuda"))

    # Initialize TorchComm
    comm = new_comm(
        backend=backend,
        device=device,
        name="main_comm",
        timeout=timedelta(seconds=300),
    )

    rank = comm.get_rank()
    world_size = comm.get_size()

    # Calculate device ID
    num_devices = torch.cuda.device_count()
    device_id = rank % num_devices
    target_device = torch.device(f"cuda:{device_id}")

    print(f"Rank {rank}/{world_size}: Running on device {device_id}")

    # Create FlightRecorderHook
    recorder = FlightRecorderHook(max_entries=100)
    recorder.register_with_comm(comm)

    # Create a tensor with rank-specific data
    tensor = torch.full(
        (1024,),
        float(rank + 1),
        dtype=torch.float32,
        device=target_device,
    )

    print(f"Rank {rank}: Before AllReduce: {tensor[0].item()}")

    # Perform multiple collective operations
    for _ in range(5):
        comm.all_reduce(tensor, ReduceOp.SUM, async_op=False)

    # Broadcast from rank 0
    comm.broadcast(tensor, root=0, async_op=False)

    # Synchronize CUDA stream
    torch.cuda.current_stream().synchronize()

    print(f"Rank {rank}: After AllReduce: {tensor[0].item()}")

    # Dump traces using dump_file API
    # The output location is controlled by TORCHCOMM_FR_DUMP_TEMP_FILE env var
    # Files are written as <TORCHCOMM_FR_DUMP_TEMP_FILE><rank>
    recorder.dump_file(rank)

    dump_prefix = os.environ.get(
        "TORCHCOMM_FR_DUMP_TEMP_FILE", "~/.cache/torchcomm_fr_trace_"
    )
    trace_file = f"{dump_prefix}{rank}"
    print(f"Rank {rank}: Flight recorder trace dumped to {trace_file}")

    # Cleanup
    recorder.unregister()
    comm.finalize()

    if rank == 0:
        dump_dir = os.path.dirname(dump_prefix) or "."
        print(f"\n=== Flight Recorder traces saved to: {dump_dir} ===")
        print("To analyze the traces, run:")
        print(f"  python -m torch.distributed.flight_recorder.fr_trace -j {dump_dir}")


if __name__ == "__main__":
    main()
