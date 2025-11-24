#!/usr/bin/env python3
# pyre-strict
# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Simple example demonstrating synchronous AllReduce communication using TorchComm.

This example shows how to perform collective AllReduce operations between all ranks
in a synchronous manner, where the operation blocks until completion.
"""

import torch
from torchcomms import new_comm, ReduceOp


def main() -> None:
    # Initialize TorchComm
    device = torch.device("cuda")
    torchcomm = new_comm("ncclx", device, name="main_comm")

    # Get rank and world size
    rank = torchcomm.get_rank()
    world_size = torchcomm.get_size()

    # Calculate device ID based on rank and number of available devices
    num_devices = torch.cuda.device_count()
    device_id = rank % num_devices
    target_device = torch.device(f"cuda:{device_id}")

    print(
        f"Rank {rank}/{world_size}: Starting synchronous AllReduce example on device {device_id}"
    )

    # Create a tensor with rank-specific data
    tensor_size = 1024
    tensor = torch.full(
        (tensor_size,), float(rank + 1), dtype=torch.float32, device=target_device
    )

    print(f"Rank {rank}: Initial tensor value: {tensor[0].item()}")

    # Perform synchronous AllReduce operation (sum reduction)
    torchcomm.all_reduce(tensor, ReduceOp.SUM, async_op=False)

    # Synchronize CUDA stream
    torch.cuda.current_stream().synchronize()

    torchcomm.finalize()
    print(f"Rank {rank}: Synchronous AllReduce example completed")


if __name__ == "__main__":
    main()
