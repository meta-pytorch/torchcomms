#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Simple example demonstrating asynchronous AllReduce communication using TorchComm.

This example shows how to perform collective AllReduce operations between all ranks
in an asynchronous manner, where the operation returns immediately and can be
waited on later for completion.
"""

import torch
from torchcomms import new_comm, ReduceOp


def main():
    # Initialize TorchComm
    device = torch.device("hip")
    torchcomm = new_comm("rccl", device, name="main_comm")

    # Get rank and world size
    rank = torchcomm.get_rank()
    world_size = torchcomm.get_size()

    # Calculate device ID based on rank and number of available devices
    num_devices = torch.cuda.device_count()
    device_id = rank % num_devices
    target_device = torch.device(f"cuda:{device_id}")

    print(
        f"Rank {rank}/{world_size}: Starting asynchronous AllReduce example on device {device_id}"
    )

    # Create a tensor with rank-specific data
    tensor_size = 1024
    tensor = torch.full(
        (tensor_size,), float(rank + 1), dtype=torch.float32, device=target_device
    )

    print(f"Rank {rank}: Initial tensor value: {tensor[0].item()}")

    # Perform asynchronous AllReduce operation (sum reduction)
    work = torchcomm.all_reduce(tensor, ReduceOp.SUM, async_op=True)

    # Do some other work while AllReduce is in progress
    print(f"Rank {rank}: AllReduce operation started, doing other work...")

    # Wait for the AllReduce operation to complete
    work.wait()
    print(f"Rank {rank}: AllReduce operation completed")

    # Synchronize HIP stream
    torch.cuda.current_stream().synchronize()

    torchcomm.finalize()
    print(f"Rank {rank}: Asynchronous AllReduce example completed")

    print("Allreduce output: ", tensor[0].item())

    # print correctness
    if rank == 0:
        if tensor[0].item() == 36.0:
            print("Allreduce output is correct")
        else:
            print("Allreduce output is incorrect")


if __name__ == "__main__":
    main()
