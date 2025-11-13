#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Simple example demonstrating synchronous send/recv communication using TorchComm.

This example shows how to perform point-to-point communication between ranks
in a synchronous manner, where operations block until completion.
"""

import torch
from torchcomms import new_comm


def main():
    # Initialize TorchComm
    device = torch.device("hip")
    torchcomm = new_comm("rcclx", device, name="main_comm")

    # Get rank and world size
    rank = torchcomm.get_rank()
    world_size = torchcomm.get_size()

    # Calculate device ID based on rank and number of available devices
    num_devices = torch.cuda.device_count()
    device_id = rank % num_devices
    target_device = torch.device(f"cuda:{device_id}")

    print(
        f"Rank {rank}/{world_size}: Starting synchronous send/recv example on device {device_id}"
    )

    # Create a tensor with rank-specific data
    tensor_size = 1024
    send_tensor = torch.full(
        (tensor_size,), float(rank), dtype=torch.float32, device=target_device
    )
    recv_tensor = torch.zeros(tensor_size, dtype=torch.float32, device=target_device)

    # Calculate send and receive ranks (ring topology)
    send_rank = (rank + 1) % world_size
    recv_rank = (rank - 1 + world_size) % world_size

    print(f"Rank {rank}: Sending to rank {send_rank}, receiving from rank {recv_rank}")

    # Perform synchronous send/recv operations
    # Use alternating pattern to avoid deadlock
    if rank % 2 == 0:
        # Even ranks: send first, then receive
        torchcomm.send(send_tensor, send_rank, async_op=False)
        torchcomm.recv(recv_tensor, recv_rank, async_op=False)
    else:
        # Odd ranks: receive first, then send
        torchcomm.recv(recv_tensor, recv_rank, async_op=False)
        torchcomm.send(send_tensor, send_rank, async_op=False)

    # Synchronize CUDA stream
    torch.cuda.current_stream().synchronize()

    torchcomm.finalize()
    print(f"Rank {rank}: Synchronous send/recv example completed")


if __name__ == "__main__":
    main()
