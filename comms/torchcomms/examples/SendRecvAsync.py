#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Simple example demonstrating asynchronous send/recv communication using TorchComm.

This example shows how to perform point-to-point communication between ranks
in an asynchronous manner, where operations return immediately and can be
waited on later for completion.
"""

import torch
from torchcomms import new_comm


def main():
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
        f"Rank {rank}/{world_size}: Starting asynchronous send/recv example on device {device_id}"
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

    # Perform asynchronous send/recv operations
    # Use alternating pattern to avoid deadlock
    send_work = None
    recv_work = None

    if rank % 2 == 0:
        # Even ranks: send first, then receive
        send_work = torchcomm.send(send_tensor, send_rank, async_op=True)
        recv_work = torchcomm.recv(recv_tensor, recv_rank, async_op=True)
    else:
        # Odd ranks: receive first, then send
        recv_work = torchcomm.recv(recv_tensor, recv_rank, async_op=True)
        send_work = torchcomm.send(send_tensor, send_rank, async_op=True)

    # Wait for operations to complete
    if send_work is not None:
        send_work.wait()
        print(f"Rank {rank}: Send operation completed")

    if recv_work is not None:
        recv_work.wait()
        print(f"Rank {rank}: Receive operation completed")

    # Synchronize CUDA stream
    torch.cuda.current_stream().synchronize()

    torchcomm.finalize()
    print(f"Rank {rank}: Asynchronous send/recv example completed")


if __name__ == "__main__":
    main()
