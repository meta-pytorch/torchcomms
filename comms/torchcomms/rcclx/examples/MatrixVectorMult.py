#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Simple example demonstrating distributed matrix-vector multiplication using TorchComm.

This example shows how to perform distributed matrix-vector multiplication across
multiple ranks, where each rank holds a portion of the matrix and the vector
is replicated across all ranks. The result vector is computed and gathered
across all ranks.
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
        f"Rank {rank}/{world_size}: Starting distributed matrix-vector multiplication on device {device_id}"
    )

    # Matrix and vector dimensions
    matrix_rows = 128
    matrix_cols = 64  # Number of matrix columns = Size of input vector

    # In our build file, we have 8 GPUs and 1 Host, so each rank will get 16 rows
    rows_per_rank = matrix_rows // world_size

    if rank == 0:
        print(
            f"Rank {rank}: Matrix dimensions: {matrix_rows}x{matrix_cols}, Rows per rank: {rows_per_rank}"
        )

    # Create local portion of the matrix (each rank gets a set of rows)
    # Fill the entire matrix with 1s
    local_matrix = torch.ones(
        rows_per_rank, matrix_cols, dtype=torch.float32, device=target_device
    )
    # Alternative: Use random values (uncomment the line below and comment the line above)
    # local_matrix = torch.randn(rows_per_rank, matrix_cols, dtype=torch.float32, device=target_device)

    # Create the input vector (replicated across all ranks)
    # Fill the vector with 3s, only rank 0 creates the vector, others will receive it via broadcast
    if rank == 0:
        input_vector = torch.full(
            (matrix_cols,), 3.0, dtype=torch.float32, device=target_device
        )
        # Alternative: Use random values (uncomment the line below and comment the line above)
        # input_vector = torch.randn(matrix_cols, dtype=torch.float32, device=target_device)
    else:
        input_vector = torch.zeros(
            matrix_cols, dtype=torch.float32, device=target_device
        )

    # Broadcast the vector from rank 0 to ensure all ranks have the same vector
    if rank == 0:
        print(f"Rank {rank}: Broadcasting input vector to all ranks")
    torchcomm.broadcast(input_vector, 0, async_op=False)

    # Perform local matrix-vector multiplication
    # With matrix of all 1s and vector of all 3s, each element of result should be 3 * matrix_cols
    local_result = torch.matmul(local_matrix, input_vector)

    # Gather all local results to get the complete result vector
    # Each rank will have a portion of the final result
    result_list = [
        torch.zeros(rows_per_rank, dtype=torch.float32, device=target_device)
        for _ in range(world_size)
    ]

    # AllGather to collect all partial results
    torchcomm.all_gather(result_list, local_result, async_op=False)

    # Concatenate all results to form the complete result vector
    global_result = torch.cat(result_list, dim=0)
    print(f"Rank {rank}: Global result shape: {global_result.shape}")
    print(
        f"Rank {rank}: Global result first few elements: {global_result[:3].tolist()}"
    )

    # Verify correctness by running the above computation on a single rank
    if rank == 0:
        print(f"Rank {rank}: Verifying distributed computation...")

        # Gather all matrix portions to rank 0 for verification
        matrix_list = [
            torch.zeros(
                rows_per_rank, matrix_cols, dtype=torch.float32, device=target_device
            )
            for _ in range(world_size)
        ]

        # Rank 0 keeps its own matrix and receives rest of the portions from others
        matrix_list[0] = local_matrix.clone()
        for src_rank in range(1, world_size):
            recv_matrix = torch.zeros(
                rows_per_rank, matrix_cols, dtype=torch.float32, device=target_device
            )
            torchcomm.recv(recv_matrix, src_rank, async_op=False)
            matrix_list[src_rank] = recv_matrix

        # Reconstruct full matrix and compute reference result
        full_matrix = torch.cat(matrix_list, dim=0)
        reference_result = torch.matmul(full_matrix, input_vector)

        # Check if results match element by element
        diff_tensor = torch.abs(global_result - reference_result)
        max_diff = torch.max(diff_tensor).item()
        has_large_diff = torch.any(diff_tensor > 1e-5)

        print(
            f"Rank {rank}: Maximum difference between distributed and reference: {max_diff:.6f}"
        )

        if not has_large_diff:
            print(f"Rank {rank}: Distributed matrix-vector multiplication is correct!")
        else:
            print(f"Rank {rank}: Warning: Results differ more than expected")
            # For debugging, print first few elements
            print(
                f"Rank {rank}: Reference result first few elements: {reference_result[:5].tolist()}"
            )
            print(
                f"Rank {rank}: Global result first few elements: {global_result[:5].tolist()}"
            )
    else:
        # Send local matrix to rank 0 for verification
        torchcomm.send(local_matrix, 0, async_op=False)

    # Synchronize HIP stream
    torch.cuda.current_stream().synchronize()

    torchcomm.finalize()
    print(f"Rank {rank}: Distributed matrix-vector multiplication example completed")


if __name__ == "__main__":
    main()
