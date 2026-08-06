# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-unsafe

import torch
from torchcomms.tests.perf.py.perf_test_helpers import (
    BenchRecord,
    CommAdapter,
    create_tensor,
    run_bench_sweep,
)


def run_reduce_scatter_perf(
    comm: CommAdapter,
    record: BenchRecord,
    device: torch.device,
) -> None:
    op = comm.get_reduce_op(record.params.reduce_op)

    def setup(num_elements, rank, num_ranks, device, dtype):
        output_tensor = create_tensor(num_elements, rank, device, dtype)
        input_list = [
            create_tensor(num_elements, rank, device, dtype) for _ in range(num_ranks)
        ]
        return (output_tensor, input_list, op), {}

    def bus_bw(num_elements, element_size, num_ranks, avg_time_us):
        # ReduceScatter: (n-1) / n * total_size / time
        total_size = num_elements * num_ranks * element_size
        algo_bw = total_size / avg_time_us
        return algo_bw * (num_ranks - 1) / num_ranks / 1000.0

    run_bench_sweep(comm, record, device, "reduce_scatter", setup, bus_bw)
