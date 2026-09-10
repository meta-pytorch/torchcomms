# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-unsafe

import torch
from torchcomms.tests.perf.py.perf_test_helpers import (
    BenchRecord,
    CommAdapter,
    create_tensor,
    run_bench_sweep,
)


def run_all_to_all_single_perf(
    comm: CommAdapter,
    record: BenchRecord,
    device: torch.device,
) -> None:
    def setup(num_elements, rank, num_ranks, device, dtype):
        input_tensor = create_tensor(num_elements * num_ranks, rank, device, dtype)
        output_tensor = create_tensor(num_elements * num_ranks, rank, device, dtype)
        return (output_tensor, input_tensor), {}

    def bus_bw(num_elements, element_size, num_ranks, avg_time_us):
        # AllToAllSingle: (n-1) / n * total_size / time
        total_size = num_elements * num_ranks * element_size
        algo_bw = total_size / avg_time_us
        return algo_bw * (num_ranks - 1) / num_ranks / 1000.0

    run_bench_sweep(comm, record, device, "all_to_all_single", setup, bus_bw)
