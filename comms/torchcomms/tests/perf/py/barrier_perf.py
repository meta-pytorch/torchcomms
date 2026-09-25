# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-unsafe

import torch
from torchcomms.tests.perf.py.perf_test_helpers import (
    BenchRecord,
    CommAdapter,
    log_perf_header,
    log_perf_result,
    PerfTimer,
    sync_device,
)


def run_barrier_perf(
    comm: CommAdapter,
    record: BenchRecord,
    device: torch.device,
) -> None:
    rank = comm.get_rank()
    num_ranks = comm.get_size()
    params = record.params
    config = record.config

    # Barrier has no message size, run once
    # Warmup
    for _ in range(params.warmup_iterations):
        comm.run_collective("barrier", async_op=params.async_op)

    # Synchronize all ranks before measurement
    comm.run_collective("barrier", async_op=False)

    # Measure
    timer = PerfTimer()
    sync_device()
    timer.start()

    for i in range(params.measure_iterations):
        comm.run_collective("barrier", async_op=params.async_op)

        if params.sync_interval > 0 and (i + 1) % params.sync_interval == 0:
            sync_device()

    sync_device()
    timer.stop()

    # Calculate statistics
    total = timer.elapsed_us()
    avg_time = total / params.measure_iterations

    # Barrier has no data transfer, so bus bandwidth is 0
    record.metrics.total_time_us = total
    record.metrics.avg_time_us = avg_time
    record.metrics.min_time_us = avg_time
    record.metrics.max_time_us = avg_time
    record.metrics.bus_bw_gbps = 0.0

    log_perf_result(record, rank)
