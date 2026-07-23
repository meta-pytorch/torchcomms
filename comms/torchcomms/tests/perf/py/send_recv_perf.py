# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-unsafe

import torch
from torchcomms.tests.perf.py.perf_test_helpers import (
    BenchRecord,
    CommAdapter,
    create_tensor,
    log_perf_header,
    log_perf_result,
    PerfTimer,
    sync_device,
)


def _do_ping_pong(
    send_call,
    send_args,
    send_kwargs,
    recv_call,
    recv_args,
    recv_kwargs,
    rank: int,
    peer: int,
) -> None:
    """Execute one ping-pong iteration between rank and peer."""
    if rank < peer:
        send_call(*send_args, **send_kwargs)
        recv_call(*recv_args, **recv_kwargs)
    else:
        recv_call(*recv_args, **recv_kwargs)
        send_call(*send_args, **send_kwargs)


def run_send_recv_perf(
    comm: CommAdapter,
    record: BenchRecord,
    device: torch.device,
) -> None:
    rank = comm.get_rank()
    num_ranks = comm.get_size()
    params = record.params
    config = record.config

    if num_ranks < 2:
        if rank == 0:
            print("SendRecv test requires at least 2 ranks, skipping")
        return
    if num_ranks % 2 != 0:
        if rank == 0:
            print("SendRecv test requires an even number of ranks, skipping")
        return

    # Pair adjacent ranks (0<->1, 2<->3, …) so all pairs run concurrently, matching
    # the nccl-tests / OSU send-recv benchmark patterns:
    #   XOR with 1 flips the least significant bit, giving each rank its peer.
    #   With an odd rank count the last rank computes a peer >= num_ranks so we skip it.
    peer = rank ^ 1
    if peer >= num_ranks:
        return
    element_size = torch.tensor([], dtype=params.dtype).element_size()

    msg_size = config.min_size
    while msg_size <= config.max_size:
        num_elements = msg_size // element_size
        if num_elements == 0:
            num_elements = 1
        tensor = create_tensor(num_elements, rank, device, params.dtype)

        # Pre-resolve send/recv once so the timing loop avoids per-call dispatch.
        send_call, send_args, send_kwargs = comm.resolve_collective(
            "send", tensor, peer, async_op=params.async_op
        )
        recv_call, recv_args, recv_kwargs = comm.resolve_collective(
            "recv", tensor, peer, async_op=params.async_op
        )

        # Warmup
        for _ in range(params.warmup_iterations):
            _do_ping_pong(
                send_call,
                send_args,
                send_kwargs,
                recv_call,
                recv_args,
                recv_kwargs,
                rank,
                peer,
            )

        comm.run_collective("barrier", async_op=False)

        # Measure ping-pong latency
        timer = PerfTimer()
        sync_device()
        timer.start()

        for i in range(params.measure_iterations):
            _do_ping_pong(
                send_call,
                send_args,
                send_kwargs,
                recv_call,
                recv_args,
                recv_kwargs,
                rank,
                peer,
            )

            if params.sync_interval > 0 and (i + 1) % params.sync_interval == 0:
                sync_device()

        sync_device()
        timer.stop()

        total = timer.elapsed_us()
        avg_time = total / params.measure_iterations

        # SendRecv bus bandwidth: size / one-way time
        one_way_time = avg_time / 2
        algo_bw = (num_elements * element_size) / one_way_time  # bytes/us = MB/s
        bus_bw_gbps = algo_bw / 1000.0  # Convert to GB/s

        record.metrics.message_size_bytes = num_elements * element_size
        record.metrics.total_time_us = total
        record.metrics.avg_time_us = avg_time / 2
        record.metrics.min_time_us = avg_time / 2
        record.metrics.max_time_us = avg_time / 2
        record.metrics.bus_bw_gbps = bus_bw_gbps

        log_perf_result(record, rank)

        msg_size *= config.size_scaling_factor
