# Copyright (c) Meta Platforms, Inc. and affiliates.
# Confidential and proprietary.
# pyre-unsafe
"""
Benchmark for Triton LL P2P send/recv over NVLink.

Sweeps message sizes from 8B to 256KB, measuring unidirectional and
bidirectional bandwidth (GB/s) and latency (us).

Usage:
    buck2 run @fbcode//mode/opt \
        -c hpc_comms.use_ncclx=stable \
        //comms/pipes/ll/triton/tests:benchmark_ll_p2p

    # With GPU offset (e.g., use GPUs 2 and 3):
    buck2 run ... :benchmark_ll_p2p -- --gpu-offset 2

    # Bidirectional mode:
    buck2 run ... :benchmark_ll_p2p -- --bidirectional
"""

import os
import socket
import sys

os.environ.setdefault("NCCL_GIN_ENABLE", "1")
os.environ.setdefault("NCCL_GIN_TYPE", "-1")

import torch
import torch.multiprocessing as mp
import torchcomms
from comms.pipes.ll.triton.ll_p2p_op import LlP2pOp


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def format_size(nbytes):
    if nbytes >= 1024 * 1024:
        return f"{nbytes / (1024**2):.0f}MB"
    if nbytes >= 1024:
        return f"{nbytes / 1024:.0f}KB"
    return f"{nbytes}B"


MSG_SIZES = [8, 64, 256, 1024, 4096, 16384, 65536, 131072, 262144]

WARMUP_ITERS = 20
TIMED_ITERS = 200


def _do_ops(op, local_rank, peer_rank, fused, bidirectional, src, dst, nbytes):
    if fused:
        op.sendrecv(peer=peer_rank, src_tensor=src, dst_tensor=dst, nbytes=nbytes)
    elif bidirectional:
        op.send(peer=peer_rank, src_tensor=src, nbytes=nbytes)
        op.recv(peer=peer_rank, dst_tensor=dst, nbytes=nbytes)
    else:
        if local_rank == 0:
            op.send(peer=peer_rank, src_tensor=src, nbytes=nbytes)
        else:
            op.recv(peer=peer_rank, dst_tensor=dst, nbytes=nbytes)


def _bench_eager(op, comm, local_rank, peer_rank, gpu_id, fused, bidirectional):
    results = {}
    for nbytes in MSG_SIZES:
        num_elements = nbytes // 4
        src = torch.randn(num_elements, dtype=torch.float32, device=f"cuda:{gpu_id}")
        dst = torch.zeros(num_elements, dtype=torch.float32, device=f"cuda:{gpu_id}")

        for _ in range(WARMUP_ITERS):
            _do_ops(op, local_rank, peer_rank, fused, bidirectional, src, dst, nbytes)
        torch.cuda.synchronize()
        comm.barrier(False)

        iters = TIMED_ITERS if nbytes <= 16384 else TIMED_ITERS // 2
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)

        comm.barrier(False)
        start_ev.record()
        for _ in range(iters):
            _do_ops(op, local_rank, peer_rank, fused, bidirectional, src, dst, nbytes)
        end_ev.record()
        torch.cuda.synchronize()

        elapsed_ms = start_ev.elapsed_time(end_ev)
        results[nbytes] = (elapsed_ms * 1000.0) / iters
        del src, dst
    return results


def _bench_graph(op, comm, local_rank, peer_rank, gpu_id, fused, bidirectional):
    results = {}
    for nbytes in MSG_SIZES:
        num_elements = nbytes // 4
        src = torch.randn(num_elements, dtype=torch.float32, device=f"cuda:{gpu_id}")
        dst = torch.zeros(num_elements, dtype=torch.float32, device=f"cuda:{gpu_id}")

        for _ in range(WARMUP_ITERS):
            _do_ops(op, local_rank, peer_rank, fused, bidirectional, src, dst, nbytes)
        torch.cuda.synchronize()
        comm.barrier(False)

        iters = TIMED_ITERS if nbytes <= 16384 else TIMED_ITERS // 2

        graph_stream = torch.cuda.Stream()
        with torch.cuda.stream(graph_stream):
            graph = torch.cuda.CUDAGraph()
            # pyre-fixme[6]: pool handle type
            with torch.cuda.graph(graph, pool=op.get_graph_pool_id()):  # type: ignore[arg-type]
                for _ in range(iters):
                    _do_ops(
                        op,
                        local_rank,
                        peer_rank,
                        fused,
                        bidirectional,
                        src,
                        dst,
                        nbytes,
                    )

        with torch.cuda.stream(graph_stream):
            for _ in range(WARMUP_ITERS):
                graph.replay()
        torch.cuda.synchronize()
        comm.barrier(False)

        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(graph_stream):
            start_ev.record()
            graph.replay()
            end_ev.record()
        torch.cuda.synchronize()

        elapsed_ms = start_ev.elapsed_time(end_ev)
        results[nbytes] = (elapsed_ms * 1000.0) / iters
        del src, dst, graph
    return results


def run_benchmark_worker(
    local_rank,
    master_port,
    bidirectional,
    fused,
    cuda_graph,
    gpu_offset,
):
    gpu_id = local_rank + gpu_offset
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(local_rank)
    os.environ["LOCAL_RANK"] = str(gpu_id)
    os.environ["WORLD_SIZE"] = "2"

    peer_rank = 1 - local_rank
    torch.cuda.set_device(gpu_id)

    comm = torchcomms.new_comm(
        "ncclx",
        torch.device(f"cuda:{gpu_id}"),
        name="ll_p2p_bench",
    )

    max_nbytes = max(MSG_SIZES)
    op = LlP2pOp(
        comm=comm,
        max_nbytes=max_nbytes,
        device=torch.device(f"cuda:{gpu_id}"),
    )
    op.setup()

    if fused:
        mode_str = "Fused SendRecv"
    elif bidirectional:
        mode_str = "Bidirectional"
    else:
        mode_str = "Unidirectional"

    for prewarm_nbytes in [8, 1024]:
        prewarm_elements = prewarm_nbytes // 4
        prewarm_src = torch.randn(
            prewarm_elements, dtype=torch.float32, device=f"cuda:{gpu_id}"
        )
        prewarm_dst = torch.zeros(
            prewarm_elements, dtype=torch.float32, device=f"cuda:{gpu_id}"
        )
        _do_ops(
            op,
            local_rank,
            peer_rank,
            fused,
            bidirectional,
            prewarm_src,
            prewarm_dst,
            prewarm_nbytes,
        )
        torch.cuda.synchronize()
        comm.barrier(False)
        del prewarm_src, prewarm_dst

    eager_results = _bench_eager(
        op,
        comm,
        local_rank,
        peer_rank,
        gpu_id,
        fused,
        bidirectional,
    )

    if cuda_graph:
        graph_results = _bench_graph(
            op,
            comm,
            local_rank,
            peer_rank,
            gpu_id,
            fused,
            bidirectional,
        )

    if local_rank == 0:
        if cuda_graph:
            print(
                f"Triton LL P2P Benchmark ({mode_str} + CUDA Graph, 2 GPUs)",
                flush=True,
            )
        else:
            print(f"Triton LL P2P Benchmark ({mode_str}, 2 GPUs)", flush=True)
        print(f"GPU {gpu_id} <-> GPU {gpu_id + 1 - 2 * local_rank}", flush=True)
        print(flush=True)

        if cuda_graph:
            print(
                f"{'Size':<10} | {'Eager (us)':<14} | {'Graph (us)':<14} | "
                f"{'Speedup':<8} | {'Iters':<6}",
                flush=True,
            )
            print("-" * 68, flush=True)
            for nbytes in MSG_SIZES:
                iters = TIMED_ITERS if nbytes <= 16384 else TIMED_ITERS // 2
                eager_us = eager_results[nbytes]
                graph_us = graph_results[nbytes]
                speedup = eager_us / graph_us if graph_us > 0 else 0.0
                print(
                    f"{format_size(nbytes):<10} | {eager_us:<14.2f} | "
                    f"{graph_us:<14.2f} | {speedup:<8.2f}x | {iters:<6}",
                    flush=True,
                )
        else:
            print(
                f"{'Size':<10} | {'Latency (us)':<14} | "
                f"{'BW (GB/s)':<12} | {'Iters':<6}",
                flush=True,
            )
            print("-" * 56, flush=True)
            for nbytes in MSG_SIZES:
                iters = TIMED_ITERS if nbytes <= 16384 else TIMED_ITERS // 2
                avg_us = eager_results[nbytes]
                bw_gbps = (nbytes / (1024**3)) / (avg_us / 1_000_000.0)
                print(
                    f"{format_size(nbytes):<10} | {avg_us:<14.2f} | "
                    f"{bw_gbps:<12.4f} | {iters:<6}",
                    flush=True,
                )

    op.teardown()
    comm.finalize()


if __name__ == "__main__":
    bidirectional = "--bidirectional" in sys.argv
    fused = "--fused" in sys.argv
    cuda_graph = "--cuda-graph" in sys.argv
    gpu_offset = 0
    for i, a in enumerate(sys.argv):
        if a == "--gpu-offset" and i + 1 < len(sys.argv):
            gpu_offset = int(sys.argv[i + 1])

    port = find_free_port()
    mp.spawn(
        run_benchmark_worker,
        args=(port, bidirectional, fused, cuda_graph, gpu_offset),
        nprocs=2,
        join=True,
    )
