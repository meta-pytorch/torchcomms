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


def run_benchmark_worker(local_rank, master_port, bidirectional, fused, gpu_offset):
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
    # Pre-warm Triton JIT compilation for all BLOCK_SIZE variants before the
    # timed benchmark loop. Auto-tune uses BLOCK_SIZE=32 for small messages
    # (<=256B) and BLOCK_SIZE=512 for larger ones. Each unique BLOCK_SIZE
    # (a tl.constexpr) triggers a separate Triton compilation that can take
    # 30+ seconds. Both ranks must finish compiling before the fused kernel
    # can run, so we pre-warm with sync+barrier after each variant.
    for prewarm_nbytes in [8, 1024]:
        prewarm_elements = prewarm_nbytes // 4
        prewarm_src = torch.randn(
            prewarm_elements, dtype=torch.float32, device=f"cuda:{gpu_id}"
        )
        prewarm_dst = torch.zeros(
            prewarm_elements, dtype=torch.float32, device=f"cuda:{gpu_id}"
        )
        if fused:
            op.sendrecv(
                peer=peer_rank,
                src_tensor=prewarm_src,
                dst_tensor=prewarm_dst,
                nbytes=prewarm_nbytes,
            )
        elif bidirectional:
            op.send(peer=peer_rank, src_tensor=prewarm_src, nbytes=prewarm_nbytes)
            op.recv(peer=peer_rank, dst_tensor=prewarm_dst, nbytes=prewarm_nbytes)
        else:
            if local_rank == 0:
                op.send(peer=peer_rank, src_tensor=prewarm_src, nbytes=prewarm_nbytes)
            else:
                op.recv(peer=peer_rank, dst_tensor=prewarm_dst, nbytes=prewarm_nbytes)
        torch.cuda.synchronize()
        comm.barrier(False)
        del prewarm_src, prewarm_dst

    if local_rank == 0:
        print(f"Triton LL P2P Benchmark ({mode_str}, 2 GPUs)", flush=True)
        print(f"GPU {gpu_id} <-> GPU {gpu_id + 1 - 2 * local_rank}", flush=True)
        print(flush=True)
        print(
            f"{'Size':<10} | {'Latency (us)':<14} | {'BW (GB/s)':<12} | {'Iters':<6}",
            flush=True,
        )
        print("-" * 56, flush=True)

    for nbytes in MSG_SIZES:
        num_elements = nbytes // 4  # int32
        src = torch.randn(num_elements, dtype=torch.float32, device=f"cuda:{gpu_id}")
        dst = torch.zeros(num_elements, dtype=torch.float32, device=f"cuda:{gpu_id}")

        # Warmup
        for _ in range(WARMUP_ITERS):
            if fused:
                op.sendrecv(
                    peer=peer_rank, src_tensor=src, dst_tensor=dst, nbytes=nbytes
                )
            elif bidirectional:
                op.send(peer=peer_rank, src_tensor=src, nbytes=nbytes)
                op.recv(peer=peer_rank, dst_tensor=dst, nbytes=nbytes)
            else:
                if local_rank == 0:
                    op.send(peer=peer_rank, src_tensor=src, nbytes=nbytes)
                else:
                    op.recv(peer=peer_rank, dst_tensor=dst, nbytes=nbytes)
        torch.cuda.synchronize()
        comm.barrier(False)

        # Fewer iterations for larger messages
        iters = TIMED_ITERS if nbytes <= 16384 else TIMED_ITERS // 2

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        comm.barrier(False)
        start_event.record()
        for _ in range(iters):
            if fused:
                op.sendrecv(
                    peer=peer_rank, src_tensor=src, dst_tensor=dst, nbytes=nbytes
                )
            elif bidirectional:
                op.send(peer=peer_rank, src_tensor=src, nbytes=nbytes)
                op.recv(peer=peer_rank, dst_tensor=dst, nbytes=nbytes)
            else:
                if local_rank == 0:
                    op.send(peer=peer_rank, src_tensor=src, nbytes=nbytes)
                else:
                    op.recv(peer=peer_rank, dst_tensor=dst, nbytes=nbytes)
        end_event.record()
        torch.cuda.synchronize()

        elapsed_ms = start_event.elapsed_time(end_event)
        avg_us = (elapsed_ms * 1000.0) / iters
        bw_gbps = (nbytes / (1024**3)) / (avg_us / 1_000_000.0)

        if local_rank == 0:
            print(
                f"{format_size(nbytes):<10} | {avg_us:<14.2f} | "
                f"{bw_gbps:<12.4f} | {iters:<6}"
            )

        del src, dst

    op.teardown()
    comm.finalize()


if __name__ == "__main__":
    bidirectional = "--bidirectional" in sys.argv
    fused = "--fused" in sys.argv
    gpu_offset = 0
    for i, a in enumerate(sys.argv):
        if a == "--gpu-offset" and i + 1 < len(sys.argv):
            gpu_offset = int(sys.argv[i + 1])

    port = find_free_port()
    mp.spawn(
        run_benchmark_worker,
        args=(port, bidirectional, fused, gpu_offset),
        nprocs=2,
        join=True,
    )
