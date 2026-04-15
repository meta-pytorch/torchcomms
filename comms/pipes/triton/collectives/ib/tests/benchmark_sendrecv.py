# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
Benchmark for Triton pipelined sendrecv.

Sweeps through message sizes, pipeline depths, chunk sizes, and block counts
to characterize bandwidth and latency.

Usage:
    buck2 run ... :benchmark_sendrecv                # original kernel
    buck2 run ... :benchmark_sendrecv -- --parallel   # tile-parallel kernel
"""

import os
import socket

# Enable GIN (GPU-Initiated Networking) for device-side window operations
os.environ.setdefault("NCCL_GIN_ENABLE", "1")
os.environ.setdefault("NCCL_GIN_TYPE", "-1")
# Disable P2P to force IB transport even on single-node
os.environ.setdefault("NCCL_P2P_DISABLE", "1")
# Large QP depth for parallel kernel (many WQEs per step)
os.environ.setdefault("NCCL_GIN_GDAKI_QP_DEPTH", "1024")
os.environ.setdefault("NCCL_DEBUG", "INFO")

import sys

import torch
import torch.multiprocessing as mp
import torchcomms
from comms.pipes.triton.collectives.ib.sendrecv_op import SendRecvOp


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


M = 1024 * 1024
TOTAL = 512 * M  # Default message size for sweep (overridable via --total)


def generate_sweep_configs(block_filter=None):
    """
    Generate comprehensive parameter sweep for Pareto analysis.

    Sweep dimensions:
      - num_blocks: [16, 32, 64, 128] (or filtered via --blocks N)
      - section_size: powers of 2 from 32MB up to total
      - pipeline_depth: [1, 2, 4] capped at total_steps

    For each config, compute:
      - tile_per_put = section / blocks  (NIC efficiency driver)
      - staging_memory = pd * section * 2  (send + recv staging)
    """
    configs = []
    block_counts = [block_filter] if block_filter else [16, 32, 64, 128]
    # Section sizes from 32MB to 2GB (powers of 2)
    section_sizes = [32 * M, 64 * M, 128 * M, 256 * M, 512 * M, 1024 * M, 2048 * M]
    pd_values = [1, 2, 4, 8, 16]

    for nblocks in block_counts:
        for sec in section_sizes:
            if sec > TOTAL:
                continue
            total_steps = TOTAL // sec
            for pd in pd_values:
                if pd > total_steps:
                    continue
                name = f"b{nblocks}_s{format_size(sec)}_p{pd}"
                configs.append((name, TOTAL, sec, pd, nblocks))

    return configs


# Parse CLI filters: --blocks N --section S --pd P --total T (all in MB) --gpu-offset G
_block_filter = None
_section_filter = None
_pd_filter = None
_gpu_offset = 0
for i, a in enumerate(sys.argv):
    if a == "--blocks" and i + 1 < len(sys.argv):
        _block_filter = int(sys.argv[i + 1])
    elif a == "--section" and i + 1 < len(sys.argv):
        _section_filter = int(sys.argv[i + 1]) * M
    elif a == "--pd" and i + 1 < len(sys.argv):
        _pd_filter = int(sys.argv[i + 1])
    elif a == "--total" and i + 1 < len(sys.argv):
        TOTAL = int(sys.argv[i + 1]) * M
    elif a == "--gpu-offset" and i + 1 < len(sys.argv):
        _gpu_offset = int(sys.argv[i + 1])

if _section_filter and _pd_filter and _block_filter:
    # Single config mode
    _name = f"b{_block_filter}_s{format_size(_section_filter)}_p{_pd_filter}"
    CONFIGS = [(_name, TOTAL, _section_filter, _pd_filter, _block_filter)]
else:
    CONFIGS = generate_sweep_configs(_block_filter)


def run_benchmark_worker(local_rank, master_port, use_parallel, gpu_offset):
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
        name="sendrecv_bench",
    )

    if local_rank == 0:
        mode = "PARALLEL (per-block RDMA)" if use_parallel else "ORIGINAL (cooperative)"
        print(f"Bidirectional Triton SendRecv Benchmark (2 GPUs) — {mode}")
        print(f"GPU {gpu_id} <-> GPU {gpu_id + 1 - 2 * local_rank}")
        print(f"Total message: {format_size(TOTAL)}")
        print()
        # CSV-friendly header for easy parsing
        print(
            f"{'Name':<22} | {'Blocks':<6} | {'Section':<10} | "
            f"{'PD':<4} | {'Tile':<10} | {'Steps':<6} | "
            f"{'Staging':<10} | "
            f"{'Lat (us)':<12} | {'BW (GB/s)':<12}"
        )
        print("-" * 120)

    for name, msg_bytes, sec_bytes, pd, nblocks in CONFIGS:
        total_elements = msg_bytes // 4

        src = torch.randn(total_elements, dtype=torch.float32, device=f"cuda:{gpu_id}")
        dst = torch.zeros(total_elements, dtype=torch.float32, device=f"cuda:{gpu_id}")

        op = SendRecvOp(comm, pd, sec_bytes, nblocks, parallel=use_parallel)

        # Warmup
        for _ in range(20):
            op(src, dst, peer_rank)
        torch.cuda.synchronize()
        comm.barrier(False)

        # Timed runs
        iters = 200 if msg_bytes < 64 * M else 100

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        comm.barrier(False)
        start_event.record()
        for _ in range(iters):
            op(src, dst, peer_rank)
        end_event.record()
        torch.cuda.synchronize()

        elapsed_ms = start_event.elapsed_time(end_event)
        avg_us = (elapsed_ms * 1000.0) / iters

        bw_unidir = (msg_bytes / (1024**3)) / (avg_us / 1_000_000.0)
        bw_bidir = 2.0 * bw_unidir

        if local_rank == 0:
            tile_bytes = sec_bytes // nblocks
            total_steps = msg_bytes // sec_bytes
            staging_bytes = pd * sec_bytes * 2  # send + recv
            print(
                f"{name:<22} | {nblocks:<6} | {format_size(sec_bytes):<10} | "
                f"{pd:<4} | {format_size(tile_bytes):<10} | {total_steps:<6} | "
                f"{format_size(staging_bytes):<10} | "
                f"{avg_us:<12.2f} | {bw_unidir:<12.2f}"
            )

        op.teardown()
        del op, src, dst
        torch.cuda.empty_cache()

    comm.finalize()


if __name__ == "__main__":
    use_parallel = "--parallel" in sys.argv
    port = find_free_port()
    mp.spawn(
        run_benchmark_worker,
        args=(port, use_parallel, _gpu_offset),
        nprocs=2,
        join=True,
    )
