# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
Benchmark for put_block bandwidth microbenchmark.

Sweeps three modes — fire-and-forget, pipelined, and multi-block — to
characterize raw RDMA put throughput without copy overhead.

Usage:
    buck2 run ... :benchmark_put_bw                    # all experiments
    buck2 run ... :benchmark_put_bw -- fireforget       # fire-and-forget only
    buck2 run ... :benchmark_put_bw -- pipelined        # pipelined only
    buck2 run ... :benchmark_put_bw -- multiblock       # multi-block only
"""

import os
import socket

# Enable GIN (GPU-Initiated Networking) for device-side window operations
os.environ.setdefault("NCCL_GIN_ENABLE", "1")
os.environ.setdefault("NCCL_GIN_TYPE", "-1")
# Disable P2P to force IB transport even on single-node
os.environ.setdefault("NCCL_P2P_DISABLE", "1")
# Large QP depth for multi-block kernel (many WQEs per step)
os.environ.setdefault("NCCL_GIN_GDAKI_QP_DEPTH", "1024")
os.environ.setdefault("NCCL_DEBUG", "INFO")

import sys

import torch
import torch.multiprocessing as mp
import torchcomms
from comms.pipes.triton.collectives.ib.put_bw_op import PutBwOp


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

# ── Experiment 1: Fire-and-forget put sweep (no flow control) ──
# (name, total_bytes, section_bytes)
FIREFORGET_CONFIGS = [
    ("256KB", 256 * 1024, 256 * 1024),  # 1 step
    ("1MB", 64 * M, 1 * M),  # 64 steps
    ("4MB", 256 * M, 4 * M),  # 64 steps
    ("16MB", 256 * M, 16 * M),  # 16 steps
    ("64MB", 512 * M, 64 * M),  # 8 steps
    ("128MB", 1024 * M, 128 * M),  # 8 steps
]

# ── Experiment 2: Pipelined put sweep (with flow control) ──
# (name, total_bytes, section_bytes, pipeline_depth)
PIPELINED_CONFIGS = [
    ("256KB", 256 * M, 256 * 1024, 8),
    ("1MB", 512 * M, 1 * M, 8),
    ("4MB", 512 * M, 4 * M, 8),
    ("16MB", 1024 * M, 16 * M, 4),
    ("64MB", 1024 * M, 64 * M, 4),
    ("128MB", 1024 * M, 128 * M, 4),
]

# ── Experiment 3: Multi-block put sweep (fixed section, sweep blocks) ──
# (name, total_bytes, section_bytes, pipeline_depth, num_blocks)
MULTIBLOCK_CONFIGS = [
    ("1blk", 1024 * M, 64 * M, 4, 1),
    ("2blk", 1024 * M, 64 * M, 4, 2),
    ("4blk", 1024 * M, 64 * M, 4, 4),
    ("8blk", 1024 * M, 64 * M, 4, 8),
    ("16blk", 1024 * M, 64 * M, 4, 16),
    ("32blk", 1024 * M, 64 * M, 4, 32),
    ("64blk", 1024 * M, 64 * M, 4, 64),
    ("128blk", 1024 * M, 64 * M, 4, 128),
]

# ── Diagnostic configs for debugging 16-block deadlock ──
# Each tests a specific hypothesis about the root cause.
# (name, total_bytes, section_bytes, pipeline_depth, num_blocks)
DIAGNOSTIC_CONFIGS = {
    # --- Step count sweep (16 blocks, PD=total_steps, no flow control) ---
    "D1": ("D1_16blk_1step", 64 * M, 64 * M, 1, 16),
    "D2": ("D2_16blk_2step", 128 * M, 64 * M, 2, 16),
    "D3": ("D3_16blk_4step", 256 * M, 64 * M, 4, 16),
    "D4": ("D4_16blk_8step", 512 * M, 64 * M, 8, 16),
    "D5": ("D5_16blk_16step", 1024 * M, 64 * M, 16, 16),
    # --- Block count threshold (PD=4, total_steps=16, with flow control) ---
    "D6": ("D6_9blk_fc", 1024 * M, 64 * M, 4, 9),
    "D7": ("D7_10blk_fc", 1024 * M, 64 * M, 4, 10),
    "D8": ("D8_12blk_fc", 1024 * M, 64 * M, 4, 12),
    "D9": ("D9_14blk_fc", 1024 * M, 64 * M, 4, 14),
    "D10": ("D10_15blk_fc", 1024 * M, 64 * M, 4, 15),
    "D11": ("D11_16blk_fc", 1024 * M, 64 * M, 4, 16),
}


def run_fireforget(comm, local_rank, configs):
    peer_rank = 1 - local_rank
    if local_rank == 0:
        print("\n=== Experiment 1: Fire-and-Forget Put (no flow control) ===")
        print(
            f"{'PutSize':<12} | {'Total':<10} | {'Steps':<8} | "
            f"{'Lat (us)':<12} | {'BW (GB/s)':<12}"
        )
        print("-" * 65)

    for name, total_bytes, section_bytes in configs:
        total_steps = total_bytes // section_bytes
        # For fireforget: pipeline_depth = total_steps (unique slot per step)
        op = PutBwOp(
            comm,
            "fireforget",
            total_steps,
            section_bytes,
            total_bytes=total_bytes,
        )

        # Warmup
        for _ in range(3):
            op(total_bytes, peer_rank)
        torch.cuda.synchronize()
        comm.barrier(False)

        # Timed runs
        iters = 50 if total_bytes >= 256 * M else 200
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        comm.barrier(False)
        start.record()
        for _ in range(iters):
            op(total_bytes, peer_rank)
        end.record()
        torch.cuda.synchronize()

        elapsed_ms = start.elapsed_time(end)
        avg_us = (elapsed_ms * 1000.0) / iters
        bw = (total_bytes / (1024**3)) / (avg_us / 1e6)

        if local_rank == 0:
            print(
                f"{name:<12} | {format_size(total_bytes):<10} | "
                f"{total_steps:<8} | {avg_us:<12.2f} | {bw:<12.2f}"
            )

        op.teardown()
        torch.cuda.empty_cache()
        comm.barrier(False)


def run_pipelined(comm, local_rank, configs):
    peer_rank = 1 - local_rank
    if local_rank == 0:
        print("\n=== Experiment 2: Pipelined Put (with flow control) ===")
        print(
            f"{'PutSize':<12} | {'Total':<10} | {'PD':<4} | {'Steps':<8} | "
            f"{'Lat (us)':<12} | {'BW (GB/s)':<12}"
        )
        print("-" * 70)

    for name, total_bytes, section_bytes, pipeline_depth in configs:
        total_steps = total_bytes // section_bytes
        op = PutBwOp(comm, "pipelined", pipeline_depth, section_bytes)

        # Warmup
        for _ in range(3):
            op(total_bytes, peer_rank)
        torch.cuda.synchronize()
        comm.barrier(False)

        # Timed runs
        iters = 50 if total_bytes >= 256 * M else 200
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        comm.barrier(False)
        start.record()
        for _ in range(iters):
            op(total_bytes, peer_rank)
        end.record()
        torch.cuda.synchronize()

        elapsed_ms = start.elapsed_time(end)
        avg_us = (elapsed_ms * 1000.0) / iters
        bw = (total_bytes / (1024**3)) / (avg_us / 1e6)

        if local_rank == 0:
            print(
                f"{name:<12} | {format_size(total_bytes):<10} | "
                f"{pipeline_depth:<4} | {total_steps:<8} | "
                f"{avg_us:<12.2f} | {bw:<12.2f}"
            )

        op.teardown()
        torch.cuda.empty_cache()
        comm.barrier(False)


def run_multiblock(comm, local_rank, configs):
    peer_rank = 1 - local_rank
    if local_rank == 0:
        print("\n=== Experiment 3: Multi-Block Put (sweep block count) ===")
        print(
            f"{'Blocks':<12} | {'Total':<10} | {'PD':<4} | {'Steps':<8} | "
            f"{'Lat (us)':<12} | {'BW (GB/s)':<12}"
        )
        print("-" * 70)

    for name, total_bytes, section_bytes, pipeline_depth, num_blocks in configs:
        total_steps = total_bytes // section_bytes
        op = PutBwOp(
            comm,
            "multiblock",
            pipeline_depth,
            section_bytes,
            num_blocks=num_blocks,
        )

        # Warmup
        for _ in range(3):
            op(total_bytes, peer_rank)
        torch.cuda.synchronize()
        comm.barrier(False)

        # Timed runs
        iters = 50 if total_bytes >= 256 * M else 200
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        comm.barrier(False)
        start.record()
        for _ in range(iters):
            op(total_bytes, peer_rank)
        end.record()
        torch.cuda.synchronize()

        elapsed_ms = start.elapsed_time(end)
        avg_us = (elapsed_ms * 1000.0) / iters
        bw = (total_bytes / (1024**3)) / (avg_us / 1e6)

        if local_rank == 0:
            print(
                f"{name:<12} | {format_size(total_bytes):<10} | "
                f"{pipeline_depth:<4} | {total_steps:<8} | "
                f"{avg_us:<12.2f} | {bw:<12.2f}"
            )

        op.teardown()
        torch.cuda.empty_cache()
        comm.barrier(False)


def run_diagnostic(comm, local_rank, config_id):
    """Run a single diagnostic config by ID (e.g. 'D1')."""
    if config_id not in DIAGNOSTIC_CONFIGS:
        if local_rank == 0:
            print(f"Unknown diag config: {config_id}")
            print(f"Available: {', '.join(sorted(DIAGNOSTIC_CONFIGS.keys()))}")
        return

    name, total_bytes, section_bytes, pipeline_depth, num_blocks = DIAGNOSTIC_CONFIGS[
        config_id
    ]
    total_steps = total_bytes // section_bytes
    peer_rank = 1 - local_rank

    if local_rank == 0:
        print(
            f"DIAG {name}: total={format_size(total_bytes)} "
            f"section={format_size(section_bytes)} PD={pipeline_depth} "
            f"steps={total_steps} blocks={num_blocks}"
        )
        sys.stdout.flush()

    op = PutBwOp(
        comm,
        "multiblock",
        pipeline_depth,
        section_bytes,
        num_blocks=num_blocks,
    )

    # Warmup
    for i in range(3):
        if local_rank == 0:
            print(f"  warmup {i + 1}/3...", end="", flush=True)
        op(total_bytes, peer_rank)
        if local_rank == 0:
            print(" done", flush=True)
    torch.cuda.synchronize()
    comm.barrier(False)

    # Timed runs
    iters = 10
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    comm.barrier(False)
    start.record()
    for _ in range(iters):
        op(total_bytes, peer_rank)
    end.record()
    torch.cuda.synchronize()

    elapsed_ms = start.elapsed_time(end)
    avg_us = (elapsed_ms * 1000.0) / iters
    bw = (total_bytes / (1024**3)) / (avg_us / 1e6)

    if local_rank == 0:
        print(f"  RESULT: OK  {bw:.2f} GB/s  (lat={avg_us:.2f} us)")

    op.teardown()
    torch.cuda.empty_cache()
    comm.barrier(False)


def run_benchmark_worker(local_rank, master_port, experiment):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(local_rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = "2"

    torch.cuda.set_device(local_rank)

    comm = torchcomms.new_comm(
        "ncclx",
        torch.device(f"cuda:{local_rank}"),
        name="put_bw_bench",
    )

    if local_rank == 0:
        print("Put-Block Bandwidth Microbenchmark (2 GPUs)")
        print(f"GPU {local_rank} <-> GPU {1 - local_rank}")

    if experiment in ("fireforget", "all"):
        run_fireforget(comm, local_rank, FIREFORGET_CONFIGS)

    if experiment in ("pipelined", "all"):
        run_pipelined(comm, local_rank, PIPELINED_CONFIGS)

    if experiment in ("multiblock", "all"):
        run_multiblock(comm, local_rank, MULTIBLOCK_CONFIGS)

    if experiment.startswith("D") and experiment in DIAGNOSTIC_CONFIGS:
        run_diagnostic(comm, local_rank, experiment)

    comm.finalize()


if __name__ == "__main__":
    experiment = "all"
    for arg in sys.argv[1:]:
        if arg in ("fireforget", "pipelined", "multiblock", "all"):
            experiment = arg
        elif arg.startswith("D") and arg in DIAGNOSTIC_CONFIGS:
            experiment = arg

    port = find_free_port()
    mp.spawn(
        run_benchmark_worker,
        args=(port, experiment),
        nprocs=2,
        join=True,
    )
