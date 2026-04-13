#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-unsafe
#
# CPU micro-benchmark comparing Gloo, UCC (shared memory), and UCC (TCP-only).
#
# Single rank:
#   MASTER_ADDR=localhost MASTER_PORT=0 WORLD_SIZE=1 RANK=0 \
#     python comms/torchcomms/tests/perf/py/gloo_vs_ucc_bench.py
#
# Multi-rank:
#   torchrun --nproc_per_node=8 comms/torchcomms/tests/perf/py/gloo_vs_ucc_bench.py

import os
import time

import torch
import torchcomms


WARMUP = 2
MIN_ITERS = 3
TARGET_SECS = 0.3  # run each (backend, collective, size) for ~0.3s
DTYPE = torch.float32
ELEMENT_SIZE = 4

SIZES = [64, 4096, 262144, 4194304, 67108864, 104857600]
COLLECTIVES = ["all_reduce", "all_gather", "broadcast", "barrier"]


def bench_collective(comm, coord_comm, name, size_bytes):
    """Benchmark a collective. Uses coord_comm (separate backend) to synchronize
    iteration counts across ranks so no rank exits early."""
    n_elem = max(size_bytes // ELEMENT_SIZE, 1)
    rank = comm.get_rank()
    n_ranks = comm.get_size()
    tensor = torch.ones(n_elem, dtype=DTYPE) * float(rank + 1)

    if name == "all_reduce":
        def op():
            comm.all_reduce(tensor, torchcomms.ReduceOp.SUM, False)
    elif name == "all_gather":
        out = [torch.zeros(n_elem, dtype=DTYPE) for _ in range(n_ranks)]
        def op():
            comm.all_gather(out, tensor, False)
    elif name == "broadcast":
        def op():
            comm.broadcast(tensor, 0, False)
    elif name == "barrier":
        def op():
            comm.barrier(False)
    else:
        return None

    for _ in range(WARMUP):
        op()
    comm.barrier(False)

    # Calibrate: run MIN_ITERS, measure time, compute iters for TARGET_SECS
    start = time.perf_counter()
    for _ in range(MIN_ITERS):
        op()
    cal_elapsed = time.perf_counter() - start
    avg_per_iter = cal_elapsed / MIN_ITERS
    target_iters = max(MIN_ITERS, int(TARGET_SECS / avg_per_iter)) if avg_per_iter > 0 else MIN_ITERS

    # Agree on iteration count across ranks (use MAX so slower ranks set the pace)
    iters_tensor = torch.tensor([target_iters], dtype=torch.int64)
    coord_comm.all_reduce(iters_tensor, torchcomms.ReduceOp.MAX, False)
    n_iters = iters_tensor.item()

    # Run the agreed-upon number of iterations (subtract calibration iters already done)
    remaining = n_iters - MIN_ITERS
    comm.barrier(False)
    start = time.perf_counter()
    for _ in range(remaining):
        op()
    elapsed = time.perf_counter() - start

    # Total time includes calibration
    total_elapsed = cal_elapsed + elapsed
    return (total_elapsed / n_iters) * 1e6, n_iters


def human_size(b):
    if b >= 1048576:
        return f"{b / 1048576:.0f}MB"
    if b >= 1024:
        return f"{b / 1024:.0f}KB"
    return f"{b}B"


def make_ucc_tcp_comm(name):
    """Create a UCC comm with shared memory disabled, forcing TCP transport."""
    # Save and override UCX env to disable shared memory transports
    saved = {}
    overrides = {
        "UCX_TLS": "tcp",
        "UCX_NET_DEVICES": "all",
    }
    for k, v in overrides.items():
        saved[k] = os.environ.get(k)
        os.environ[k] = v

    try:
        comm = torchcomms.new_comm("ucc", torch.device("cpu"), name)
    finally:
        # Restore original env
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    return comm


def main():
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("RANK", "0")

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])

    comms = {}
    labels = {
        "gloo": "gloo",
        "ucc": "ucc(shm)",
        "ucc_tcp": "ucc(tcp)",
    }

    # Gloo
    try:
        comms["gloo"] = torchcomms.new_comm("gloo", torch.device("cpu"), "bench_gloo")
    except Exception:
        if rank == 0:
            print("WARNING: gloo backend not available")

    # UCC with default transports (shared memory + tcp)
    try:
        comms["ucc"] = torchcomms.new_comm("ucc", torch.device("cpu"), "bench_ucc")
    except Exception:
        if rank == 0:
            print("WARNING: ucc backend not available")

    # UCC with TCP only (no shared memory)
    try:
        comms["ucc_tcp"] = make_ucc_tcp_comm("bench_ucc_tcp")
    except Exception:
        if rank == 0:
            print("WARNING: ucc(tcp) backend not available")

    backends = list(comms.keys())
    if not backends:
        if rank == 0:
            print("No backends available.")
        return

    # Coordination comm for synchronizing iteration counts across ranks
    coord_comm = torchcomms.new_comm("gloo", torch.device("cpu"), "bench_coord")

    if rank == 0:
        print(f"Backends: {', '.join(labels[b] for b in backends)}  Ranks: {world}")
        print(f"Warmup: {WARMUP}  Target: {TARGET_SECS}s/test  Dtype: {DTYPE}")
        print()

    for coll_name in COLLECTIVES:
        sizes = [0] if coll_name == "barrier" else SIZES

        if rank == 0:
            print(f"=== {coll_name} ===")
            header = f"{'Size':<10}{'Iters':>8}"
            for b in backends:
                header += f"{labels[b]:>14}(us)"
            if "gloo" in comms and "ucc" in comms:
                header += "  gloo/shm"
            if "gloo" in comms and "ucc_tcp" in comms:
                header += "  gloo/tcp"
            if "ucc" in comms and "ucc_tcp" in comms:
                header += "   shm/tcp"
            print(header)
            print("-" * len(header))

        for size in sizes:
            results = {}
            iters_used = 0
            for backend in backends:
                avg_us, iters = bench_collective(
                    comms[backend], coord_comm, coll_name, size
                )
                results[backend] = avg_us
                iters_used = max(iters_used, iters)

            if rank == 0:
                size_str = "N/A" if coll_name == "barrier" else human_size(size)
                line = f"{size_str:<10}{iters_used:>8}"
                for b in backends:
                    line += f"{results[b]:>16.2f} "
                # Ratios
                if "gloo" in results and "ucc" in results and results["ucc"] > 0:
                    line += f"  {results['gloo'] / results['ucc']:>7.2f}x"
                if "gloo" in results and "ucc_tcp" in results and results["ucc_tcp"] > 0:
                    line += f"  {results['gloo'] / results['ucc_tcp']:>7.2f}x"
                if "ucc" in results and "ucc_tcp" in results and results["ucc_tcp"] > 0:
                    line += f"  {results['ucc'] / results['ucc_tcp']:>7.2f}x"
                print(line)

        if rank == 0:
            print()

    coord_comm.finalize()
    for c in comms.values():
        c.finalize()


if __name__ == "__main__":
    main()
