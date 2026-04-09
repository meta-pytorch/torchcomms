#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-unsafe
#
# CPU micro-benchmark for the MPI backend.
# Must be launched via mpirun (not torchrun):
#
#   LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH \
#     mpirun --oversubscribe -np 8 \
#     python comms/torchcomms/tests/perf/py/mpi_bench.py

import atexit
import ctypes
import time

import torch
import torchcomms


def _mpi_finalize_at_exit():
    """Call MPI_Finalize on process exit to avoid prterun warnings."""
    try:
        libmpi = ctypes.CDLL("libmpi.so", mode=ctypes.RTLD_GLOBAL)
        finalized = ctypes.c_int(0)
        libmpi.MPI_Finalized(ctypes.byref(finalized))
        if not finalized.value:
            libmpi.MPI_Finalize()
    except Exception:
        pass


atexit.register(_mpi_finalize_at_exit)


WARMUP = 2
MIN_ITERS = 3
TARGET_SECS = 0.3
DTYPE = torch.float32
ELEMENT_SIZE = 4

SIZES = [64, 4096, 262144, 4194304, 67108864, 104857600]
COLLECTIVES = ["all_reduce", "all_gather", "broadcast", "barrier"]


def bench_collective(comm, coord_comm, name, size_bytes):
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

    # Calibrate
    start = time.perf_counter()
    for _ in range(MIN_ITERS):
        op()
    cal_elapsed = time.perf_counter() - start
    avg_per_iter = cal_elapsed / MIN_ITERS
    target_iters = (
        max(MIN_ITERS, int(TARGET_SECS / avg_per_iter))
        if avg_per_iter > 0
        else MIN_ITERS
    )

    # Agree on iteration count via coord_comm
    iters_tensor = torch.tensor([target_iters], dtype=torch.int64)
    coord_comm.all_reduce(iters_tensor, torchcomms.ReduceOp.MAX, False)
    n_iters = iters_tensor.item()

    remaining = n_iters - MIN_ITERS
    comm.barrier(False)
    start = time.perf_counter()
    for _ in range(remaining):
        op()
    elapsed = time.perf_counter() - start

    total_elapsed = cal_elapsed + elapsed
    return (total_elapsed / n_iters) * 1e6, n_iters


def human_size(b):
    if b >= 1048576:
        return f"{b / 1048576:.0f}MB"
    if b >= 1024:
        return f"{b / 1024:.0f}KB"
    return f"{b}B"


def main():
    comm = torchcomms.new_comm("mpi", torch.device("cpu"), "bench_mpi")
    coord_comm = torchcomms.new_comm("mpi", torch.device("cpu"), "bench_coord")

    rank = comm.get_rank()
    world = comm.get_size()

    if rank == 0:
        print(f"Backend: mpi  Ranks: {world}")
        print(f"Warmup: {WARMUP}  Target: {TARGET_SECS}s/test  Dtype: {DTYPE}")
        print()

    for coll_name in COLLECTIVES:
        sizes = [0] if coll_name == "barrier" else SIZES

        if rank == 0:
            print(f"=== {coll_name} ===")
            print(f"{'Size':<10}{'Iters':>8}{'mpi(us)':>16}")
            print("-" * 34)

        for size in sizes:
            result = bench_collective(comm, coord_comm, coll_name, size)
            if result is None:
                if rank == 0:
                    size_str = "N/A" if coll_name == "barrier" else human_size(size)
                    print(f"{size_str:<10}{'SKIP':>8}{'N/A':>16}")
                continue
            avg_us, iters = result
            if rank == 0:
                size_str = "N/A" if coll_name == "barrier" else human_size(size)
                print(f"{size_str:<10}{iters:>8}{avg_us:>16.2f}")

        if rank == 0:
            print()

    coord_comm.finalize()
    comm.finalize()


if __name__ == "__main__":
    main()
