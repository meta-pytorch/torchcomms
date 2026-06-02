#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Flight Recorder Hang Debugging Demo

Demonstrates debugging a rank-hang scenario

Scenario
--------
  - All ranks complete initial collectives together (baseline)
  - One rank enters a long sleep (simulating a compute hang)
  - Other ranks attempt another collective and eventually timeout
  - The abort hook marks the timed-out rank as unhealthy
  - The debug server's health check detects the unhealthy rank and
    triggers per-rank pickle dumps automatically
  - Periodic aggregated text dumps capture the divergent state
  - The FR CLI performs cross-rank mismatch detection on the pickle files

Usage
-----
  # Basic run (last rank hangs, 30s timeout):
  torchrun --nproc_per_node=2 verify_flight_recorder.py

  # Customise dump directory, timeout, and which rank hangs:
  FR_DUMP_DIR=/tmp/fr_hang_debug \\
  COMM_TIMEOUT=20 \\
  HANGING_RANK=0 \\
  torchrun --nproc_per_node=4 verify_flight_recorder.py

  # After the job fails, analyse the per-rank pickle traces:
  python -m torch.distributed.flight_recorder.fr_trace \\
      /tmp/fr_hang_debug/per_rank -p rank_

Environment Variables
---------------------
  FR_DUMP_DIR         Root dump directory (default: /tmp/fr_hang_debug)
  FR_DUMP_INTERVAL    Seconds between periodic dumps (default: 5)
  COMM_TIMEOUT        Communicator timeout in seconds (default: 30)
  HANGING_RANK        Which rank to hang; -1 = last rank (default: -1)
  TEST_BACKEND        Communication backend (default: gloo)
  TEST_DEVICE         Tensor device (default: cuda)

Output Layout
-------------
  FR_DUMP_DIR/
  ├── torchcomms_fr_trace_<ts>.txt        ← aggregated text dumps
  ├── torchcomms_health_check_<ts>.txt    ← health check status
  ├── stacks_<ts>.txt                     ← Python stack dumps
  └── per_rank/                           ← per-rank pickle dumps (triggered by health check)
      ├── rank_0
      └── rank_1
"""

import os
import time
from datetime import timedelta

import torch
from torch.distributed.debug import start_debug_server
from torchcomms import new_comm, ReduceOp
from torchcomms.hooks import FlightRecorderHook


def main() -> None:
    backend = os.environ.get("TEST_BACKEND", "gloo")
    device = torch.device(os.environ.get("TEST_DEVICE", "cuda"))

    dump_dir = os.environ.get("FR_DUMP_DIR", "/tmp/fr_hang_debug")
    dump_interval = float(os.environ.get("FR_DUMP_INTERVAL", "5"))
    timeout_seconds = int(os.environ.get("COMM_TIMEOUT", "30"))
    hanging_rank = int(os.environ.get("HANGING_RANK", "-1"))

    # The abort hook waits this long for the health check to trigger
    # FR dumps before the exception propagates.  Default to 3× the dump
    # interval so at least one dump cycle can detect the unhealthy state.
    health_check_wait = int(
        os.environ.get(
            "TORCHCOMM_HEALTH_CHECK_WAIT_SECONDS", str(int(dump_interval) * 3)
        )
    )
    os.environ.setdefault("TORCHCOMM_HEALTH_CHECK_WAIT_SECONDS", str(health_check_wait))

    os.makedirs(dump_dir, exist_ok=True)

    # Per-rank pickle dumps go into a dedicated subdirectory so that the
    # FR CLI can operate on a clean folder containing only rank_<N> files,
    # separate from the debug-server's aggregated text dumps.
    per_rank_dir = os.path.join(dump_dir, "per_rank")
    os.makedirs(per_rank_dir, exist_ok=True)
    dump_prefix = os.path.join(per_rank_dir, "rank_")
    os.environ["TORCHCOMM_FR_DUMP_TEMP_FILE"] = dump_prefix

    comm = new_comm(
        backend=backend,
        device=device,
        name="main_comm",
        timeout=timedelta(seconds=timeout_seconds),
    )

    rank = comm.get_rank()
    world_size = comm.get_size()

    if hanging_rank < 0:
        hanging_rank = world_size - 1

    num_devices = torch.accelerator.device_count()
    device_id = rank % num_devices
    target_device = torch.device(f"{device.type}:{device_id}")

    print(
        f"[Rank {rank}/{world_size}] device={device_id}, "
        f"hanging_rank={hanging_rank}, timeout={timeout_seconds}s"
    )

    # ── Debug Server with Periodic Dumps ──────────────────────────────
    # The debug server runs three periodic dump handlers:
    #   1) torchcomms_fr_trace  → aggregated text tables in dump_dir
    #   2) torchcomms_health_check → detects unhealthy ranks (watchdog
    #      timeouts) and triggers per-rank pickle dumps → per_rank/rank_<N>.
    #      These pickles are consumed by the FR CLI for automated
    #      cross-rank mismatch analysis.
    #   3) stacks → Python stack traces for every rank
    start_debug_server(
        port=25999,
        dump_dir=dump_dir,
        dump_interval=dump_interval,
        enabled_dumps={"torchcomms_fr_trace", "torchcomms_health_check", "stacks"},
    )
    if rank == 0:
        print(f"[Rank {rank}] Debug server: http://localhost:25999")
        print(f"[Rank {rank}] Periodic dumps every {dump_interval}s → {dump_dir}")
        print(f"[Rank {rank}] Health check triggers per-rank pickles → {per_rank_dir}")

    # ── Flight Recorder Hook ─────────────────────────────────────────
    recorder = FlightRecorderHook(max_entries=100)
    recorder.register_with_comm(comm)

    tensor = torch.full(
        (1024,),
        float(rank + 1),
        dtype=torch.float32,
        device=target_device,
    )

    # ── Phase 1: Successful collectives (all ranks) ──────────────────
    print(f"[Rank {rank}] Phase 1: Running 3 all_reduce + 1 broadcast")
    for _i in range(3):
        comm.all_reduce(tensor, ReduceOp.SUM, async_op=False)
    comm.broadcast(tensor, root=0, async_op=False)
    torch.accelerator.current_stream().synchronize()
    print(f"[Rank {rank}] Phase 1 complete")

    # ── Phase 2: One rank hangs ──────────────────────────────────────
    # The hanging rank sleeps long enough for the other ranks' abort
    # hook to fire and the health check to trigger dumps.
    hang_duration = timeout_seconds + health_check_wait + int(dump_interval)
    if rank == hanging_rank:
        print(f"[Rank {rank}] >>> HANGING – sleeping {hang_duration}s <<<")
        print(
            f"[Rank {rank}] Periodic dumps will show this rank's ops stuck at "
            "collective_seq_id=4 while others advance."
        )
        time.sleep(hang_duration)
        print(f"[Rank {rank}] Hang period over, exiting.")
        comm.finalize()
        return

    print(
        f"[Rank {rank}] Phase 2: all_reduce (rank {hanging_rank} will NOT participate)"
    )
    print(f"[Rank {rank}] Expecting timeout in ~{timeout_seconds}s ...")

    # When the collective times out, the abort hook fires: it calls
    # setTimedOut() and sleeps for TORCHCOMM_HEALTH_CHECK_WAIT_SECONDS,
    # giving the health check time to detect the unhealthy rank and
    # trigger per-rank pickle dumps.  The exception then propagates here.
    try:
        comm.all_reduce(tensor, ReduceOp.SUM, async_op=False)
    except RuntimeError as e:
        # torchcomms raises RuntimeError on collective timeout (the case
        # this script is intentionally exercising). Print analysis hints
        # and continue to finalize() so the rank exits cleanly. Any other
        # exception type indicates a real bug and should propagate.
        print(f"[Rank {rank}] Caught timeout: {type(e).__name__}: {e}")
        _print_analysis_instructions(rank, dump_dir, per_rank_dir)

    comm.finalize()


def _print_analysis_instructions(rank: int, dump_dir: str, per_rank_dir: str) -> None:
    if rank != 0:
        return
    print()
    print("=" * 70)
    print("  HOW TO DEBUG THIS HANG")
    print("=" * 70)
    print()
    print("  1) Aggregated text dumps (human-readable):")
    print(f"     ls {dump_dir}/torchcomms_fr_trace_*.txt")
    print()
    print("  2) Health check status (triggers per-rank pickle dumps):")
    print(f"     ls {dump_dir}/torchcomms_health_check_*.txt")
    print()
    print("  3) Per-rank pickle dumps (FR CLI cross-rank analysis):")
    print(
        f"     python -m torch.distributed.flight_recorder.fr_trace"
        f" {per_rank_dir} -p rank_"
    )
    print()
    print("  Detailed per-entry output with stack traces:")
    print(
        f"     python -m torch.distributed.flight_recorder.fr_trace"
        f" {per_rank_dir} -p rank_ -j --print_stack_trace"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
