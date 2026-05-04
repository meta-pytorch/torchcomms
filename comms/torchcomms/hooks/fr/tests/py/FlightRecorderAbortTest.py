# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-unsafe
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import glob
import os
import pickle
import unittest
from datetime import timedelta

os.environ["TORCHCOMM_FR_DUMP_DYNAMIC_FILE_NAME"] = "1"

import torch
import torchcomms
from torchcomms.hooks import FlightRecorderHook
from torchcomms.tests.integration.py.TorchCommTestHelpers import (
    get_rank_and_size,
    skip_backend,
)


class TestFlightRecorderAbort(unittest.TestCase):
    """Abort-related FlightRecorder tests.

    These tests intentionally trigger NCCL aborts, which corrupts
    process-level NCCL state. They must run in a separate process
    from other FlightRecorder tests.
    """

    backend = os.environ["TEST_BACKEND"]
    device = torch.device(os.environ.get("TEST_DEVICE", "cuda"))

    @skip_backend("xccl", "XCCL backend does not support comm abort")
    def test_fr_abort_hook_writes_traces_on_simulated_rank_failure(self) -> None:
        """Test abort hook writes traces when simulating a rank failure with threads.

        This test uses threads to simulate a rank crash:
        - Each rank spawns a thread that runs a collective
        - On rank 0, the thread exits early (simulating a crash)
        - On other ranks, the collective times out waiting for rank 0
        - The timeout triggers the abort hook, writing traces

        Note: Uses abort_process_on_timeout_or_error=False so the process doesn't
        actually exit, allowing us to verify the traces were written.
        """
        import threading

        rank, size = get_rank_and_size()
        if size < 2:
            self.skipTest("This test requires at least 2 ranks")

        trace_dir = "/tmp/fr_thread_crash_test_traces"
        os.makedirs(trace_dir, exist_ok=True)
        expected_trace_file = os.path.join(trace_dir, f"fr_crash_trace_{rank}")

        original_dump_file = os.environ.get("TORCHCOMM_FR_DUMP_TEMP_FILE")
        os.environ["TORCHCOMM_FR_DUMP_TEMP_FILE"] = expected_trace_file

        collective_exception: list[Exception] = []

        def run_collective_in_thread(
            comm: torchcomms.TorchComm,
            should_exit_early: bool,
        ) -> None:
            """Run a collective in a thread, optionally exiting early to simulate crash."""
            try:
                t = torch.rand(10, 10, device=self.device)
                if should_exit_early:
                    return
                comm.all_reduce(t, op=torchcomms.ReduceOp.SUM, async_op=False)
            except Exception as e:
                collective_exception.append(e)

        try:
            comm = torchcomms.new_comm(
                backend=self.backend,
                device=self.device,
                name="test_comm_thread_crash",
                timeout=timedelta(milliseconds=2000),
                abort_process_on_timeout_or_error=False,
            )

            recorder = FlightRecorderHook(max_entries=100, isolated=True)
            recorder.register_with_comm(comm)

            t = torch.rand(10, 10, device=self.device)
            comm.all_reduce(t, op=torchcomms.ReduceOp.SUM, async_op=False)

            should_crash = rank == 0
            collective_thread = threading.Thread(
                target=run_collective_in_thread,
                args=(comm, should_crash),
                name=f"collective-thread-rank-{rank}",
            )
            collective_thread.start()
            collective_thread.join(timeout=10)

            if rank != 0:
                self.assertGreater(
                    len(collective_exception),
                    0,
                    "Non-rank-0 should have experienced a timeout/error",
                )

            recorder.dump_file(rank)

            self.assertTrue(
                os.path.exists(expected_trace_file),
                f"Expected trace file {expected_trace_file} was not created",
            )

            with open(expected_trace_file, "rb") as f:
                data = pickle.load(f)

            self.assertIn("version", data)
            self.assertEqual(data["version"], "2.10")
            self.assertIn("entries", data)

            entries = data.get("entries", [])
            self.assertGreater(
                len(entries),
                0,
                f"Trace file for rank {rank} should have entries",
            )

            has_all_reduce = any(
                entry.get("profiling_name") == f"{self.backend}:all_reduce"
                for entry in entries
            )
            self.assertTrue(has_all_reduce, "Trace should contain all_reduce entry")

            recorder.unregister()
            comm.finalize()

        finally:
            if original_dump_file is not None:
                os.environ["TORCHCOMM_FR_DUMP_TEMP_FILE"] = original_dump_file
            elif "TORCHCOMM_FR_DUMP_TEMP_FILE" in os.environ:
                del os.environ["TORCHCOMM_FR_DUMP_TEMP_FILE"]

            for trace_file in glob.glob(f"{expected_trace_file}*"):
                try:
                    os.remove(trace_file)
                except OSError:
                    pass
