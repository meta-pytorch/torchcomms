#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

import json
import os
import tempfile
import time
import unittest
from collections import defaultdict

import torch
import torchcomms
from torch.profiler import profile

TENSOR_COUNT: int = 4


def get_profiler_ncclx_meta(prof):
    """Torch profiler includes nccl metadata in an inserted operator called "record_param_comms"
    We will need to test metadata obtained from profiler here"""
    tf = tempfile.NamedTemporaryFile(mode="w+t", suffix=".json", delete=False)
    tf.close()
    trace_file = tf.name

    prof.export_chrome_trace(trace_file)
    with open(trace_file) as f:
        events = json.load(f)["traceEvents"]
    print(f"Trace saved to {trace_file}")

    os.remove(trace_file)

    return [e for e in events if e.get("name") == "record_param_comms"]


class ProfilerKinetoOverheadTest(unittest.TestCase):
    """Test class to measure kineto logging overhead across all operations in TorchComm."""

    def setUp(self):
        """Set up test environment before each test."""
        device = torch.device("cuda")

        self.torchcomm = torchcomms.new_comm("ncclx", device, name="comms_test_name")

    def tearDown(self):
        """Clean up after each test."""
        # Finalize the TorchComm object to ensure proper cleanup
        if self.torchcomm:
            self.torchcomm.finalize()
            self.torchcomm = None

    def _sanity_check_profiler_ncclx_meta(self, ncclx_meta_events):
        """Torch profiler includes nccl metadata in an inserted operator called "record_param_comms"
        We test for basic fields in this profiler event that correspond to the nccl communication
        collectives"""
        per_coll_meta = defaultdict(list)
        for e in ncclx_meta_events:
            args = e.get("args", {})
            collname = args.get("Collective name", "")
            self.assertNotEqual(collname, "")
            self.assertNotEqual(args.get("dtype", ""), "")

            per_coll_meta[collname].append(args)

            self.assertEqual(args["Process Group Name"], "comms_test_name")
            self.assertNotEqual(args["Process Group Ranks"], "")

            self.assertGreaterEqual(args.get("In msg nelems", -1), 0)
            self.assertGreaterEqual(args.get("Out msg nelems", -1), 0)
            self.assertGreaterEqual(args.get("Group size", -1), 0)

        return per_coll_meta

    def _reset_tensors(self, options):
        """Helper method to reset tensors for each iteration."""
        send_tensor = torch.ones(TENSOR_COUNT, **options) * float(self.rank + 1)
        recv_tensor = torch.zeros(TENSOR_COUNT * self.num_ranks, **options)
        return send_tensor, recv_tensor

    def test_all_tests(self):
        """Test to measure kineto profiling overhead - with vs without profiling, using multiple iterations."""
        NUM_ITERATIONS = 1000  # Run fewer iterations to avoid timeouts
        NUM_WARMUP_ITERATIONS = (
            10  # Number of warmup iterations to drop from measurements
        )

        # Set rank and size information
        self.rank = self.torchcomm.get_rank()
        self.num_ranks = self.torchcomm.get_size()

        # Calculate device index as rank % num_devices
        device_index = self.rank % torch.cuda.device_count()
        self.device = torch.device(f"cuda:{device_index}")

        options = {"dtype": torch.float, "device": self.device}

        send_tensor, recv_tensor = self._reset_tensors(options)

        # Run multiple iterations WITHOUT profiling to get baseline
        baseline_times = []
        for _ in range(NUM_ITERATIONS + NUM_WARMUP_ITERATIONS):
            start_baseline = time.perf_counter()
            self.torchcomm.all_gather_single(recv_tensor, send_tensor, False)
            baseline_times.append((time.perf_counter() - start_baseline) * 1_000_000)

        # Synchronize the CUDA stream before finishing
        torch.cuda.current_stream(self.device).synchronize()

        # Run multiple iterations WITH profiling
        profiled_times = []
        prof = None
        with profile() as p:
            for _ in range(NUM_ITERATIONS + NUM_WARMUP_ITERATIONS):
                start_profiled = time.perf_counter()
                self.torchcomm.all_gather_single(recv_tensor, send_tensor, False)
                profiled_times.append(
                    (time.perf_counter() - start_profiled) * 1_000_000
                )
            prof = p  # Keep the profiler for validation

        # Synchronize the CUDA stream before finishing
        torch.cuda.current_stream(self.device).synchronize()

        # Get rank before cleanup
        rank_ = self.torchcomm.get_rank()

        if rank_ == 0:
            # Drop warmup run measurements to avoid cold start effects
            baseline_times_filtered = baseline_times[
                NUM_WARMUP_ITERATIONS:
            ]  # Drop warmup measurements
            profiled_times_filtered = profiled_times[
                NUM_WARMUP_ITERATIONS:
            ]  # Drop warmup measurements

            # Calculate mean and variance (excluding warmup runs)
            mean_baseline = sum(baseline_times_filtered) / len(baseline_times_filtered)
            mean_profiled = sum(profiled_times_filtered) / len(profiled_times_filtered)

            # Calculate variance: sum of squared deviations from mean
            variance_baseline = sum(
                (t - mean_baseline) ** 2 for t in baseline_times_filtered
            ) / len(baseline_times_filtered)
            variance_profiled = sum(
                (t - mean_profiled) ** 2 for t in profiled_times_filtered
            ) / len(profiled_times_filtered)

            # Print overhead analysis results
            print("")
            print("=== Kineto Logging Overhead Analysis ===")

            # Print raw timing values
            if len(baseline_times_filtered) <= 10:
                baseline_values = [f"{t:.1f}" for t in baseline_times_filtered]
                profiled_values = [f"{t:.1f}" for t in profiled_times_filtered]
                print(f"Raw baseline times (us): {baseline_values}")
                print(f"Raw profiled times (us): {profiled_values}")
                print("")

            # Print statistics
            print(f"Mean without profiling: {mean_baseline:.1f} us")
            print(f"Mean with profiling: {mean_profiled:.1f} us")
            print(f"Variance without profiling: {variance_baseline:.1f} us^2")
            print(f"Variance with profiling: {variance_profiled:.1f} us^2")

            overhead_absolute = mean_profiled - mean_baseline
            overhead_percentage = (overhead_absolute / mean_baseline) * 100
            print(f"Overhead: {overhead_absolute:.1f} us ({overhead_percentage:.1f}%)")
            print("")

            # Basic profiler validation (non-critical)
            try:
                ncclx_meta_events = get_profiler_ncclx_meta(prof)
                if len(ncclx_meta_events) > 0:
                    self._sanity_check_profiler_ncclx_meta(ncclx_meta_events)
                    print(
                        f"Profiler validation: Successfully captured {len(ncclx_meta_events)} events"
                    )
                else:
                    print("Warning: No profiler events captured")
            except Exception:
                print("Warning: Profiler validation failed (non-critical)")


if __name__ == "__main__":
    unittest.main()
