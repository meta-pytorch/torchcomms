#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Performance benchmark for CPU collectives via TorchComms MCCL backend.

Measures latency of CPU collectives on 8 ranks on a single host across
multiple message sizes. Uses the unified APIs with CPU tensor auto-detection.

Usage:
  buck test fbcode//comms/torchcomms/mccl/tests/integration/py:CpuCollectiveBenchmark_1x8_backend_mccl
"""

import os
import time
import unittest

import torch
import torchcomms
from torchcomms.tests.integration.helpers.TorchCommTestHelpers import (
    TorchCommTestWrapper,
)


class CpuCollectiveBenchmark(unittest.TestCase):
    """Benchmark CPU collectives via unified TorchComms APIs."""

    MSG_SIZES = [64, 1024, 65536, 1048576]
    NUM_WARMUP = 2
    NUM_ITERS = 10

    def setUp(self):
        os.environ["NCCL_SOCKET_IFNAME"] = "lo"
        self.wrapper = TorchCommTestWrapper()
        self.torchcomm = self.wrapper.get_torchcomm()
        self.rank = self.torchcomm.get_rank()
        self.num_ranks = self.torchcomm.get_size()

    def tearDown(self):
        self.torchcomm = None
        self.wrapper = None

    def _bench(self, name, fn, num_warmup=None, num_iters=None):
        warmup = num_warmup if num_warmup is not None else self.NUM_WARMUP
        iters = num_iters if num_iters is not None else self.NUM_ITERS

        for _ in range(warmup):
            fn()

        latencies = []
        for _ in range(iters):
            start = time.monotonic()
            fn()
            elapsed_us = (time.monotonic() - start) * 1e6
            latencies.append(elapsed_us)

        avg_us = sum(latencies) / len(latencies)
        min_us = min(latencies)
        max_us = max(latencies)
        if self.rank == 0:
            print(
                f"  {name}: avg={avg_us:.0f}us  min={min_us:.0f}us  max={max_us:.0f}us"
            )

    def test_barrier_cpu(self):
        """Benchmark CPU barrier via use_cpu_barrier comm hint."""
        if self.rank == 0:
            print("\n=== CPU Barrier ===")

        wrapper = TorchCommTestWrapper(hints={"use_cpu_barrier": "true"})
        torchcomm = wrapper.get_torchcomm()

        def barrier_fn():
            torchcomm.barrier(False)

        self._bench("barrier_cpu", barrier_fn)

    def test_broadcast_cpu(self):
        """Benchmark CPU broadcast via tensor auto-detection."""
        if self.rank == 0:
            print("\n=== CPU Broadcast ===")

        for size in self.MSG_SIZES:
            num_elements = size // 4  # float32
            tensor = torch.zeros(num_elements, dtype=torch.float32, device="cpu")
            if self.rank == 0:
                tensor.fill_(1.0)

            def broadcast_fn(t=tensor):
                self.torchcomm.broadcast(t, 0, False)

            self._bench(f"broadcast_cpu_{size}B", broadcast_fn)

    def test_all_gather_cpu(self):
        """Benchmark CPU all_gather_single via tensor auto-detection."""
        if self.rank == 0:
            print("\n=== CPU AllGather ===")

        for size in self.MSG_SIZES:
            num_elements = size // 4
            input_tensor = torch.full(
                (num_elements,), float(self.rank), dtype=torch.float32, device="cpu"
            )
            output_tensor = torch.zeros(
                num_elements * self.num_ranks, dtype=torch.float32, device="cpu"
            )

            def allgather_fn(inp=input_tensor, out=output_tensor):
                self.torchcomm.all_gather_single(out, inp, False)

            self._bench(f"all_gather_cpu_{size}B", allgather_fn)

    def test_all_reduce_cpu(self):
        """Benchmark CPU all_reduce via tensor auto-detection."""
        if self.rank == 0:
            print("\n=== CPU AllReduce ===")

        for size in self.MSG_SIZES:
            num_elements = size // 4
            tensor = torch.full(
                (num_elements,),
                float(self.rank + 1),
                dtype=torch.float32,
                device="cpu",
            )

            def allreduce_fn(t=tensor):
                self.torchcomm.all_reduce(t, torchcomms.ReduceOp.SUM, False)

            self._bench(f"all_reduce_cpu_{size}B", allreduce_fn)


if __name__ == "__main__":
    unittest.main()
