#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
import unittest

import torch
from torch.distributed import BarrierOptions
from torchcomms.device_mesh import _create_torchcomm_process_group
from torchcomms.tests.integration.helpers.TorchCommTestHelpers import (
    TorchCommTestWrapper,
)


class BarrierTest(unittest.TestCase):
    """Test class for barrier operations in TorchComm."""

    # Class variables for test parameters
    num_replays = 4

    def get_wrapper(self):
        return TorchCommTestWrapper()

    def setUp(self):
        """Set up test environment before each test."""
        self.wrapper = self.get_wrapper()
        self.torchcomm = self.wrapper.get_torchcomm()
        self.rank = self.torchcomm.get_rank()
        self.num_ranks = self.torchcomm.get_size()
        self.device = self.torchcomm.get_device()

    def tearDown(self):
        """Clean up after each test."""
        # Explicitly reset the TorchComm object to ensure proper cleanup
        self.torchcomm = None
        self.wrapper = None

    def _sync_barrier(self):
        """Test synchronous barrier with work object."""
        print("Testing sync barrier")

        # Call barrier
        work = self.torchcomm.barrier(False)
        work.wait()

        # No explicit verification needed for barrier, just ensure it completes

    def _sync_barrier_no_work(self):
        """Test synchronous barrier without work object."""
        print("Testing sync barrier without work object")

        # Call barrier without keeping the work object
        self.torchcomm.barrier(False)

    def _async_barrier(self):
        """Test asynchronous barrier with wait."""
        print("Testing async barrier")

        # Call barrier
        work = self.torchcomm.barrier(True)

        # Wait for the barrier to complete
        work.wait()

    def _async_barrier_early_reset(self):
        """Test asynchronous barrier with early reset."""
        print("Testing async barrier with early reset")

        # Call barrier
        work = self.torchcomm.barrier(True)

        # Wait for the work to complete before resetting
        work.wait()

        # Reset the work object
        work = None

    def _graph_barrier(self):
        """Test CUDA Graph barrier."""
        print("Testing CUDA Graph barrier")

        # Create a non-default CUDA stream (required for CUDA graph capture)
        stream = torch.cuda.Stream()

        # Set the stream as current for graph capture
        with torch.cuda.stream(stream):
            # Create PyTorch CUDA graph
            graph = torch.cuda.CUDAGraph()

            # Capture the barrier operation in the graph
            with torch.cuda.graph(graph):
                # Call barrier without keeping the work object
                self.torchcomm.barrier(False)

            # Replay the captured graph multiple times
            for _ in range(self.num_replays):
                graph.replay()

                # No explicit verification needed for barrier, just ensure it completes

    def _sync_barrier_blocks_host_on_stream(self):
        """A synchronous c10d barrier must block the host until prior work on the
        current stream has completed.

        This mirrors stock ProcessGroupNCCL, whose barrier host-blocks the CPU
        thread. It is a regression test for a trtllm flashinfer one-shot Lamport
        all_reduce deadlock: that code clears its IPC buffers asynchronously on
        the stream and then issues a synchronous barrier before the first
        all_reduce, relying on the barrier to flush that clear. If the barrier
        only inserts a stream-ordered wait (no host sync), the first all_reduce
        races the clear and both ranks spin forever. The host block is applied by
        BackendWrapper (the c10d layer), so the barrier is driven through a c10d
        ProcessGroup rather than the native torchcomm.barrier().
        """
        pg = _create_torchcomm_process_group(self.torchcomm, "barrier_host_block_pg")
        with torch.cuda.device(self.device):
            stream = torch.cuda.current_stream()

            # Enqueue a long-running kernel on the current stream so the host
            # would observe an unfinished stream if the barrier did not
            # synchronize it.
            torch.cuda._sleep(1_000_000_000)
            self.assertFalse(
                stream.query(),
                "precondition: enqueued work should leave the stream busy",
            )

            # Synchronous c10d barrier + wait() must block the host until the
            # stream is drained.
            opts = BarrierOptions()
            opts.asyncOp = False
            opts.device = self.device
            work = pg.barrier(opts=opts)
            work.wait()

            self.assertTrue(
                stream.query(),
                "synchronous barrier must block the host until prior stream work completes",
            )
        print(f"[Rank {self.rank}] sync barrier blocked host until stream drained")

    @unittest.skipIf(
        os.getenv("TEST_BACKEND") not in ("nccl", "ncclx"),
        "Host-blocking synchronous barrier is validated for the GPU backends "
        "(nccl, ncclx)",
    )
    def test_sync_barrier_blocks_host_on_stream(self):
        """Synchronous barrier + wait() must block the host until prior stream work done."""
        self._sync_barrier_blocks_host_on_stream()

    def test_sync_barrier(self):
        """Test synchronous barrier with work object."""
        self._sync_barrier()

    def test_sync_barrier_no_work(self):
        """Test synchronous barrier without work object."""
        self._sync_barrier_no_work()

    def test_async_barrier(self):
        """Test asynchronous barrier with wait."""
        self._async_barrier()

    def test_async_barrier_early_reset(self):
        """Test asynchronous barrier with early reset."""
        self._async_barrier_early_reset()

    @unittest.skipIf(
        os.getenv("TEST_BACKEND") != "ncclx", "Skipping NCCLX-only scatter tests"
    )
    def test_graph_barrier(self):
        """Test CUDA Graph barrier."""
        self._graph_barrier()


if __name__ == "__main__":
    unittest.main()
