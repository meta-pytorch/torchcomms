#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Tests for the reconfigure() Fault Tolerance API in TorchComms.

This test verifies:
1. Backends that don't implement reconfigure() raise RuntimeError
2. Backends that implement reconfigure() can successfully reconfigure the
   communicator with a new set of peers
3. After successful reconfigure, collective operations are permitted
"""

import os
import unittest
from datetime import timedelta

import torch
from torch.distributed import TCPStore
from torchcomms.tests.integration.py.TorchCommTestHelpers import TorchCommTestWrapper


class ReconfigureTest(unittest.TestCase):
    """Test class for reconfigure() fault tolerance API."""

    # Backends that implement reconfigure()
    SUPPORTED_BACKENDS = {"mccl"}

    # Class-level shared store to avoid port conflicts between tests
    _shared_store = None

    def get_wrapper(self):
        return TorchCommTestWrapper()

    def setUp(self):
        """Set up test environment before each test."""
        self.backend = os.getenv("TEST_BACKEND", "")

        # For supported backends, we need CUDA and a shared TCPStore
        if self._is_supported_backend():
            if not torch.cuda.is_available():
                self.skipTest("CUDA is not available")

            # Get rank and world size from environment
            self.rank = int(
                os.environ.get("RANK", os.environ.get("OMPI_COMM_WORLD_RANK", 0))
            )
            self.world_size = int(
                os.environ.get("WORLD_SIZE", os.environ.get("OMPI_COMM_WORLD_SIZE", 1))
            )

            # Setup shared TCP store - create once and reuse for all tests
            # Each test uses unique key prefixes to avoid conflicts
            if ReconfigureTest._shared_store is None:
                master_addr = os.environ.get("MASTER_ADDR", "localhost")
                master_port = int(os.environ.get("MASTER_PORT", "29500"))

                ReconfigureTest._shared_store = TCPStore(
                    host_name=master_addr,
                    port=master_port,
                    world_size=self.world_size,
                    is_master=(self.rank == 0),
                    timeout=timedelta(seconds=30),
                )

            self.store = ReconfigureTest._shared_store

            device_id = self.rank % torch.cuda.device_count()
            self.device = torch.device(f"cuda:{device_id}")
        else:
            # For unsupported backends, use TorchCommTestWrapper
            self.wrapper = self.get_wrapper()
            self.torchcomm = self.wrapper.get_torchcomm()
            self.rank = self.torchcomm.get_rank()
            self.world_size = self.torchcomm.get_size()
            self.device = self.torchcomm.get_device()

    def tearDown(self):
        """Clean up after each test."""
        if not self._is_supported_backend():
            self.torchcomm = None
            self.wrapper = None

    def _is_supported_backend(self):
        """Check if current backend supports reconfigure()."""
        return self.backend in self.SUPPORTED_BACKENDS

    def test_reconfigure_unsupported_backend(self):
        """Test reconfigure() raises RuntimeError for unsupported backends."""
        if self._is_supported_backend():
            self.skipTest(
                f"Backend {self.backend} supports reconfigure(), skipping negative test"
            )

        # Backend doesn't support reconfigure() - should raise RuntimeError
        with self.assertRaises(RuntimeError) as context:
            self.torchcomm.reconfigure(
                uuid=0,
                init_handles=["test_handle"],
                timeout=timedelta(milliseconds=5000),
            )

        self.assertIn("reconfigure not implemented", str(context.exception))
        print(f"[Rank {self.rank}] Expected RuntimeError raised: {context.exception}")

    def test_reconfigure_basic(self):
        """Test basic reconfigure with ordered handles (list)."""
        if not self._is_supported_backend():
            self.skipTest(f"Backend {self.backend} does not support reconfigure()")

        import torchcomms

        # Create communicator in dynamic regime
        hints = {"initDynamicRegime": "true"}
        comm = torchcomms.new_comm(
            "mccl", self.device, "reconfigure_basic", hints=hints
        )

        # Get init handle and exchange with all ranks
        my_handle = comm.get_init_handle()
        self.assertIsNotNone(my_handle)
        self.assertNotEqual(my_handle, "")

        # Store handle in TCP store
        key = f"test_reconfigure_basic_{self.rank}"
        self.store.set(key, my_handle)

        # Collect all handles (ordered by rank)
        all_handles = []
        for i in range(self.world_size):
            url_key = f"test_reconfigure_basic_{i}"
            handle = self.store.get(url_key).decode("utf-8")
            all_handles.append(handle)

        print(f"[Rank {self.rank}] Collected handles: {all_handles}")

        # Reconfigure with ordered handles (list)
        work = comm.reconfigure(
            uuid=0,
            init_handles=all_handles,
            timeout=timedelta(milliseconds=5000),
        )
        self.assertIsNotNone(work)

        # Wait for reconfigure to complete
        work.wait_blocking()

        # Verify rank and size are now correct
        self.assertEqual(comm.get_rank(), self.rank)
        self.assertEqual(comm.get_size(), self.world_size)

        print(f"[Rank {self.rank}] Reconfigure completed successfully")

        comm.finalize()

    def test_reconfigure_unordered_handles(self):
        """Test reconfigure with unordered handles (set)."""
        if not self._is_supported_backend():
            self.skipTest(f"Backend {self.backend} does not support reconfigure()")

        import torchcomms

        # Create communicator in dynamic regime
        hints = {"initDynamicRegime": "true"}
        comm = torchcomms.new_comm(
            "mccl", self.device, "reconfigure_unordered", hints=hints
        )

        # Get init handle and exchange with all ranks
        my_handle = comm.get_init_handle()

        # Store handle in TCP store
        key = f"test_reconfigure_unordered_{self.rank}"
        self.store.set(key, my_handle)

        # Collect all handles as a set (unordered)
        all_handles = set()
        for i in range(self.world_size):
            url_key = f"test_reconfigure_unordered_{i}"
            handle = self.store.get(url_key).decode("utf-8")
            all_handles.add(handle)

        print(f"[Rank {self.rank}] Collected handles (set): {all_handles}")

        # Reconfigure with unordered handles (set)
        work = comm.reconfigure(
            uuid=1,
            init_handles=all_handles,
            timeout=timedelta(milliseconds=5000),
        )

        work.wait_blocking()

        # Verify communicator is initialized (rank may differ from original)
        self.assertGreaterEqual(comm.get_rank(), 0)
        self.assertEqual(comm.get_size(), self.world_size)

        print(
            f"[Rank {self.rank}] Reconfigure with set completed, "
            f"assigned rank: {comm.get_rank()}"
        )

        comm.finalize()

    def test_reconfigure_then_collective(self):
        """Test that collective operations work after reconfigure."""
        if not self._is_supported_backend():
            self.skipTest(f"Backend {self.backend} does not support reconfigure()")

        import torchcomms

        # Create communicator in dynamic regime
        hints = {"initDynamicRegime": "true"}
        comm = torchcomms.new_comm(
            "mccl", self.device, "reconfigure_collective", hints=hints
        )

        # Get init handle and exchange with all ranks
        my_handle = comm.get_init_handle()

        key = f"test_reconfigure_collective_{self.rank}"
        self.store.set(key, my_handle)

        all_handles = []
        for i in range(self.world_size):
            url_key = f"test_reconfigure_collective_{i}"
            handle = self.store.get(url_key).decode("utf-8")
            all_handles.append(handle)

        # Reconfigure
        work = comm.reconfigure(
            uuid=2,
            init_handles=all_handles,
            timeout=timedelta(milliseconds=5000),
        )
        work.wait_blocking()

        # Test send/recv after reconfigure
        if self.world_size > 1:
            count = 4
            send_tensor = torch.ones(count, dtype=torch.float, device=self.device) * (
                self.rank + 1
            )
            recv_tensor = torch.zeros(count, dtype=torch.float, device=self.device)

            send_rank = (self.rank + 1) % self.world_size
            recv_rank = (self.rank - 1 + self.world_size) % self.world_size

            # Alternate send/recv order based on rank to avoid deadlock
            if self.rank % 2 == 0:
                send_work = comm.send(send_tensor, send_rank, async_op=True)
                recv_work = comm.recv(recv_tensor, recv_rank, async_op=True)
            else:
                recv_work = comm.recv(recv_tensor, recv_rank, async_op=True)
                send_work = comm.send(send_tensor, send_rank, async_op=True)

            send_work.wait()
            recv_work.wait()

            torch.cuda.current_stream().synchronize()

            # Verify received data
            expected = torch.ones(count, dtype=torch.float, device="cpu") * (
                recv_rank + 1
            )
            self.assertTrue(
                torch.allclose(recv_tensor.cpu(), expected),
                f"[Rank {self.rank}] Send/recv after reconfigure failed",
            )

            print(f"[Rank {self.rank}] Send/recv after reconfigure succeeded")

        comm.finalize()


if __name__ == "__main__":
    unittest.main()
