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

    SUPPORTED_BACKENDS = {"mccl", "gloo", "nccl", "ncclx", "rccl", "rcclx"}

    _shared_store = None

    def get_wrapper(self):
        return TorchCommTestWrapper()

    def setUp(self):
        """Set up test environment before each test."""
        self.backend = os.getenv("TEST_BACKEND", "")

        if self._is_supported_backend():
            self.rank = int(
                os.environ.get("RANK", os.environ.get("OMPI_COMM_WORLD_RANK", 0))
            )
            self.world_size = int(
                os.environ.get("WORLD_SIZE", os.environ.get("OMPI_COMM_WORLD_SIZE", 1))
            )

            if self.backend == "gloo":
                self.device = torch.device(os.environ.get("TEST_DEVICE", "cpu"))
            else:
                if not torch.cuda.is_available():
                    self.skipTest("CUDA is not available")

                if self.backend == "mccl":
                    os.environ["NCCL_COMM_STATE_DEBUG_TOPO"] = "nolocal"
                    os.environ["NCCL_IGNORE_TOPO_LOAD_FAILURE"] = "1"

                device_id = self.rank % torch.cuda.device_count()
                self.device = torch.device(f"cuda:{device_id}")

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
        else:
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

    def _get_store_for_comm(self):
        """Get the store to pass to new_comm for NCCL/NCCLx bootstrap."""
        return getattr(self, "store", None)

    def _collect_handles(self, comm, key_prefix):
        """Collect init handles from all ranks."""
        my_handle = comm.get_init_handle()
        if self.backend == "gloo":
            return [my_handle] * self.world_size
        key = f"{key_prefix}_{self.rank}"
        self.store.set(key, my_handle)
        handles = []
        for i in range(self.world_size):
            handle = self.store.get(f"{key_prefix}_{i}").decode("utf-8")
            handles.append(handle)
        return handles

    def test_reconfigure_unsupported_backend(self):
        """Test reconfigure() raises RuntimeError for unsupported backends."""
        if self._is_supported_backend():
            self.skipTest(
                f"Backend {self.backend} supports reconfigure(), skipping negative test"
            )

        with self.assertRaises(RuntimeError) as context:
            self.torchcomm.reconfigure(
                uuid=0,
                init_handles=["test_handle"],
                timeout=timedelta(milliseconds=5000),
            )

        self.assertIn("reconfigure not implemented", str(context.exception))
        print(f"[Rank {self.rank}] Expected RuntimeError raised: {context.exception}")

    def test_enable_reconfigure_unsupported_backend(self):
        """Test enable_reconfigure=True raises RuntimeError for unsupported backends."""
        if self._is_supported_backend():
            self.skipTest(
                f"Backend {self.backend} supports reconfigure, skipping negative test"
            )

        import torchcomms

        with self.assertRaises(RuntimeError) as context:
            torchcomms.new_comm(
                self.backend,
                self.device,
                "test_reconfigure_unsupported",
                enable_reconfigure=True,
            )

        print(
            f"[Rank {self.rank}] Expected RuntimeError for enable_reconfigure=True: "
            f"{context.exception}"
        )

    def test_reconfigure_basic(self):
        """Test basic reconfigure with ordered handles (list)."""
        if not self._is_supported_backend():
            self.skipTest(f"Backend {self.backend} does not support reconfigure()")

        import torchcomms

        comm = torchcomms.new_comm(
            self.backend,
            self.device,
            "reconfigure_basic",
            enable_reconfigure=True,
            store=self._get_store_for_comm(),
        )

        all_handles = self._collect_handles(comm, "test_reconfigure_basic")
        self.assertGreater(len(all_handles), 0)

        print(f"[Rank {self.rank}] Collected handles: {all_handles}")

        work = comm.reconfigure(
            uuid=0,
            init_handles=all_handles,
            timeout=timedelta(milliseconds=30000),
        )
        self.assertIsNotNone(work)

        work.wait()

        self.assertGreaterEqual(comm.get_rank(), 0)
        self.assertLess(comm.get_rank(), self.world_size)
        self.assertEqual(comm.get_size(), self.world_size)

        print(f"[Rank {self.rank}] Reconfigure completed successfully")

        comm.finalize()

    def test_reconfigure_unordered_handles(self):
        """Test reconfigure with unordered handles (set)."""
        if not self._is_supported_backend():
            self.skipTest(f"Backend {self.backend} does not support reconfigure()")

        if self.backend == "gloo":
            self.skipTest("Gloo uses identical handles; unordered set collapses to 1")

        import torchcomms

        comm = torchcomms.new_comm(
            self.backend,
            self.device,
            "reconfigure_unordered",
            enable_reconfigure=True,
            store=self._get_store_for_comm(),
        )

        all_handles = set(self._collect_handles(comm, "test_reconfigure_unordered"))

        print(f"[Rank {self.rank}] Collected handles (set): {all_handles}")

        work = comm.reconfigure(
            uuid=1,
            init_handles=all_handles,
            timeout=timedelta(milliseconds=30000),
        )

        work.wait()

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

        comm = torchcomms.new_comm(
            self.backend,
            self.device,
            "reconfigure_collective",
            enable_reconfigure=True,
            store=self._get_store_for_comm(),
        )

        all_handles = self._collect_handles(comm, "test_reconfigure_collective")

        work = comm.reconfigure(
            uuid=2,
            init_handles=all_handles,
            timeout=timedelta(milliseconds=30000),
        )
        work.wait()

        if self.world_size > 1:
            my_rank = comm.get_rank()
            count = 4
            send_tensor = torch.ones(count, dtype=torch.float, device=self.device) * (
                my_rank + 1
            )
            recv_tensor = torch.zeros(count, dtype=torch.float, device=self.device)

            send_rank = (my_rank + 1) % self.world_size
            recv_rank = (my_rank - 1 + self.world_size) % self.world_size

            if my_rank % 2 == 0:
                send_work = comm.send(send_tensor, send_rank, async_op=True)
                recv_work = comm.recv(recv_tensor, recv_rank, async_op=True)
            else:
                recv_work = comm.recv(recv_tensor, recv_rank, async_op=True)
                send_work = comm.send(send_tensor, send_rank, async_op=True)

            send_work.wait()
            recv_work.wait()

            if self.device.type == "cuda":
                torch.cuda.current_stream().synchronize()

            expected = torch.ones(count, dtype=torch.float, device="cpu") * (
                recv_rank + 1
            )
            self.assertTrue(
                torch.allclose(recv_tensor.cpu(), expected),
                f"[Rank {my_rank}] Send/recv after reconfigure failed",
            )

            print(f"[Rank {my_rank}] Send/recv after reconfigure succeeded")

        comm.finalize()

    def test_reconfigure_then_allreduce(self):
        """Test that allreduce works after reconfigure."""
        if not self._is_supported_backend():
            self.skipTest(f"Backend {self.backend} does not support reconfigure()")

        import torchcomms

        comm = torchcomms.new_comm(
            self.backend,
            self.device,
            "reconfigure_allreduce",
            enable_reconfigure=True,
            store=self._get_store_for_comm(),
        )

        all_handles = self._collect_handles(comm, "test_reconfigure_allreduce")

        work = comm.reconfigure(
            uuid=3,
            init_handles=all_handles,
            timeout=timedelta(milliseconds=30000),
        )
        work.wait()

        my_rank = comm.get_rank()
        tensor = torch.ones(4, dtype=torch.float, device=self.device) * (my_rank + 1)
        comm.all_reduce(tensor, torchcomms.ReduceOp.SUM, async_op=False)

        if self.device.type == "cuda":
            torch.cuda.current_stream().synchronize()

        expected_value = sum(range(1, self.world_size + 1))
        expected = torch.ones(4, dtype=torch.float, device="cpu") * expected_value
        self.assertTrue(
            torch.allclose(tensor.cpu(), expected),
            f"[Rank {my_rank}] AllReduce after reconfigure failed: "
            f"got {tensor.cpu()}, expected {expected}",
        )

        print(f"[Rank {my_rank}] AllReduce after reconfigure succeeded")

        comm.finalize()

    def _create_reconfigured_comm(self, name, uuid):
        """Helper: create a comm and perform initial reconfigure."""
        import torchcomms

        comm = torchcomms.new_comm(
            self.backend,
            self.device,
            name,
            enable_reconfigure=True,
            store=self._get_store_for_comm(),
            timeout=timedelta(seconds=120),
        )

        init_handles = self._collect_handles(comm, f"{name}_init")
        work = comm.reconfigure(
            uuid=uuid,
            init_handles=init_handles,
            timeout=timedelta(seconds=30),
        )
        work.wait()

        post_handles = self._collect_handles(comm, f"{name}_post")
        return comm, post_handles

    def test_shrink_basic(self):
        """Test shrink: exclude last rank, allreduce on child comm."""
        if not self._is_supported_backend():
            self.skipTest(f"Backend {self.backend} does not support reconfigure()")
        if self.world_size < 3:
            self.skipTest("Need at least 3 ranks for shrink test")

        comm, all_handles = self._create_reconfigured_comm("shrink_basic", 100)
        exclude_rank = self.world_size - 1

        if self.rank == exclude_rank:
            comm.finalize()
            return

        surviving_handles = [h for i, h in enumerate(all_handles) if i != exclude_rank]
        work = comm.reconfigure(
            uuid=101,
            init_handles=surviving_handles,
            timeout=timedelta(seconds=30),
        )
        work.wait()

        new_size = comm.get_size()
        self.assertEqual(new_size, self.world_size - 1)

        my_rank = comm.get_rank()
        tensor = torch.ones(1024, dtype=torch.float, device=self.device) * (my_rank + 1)
        comm.all_reduce(tensor, op=self._reduce_op_sum(), async_op=False)

        expected = sum(range(1, new_size + 1))
        self.assertTrue(
            torch.allclose(tensor, torch.full_like(tensor, expected)),
            f"AllReduce post-shrink failed: got {tensor[0].item()}, expected {expected}",
        )

        print(f"[Rank {my_rank}] Shrink {self.world_size} -> {new_size}, allreduce OK")
        comm.finalize()

    def test_shrink_multiple_collectives(self):
        """Test multiple allreduces + broadcast on child comm after shrink."""
        if not self._is_supported_backend():
            self.skipTest(f"Backend {self.backend} does not support reconfigure()")
        if self.world_size < 3:
            self.skipTest("Need at least 3 ranks for shrink test")

        comm, all_handles = self._create_reconfigured_comm("shrink_multi", 200)
        exclude_rank = self.world_size - 1

        if self.rank == exclude_rank:
            comm.finalize()
            return

        surviving_handles = [h for i, h in enumerate(all_handles) if i != exclude_rank]
        work = comm.reconfigure(
            uuid=201,
            init_handles=surviving_handles,
            timeout=timedelta(seconds=30),
        )
        work.wait()

        new_size = comm.get_size()
        my_rank = comm.get_rank()

        for i in range(3):
            tensor = torch.ones(2048, dtype=torch.float, device=self.device) * (
                my_rank + 1 + i
            )
            comm.all_reduce(tensor, op=self._reduce_op_sum(), async_op=False)
            expected = sum(range(1 + i, new_size + 1 + i))
            self.assertTrue(
                torch.allclose(tensor, torch.full_like(tensor, expected)),
                f"AllReduce iteration {i} failed",
            )

        tensor = torch.zeros(1024, dtype=torch.float, device=self.device)
        if my_rank == 0:
            tensor.fill_(42.0)
        comm.broadcast(tensor, root=0, async_op=False)
        self.assertTrue(
            torch.allclose(tensor, torch.full_like(tensor, 42.0)),
            "Broadcast after shrink failed",
        )

        print(
            f"[Rank {my_rank}] 3 AllReduce + 1 Broadcast on child comm OK"
        )
        comm.finalize()

    def test_shrink_exclude_middle_rank(self):
        """Test shrink excluding a middle rank, verify rank renumbering."""
        if not self._is_supported_backend():
            self.skipTest(f"Backend {self.backend} does not support reconfigure()")
        if self.world_size < 3:
            self.skipTest("Need at least 3 ranks for shrink test")

        comm, all_handles = self._create_reconfigured_comm("shrink_middle", 300)
        exclude_rank = self.world_size // 2

        if self.rank == exclude_rank:
            comm.finalize()
            return

        surviving_handles = [h for i, h in enumerate(all_handles) if i != exclude_rank]
        work = comm.reconfigure(
            uuid=301,
            init_handles=surviving_handles,
            timeout=timedelta(seconds=30),
        )
        work.wait()

        new_size = comm.get_size()
        my_rank = comm.get_rank()
        self.assertEqual(new_size, self.world_size - 1)
        self.assertGreaterEqual(my_rank, 0)
        self.assertLess(my_rank, new_size)

        tensor = torch.ones(1024, dtype=torch.float, device=self.device) * (my_rank + 1)
        comm.all_reduce(tensor, op=self._reduce_op_sum(), async_op=False)

        expected = sum(range(1, new_size + 1))
        self.assertTrue(
            torch.allclose(tensor, torch.full_like(tensor, expected)),
            f"AllReduce after middle-rank shrink failed",
        )

        print(
            f"[Rank {my_rank}] Shrink excluding middle rank {exclude_rank}, "
            f"new size={new_size}, allreduce OK"
        )
        comm.finalize()

    def _reduce_op_sum(self):
        import torchcomms

        return torchcomms.ReduceOp.SUM


if __name__ == "__main__":
    unittest.main()
