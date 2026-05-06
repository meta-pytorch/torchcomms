#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
import unittest
from datetime import timedelta

import torch
import torchcomms
from torch.distributed import TCPStore


class AbortTest(unittest.TestCase):
    """Test the abort() API for TorchComm."""

    SUPPORTED_BACKENDS = {"nccl", "ncclx", "rccl", "rcclx"}

    _shared_store = None

    def setUp(self):
        self.backend = os.getenv("TEST_BACKEND", "")

        if self.backend not in self.SUPPORTED_BACKENDS:
            self.skipTest(f"Backend {self.backend} does not support abort()")

        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available")

        self.rank = int(
            os.environ.get("RANK", os.environ.get("OMPI_COMM_WORLD_RANK", 0))
        )
        self.world_size = int(
            os.environ.get("WORLD_SIZE", os.environ.get("OMPI_COMM_WORLD_SIZE", 1))
        )
        device_id = self.rank % torch.cuda.device_count()
        self.device = torch.device(f"cuda:{device_id}")

        if AbortTest._shared_store is None:
            master_addr = os.environ.get("MASTER_ADDR", "localhost")
            master_port = int(os.environ.get("MASTER_PORT", "29500"))
            AbortTest._shared_store = TCPStore(
                host_name=master_addr,
                port=master_port,
                world_size=self.world_size,
                is_master=(self.rank == 0),
                timeout=timedelta(seconds=30),
            )
        self.store = AbortTest._shared_store

    def _create_reconfigurable_comm(self, name, uuid):
        comm = torchcomms.new_comm(
            self.backend,
            self.device,
            name,
            enable_reconfigure=True,
            store=self.store,
        )
        my_handle = comm.get_init_handle()
        self.store.set(f"{name}_{self.rank}", my_handle)
        handles = []
        for i in range(self.world_size):
            h = self.store.get(f"{name}_{i}").decode("utf-8")
            handles.append(h)
        work = comm.reconfigure(
            uuid=uuid,
            init_handles=handles,
            timeout=timedelta(seconds=30),
        )
        work.wait()
        return comm

    def test_abort(self):
        """After abort(), subsequent operations should raise RuntimeError."""
        comm = self._create_reconfigurable_comm("abort_error_state", 0)

        comm.abort()
        comm.finalize()

    def test_abort_then_reconfigure(self):
        """After abort(), reconfigure() should recover the communicator."""
        comm = self._create_reconfigurable_comm("abort_recover", 10)

        comm.abort()

        my_handle = comm.get_init_handle()
        self.store.set(f"abort_recover2_{self.rank}", my_handle)
        handles = []
        for i in range(self.world_size):
            h = self.store.get(f"abort_recover2_{i}").decode("utf-8")
            handles.append(h)

        work = comm.reconfigure(
            uuid=11,
            init_handles=handles,
            timeout=timedelta(seconds=30),
        )
        work.wait()

        tensor = torch.ones(4, dtype=torch.float, device=self.device) * (
            comm.get_rank() + 1
        )
        comm.all_reduce(tensor, torchcomms.ReduceOp.SUM, async_op=False)
        torch.cuda.synchronize()

        expected_value = sum(range(1, self.world_size + 1))
        expected = torch.ones(4, dtype=torch.float, device="cpu") * expected_value
        self.assertTrue(torch.allclose(tensor.cpu(), expected))

        comm.finalize()


if __name__ == "__main__":
    unittest.main()
