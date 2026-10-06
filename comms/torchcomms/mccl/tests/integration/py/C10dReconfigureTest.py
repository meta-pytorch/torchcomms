#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.

"""End-to-end coverage for MCCL reconfiguration through c10d."""

import os
import unittest
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.distributed.config as dist_config
from torchcomms._comms import _BackendWrapper


class C10dReconfigureTest(unittest.TestCase):
    TIMEOUT = timedelta(seconds=60)
    INITIAL_UUID = 1 << 62
    SHRINK_UUID = INITIAL_UUID + 1

    def setUp(self) -> None:
        if os.environ.get("TEST_BACKEND") != "mccl":
            self.skipTest("This test requires the MCCL backend")
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available")

        os.environ["NCCL_COMM_STATE_DEBUG_TOPO"] = "nolocal"
        os.environ["NCCL_IGNORE_TOPO_LOAD_FAILURE"] = "1"

        self.rank = int(
            os.environ.get("RANK", os.environ.get("OMPI_COMM_WORLD_RANK", "0"))
        )
        self.world_size = int(
            os.environ.get("WORLD_SIZE", os.environ.get("OMPI_COMM_WORLD_SIZE", "1"))
        )
        self.assertGreaterEqual(self.world_size, 3)

        local_rank = int(
            os.environ.get(
                "LOCAL_RANK", os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", self.rank)
            )
        )
        self.device = torch.device("cuda", local_rank)
        torch.cuda.set_device(self.device)

        self.store = dist.TCPStore(
            host_name=os.environ.get("MASTER_ADDR", "localhost"),
            port=int(os.environ.get("MASTER_PORT", "29500")),
            world_size=self.world_size,
            is_master=self.rank == 0,
            timeout=self.TIMEOUT,
        )

    def tearDown(self) -> None:
        if dist.is_initialized():
            dist.destroy_process_group()

    def _collect_handles(self, prefix: str) -> list[str]:
        self.store.set(f"{prefix}_{self.rank}", dist._get_reconfigure_handle())
        return [
            self.store.get(f"{prefix}_{rank}").decode("utf-8")
            for rank in range(self.world_size)
        ]

    def _store_barrier(self, prefix: str) -> None:
        self.store.set(f"{prefix}_{self.rank}", "1")
        for rank in range(self.world_size):
            self.store.get(f"{prefix}_{rank}")

    def _reconfigure(self, uuid: int, handles: list[str]) -> None:
        work = dist._reconfigure(
            uuid=uuid,
            handles=handles,
            timeout=self.TIMEOUT,
            hints={"reconfigureQuorumId": f"c10d-mccl-{uuid}"},
        )
        self.assertIsNotNone(work)
        self.assertTrue(work.wait())

    def _assert_all_reduce(self, pool: torch.cuda.MemPool) -> None:
        rank = dist.get_rank()
        size = dist.get_world_size()
        with torch.cuda.use_mem_pool(pool):
            tensor = torch.full(
                (16,), float(rank + 1), dtype=torch.float32, device=self.device
            )

        dist.all_reduce(tensor)
        torch.cuda.synchronize(self.device)
        expected = float(size * (size + 1) // 2)
        torch.testing.assert_close(
            tensor,
            torch.full_like(tensor, expected),
            rtol=0,
            atol=0,
        )

    def test_initial_config_and_shrink_through_backend_wrapper(self) -> None:
        dist_config.use_torchcomms = False
        self.assertNotIn("MCCL", dist.Backend._plugins)

        dist.init_process_group(
            backend="mccl",
            store=self.store,
            rank=self.rank,
            world_size=self.world_size,
            timeout=self.TIMEOUT,
            device_id=self.device,
            enable_reconfigure=True,
        )

        backend = dist.get_backend_impl(device=self.device)
        self.assertFalse(dist_config.use_torchcomms)
        self.assertIsInstance(backend, _BackendWrapper)
        self.assertEqual(dist.get_backend(), "mccl")
        self.assertEqual(backend.get_comm().get_backend(), "mccl")
        self.assertTrue(dist._supports_reconfigure())
        self.assertEqual(backend.rank(), -1)
        self.assertEqual(backend.size(), -1)
        self.assertEqual(dist.get_rank(), -1)
        self.assertEqual(dist.get_world_size(), -1)

        tensor = torch.ones((1,), dtype=torch.float32, device=self.device)
        with self.assertRaisesRegex(RuntimeError, "TorchCommMCCL not initialized"):
            dist.all_reduce(tensor)

        handles = self._collect_handles("c10d_mccl_initial")
        self._reconfigure(self.INITIAL_UUID, handles)
        self.assertEqual(dist.get_rank(), self.rank)
        self.assertEqual(dist.get_world_size(), self.world_size)
        self.assertEqual(backend.rank(), self.rank)
        self.assertEqual(backend.size(), self.world_size)

        pool = torch.cuda.MemPool(backend.get_mem_allocator())
        self._assert_all_reduce(pool)

        handles = self._collect_handles("c10d_mccl_shrink")
        excluded_rank = self.world_size // 2
        if self.rank != excluded_rank:
            surviving_handles = [
                handle for rank, handle in enumerate(handles) if rank != excluded_rank
            ]
            self._reconfigure(self.SHRINK_UUID, surviving_handles)
            expected_rank = self.rank if self.rank < excluded_rank else self.rank - 1
            self.assertEqual(dist.get_rank(), expected_rank)
            self.assertEqual(dist.get_world_size(), self.world_size - 1)
            self.assertEqual(backend.rank(), expected_rank)
            self.assertEqual(backend.size(), self.world_size - 1)
            self._assert_all_reduce(pool)
        else:
            self.assertEqual(dist.get_rank(), self.rank)
            self.assertEqual(dist.get_world_size(), self.world_size)

        self._store_barrier("c10d_mccl_shrink_done")


if __name__ == "__main__":
    unittest.main()
