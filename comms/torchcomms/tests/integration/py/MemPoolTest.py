#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

import datetime
import os
import unittest
from typing import Any

import torch
import torch.distributed as dist
import torchcomms
from torchcomms._comms import _BackendWrapper
from torchcomms.device_mesh import _create_torchcomm_process_group, _get_store_for_pg


class MemPoolTorchCommTest(unittest.TestCase):
    @property
    def world_size(self) -> int:
        return dist.get_world_size()

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def create_mem_pool(self, pg: dist.ProcessGroup) -> torch.cuda.MemPool:
        """
        Lazy create a mem pool for the given process group.
        """
        assert dist.is_initialized()
        pg = pg or dist.distributed_c10d._get_default_group()
        rank = dist.get_rank()
        device = torch.device(f"cuda:{rank}")
        current_device = torch.cuda.current_device()
        assert (
            current_device == rank
        ), f"torch.cuda current device is not set to rank: {current_device}"

        comm_wrapper = pg._get_backend(device)
        assert isinstance(comm_wrapper, _BackendWrapper)
        comm = comm_wrapper.get_comm()
        return torch.cuda.MemPool(comm.mem_allocator)

    @unittest.skipIf(
        os.getenv("TEST_BACKEND") not in ["ncclx", "nccl"],
        "Skipping NCCLX/NCCL-only mem pool tests",
    )
    def test_mem_pool(self) -> None:
        """
        Test mem pool with torchComm.
        """
        device = torch.device(os.environ.get("TEST_DEVICE", "cuda"))
        backend = os.environ["TEST_BACKEND"]
        num_comms = 3
        comms = []
        for i in range(num_comms):
            comms.append(
                torchcomms.new_comm(
                    backend,
                    device,
                    name=f"test_mem_pool_{i}",
                    timeout=datetime.timedelta(seconds=60),
                )
            )
        store = _get_store_for_pg()
        pgs = []
        for comm in comms:
            pgs.append(
                _create_torchcomm_process_group(
                    comm=comm,
                    group_name=comm.get_name(),
                    prefix_store=dist.PrefixStore(comm.get_name(), store),
                    global_ranks_mapping=None,  # Will use default mapping
                )
            )
        dist.distributed_c10d.GroupMember.WORLD = pgs[0]

        torch.cuda.set_device(comm.get_device())
        torch.set_default_device(comm.get_device())
        for i in range(num_comms):
            pg = pgs[i]
            self.assertEqual(pg.group_name, f"test_mem_pool_{i}")

        expected = self.world_size * torch.ones(1024 * 1024, device=device)
        # Use memory pool with multiple comms
        for pg in pgs:
            pool = self.create_mem_pool(pg)

            # Now the pool we created should be registered to all process groups
            for pg in pgs:
                # allocate memory with ncclMemAlloc
                with torch.cuda.use_mem_pool(pool):
                    tensor = torch.ones(1024 * 1024, device=device)
                pg.allreduce(tensor).wait()
                torch.cuda.synchronize(device=device)
                torch.testing.assert_close(
                    tensor,
                    expected,
                    msg=f"Rank {dist.get_rank()} Expected {expected} but got {tensor}",
                )
                del tensor

            # clean up memory
            del pool
        for comm in comms:
            comm.finalize()


if __name__ == "__main__":
    unittest.main()
