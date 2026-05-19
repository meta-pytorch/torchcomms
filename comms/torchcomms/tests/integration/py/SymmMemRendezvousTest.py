# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict
"""
Symmetric memory rendezvous over a torchcomms-backed c10d ProcessGroup.

Builds a torchcomms ncclx comm directly, wraps it in _BackendWrapper, and
registers it onto a bare dist.ProcessGroup as the NCCL backend type. Then
publishes the underlying ncclComm into NCCLDevCommManager via
register_with_symm_mem, allocates a symmetric-memory buffer, rendezvouses
on it, and verifies every rank can read its peers' buffers through the
returned handle.

Exercises the symm_mem path that no longer dynamic_casts to
ProcessGroupNCCL — see PR P2321051149.
"""

from __future__ import annotations

import os
import unittest

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import torchcomms
from torchcomms._comms import _BackendWrapper
from torchcomms.tests.integration.py.TorchCommTestHelpers import (
    get_device,
    get_rank_and_size,
)


GROUP_NAME = "symm_mem_torchcomms_ncclx"


def _is_missing_symm_mem_device_support(error: RuntimeError) -> bool:
    return "register_with_symm_mem requires" in str(error)


def _set_group_name(pg: dist.ProcessGroup, name: str) -> None:
    from torch._C._distributed_c10d import _register_process_group  # @manual

    pg._set_group_name(name)
    _register_process_group(name, pg)


def _populate_world_tables(pg: dist.ProcessGroup, name: str, world_size: int) -> None:
    """symm_mem reads c10d._world.pg_group_ranks, normally populated by
    init_process_group. Bare PGs need this filled in manually."""
    import torch.distributed.distributed_c10d as _c10d  # @manual

    _c10d._world.pg_group_ranks[pg] = {r: r for r in range(world_size)}
    _c10d._world.pg_names[pg] = name
    _c10d._world.pg_map[pg] = (None, None)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required for symm_mem")
class TestSymmMemRendezvous(unittest.TestCase):
    """Symm-mem rendezvous over a torchcomms _BackendWrapper PG."""

    @classmethod
    def setUpClass(cls) -> None:
        if not torchcomms.is_backend_built("ncclx"):
            raise unittest.SkipTest("torchcomms ncclx backend not built")

        cls.rank, cls.world_size = get_rank_and_size()
        if cls.world_size < 2:
            raise unittest.SkipTest("symm_mem rendezvous needs world_size >= 2")

        cls.device = get_device("ncclx", cls.rank)
        torch.cuda.set_device(cls.device)

        # Shared FileStore so all ranks meet at the same ncclx rendezvous.
        store_path = os.environ.get(
            "TORCHCOMM_STORE_PATH",
            f"/tmp/SymmMemRendezvousTest_{os.environ.get('MASTER_PORT', '0')}",
        )
        file_store = dist.FileStore(store_path, cls.world_size)
        os.environ["TORCHCOMM_RANK"] = str(cls.rank)
        os.environ["TORCHCOMM_SIZE"] = str(cls.world_size)

        cls.ncclx_comm = torchcomms.new_comm(
            "ncclx",
            cls.device,
            name="symm_mem_ncclx_comm",
            store=dist.PrefixStore("ncclx/", file_store),
            hints={"persistent_store": "true"},
        )

        pg_store = dist.PrefixStore("pg/", file_store)
        cls.pg = dist.ProcessGroup(
            pg_store, cls.ncclx_comm.get_rank(), cls.ncclx_comm.get_size()
        )
        # symm_mem's NCCL path keys off pg._get_backend(cuda).type() == NCCL.
        cls.pg._register_backend(
            cls.ncclx_comm.get_device(),
            dist.ProcessGroup.BackendType.NCCL,
            _BackendWrapper(cls.ncclx_comm),
        )
        _set_group_name(cls.pg, GROUP_NAME)
        _populate_world_tables(cls.pg, GROUP_NAME, cls.world_size)

        symm_mem.set_backend("NCCL")

        try:
            cls.ncclx_comm.register_with_symm_mem(GROUP_NAME)
        except RuntimeError as error:
            if not _is_missing_symm_mem_device_support(error):
                raise
            raise unittest.SkipTest(
                "torchcomms ncclx was built without NCCL device API support "
                "(TORCHCOMMS_HAS_NCCL_DEVICE_API off); cannot exercise "
                "symm_mem rendezvous."
            )

    def _pg_barrier(self) -> None:
        """Stand-in for handle.barrier() — NCCLSymmetricMemory::barrier is
        currently NYI on both ProcessGroupNCCL and BackendWrapper paths."""
        tok = torch.zeros(1, device=self.device)
        self.pg.allreduce([tok]).wait()
        torch.cuda.synchronize()

    def test_sanity_allreduce(self) -> None:
        """The PG dispatch path must be healthy before we drive symm_mem
        collectives over it."""
        sanity = torch.ones(4, device=self.device)
        self.pg.allreduce([sanity]).wait()
        torch.cuda.synchronize()
        expected = torch.full((4,), float(self.world_size), device=self.device)
        self.assertTrue(torch.equal(sanity, expected))

    def test_rendezvous_handle_matches_world(self) -> None:
        buf = symm_mem.empty(1024, dtype=torch.float32, device=self.device)
        handle = symm_mem.rendezvous(buf, group=GROUP_NAME)
        self.assertEqual(handle.world_size, self.world_size)
        self.assertEqual(handle.rank, self.rank)

    def test_peer_buffer_reads(self) -> None:
        """Each rank fills its symm-mem buffer with float(rank); after
        rendezvous, every rank reads every peer's buffer via the handle
        and must see uniform float(peer) values."""
        num_elem = 1024
        dtype = torch.float32
        buf = symm_mem.empty(num_elem, dtype=dtype, device=self.device)
        buf.fill_(float(self.rank))
        handle = symm_mem.rendezvous(buf, group=GROUP_NAME)

        self._pg_barrier()
        for peer in range(self.world_size):
            peer_buf = handle.get_buffer(peer, (num_elem,), dtype)
            expected = float(peer)
            self.assertEqual(peer_buf.min().item(), expected)
            self.assertEqual(peer_buf.max().item(), expected)
        self._pg_barrier()


if __name__ == "__main__":
    unittest.main()
