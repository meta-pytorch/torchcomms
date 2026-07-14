# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict
"""
Symmetric memory rendezvous over a torchcomms-backed c10d ProcessGroup.

Builds a torchcomms nccl comm directly, wraps it in _BackendWrapper, and
registers it onto a bare dist.ProcessGroup as the NCCL backend type. Then
publishes the underlying ncclComm into NCCLDevCommManager via PyTorch's
_symmetric_memory._nccl.register_external_nccl_comm (passing
comm.get_nccl_comm_ptr()),
allocates a symmetric-memory buffer, rendezvouses on it, and verifies every
rank can read its peers' buffers through the returned handle.

Exercises the symm_mem path that no longer dynamic_casts to
ProcessGroupNCCL — see PR P2321051149. PyTorch owns the registration
plumbing; torchcomms only exposes the raw comm pointer via get_nccl_comm_ptr.
"""

from __future__ import annotations

import os
import unittest

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import torchcomms
from torch.distributed.distributed_c10d import GroupName
from torchcomms._comms import _BackendWrapper
from torchcomms.tests.integration.helpers.TorchCommTestHelpers import (
    get_device,
    get_rank_and_size,
)


GROUP_NAME = GroupName("symm_mem_torchcomms_nccl")


def _symm_mem_peer_access_supported() -> bool:
    """True iff the local CUDA device supports NVLink multicast.

    NCCL symmetric memory rendezvous resolves peer device pointers via
    ``ncclGetPeerDevicePointer``, which requires the GPUs to share an LSA
    (NVLink) domain; PCIe-only multi-GPU hosts (e.g. A10G / AWS g5) fail
    rendezvous with "invalid argument". Multicast support implies an NVLink
    domain, so PyTorch's own symm_mem suite gates on it (see
    ``torch.testing._internal.common_distributed.requires_multicast_support``).
    Using the same hardware signal lets this suite skip declaratively rather
    than discover the limitation by catching a rendezvous error.
    """
    if not torch.cuda.is_available():
        return False
    try:
        from torch._C._autograd import DeviceType
        from torch._C._distributed_c10d import _SymmetricMemory

        return _SymmetricMemory.has_multicast_support(DeviceType.CUDA, 0)
    except (ImportError, AttributeError):
        return False


def _set_group_name(pg: dist.ProcessGroup, name: GroupName) -> None:
    from torch._C._distributed_c10d import _register_process_group  # @manual

    pg._set_group_name(name)
    _register_process_group(name, pg)


def _populate_world_tables(
    pg: dist.ProcessGroup, name: GroupName, world_size: int
) -> None:
    """symm_mem reads c10d._world.pg_group_ranks, normally populated by
    init_process_group. Bare PGs need this filled in manually."""
    import torch.distributed.distributed_c10d as _c10d  # @manual

    _c10d._world.pg_group_ranks[pg] = {r: r for r in range(world_size)}
    _c10d._world.pg_names[pg] = name
    # Bare PG has no (backend, store) entry; symm_mem only reads the keys.
    _c10d._world.pg_map[pg] = (None, None)  # pyre-ignore[6]


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required for symm_mem")
@unittest.skipUnless(
    _symm_mem_peer_access_supported(),
    "NCCL symmetric memory needs NVLink/LSA peer access (no multicast support "
    "on this host; PCIe-only multi-GPU like A10G/AWS g5 cannot rendezvous)",
)
class TestSymmMemRendezvous(unittest.TestCase):
    """Symm-mem rendezvous over a torchcomms nccl _BackendWrapper PG."""

    @classmethod
    def setUpClass(cls) -> None:
        if not torchcomms.is_backend_built("nccl"):
            raise unittest.SkipTest("torchcomms nccl backend not built")

        cls.rank, cls.world_size = get_rank_and_size()
        if cls.world_size < 2:
            raise unittest.SkipTest("symm_mem rendezvous needs world_size >= 2")

        cls.device = get_device("nccl", cls.rank)
        torch.cuda.set_device(cls.device)

        # Shared FileStore so all ranks meet at the same nccl rendezvous.
        store_path = os.environ.get(
            "TORCHCOMM_STORE_PATH",
            f"/tmp/SymmMemRendezvousTest_{os.environ.get('MASTER_PORT', '0')}",
        )
        file_store = dist.FileStore(store_path, cls.world_size)
        os.environ["TORCHCOMM_RANK"] = str(cls.rank)
        os.environ["TORCHCOMM_SIZE"] = str(cls.world_size)

        cls.nccl_comm = torchcomms.new_comm(
            "nccl",
            cls.device,
            name="symm_mem_nccl_comm",
            store=dist.PrefixStore("nccl/", file_store),
        )

        pg_store = dist.PrefixStore("pg/", file_store)
        cls.pg = dist.ProcessGroup(
            pg_store, cls.nccl_comm.get_rank(), cls.nccl_comm.get_size()
        )
        # symm_mem's NCCL path keys off pg._get_backend(cuda).type() == NCCL.
        cls.pg._register_backend(
            cls.nccl_comm.get_device(),
            dist.ProcessGroup.BackendType.NCCL,
            _BackendWrapper(cls.nccl_comm),  # pyre-ignore[6]
        )
        _set_group_name(cls.pg, GROUP_NAME)
        _populate_world_tables(cls.pg, GROUP_NAME, cls.world_size)

        symm_mem.set_backend("NCCL")

        # PyTorch owns the registration plumbing now: hand it the raw
        # ncclComm pointer and keep the returned handle alive for as long as
        # symm_mem should use this comm (dropped in tearDownClass). The import
        # and C++ entry point only exist in recent NCCL builds with symm-mem
        # device support, so a missing symbol means this build can't exercise
        # it. Import lazily here (not at module scope) so older/stable PyTorch
        # skips the suite cleanly instead of erroring during test collection.
        try:
            from torch.distributed._symmetric_memory._nccl import (
                register_external_nccl_comm,
            )

            # get_nccl_comm_ptr is bound on the backend impl (TorchCommNCCL),
            # not the TorchComm wrapper.
            comm_ptr = cls.nccl_comm.get_backend_impl().get_nccl_comm_ptr()
            cls._symm_mem_reg = register_external_nccl_comm(
                GROUP_NAME, comm_ptr, cls.device, comm=cls.nccl_comm
            )
        except ImportError:
            raise unittest.SkipTest(
                "PyTorch was built without NCCL symmetric-memory device "
                "support (register_external_nccl_comm unavailable); cannot "
                "exercise symm_mem rendezvous."
            )

    @classmethod
    def tearDownClass(cls) -> None:
        # Drop the symm_mem registration first (it borrows nccl_comm's
        # pointer), then finalize the comm so its timeout-watchdog thread is
        # stopped and all CUDA work is drained *before* interpreter exit.
        # Without the finalize, the watchdog keeps polling CUDA events while
        # atexit tears down the CUDA driver, hits "driver shutting down", and
        # aborts the process with a SIGSEGV after the tests have passed.
        reg = getattr(cls, "_symm_mem_reg", None)
        if reg is not None:
            reg.unregister()
            cls._symm_mem_reg = None
        comm = getattr(cls, "nccl_comm", None)
        if comm is not None:
            torch.cuda.synchronize()
            comm.finalize()
            cls.nccl_comm = None

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

    def _expected_allreduce_sum(self) -> float:
        """Each rank seeds its buffer with float(rank), so a sum-reduce over
        the world yields 0 + 1 + ... + (world_size - 1) on every element."""
        return float(sum(range(self.world_size)))

    def test_one_shot_all_reduce(self) -> None:
        """symm_mem one-shot all-reduce kernel over the NCCL-backed group: a
        single fused kernel reads every peer's symm buffer and returns the
        elementwise sum (out-of-place). Exercises the P2P-pointer path that
        the rendezvous wired up."""
        num_elem = 1024
        dtype = torch.float32
        inp = symm_mem.empty(num_elem, dtype=dtype, device=self.device)
        inp.fill_(float(self.rank))
        symm_mem.rendezvous(inp, group=GROUP_NAME)

        self._pg_barrier()
        res = torch.ops.symm_mem.one_shot_all_reduce(inp, "sum", GROUP_NAME)
        torch.cuda.synchronize()

        expected = self._expected_allreduce_sum()
        self.assertEqual(res.shape, inp.shape)
        self.assertEqual(res.min().item(), expected)
        self.assertEqual(res.max().item(), expected)
        self._pg_barrier()

    def test_two_shot_all_reduce(self) -> None:
        """symm_mem two-shot (in-place) all-reduce kernel: each rank ends with
        the elementwise sum across the world in its own symm buffer."""
        num_elem = 1024
        dtype = torch.float32
        buf = symm_mem.empty(num_elem, dtype=dtype, device=self.device)
        buf.fill_(float(self.rank))
        symm_mem.rendezvous(buf, group=GROUP_NAME)

        self._pg_barrier()
        torch.ops.symm_mem.two_shot_all_reduce_(buf, "sum", GROUP_NAME)
        torch.cuda.synchronize()

        expected = self._expected_allreduce_sum()
        self.assertEqual(buf.min().item(), expected)
        self.assertEqual(buf.max().item(), expected)
        self._pg_barrier()

    def test_multimem_all_reduce(self) -> None:
        """symm_mem multimem (NVLink-multicast) in-place all-reduce. Skipped
        unless the rendezvous'd allocation actually has a multicast pointer —
        i.e. the backend set up a multicast window and the fabric supports it.
        On backends/links without multicast this is a no-op skip rather than a
        failure."""
        num_elem = 1024
        dtype = torch.float32
        buf = symm_mem.empty(num_elem, dtype=dtype, device=self.device)
        buf.fill_(float(self.rank))
        handle = symm_mem.rendezvous(buf, group=GROUP_NAME)
        if handle.multicast_ptr == 0:
            self.skipTest(
                "symm_mem allocation has no multicast pointer "
                "(NCCL backend or no NVLink multicast support)"
            )

        self._pg_barrier()
        torch.ops.symm_mem.multimem_all_reduce_(buf, "sum", GROUP_NAME)
        torch.cuda.synchronize()

        expected = self._expected_allreduce_sum()
        self.assertEqual(buf.min().item(), expected)
        self.assertEqual(buf.max().item(), expected)
        self._pg_barrier()


if __name__ == "__main__":
    unittest.main()
