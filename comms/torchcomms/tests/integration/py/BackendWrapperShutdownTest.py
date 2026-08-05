#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Verify ``BackendWrapper::shutdown`` and ``::abort`` close the underlying
``TorchComm`` cleanly when ``torch.distributed.destroy_process_group()``
is called.

Two regressions guarded against:

1. **Deadlock on destroy**: without a ``shutdown`` override the wrapper's
   destructor would invoke ``ncclCommDestroy`` synchronously and could
   deadlock against the NCCL GC thread.

2. **Double-finalize raise**: a mixed ``cpu:gloo,cuda:nccl`` PG ends up
   with two BackendWrappers sharing one underlying ``TorchComm`` (via the
   BackendType-to-wrapper dedup). ``destroy_process_group`` calls
   ``shutdown`` on each backend; ``TorchComm::finalize`` is not idempotent
   and would raise ``RuntimeError: TorchCommNCCL already finalized`` on
   the second call. The wrapper swallows the exception so destroy is safe
   to call any number of times.
"""

from __future__ import annotations

import os
import unittest

import torch
import torch.distributed as dist
from packaging.version import InvalidVersion, Version
from torchcomms.tests.helpers.py.test_helpers import skip_if_ncclx
from torchcomms.tests.integration.helpers.TorchCommTestHelpers import (
    get_device,
    get_rank_and_size,
)

_TORCHCOMMS_CONFIG_AVAILABLE = hasattr(dist, "config") and hasattr(
    dist.config, "use_torchcomms"
)

_PR_182057_TORCH_VERSION = Version("2.13.0.dev20260502")


def _torch_predates_pr_182057() -> bool:
    try:
        return Version(torch.__version__) < _PR_182057_TORCH_VERSION
    except InvalidVersion:
        # Unparsable version string — assume newer to avoid silently
        # skipping on something we can't classify.
        return False


def _local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", get_rank_and_size()[0]))


_root_store: dist.TCPStore | None = None


def _create_isolated_store(store_name: str) -> dist.Store:
    """Give each test its own rendezvous namespace.

    Both tests call ``init_process_group`` in the same process, and c10d
    restarts its internal group-name numbering after
    ``destroy_process_group``, so the second PG can collide with keys the
    first one left behind. A per-test ``PrefixStore`` isolates them.

    Isolation is by *namespace*, not by port: the root ``TCPStore`` is bound
    once to the launcher's ``MASTER_ADDR``/``MASTER_PORT``, which is the only
    rendezvous endpoint every rank knows without a cross-rank hand-off. That
    is what makes this multi-node safe -- an ephemeral port has to be
    communicated somehow, and a node-local file (the previous approach) is
    invisible to ranks on other hosts.

    Keyed on ``store_name`` rather than a per-process counter so that ranks
    agree on the namespace regardless of how many stores each has created.
    """
    global _root_store
    if _root_store is None:
        rank, _ = get_rank_and_size()
        _root_store = dist.TCPStore(
            host_name=os.environ["MASTER_ADDR"],
            port=int(os.environ["MASTER_PORT"]),
            is_master=(rank == 0),
            wait_for_workers=False,
        )
    return dist.PrefixStore(store_name, _root_store)


@unittest.skipUnless(
    _TORCHCOMMS_CONFIG_AVAILABLE,
    "dist.config.use_torchcomms not available in this PyTorch version",
)
@skip_if_ncclx
class TestBackendWrapperShutdown(unittest.TestCase):
    """Each test creates its own PG, runs a small collective, then tears
    it down — no shared setUpClass, since the goal is to exercise the
    init+destroy cycle itself."""

    def _init_pg(self, backend: str, store_name: str) -> None:
        rank, world_size = get_rank_and_size()
        # NCCL requires a CUDA device to be bound to this rank before
        # ``init_process_group`` so that the per-rank communicator gets
        # the right device. Without this, all ranks default to cuda:0
        # and NCCL bootstrap fails. ``get_device`` can return
        # ``torch.device("cuda")`` with no index when ``TEST_DEVICE=cuda``
        # is set, so resolve the index explicitly from LOCAL_RANK / rank.
        device = get_device(os.environ["TEST_BACKEND"], rank)
        if torch.accelerator.is_available():
            torch.accelerator.set_device_index(_local_rank())
        dist.config.use_torchcomms = True
        dist.init_process_group(
            backend=backend,
            store=_create_isolated_store(store_name),
            rank=rank,
            world_size=world_size,
        )
        torch.set_default_device(device)

    def test_destroy_after_collective_no_hang(self):
        """A simple init → all_reduce → destroy cycle finishes without
        hanging. Catches the original ``ncclCommDestroy`` deadlock."""
        store_name = "destroy_after_collective_no_hang"
        self._init_pg(os.environ["TEST_BACKEND"], store_name)
        try:
            tensor = torch.ones(8, dtype=torch.float32)
            dist.all_reduce(tensor)
            self.assertEqual(tensor[0].item(), float(dist.get_world_size()))
        finally:
            dist.destroy_process_group()

    def test_mixed_backend_destroy_idempotent(self):
        """Mixed ``cpu:gloo,cuda:nccl`` PG: ``destroy_process_group``
        shuts down both sub-backends, which share one ``TorchComm``.
        Without idempotent ``shutdown``, the second call raises
        ``TorchCommNCCL already finalized``."""
        if os.environ["TEST_BACKEND"] not in ["nccl", "xccl"]:
            self.skipTest("mixed backend test is nccl/xccl-specific")
        if _torch_predates_pr_182057():
            self.skipTest(
                f"torch {torch.__version__} predates pytorch/pytorch#182057 "
                f"({_PR_182057_TORCH_VERSION}); mixed cpu:gloo,cuda:nccl + "
                "device_id= trips the ProcessGroup::setBackend "
                "bound_device_id check on init"
            )

        rank, world_size = get_rank_and_size()
        local_rank = _local_rank()
        torch.accelerator.set_device_index(local_rank)
        dist.config.use_torchcomms = True
        store_name = "mixed_backend_destroy_idempotent"
        backend_str = os.environ["TEST_BACKEND"]
        device_str = "cuda" if backend_str == "nccl" else "xpu"
        local_device_str = f"{device_str}:{local_rank}"
        dist.init_process_group(
            backend=f"cpu:gloo,{device_str}:{backend_str}",
            store=_create_isolated_store(store_name),
            rank=rank,
            world_size=world_size,
            device_id=torch.device(local_device_str),
        )
        try:
            torch.set_default_device(local_device_str)
            cpu_tensor = torch.ones(4, dtype=torch.float32, device="cpu")
            gpu_tensor = torch.ones(4, dtype=torch.float32, device=local_device_str)
            dist.all_reduce(cpu_tensor)
            dist.all_reduce(gpu_tensor)
            self.assertEqual(cpu_tensor[0].item(), float(world_size))
            self.assertEqual(gpu_tensor[0].item(), float(world_size))
        finally:
            # Must not raise even though both sub-backends share the comm.
            dist.destroy_process_group()


if __name__ == "__main__":
    unittest.main()
