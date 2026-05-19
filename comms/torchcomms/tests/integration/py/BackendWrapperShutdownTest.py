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

import os
import unittest

import torch
import torch.distributed as dist
from torchcomms.tests.helpers.py.test_helpers import skip_if_ncclx
from torchcomms.tests.integration.helpers.TorchCommTestHelpers import (
    get_device,
    get_rank_and_size,
)

_TORCHCOMMS_CONFIG_AVAILABLE = hasattr(dist, "config") and hasattr(
    dist.config, "use_torchcomms"
)


@unittest.skipUnless(
    _TORCHCOMMS_CONFIG_AVAILABLE,
    "dist.config.use_torchcomms not available in this PyTorch version",
)
@skip_if_ncclx
class TestBackendWrapperShutdown(unittest.TestCase):
    """Each test creates its own PG, runs a small collective, then tears
    it down — no shared setUpClass, since the goal is to exercise the
    init+destroy cycle itself."""

    def _init_pg(self, backend: str) -> None:
        rank, world_size = get_rank_and_size()
        dist.config.use_torchcomms = True
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        device = get_device(os.environ["TEST_BACKEND"], dist.get_rank())
        torch.set_default_device(device)

    def test_destroy_after_collective_no_hang(self):
        """A simple init → all_reduce → destroy cycle finishes without
        hanging. Catches the original ``ncclCommDestroy`` deadlock."""
        self._init_pg(os.environ["TEST_BACKEND"])
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
        if os.environ["TEST_BACKEND"] != "nccl":
            self.skipTest("mixed backend test is nccl-specific")

        rank, world_size = get_rank_and_size()
        dist.config.use_torchcomms = True
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        dist.init_process_group(
            backend="cpu:gloo,cuda:nccl",
            rank=rank,
            world_size=world_size,
            device_id=torch.device(f"cuda:{local_rank}"),
        )
        try:
            torch.set_default_device(f"cuda:{local_rank}")
            cpu_tensor = torch.ones(4, dtype=torch.float32, device="cpu")
            cuda_tensor = torch.ones(
                4, dtype=torch.float32, device=f"cuda:{local_rank}"
            )
            dist.all_reduce(cpu_tensor)
            dist.all_reduce(cuda_tensor)
            self.assertEqual(cpu_tensor[0].item(), float(world_size))
            self.assertEqual(cuda_tensor[0].item(), float(world_size))
        finally:
            # Must not raise even though both sub-backends share the comm.
            dist.destroy_process_group()


if __name__ == "__main__":
    unittest.main()
