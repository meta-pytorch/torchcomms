#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Test BackendWrapper::monitoredBarrier (the gloo health-checking barrier).

torch.distributed.monitored_barrier is only implemented for the gloo backend.
Under TorchComms the gloo ProcessGroup is backed by BackendWrapper, which now
reimplements ProcessGroupGloo::monitoredBarrier on top of TorchComms P2P: rank 0
acts as the coordinator, collecting a check-in from every other rank within the
timeout and raising a RuntimeError that names the rank(s) that failed to report.

This test wraps a real "gloo" TorchComm in a c10d ProcessGroup and drives
``pg.monitored_barrier`` directly (the same entry point dist.monitored_barrier
dispatches to), covering the all-pass path plus straggler detection in both the
wait_all_ranks (report every straggler) and fast-fail (report the first) modes.

monitored_barrier is gloo-only, so this test skips on any other backend rather
than failing (the Buck target is also restricted to ``backends = ["gloo"]``).
"""

import datetime
import os
import time
import unittest

import torch
import torch.distributed as dist
import torchcomms
from torchcomms import new_comm
from torchcomms.device_mesh import _create_torchcomm_process_group
from torchcomms.tests.integration.helpers.TorchCommTestHelpers import (
    create_store,
    get_rank_and_size,
)


def _make_wrapped_gloo_pg(
    name: str,
) -> tuple[torchcomms.TorchComm, dist.ProcessGroup]:
    """Build a gloo TorchComm wrapped in a c10d ProcessGroup.

    Reuses ``device_mesh._create_torchcomm_process_group`` for the wrap +
    register so this test does not duplicate the _BackendWrapper /
    _register_backend boilerplate. The comm carries its own store-bootstrapped
    gloo context (used by monitoredBarrier's P2P); the ProcessGroup uses a
    throwaway HashStore internally.

    The comm is created on CPU on purpose: monitoredBarrier is a CPU-only gloo
    op, and c10d dispatches pg.monitored_barrier to whichever backend is
    registered for the CPU device. _create_torchcomm_process_group registers the
    wrapper under comm.get_device(), so on a GPU host a cuda comm would leave the
    CPU device unbacked and monitored_barrier would raise "No backend type
    associated with device type cpu".
    """
    backend = os.environ["TEST_BACKEND"]
    comm = new_comm(backend, torch.device("cpu"), store=create_store(), name=name)
    pg = _create_torchcomm_process_group(comm, name)
    return comm, pg


class MonitoredBarrierTest(unittest.TestCase):
    def setUp(self) -> None:
        if os.environ.get("TEST_BACKEND") != "gloo":
            self.skipTest("monitored_barrier is only implemented for the gloo backend")
        self.rank, self.num_ranks = get_rank_and_size()

    def test_all_ranks_pass(self) -> None:
        """When every rank checks in, monitoredBarrier returns without error."""
        comm, pg = _make_wrapped_gloo_pg("mb_all_pass")
        passed = False
        try:
            # pyre-fixme[16]: ProcessGroup.monitored_barrier exists at runtime
            # (bound from c10d) but is missing from the type stub.
            pg.monitored_barrier(
                timeout=datetime.timedelta(seconds=30),
                wait_all_ranks=True,
            )
            passed = True
        finally:
            comm.finalize()
        self.assertTrue(passed)

    def test_straggler_detection(self) -> None:
        """wait_all_ranks: rank 0 names every rank that fails to reach the barrier."""
        if self.num_ranks < 2:
            self.skipTest("straggler detection requires at least 2 ranks")

        comm: torchcomms.TorchComm | None = None
        timeout = datetime.timedelta(seconds=1)
        try:
            comm, pg = _make_wrapped_gloo_pg("mb_straggler")
            if self.rank == 0:
                # Every other rank deliberately never checks in, so rank 0 must
                # time out on each and report all of them.
                with self.assertRaises(RuntimeError) as ctx:
                    # pyre-fixme[16]: ProcessGroup.monitored_barrier exists at
                    # runtime (bound from c10d) but is missing from the stub.
                    pg.monitored_barrier(timeout=timeout, wait_all_ranks=True)
                message = str(ctx.exception)
                self.assertIn("monitoredBarrier", message)
                # Match the distinctive "Ranks <list> failed" segment rather
                # than bare rank ids: small ids like "1" would also match the
                # timeout value ("1000 ms") or the "[Rank 0]" prefix, so a
                # malformed/empty straggler list could still pass.
                expected_ranks = ", ".join(str(r) for r in range(1, self.num_ranks))
                self.assertIn(f"Ranks {expected_ranks} failed", message)
            else:
                # Stay alive (keeping the gloo context up) long enough for rank
                # 0 to time out on us, then exit. wait_all_ranks spends the full
                # timeout per rank sequentially, so allow num_ranks * timeout.
                time.sleep(timeout.total_seconds() * self.num_ranks + 5)
        finally:
            if comm is not None:
                comm.finalize()

    def test_straggler_detection_fast_fail(self) -> None:
        """Fast-fail (wait_all_ranks=False): rank 0 raises on the first straggler."""
        if self.num_ranks < 2:
            self.skipTest("straggler detection requires at least 2 ranks")

        comm: torchcomms.TorchComm | None = None
        timeout = datetime.timedelta(seconds=1)
        try:
            comm, pg = _make_wrapped_gloo_pg("mb_straggler_fast_fail")
            if self.rank == 0:
                # No worker checks in. In fast-fail mode rank 0 stops at the
                # first straggler (rank 1) and raises without probing the rest.
                with self.assertRaises(RuntimeError) as ctx:
                    # pyre-fixme[16]: ProcessGroup.monitored_barrier exists at
                    # runtime (bound from c10d) but is missing from the stub.
                    pg.monitored_barrier(timeout=timeout, wait_all_ranks=False)
                message = str(ctx.exception)
                self.assertIn("monitoredBarrier", message)
                # Fast-fail reports the first straggler only. Match the full
                # "Rank 1 failed" phrase, not a bare id that could also match
                # the timeout value or the "[Rank 0]" prefix.
                self.assertIn("Rank 1 failed", message)
            else:
                # rank 0 bails after ~one timeout (it stops at rank 1), but keep
                # the context up a bit longer to avoid tearing down mid-probe.
                time.sleep(timeout.total_seconds() * self.num_ranks + 5)
        finally:
            if comm is not None:
                comm.finalize()
