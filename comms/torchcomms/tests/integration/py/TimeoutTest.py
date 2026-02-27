#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
import time
import unittest
from datetime import timedelta

import torch
import torchcomms
from torchcomms.tests.helpers.py.cuda_graph_test_helpers import (
    _wait,
    CudaGraphTestBase,
    GraphTestBuilder,
    skip_unless_ncclx,
)
from torchcomms.tests.helpers.py.fatal_state_test_helpers import FatalStateTestMixin
from torchcomms.tests.integration.py.TorchCommTestHelpers import get_rank_and_size


# ---------------------------------------------------------------------------
# Scenario functions — executed in subprocesses via sentinel env vars.
# Each scenario sets up a comm, triggers a timeout, and expects the process
# to be aborted by the watchdog. If it returns, os._exit(1) signals failure.
# ---------------------------------------------------------------------------


def _run_eager_timeout_scenario() -> None:
    """Eager barrier timeout: rank 0 sleeps, others time out and abort."""

    backend = os.environ.get("TEST_BACKEND", "")
    rank, _ = get_rank_and_size()
    device_count = torch.cuda.device_count()
    device = torch.device("cuda", rank % device_count)
    torch.cuda.set_device(device)

    comm = torchcomms.new_comm(
        backend,
        device,
        name="eager_timeout_subprocess_comm",
        abort_process_on_timeout_or_error=True,
        timeout=timedelta(milliseconds=1),
    )

    if rank == 0:
        time.sleep(10)
    comm.barrier(async_op=False)
    torch.cuda.synchronize()

    # Should not reach here — process should have been aborted
    comm.finalize()


def _run_eager_timeout_after_success_scenario() -> None:
    """First eager barrier succeeds, second times out and aborts."""

    backend = os.environ.get("TEST_BACKEND", "")
    rank, _ = get_rank_and_size()
    device_count = torch.cuda.device_count()
    device = torch.device("cuda", rank % device_count)
    torch.cuda.set_device(device)

    comm = torchcomms.new_comm(
        backend,
        device,
        name="eager_timeout_after_success_subprocess_comm",
        abort_process_on_timeout_or_error=True,
        timeout=timedelta(milliseconds=1),
    )

    # First barrier: all ranks participate, should succeed.
    comm.barrier(async_op=False)
    torch.cuda.synchronize()

    # Second barrier: rank 0 delays, causing timeout on other ranks.
    if rank == 0:
        time.sleep(10)
    comm.barrier(async_op=False)
    torch.cuda.synchronize()

    # Should not reach here — process should have been aborted
    comm.finalize()


def _run_graph_timeout_scenario() -> None:
    """Graph replay timeout: rank 0 sleeps, others time out and abort."""

    backend = os.environ.get("TEST_BACKEND", "")
    rank, _ = get_rank_and_size()
    device_count = torch.cuda.device_count()
    device = torch.device("cuda", rank % device_count)
    torch.cuda.set_device(device)

    comm = torchcomms.new_comm(
        backend,
        device,
        name="graph_timeout_subprocess_comm",
        abort_process_on_timeout_or_error=True,
    )

    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(graph):
        _wait(
            comm.barrier(
                async_op=False,
                timeout=timedelta(milliseconds=1),
            )
        )
    graph.instantiate()

    if rank == 0:
        time.sleep(10.0)
    graph.replay()
    torch.cuda.synchronize()

    # If we reach here, timeout detection failed
    graph.reset()
    comm.finalize()


def _run_graph_timeout_after_success_scenario() -> None:
    """First graph replay succeeds, second times out and aborts."""

    backend = os.environ.get("TEST_BACKEND", "")
    rank, _ = get_rank_and_size()
    device_count = torch.cuda.device_count()
    device = torch.device("cuda", rank % device_count)
    torch.cuda.set_device(device)

    comm = torchcomms.new_comm(
        backend,
        device,
        name="graph_timeout_after_success_subprocess_comm",
        abort_process_on_timeout_or_error=True,
    )

    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(graph):
        _wait(
            comm.barrier(
                async_op=False,
                timeout=timedelta(milliseconds=1),
            )
        )
    graph.instantiate()

    # First replay: all ranks participate, should succeed.
    graph.replay()
    torch.cuda.synchronize()

    # Second replay: rank 0 delays, causing timeout on other ranks.
    if rank == 0:
        time.sleep(10.0)
    graph.replay()
    torch.cuda.synchronize()

    # If we reach here, timeout detection failed
    graph.reset()
    comm.finalize()


def _run_eager_short_timeout_success_scenario() -> None:
    """Eager all_reduce with short timeout completes successfully."""

    backend = os.environ.get("TEST_BACKEND", "")
    rank, _ = get_rank_and_size()
    device_count = torch.cuda.device_count()
    device = torch.device("cuda", rank % device_count)
    torch.cuda.set_device(device)

    comm = torchcomms.new_comm(
        backend,
        device,
        name="eager_short_timeout_success_subprocess_comm",
        abort_process_on_timeout_or_error=True,
        timeout=timedelta(seconds=2),
    )
    try:
        inp = torch.ones(10, 10, device=device)
        comm.all_reduce(inp, torchcomms.ReduceOp.SUM, async_op=False)
        torch.cuda.synchronize()
        expected = torch.ones(10, 10, device=device) * comm.get_size()
        torch.testing.assert_close(inp, expected)
    finally:
        comm.finalize()


def _run_graph_short_timeout_success_scenario() -> None:
    """Graph replay without artificial delay completes without timeout."""

    class _Context(CudaGraphTestBase, unittest.TestCase):
        def runTest(self):
            pass

    ctx = _Context("runTest")
    ctx.setUp()
    try:
        for async_op in [False, True]:

            def capture(b: GraphTestBuilder, _async: bool = async_op) -> None:
                _wait(
                    b.comms[0].all_reduce(
                        b.inputs[0], torchcomms.ReduceOp.SUM, async_op=_async
                    )
                )

            def make_inputs(b: GraphTestBuilder) -> list[torch.Tensor]:
                return [torch.ones(10, 10, device=ctx.device)]

            def make_expected(b: GraphTestBuilder) -> list[torch.Tensor]:
                return [b.inputs[0] * b.comms[0].get_size()]

            GraphTestBuilder(ctx).add_capture(capture).run_serial(
                inputs=make_inputs,
                expected=make_expected,
            )
    finally:
        ctx.tearDown()


# Early exit for subprocess mode: when re-invoked with a sentinel env var,
# run the scenario and exit before the test runner discovers any test classes.
if os.environ.get("_TORCHCOMM_RUN_EAGER_TIMEOUT"):
    _run_eager_timeout_scenario()
    os._exit(1)

if os.environ.get("_TORCHCOMM_RUN_EAGER_TIMEOUT_AFTER_SUCCESS"):
    _run_eager_timeout_after_success_scenario()
    os._exit(1)

if os.environ.get("_TORCHCOMM_RUN_GRAPH_TIMEOUT"):
    _run_graph_timeout_scenario()
    os._exit(1)

if os.environ.get("_TORCHCOMM_RUN_GRAPH_TIMEOUT_AFTER_SUCCESS"):
    _run_graph_timeout_after_success_scenario()
    os._exit(1)

if os.environ.get("_TORCHCOMM_RUN_EAGER_SHORT_TIMEOUT_SUCCESS"):
    _run_eager_short_timeout_success_scenario()
    os._exit(0)

if os.environ.get("_TORCHCOMM_RUN_GRAPH_SHORT_TIMEOUT_SUCCESS"):
    _run_graph_short_timeout_success_scenario()
    os._exit(0)


class TestTimeout(FatalStateTestMixin, unittest.TestCase):
    """Tests timeout detection, abort behavior, and short timeout success verification."""

    def _run_abort_timeout_test(self, sentinel_var: str, expected_stderr: str) -> None:
        """Common logic for all subprocess abort-timeout tests.

        Spawns world_size subprocesses with sentinel, and asserts abort
        (non-rank-0) or failure (rank-0).
        """
        results = self.run_subprocesses(sentinel_var)
        for rank, result in enumerate(results):
            if rank != 0:
                self.assert_subprocess_aborted(result, expected_stderr)
            else:
                self.assert_subprocess_failed(result)

    def _run_short_timeout_success_test(self, sentinel_var: str) -> None:
        """Common logic for short-timeout-success tests.

        Spawns world_size subprocesses and asserts all succeeded.
        """
        results = self.run_subprocesses(sentinel_var)
        for result in results:
            self.assert_subprocess_succeeded(result)

    # pyre-ignore[56]
    @skip_unless_ncclx
    def test_eager_timeout(self) -> None:
        """Eager collective timeout should abort the process."""
        self._run_abort_timeout_test(
            "_TORCHCOMM_RUN_EAGER_TIMEOUT",
            "Aborting process due to timeout",
        )

    # pyre-ignore[56]
    @skip_unless_ncclx
    def test_graph_timeout(self) -> None:
        """Graph replay timeout should abort the process."""
        self._run_abort_timeout_test(
            "_TORCHCOMM_RUN_GRAPH_TIMEOUT",
            "Graph monitor: collective TIMED OUT for graph",
        )

    # pyre-ignore[56]
    @skip_unless_ncclx
    def test_eager_timeout_after_successful_op(self) -> None:
        """First eager op succeeds, second times out and aborts."""
        self._run_abort_timeout_test(
            "_TORCHCOMM_RUN_EAGER_TIMEOUT_AFTER_SUCCESS",
            "Aborting process due to timeout",
        )

    # pyre-ignore[56]
    @skip_unless_ncclx
    def test_graph_timeout_after_successful_replay(self) -> None:
        """First graph replay succeeds, second times out and aborts."""
        self._run_abort_timeout_test(
            "_TORCHCOMM_RUN_GRAPH_TIMEOUT_AFTER_SUCCESS",
            "Graph monitor: collective TIMED OUT for graph",
        )

    # pyre-ignore[56]
    @skip_unless_ncclx
    def test_eager_short_timeout_success(self) -> None:
        """Normal eager collective with short timeout completes successfully."""
        self._run_short_timeout_success_test(
            "_TORCHCOMM_RUN_EAGER_SHORT_TIMEOUT_SUCCESS"
        )

    # pyre-ignore[56]
    @skip_unless_ncclx
    def test_graph_short_timeout_success(self) -> None:
        """Graph replay without artificial delay should complete without timeout."""
        self._run_short_timeout_success_test(
            "_TORCHCOMM_RUN_GRAPH_SHORT_TIMEOUT_SUCCESS"
        )


if __name__ == "__main__":
    unittest.main()
