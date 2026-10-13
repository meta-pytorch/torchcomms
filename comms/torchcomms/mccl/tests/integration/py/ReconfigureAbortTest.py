#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Characterization test for abort() during an in-flight MCCL reconfigure().

Reproduces the shape of the PAFT v2 bug "Shrink and grow on same step causes a
300s freeze": rank 0 acts as a joining replica and calls reconfigure() on a uuid
that no other rank will ever join, so it blocks in the bootstrap barrier. A
watcher thread then calls comm.abort() to try to break it out.

The abort is rejected: McclComm::abort() begins with
MCCLCOMM_CHECK_INITIALIZED(), and reconfigureTeardown() clears initialized_
before the barrier runs, so abort() raises for the whole blocked window and only
the reconfigure deadline bounds the wait.

Note that a reconfigure failure returned through the work result -- which is
what this barrier timeout produces -- is otherwise SILENT at this layer:
TorchComm::reconfigure() does not raise for it and work.wait() is a no-op, so
work.is_completed() is the only success signal. Exceptions raised inside
TorchCommMCCL::reconfigure() do still propagate; only work-result failures are
silent.

The assertions marked CONTRACT must be flipped when abort-during-reconfigure is
implemented.
"""

import os
import threading
import time
import unittest
from datetime import timedelta

import torch
import torchcomms
from torch.distributed import TCPStore

# Deadline handed to the blocked reconfigure. Long enough that the abort at
# ABORT_DELAY_S is unambiguously mid-barrier, short enough to keep this cheap.
RECONFIGURE_TIMEOUT = timedelta(seconds=10)
# When the watcher thread fires comm.abort().
ABORT_DELAY_S = 2.0
# Allow for small differences between the configured deadline and the measured
# wall-clock duration while still proving that abort() did not shorten the wait.
RECONFIGURE_TIMEOUT_TOLERANCE_S = 1.0
# Ceiling on the blocked call. Generous so this only fires on a genuine hang.
CEILING_S = RECONFIGURE_TIMEOUT.total_seconds() * 3
# work.wait() is expected to return immediately for the pre-computed failure.
WAIT_CEILING_S = 1.0
# Ceiling - not the expected duration - on how long the incumbents stay parked.
# They are released by a store key as soon as the joiner is done.
INCUMBENT_HOLD = timedelta(seconds=60)


class ReconfigureAbortTest(unittest.TestCase):
    """Test abort() while reconfigure() is blocked in the bootstrap barrier."""

    _shared_store: TCPStore | None = None

    def setUp(self) -> None:
        self.backend = os.getenv("TEST_BACKEND", "")
        if self.backend != "mccl":
            self.skipTest(f"Backend {self.backend} is not mccl")

        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available")

        # The probe is topology-independent, but the recovery AllReduce needs
        # the single-local-rank CTRAN topology supported by this test target.
        os.environ["NCCL_COMM_STATE_DEBUG_TOPO"] = "nolocal"
        # Pin the copy path. Any zero-copy mode makes reconfigure() first run a
        # cross-rank graph-registration agreement, which the stranded joiner
        # fails before reconfigureTeardown() clears initialized_ - a different
        # blocked window than the bootstrap barrier characterized here.
        os.environ["MCCL_ALLREDUCE_ZERO_COPY_MODE"] = "off"

        self.rank = int(
            os.environ.get("RANK", os.environ.get("OMPI_COMM_WORLD_RANK", 0))
        )
        self.world_size = int(
            os.environ.get("WORLD_SIZE", os.environ.get("OMPI_COMM_WORLD_SIZE", 1))
        )
        if self.world_size < 2:
            self.skipTest("Need at least 2 ranks to strand a joiner")

        device_id = self.rank % torch.cuda.device_count()
        self.device = torch.device(f"cuda:{device_id}")

        if ReconfigureAbortTest._shared_store is None:
            ReconfigureAbortTest._shared_store = TCPStore(
                host_name=os.environ.get("MASTER_ADDR", "localhost"),
                port=int(os.environ.get("MASTER_PORT", "29500")),
                world_size=self.world_size,
                is_master=(self.rank == 0),
                timeout=timedelta(seconds=60),
            )
        self.store = ReconfigureAbortTest._shared_store

    def _collect_handles(
        self, comm: torchcomms.TorchComm, key_prefix: str
    ) -> list[str]:
        """Gather every rank's init handle, including this rank's own.

        The joiner's own handle MUST be in the list it passes to reconfigure().
        Omitting it hits `urlToRank.at()` in McclComm and surfaces as a Python
        IndexError - a different failure mode than the barrier timeout under
        test here.
        """
        self.store.set(f"{key_prefix}_{self.rank}", comm.get_init_handle())
        return [
            self.store.get(f"{key_prefix}_{i}").decode("utf-8")
            for i in range(self.world_size)
        ]

    def test_abort_during_blocked_reconfigure(self):
        comm = torchcomms.new_comm(
            self.backend,
            self.device,
            "reconfigure_abort",
            enable_reconfigure=True,
            store=self.store,
            timeout=timedelta(seconds=120),
        )

        # Bring every rank into the dynamic regime on a quorum they all join.
        handles = self._collect_handles(comm, "reconfigure_abort_init")
        comm.reconfigure(
            uuid=0, init_handles=handles, timeout=timedelta(seconds=60)
        ).wait()
        # Success is asserted via the rank accessors, as the shared
        # ReconfigureTest does; is_completed() is used below only as the
        # failure signal.
        self.assertEqual(comm.get_size(), self.world_size)
        self.assertTrue(comm.is_abort_supported())
        self.assertFalse(comm.is_aborted())

        # The probe outcome is published, not just the fact that it finished.
        # Recovery is a collective, so entering it must be all-or-nothing: if
        # rank 0 bailed on an assertion, an incumbent that proceeded alone would
        # block in _collect_handles() until the store times out, burying the
        # real failure under a secondary one. This mirrors the cross-rank gate
        # in the C++ test.
        probe_key = "reconfigure_abort_probe_ok"
        if self.rank == 0:
            probe_ok = False
            try:
                self._run_stranded_joiner(comm, handles)
                probe_ok = True
            finally:
                # Always publish: an assertion failure must still release the
                # incumbents, or one failure becomes an INCUMBENT_HOLD stall.
                self.store.set(probe_key, "1" if probe_ok else "0")
        else:
            # Stay alive holding the bootstrap socket open, without joining
            # uuid=1. This is the incumbent that bumped to a different QCN.
            # Released by the store rather than a fixed sleep.
            self.store.wait([probe_key], INCUMBENT_HOLD)

        if self.store.get(probe_key).decode("utf-8") != "1":
            self.skipTest("joiner probe failed; skipping recovery (see rank 0)")

        # Everyone rejoins on a quorum they all participate in, so the comm is
        # INITIALIZED again and finalize() is legal.
        recovery_handles = self._collect_handles(comm, "reconfigure_abort_recovery")
        comm.reconfigure(
            uuid=2, init_handles=recovery_handles, timeout=timedelta(seconds=60)
        ).wait()
        self.assertEqual(comm.get_size(), self.world_size)
        # Recovery must clear the reason latched by the timed-out attempt.
        self.assertFalse(comm.is_aborted())

        tensor = torch.ones(1024, dtype=torch.float, device=self.device) * (
            comm.get_rank() + 1
        )
        comm.all_reduce(tensor, torchcomms.ReduceOp.SUM, async_op=False)
        torch.cuda.current_stream().synchronize()
        expected = sum(range(1, self.world_size + 1))
        self.assertTrue(
            torch.allclose(tensor, torch.full_like(tensor, expected)),
            "AllReduce after recovery reconfigure failed",
        )

        comm.finalize()

    def _run_stranded_joiner(
        self, comm: torchcomms.TorchComm, handles: list[str]
    ) -> None:
        """Block in reconfigure() on a quorum nobody joins, then try to abort."""
        abort_error = []
        abort_attempt_end = []

        def fire_abort() -> None:
            time.sleep(ABORT_DELAY_S)
            try:
                comm.abort()
            except Exception as e:  # noqa: BLE001 - recording, not handling
                abort_error.append(e)
            abort_attempt_end.append(time.monotonic())

        # reconfigure/abort both release the GIL in the pybind layer, so this
        # thread makes progress while the main thread is blocked.
        watcher = threading.Thread(target=fire_abort, daemon=True)
        watcher.start()

        start = time.monotonic()
        # This does NOT raise on failure, and wait() is a no-op - see the module
        # docstring. is_completed() is the only success signal.
        work = comm.reconfigure(
            uuid=1, init_handles=handles, timeout=RECONFIGURE_TIMEOUT
        )
        reconfigure_return = time.monotonic()
        wait_start = time.monotonic()
        work.wait()
        wait_return = time.monotonic()
        reconfigure_elapsed = reconfigure_return - start
        wait_elapsed = wait_return - wait_start
        watcher.join(timeout=CEILING_S)

        # Logging the abort's offset makes the record self-evidencing: a reader
        # can see it landed inside the blocked window rather than having to
        # trust the assertions below.
        # Guard the format: if the abort thread never completed, the assertion
        # below is the informative failure, not a TypeError in this print.
        abort_at = (
            f"{abort_attempt_end[0] - start:.1f}s"
            if abort_attempt_end
            else "never completed"
        )
        print(
            f"[Rank 0] blocked reconfigure returned in {reconfigure_elapsed:.1f}s "
            f"(deadline {RECONFIGURE_TIMEOUT.total_seconds():.0f}s); "
            f"work.wait returned in {wait_elapsed:.3f}s; "
            f"abort fired at {abort_at} into it; "
            f"is_completed={work.is_completed()}; "
            f"is_aborted={comm.is_aborted()}; "
            f"abort_error={abort_error[0] if abort_error else None}"
        )

        # Validity gate: without this the test can pass having never aborted
        # in flight. A late abort would still be rejected (initState_ stays
        # UNINITIALIZED after a failed reconfigure) and is_aborted() would
        # still be True, so every assertion below would hold vacuously.
        self.assertTrue(abort_attempt_end, "abort thread never completed")
        self.assertLess(
            abort_attempt_end[0],
            reconfigure_return,
            "abort() did not complete while reconfigure was still blocked, so "
            "this run proves nothing about in-flight abort",
        )

        # Invariants that must hold however abort evolves.
        self.assertFalse(
            work.is_completed(),
            "reconfigure succeeded even though no other rank joined uuid=1",
        )
        self.assertGreaterEqual(
            reconfigure_elapsed,
            RECONFIGURE_TIMEOUT.total_seconds() - RECONFIGURE_TIMEOUT_TOLERANCE_S,
            "reconfigure returned well before its deadline even though abort() "
            "was rejected",
        )
        self.assertLess(
            reconfigure_elapsed,
            CEILING_S,
            "reconfigure did not terminate within its own deadline - this is "
            "the 300s-freeze shape from the PAFT v2 thread",
        )
        self.assertLess(
            wait_elapsed,
            WAIT_CEILING_S,
            "work.wait() blocked even though reconfigure returned a pre-computed "
            "failure",
        )
        # The failure is otherwise silent. TorchComm also clears its internal
        # ranks_ list, but getRanks() is not bound in Python, so the observable
        # blast radius is the rank/size accessors starting to throw.
        with self.assertRaises(RuntimeError):
            comm.get_rank()
        with self.assertRaises(RuntimeError):
            comm.get_size()

        # CONTRACT (flip when abort-during-reconfigure lands): abort() is
        # rejected because reconfigureTeardown() already cleared initialized_.
        # Pin the mechanism, not just "something raised".
        self.assertTrue(
            abort_error,
            "abort() during reconfigure was accepted - if intentional, update "
            "this test to assert that abort actually shortens the wait",
        )
        self.assertIsInstance(abort_error[0], RuntimeError)
        self.assertIn(
            "called before init",
            str(abort_error[0]),
            f"abort() raised for an unexpected reason: {abort_error[0]}",
        )

        # The lapsed deadline latches an abort reason. is_aborted() has no init
        # guard, so it is safe to call here - and calling it is also what
        # *materializes* the lapsed deadline into a reason.
        #
        # This cannot distinguish TIMED_OUT from ABORTED, since the reason is
        # not exposed through the Python bindings; it only confirms the comm
        # noticed. McclCommFaultToleranceTest.
        # ExternalAbortDuringReconfigureIsRejectedAndCommRecovers pins the
        # mechanism at the MCCL layer.
        self.assertTrue(
            comm.is_aborted(),
            "expected the lapsed deadline to latch an abort reason",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
