#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Regression test for the FlightRecorder gap-on-finalize bug.

``TorchComm::finalize()`` previously consumed a slot from the global
``op_id`` counter via ``GlobalOpIdGenerator::nextOpId()``, but the
``FlightRecorderHook`` short-circuits before recording finalize. That
left a hole in the recorder's index space and, because ``record()``
keys its ring-buffer slot off ``op_id % max_entries_``, the next real
collective tripped the growth-phase assertion
``entries_.size() == idx + 1``.

This test exercises the exact sequence: create comm, record a
collective, finalize, then create a *new* comm and record another
collective sharing the same recorder. Pre-fix, the second collective
trips the assertion. Post-fix, both collectives are recorded with
sequential ``op_id`` values and no gap.
"""

import json
import os
import unittest
from datetime import timedelta

import torch
import torchcomms
from torchcomms.hooks import FlightRecorderHook


class TestFlightRecorderFinalizeOpId(unittest.TestCase):
    backend = os.environ["TEST_BACKEND"]
    device = torch.device(os.environ.get("TEST_DEVICE", "cuda"))

    def test_finalize_does_not_consume_op_id(self) -> None:
        # ``isolated=True`` resets the global op_id generator so the
        # comm's first op starts at 0, making the post-fix expectation
        # below deterministic.
        recorder = FlightRecorderHook(max_entries=100, isolated=True)

        # --- First comm life cycle: one collective, then finalize. ---
        comm1 = torchcomms.new_comm(
            backend=self.backend,
            device=self.device,
            name="fr_finalize_op_id_1",
            timeout=timedelta(seconds=300),
        )
        recorder.register_with_comm(comm1)
        t = torch.rand(4, device=self.device)
        comm1.all_reduce(t, op=torchcomms.ReduceOp.SUM, async_op=False)
        comm1.finalize()

        # --- Second comm life cycle: one more collective on a fresh
        # comm sharing the same recorder. Pre-fix this is where the FR
        # growth-phase assertion fires because finalize on comm1 bumped
        # the global op_id counter without producing a recorder entry. ---
        comm2 = torchcomms.new_comm(
            backend=self.backend,
            device=self.device,
            name="fr_finalize_op_id_2",
            timeout=timedelta(seconds=300),
        )
        recorder.register_with_comm(comm2)
        comm2.all_reduce(t, op=torchcomms.ReduceOp.SUM, async_op=False)

        # Two collectives recorded; finalize must not have caused a gap.
        self.assertEqual(recorder.size(), 2)
        dump = json.loads(recorder.dump_json())
        op_ids = [entry["op_id"] for entry in dump["entries"]]
        self.assertEqual(
            len(op_ids), 2, f"expected 2 recorded ops, got entries={dump['entries']}"
        )
        op_ids.sort()
        # Sequential, no gap from finalize between them.
        self.assertEqual(
            op_ids[1] - op_ids[0],
            1,
            f"finalize on comm1 must not consume an op_id slot, but got "
            f"non-sequential op_ids={op_ids}",
        )

        comm2.finalize()


if __name__ == "__main__":
    unittest.main()
