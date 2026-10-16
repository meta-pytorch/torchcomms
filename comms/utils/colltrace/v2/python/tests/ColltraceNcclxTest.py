# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import os
import unittest

# NCCLX reads these when the communicator is created, below.
os.environ["NCCL_COLLTRACE"] = "lifecycle"
os.environ["NCCL_COLLTRACE_VERSION"] = "2"

import colltrace as ct
import torch
from torchcomms import new_comm, ReduceOp
from torchcomms.tests.integration.helpers.TorchCommTestHelpers import (
    get_device,
    get_rank_and_size,
    maybe_set_rank_envs,
)

COMM_NAME = "colltrace_ncclx_test"
NUM_ITERS = 8


class ColltraceNcclxTest(unittest.TestCase):
    def setUp(self) -> None:
        maybe_set_rank_envs()
        rank, _ = get_rank_and_size()
        self.device = get_device("ncclx", rank)
        self.torchcomm = new_comm("ncclx", self.device, name=COMM_NAME)
        major, _ = torch.cuda.get_device_capability(self.device)
        if major < 9:
            self.skipTest("colltrace v2 needs sm_90+")
        self.assertTrue(ct.enabled())

    def tearDown(self) -> None:
        self.torchcomm.finalize()
        self.torchcomm = None

    def _all_reduce(self) -> None:
        t = torch.ones(1024, device=self.device)
        self.torchcomm.all_reduce(t, ReduceOp.SUM, async_op=False)

    def test_all_reduce_start_end_per_collective(self) -> None:
        # Warm up (connection setup may launch extra work), then discard.
        self._all_reduce()
        torch.cuda.synchronize(self.device)
        ct.get_coll_events(COMM_NAME)

        before = ct.get_latest_coll_id(COMM_NAME)
        self.assertNotEqual(before, ct.INVALID_COLLID)
        for _ in range(NUM_ITERS):
            self._all_reduce()
        torch.cuda.synchronize(self.device)
        self.assertEqual(ct.get_latest_coll_id(COMM_NAME) - before, NUM_ITERS)

        events = ct.get_coll_events(COMM_NAME)
        starts = [e for e in events if e.kind == ct.EventKind.COLL_START]
        ends = [e for e in events if e.kind == ct.EventKind.COLL_END]
        expected_ids = [before + 1 + i for i in range(NUM_ITERS)]
        self.assertEqual(sorted(e.coll_id for e in starts), expected_ids)
        self.assertEqual(sorted(e.coll_id for e in ends), expected_ids)
        for coll_id in expected_ids:
            self.assertEqual(ct.get_num_kernels(coll_id), 1)
        start_ts = {e.coll_id: e.timestamp_ns for e in starts}
        end_ts = {e.coll_id: e.timestamp_ns for e in ends}
        for coll_id in expected_ids:
            self.assertLessEqual(start_ts[coll_id], end_ts[coll_id])
        self.assertEqual(ct.get_coll_events(COMM_NAME), [])


if __name__ == "__main__":
    unittest.main()
