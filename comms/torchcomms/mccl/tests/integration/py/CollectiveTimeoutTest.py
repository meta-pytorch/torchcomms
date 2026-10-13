#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
import time
import unittest
from datetime import timedelta

import torch
from torchcomms import ReduceOp
from torchcomms.tests.integration.helpers.TorchCommTestHelpers import (
    TorchCommTestWrapper,
)


class CollectiveTimeoutTest(unittest.TestCase):
    """Test that the timeout watchdog detects hung all_reduce operations."""

    def setUp(self):
        os.environ["NCCL_COMM_STATE_DEBUG_TOPO"] = "nolocal"
        # Do not abort the process on timeout thread so that we can capture the timeout.
        self.wrapper = TorchCommTestWrapper(
            abort_process_on_timeout_or_error=False,
            hints={"watchdog_timeout_ms": "1000"},
        )
        self.torchcomm = self.wrapper.get_torchcomm()
        self.rank = self.torchcomm.get_rank()
        self.num_ranks = self.torchcomm.get_size()
        self.device = self.torchcomm.get_device()

    def test_all_reduce_timeout_detected(self):
        """
        Test that a hung collective is detected by the timeout watchdog.

        Rank 1 issues a recv (from rank 0) instead of all_reduce. Since rank 0
        never sends, rank 1's recv hangs. Since rank 1 never calls all_reduce,
        the other ranks' all_reduce also hangs. The timeout watchdog detects
        the timeout on all ranks.
        """
        input_tensor = torch.ones(1024, dtype=torch.float, device=self.device)

        try:
            if self.rank != 1:
                work = self.torchcomm.all_reduce(
                    input_tensor,
                    ReduceOp.SUM,
                    True,
                    # Collective timeout is passed to CtranComm. In order to test watchdog timeout,
                    # it should be larger than watchdog's timeout
                    timeout=timedelta(seconds=10),
                )
            else:
                # Issue a recv that no rank will send to, so rank 1 also hangs and times out
                work = self.torchcomm.recv(
                    input_tensor,
                    2,
                    True,
                    timeout=timedelta(seconds=10),
                )
            work.wait()

            # Wait for the watchdog to detect the timeout.
            # The watchdog polls every 100ms and timeout is 1s,
            # so we need to wait at least 1s + some margin.
            print(f"Rank {self.rank}: waiting for timeout detection")
            time.sleep(5)
            print(f"Rank {self.rank}: done waiting")

            # After the timeout, the work should not be completed
            self.assertFalse(
                work.is_completed(),
                "Work should not be completed — it should have timed out",
            )
        except RuntimeError:
            # A RuntimeError may be thrown if the MCCL layer detects the timeout
            pass
        finally:
            # Prevent TorchCommTestWrapper.__del__ from calling finalize() again
            self.wrapper.torchcomm = None
            work = None


if __name__ == "__main__":
    unittest.main()
