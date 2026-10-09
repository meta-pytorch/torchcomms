#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

import itertools
import os
import unittest

from comms.torchcomms.tests.integration.py.AllReduceTest import (
    AllReduceTest as BaseAllReduceTest,
)


class AllReduceTest(BaseAllReduceTest):
    os.environ["NCCL_COMM_STATE_DEBUG_TOPO"] = "nolocal"

    def get_test_cases(self):
        # Create a test params
        normal_test_cases = list(itertools.product(self.counts, self.dtypes, self.ops))
        return normal_test_cases


del BaseAllReduceTest

if __name__ == "__main__":
    unittest.main()
