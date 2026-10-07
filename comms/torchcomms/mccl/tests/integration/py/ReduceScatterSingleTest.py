#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
import unittest

from comms.torchcomms.tests.integration.py.ReduceScatterSingleTest import (
    ReduceScatterSingleTest as BaseReduceScatterSingleTest,
)
from torchcomms import ReduceOp


class ReduceScatterSingleTest(BaseReduceScatterSingleTest):
    os.environ["MCCL_PRIMS_ENABLE"] = "1"
    os.environ["MCCL_CTRAN_ENABLE"] = "0"
    os.environ["NCCL_COMM_STATE_DEBUG_TOPO"] = "nolocal"
    os.environ["NCCL_IGNORE_TOPO_LOAD_FAILURE"] = "1"

    def get_test_cases(self):
        return [
            test_case
            for test_case in super().get_test_cases()
            if test_case[2] != ReduceOp.AVG
        ]


del BaseReduceScatterSingleTest

if __name__ == "__main__":
    unittest.main()
