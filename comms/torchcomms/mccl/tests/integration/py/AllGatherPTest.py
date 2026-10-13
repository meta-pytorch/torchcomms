#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
import unittest

# The shared AllGatherPTest gates test_allgatherp on NCCL_CTRAN_ENABLE via a
# method decorator evaluated at IMPORT time, so these must be set before the
# import below or the test would silently SKIP. NCCL_COMM_STATE_DEBUG_TOPO and
# NCCL_IGNORE_TOPO_LOAD_FAILURE mirror the other single-node MCCL wrappers.
os.environ["NCCL_CTRAN_ENABLE"] = "true"
os.environ["NCCL_COMM_STATE_DEBUG_TOPO"] = "nolocal"
os.environ["NCCL_IGNORE_TOPO_LOAD_FAILURE"] = "1"

from torchcomms.tests.integration.py.AllGatherPTest import (
    AllGatherPTest as BaseAllGatherPTest,
)


class AllGatherPTest(BaseAllGatherPTest):
    pass


del BaseAllGatherPTest

if __name__ == "__main__":
    unittest.main()
