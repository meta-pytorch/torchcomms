#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
import unittest

from comms.torchcomms.tests.integration.py.SplitTest import SplitTest as BaseSplitTest


class SplitTest(BaseSplitTest):
    os.environ["NCCL_COMM_STATE_DEBUG_TOPO"] = "nolocal"


del BaseSplitTest

if __name__ == "__main__":
    unittest.main()
