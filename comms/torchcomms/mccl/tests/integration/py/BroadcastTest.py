#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

import unittest

from comms.torchcomms.tests.integration.py.BroadcastTest import (
    BroadcastTest as BaseBroadcastTest,
)


class BroadcastTest(BaseBroadcastTest):
    # TODO: The mccl will fail if the numElements is 0
    counts = [4, 1024, 1024 * 1024]


del BaseBroadcastTest

if __name__ == "__main__":
    unittest.main()
