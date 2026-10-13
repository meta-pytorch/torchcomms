#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

import unittest

from comms.torchcomms.tests.integration.py.BarrierTest import (
    BarrierTest as BaseBarrierTest,
)


class BarrierTest(BaseBarrierTest):
    pass


del BaseBarrierTest

if __name__ == "__main__":
    unittest.main()
