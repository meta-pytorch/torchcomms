#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

import itertools
import os
import unittest

import torch
from torchcomms._comms_ncclx import RdmaTransport
from torchcomms.tests.integration.py.TorchCommTestHelpers import (
    get_dtype_name,
    TorchCommTestWrapper,
)


class TransportTest(unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test."""

    def tearDown(self):
        """Clean up after each test."""

    def test_basic(self) -> None:
        device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))

        transport = RdmaTransport(device)


if __name__ == "__main__" and os.environ["TEST_BACKEND"] == "ncclx":
    unittest.main()
