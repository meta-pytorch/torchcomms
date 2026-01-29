#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Test that init_process_group raises error when backend is None."""

import unittest

from torchcomms import distwrap as dist


class Test(unittest.TestCase):
    def test(self) -> None:
        with self.assertRaises(ValueError) as context:
            dist.init_process_group(backend=None)
        self.assertIn("backend must be specified", str(context.exception))


if __name__ == "__main__":
    unittest.main()
