#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

import json
import unittest

from torchcomms.tests.integration.py.TorchCommTestHelpers import TorchCommTestWrapper


class CommDumpTest(unittest.TestCase):
    def setUp(self):
        self.wrapper = TorchCommTestWrapper()
        self.torchcomm = self.wrapper.get_torchcomm()
        self.rank = self.torchcomm.get_rank()
        self.num_ranks = self.torchcomm.get_size()
        self.ncclx_backend = self.torchcomm.get_backend_impl()

    def tearDown(self):
        del self.torchcomm
        del self.wrapper

    def test_comm_dump_basic(self):
        """Test that comm_dump returns a non-empty dict with expected keys."""
        dump = self.ncclx_backend.comm_dump()

        self.assertIsInstance(dump, dict)
        self.assertGreater(len(dump), 0, "comm_dump() returned empty dict")

        # Verify expected metadata keys
        self.assertIn("commHash", dump)
        self.assertIn("rank", dump)
        self.assertIn("nRanks", dump)

        # Verify rank and nRanks values
        self.assertEqual(dump["rank"], str(self.rank))
        self.assertEqual(dump["nRanks"], str(self.num_ranks))

        # Verify all values are valid JSON
        for key, val in dump.items():
            try:
                json.loads(val)
            except json.JSONDecodeError:
                self.fail(f"Value for key '{key}' is not valid JSON: {val}")

    def test_comm_dump_split_comm(self):
        """Test comm_dump on a split communicator reflects the subgroup."""
        # Split into even and odd rank groups
        even_ranks = list(range(0, self.num_ranks, 2))
        odd_ranks = list(range(1, self.num_ranks, 2))

        if self.rank % 2 == 0:
            split_comm = self.torchcomm.split(even_ranks, name="even_split")
            expected_size = len(even_ranks)
        else:
            split_comm = self.torchcomm.split(odd_ranks, name="odd_split")
            expected_size = len(odd_ranks)

        self.assertIsNotNone(split_comm)
        split_backend = split_comm.get_backend_impl()
        dump = split_backend.comm_dump()

        self.assertIn("nRanks", dump)
        self.assertEqual(dump["nRanks"], str(expected_size))

        # commHash should differ from the parent communicator
        parent_dump = self.ncclx_backend.comm_dump()
        self.assertNotEqual(
            dump.get("commHash"),
            parent_dump.get("commHash"),
            "Split comm should have a different commHash than parent",
        )

        del split_comm


if __name__ == "__main__":
    unittest.main()
