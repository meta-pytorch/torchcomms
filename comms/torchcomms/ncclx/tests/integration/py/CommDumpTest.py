#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

import json
import unittest

from torchcomms._comms_ncclx import comm_dump_all
from torchcomms.tests.integration.helpers.TorchCommTestHelpers import (
    TorchCommTestWrapper,
)


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

    def test_comm_dump_all_basic(self):
        """Test that comm_dump_all returns a nested dict with at least one comm."""
        all_dumps = comm_dump_all()

        self.assertIsInstance(all_dumps, dict)
        self.assertGreaterEqual(
            len(all_dumps), 1, "comm_dump_all() returned no communicators"
        )

        for comm_key, dump in all_dumps.items():
            if comm_key == "GlobalInfo":
                continue
            self.assertIsInstance(dump, dict)
            self.assertGreater(len(dump), 0, f"Dump for comm {comm_key} is empty")
            self.assertIn("rank", dump, f"Missing rank for comm {comm_key}")
            self.assertIn("nRanks", dump, f"Missing nRanks for comm {comm_key}")

    def test_comm_dump_all_contains_current_comm(self):
        """Test that comm_dump_all contains both parent and split communicators."""
        # Create a split communicator so there are at least 2 comms
        even_ranks = list(range(0, self.num_ranks, 2))
        odd_ranks = list(range(1, self.num_ranks, 2))

        if self.rank % 2 == 0:
            split_comm = self.torchcomm.split(even_ranks, name="even_split")
        else:
            split_comm = self.torchcomm.split(odd_ranks, name="odd_split")

        self.assertIsNotNone(split_comm)

        parent_dump = self.ncclx_backend.comm_dump()
        split_dump = split_comm.get_backend_impl().comm_dump()
        all_dumps = comm_dump_all()

        # Both parent and split comm hashes should appear in dump_all
        parent_hash = parent_dump.get("commHash", "").strip('"')
        split_hash = split_dump.get("commHash", "").strip('"')

        parent_found = any(parent_hash in key for key in all_dumps)
        split_found = any(split_hash in key for key in all_dumps)

        self.assertTrue(
            parent_found,
            f"Parent comm hash {parent_hash} not found in comm_dump_all keys: "
            f"{list(all_dumps.keys())}",
        )
        self.assertTrue(
            split_found,
            f"Split comm hash {split_hash} not found in comm_dump_all keys: "
            f"{list(all_dumps.keys())}",
        )

        # dump_all should have at least 2 communicators
        self.assertGreaterEqual(len(all_dumps), 2)

        del split_comm

    def test_comm_dump_all_with_request_fields_comm_info(self):
        """Test that requestFields hint filters to only the requested keys."""
        all_dumps = comm_dump_all(hints={"comm_dump::requestFields": "rank;nRanks"})

        self.assertIsInstance(all_dumps, dict)
        self.assertGreaterEqual(len(all_dumps), 1)

        for comm_key, dump in all_dumps.items():
            if comm_key == "GlobalInfo":
                continue
            self.assertIn("rank", dump)
            self.assertIn("nRanks", dump)
            # Keys not requested should be absent
            self.assertNotIn("commHash", dump)
            self.assertNotIn("localRank", dump)
            self.assertNotIn("CT_pastColls", dump)

    def test_comm_dump_all_with_request_fields_global_info(self):
        """Test that requesting only GlobalInfo keys skips per-comm dumps."""
        all_dumps = comm_dump_all(
            hints={"comm_dump::requestFields": "GlobalInfo::totalCommDurPerIterationUs"}
        )

        self.assertIsInstance(all_dumps, dict)
        self.assertIn("GlobalInfo", all_dumps)
        self.assertIn("totalCommDurPerIterationUs", all_dumps["GlobalInfo"])

        # Per-communicator entries should be skipped
        for comm_key, dump in all_dumps.items():
            if comm_key == "GlobalInfo":
                continue
            self.assertEqual(
                len(dump),
                0,
                f"Per-comm dump for {comm_key} should be empty when only GlobalInfo requested",
            )

    def test_comm_dump_all_none_request_fields_dumps_everything(self):
        """Test that empty hints (default) dumps all fields."""
        all_dumps = comm_dump_all()

        self.assertIsInstance(all_dumps, dict)
        self.assertGreaterEqual(len(all_dumps), 1)

        for comm_key, dump in all_dumps.items():
            if comm_key == "GlobalInfo":
                continue
            # Comm info keys are always present
            self.assertIn("rank", dump)
            self.assertIn("nRanks", dump)
            self.assertIn("commHash", dump)
            # CollTrace keys require them to be manually enabled

    def test_comm_dump_all_mixed_per_comm_and_global_info(self):
        """Test request_fields with both per-comm and GlobalInfo keys."""
        all_dumps = comm_dump_all(
            hints={
                "comm_dump::requestFields": "rank;GlobalInfo::totalCommDurPerIterationUs"
            }
        )

        self.assertIsInstance(all_dumps, dict)

        # GlobalInfo should be present
        self.assertIn("GlobalInfo", all_dumps)
        self.assertIn("totalCommDurPerIterationUs", all_dumps["GlobalInfo"])

        # Per-communicator should have only rank
        for comm_key, dump in all_dumps.items():
            if comm_key == "GlobalInfo":
                continue
            self.assertIn("rank", dump)
            self.assertNotIn("commHash", dump)
            self.assertNotIn("CT_pastColls", dump)


if __name__ == "__main__":
    unittest.main()
