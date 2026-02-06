# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import os
import typing
import unittest
from datetime import timedelta

import torch
import torchcomms
from torch.distributed.flight_recorder.components.builder import build_db
from torch.distributed.flight_recorder.components.types import (
    Collective,
    Group,
    Membership,
)
from torchcomms import FlightRecorderHook
from torchcomms.objcol import all_gather_object


class TestFlightRecorderHook(unittest.TestCase):
    """Test FlightRecorderHook for tracking collective operations."""

    def test_create_flight_recorder(self) -> None:
        """Test creating a FlightRecorderHook."""
        recorder = FlightRecorderHook()
        self.assertIsNotNone(recorder)
        self.assertEqual(recorder.size(), 0)
        self.assertFalse(recorder.is_enabled())

    def test_create_flight_recorder_with_custom_size(self) -> None:
        """Test creating a FlightRecorderHook with custom buffer size."""
        recorder = FlightRecorderHook(max_entries=100)
        self.assertIsNotNone(recorder)
        self.assertEqual(recorder.size(), 0)

    def test_dump_json_empty(self) -> None:
        """Test dumping JSON when no operations have been recorded."""
        recorder = FlightRecorderHook()
        json_str = recorder.dump_json()

        # Parse and validate JSON structure
        data = json.loads(json_str)
        self.assertIn("version", data)
        self.assertEqual(data["version"], "2.10")
        self.assertIn("pg_config", data)
        self.assertIn("pg_status", data)

    def test_reset(self) -> None:
        """Test resetting the flight recorder."""
        recorder = FlightRecorderHook()
        recorder.reset()
        self.assertEqual(recorder.size(), 0)

    def _validate_entry_format(self, entry: dict) -> None:
        """Validate that an entry matches the OSS FlightRecorder format.

        This format is expected by the fr_trace analyzer tools from
        torch.distributed.flight_recorder.
        """
        # Required fields
        required_fields = [
            "record_id",
            "pg_id",
            "process_group",
            "collective_seq_id",
            "p2p_seq_id",
            "op_id",
            "profiling_name",
            "input_sizes",
            "input_dtypes",
            "output_sizes",
            "output_dtypes",
            "time_created_ns",
            "state",
            "is_p2p",
            "retired",
            "timeout_ms",
        ]

        for field in required_fields:
            self.assertIn(field, entry, f"Missing required field '{field}' in entry")

        # Validate types
        self.assertIsInstance(entry["record_id"], int)
        self.assertIsInstance(entry["pg_id"], int)
        self.assertIsInstance(entry["process_group"], list)
        self.assertEqual(len(entry["process_group"]), 2)  # (name, desc)
        self.assertIsInstance(entry["collective_seq_id"], int)
        self.assertIsInstance(entry["p2p_seq_id"], int)
        self.assertIsInstance(entry["op_id"], int)
        self.assertIsInstance(entry["profiling_name"], str)
        self.assertIsInstance(entry["input_sizes"], list)
        self.assertIsInstance(entry["input_dtypes"], list)
        self.assertIsInstance(entry["output_sizes"], list)
        self.assertIsInstance(entry["output_dtypes"], list)
        self.assertIsInstance(entry["time_created_ns"], int)
        self.assertIn(entry["state"], ["scheduled", "started", "completed"])
        self.assertIsInstance(entry["is_p2p"], bool)
        self.assertIsInstance(entry["retired"], bool)
        self.assertIsInstance(entry["timeout_ms"], int)

        # Validate profiling_name format (uses nccl: prefix as expected by FR analyzer)
        self.assertIsInstance(entry["profiling_name"], str)
        self.assertGreater(
            len(entry["profiling_name"]), 0, "profiling_name should not be empty"
        )
        self.assertTrue(
            entry["profiling_name"].startswith("nccl:"),
            f"profiling_name should start with 'nccl:', got: {entry['profiling_name']}",
        )

        # Validate time_created_ns is positive
        self.assertGreater(
            entry["time_created_ns"], 0, "time_created_ns should be positive"
        )

        # Validate state is one of the valid values
        valid_states = ["scheduled", "started", "completed"]
        self.assertIn(
            entry["state"],
            valid_states,
            f"state should be one of {valid_states}, got: {entry['state']}",
        )

        # Validate timeout_ms is positive
        self.assertGreater(entry["timeout_ms"], 0, "timeout_ms should be positive")

        # Validate optional duration_ms if present
        if "duration_ms" in entry:
            self.assertIsInstance(entry["duration_ms"], (int, float))
            self.assertGreaterEqual(
                entry["duration_ms"], 0, "duration_ms should be non-negative"
            )

    def _create_recorder_with_entries(
        self,
        comm_name: str,
    ) -> tuple[FlightRecorderHook, dict, typing.Any]:
        """Create a FlightRecorderHook with real entries from a gloo communicator.

        Returns:
            Tuple of (recorder, parsed_json_data, comm) or raises SkipTest if
            gloo backend is not available.
        """
        # Create communicator
        backend = os.environ["TEST_BACKEND"]
        device = torch.device(os.environ.get("TEST_DEVICE", "cuda"))
        comm = torchcomms.new_comm(
            backend=backend,
            device=device,
            name=comm_name,
            timeout=timedelta(seconds=300),
        )

        # Create and register flight recorder
        recorder = FlightRecorderHook(max_entries=100)
        recorder.register_with_comm(comm, pg_id=0, pg_desc="test_group")

        # Run a collective operation to generate entries
        t = torch.rand(10, 10, device=device)
        comm.all_reduce(t, op=torchcomms.ReduceOp.SUM, async_op=False)

        # Parse the JSON output
        json_str = recorder.dump_json()
        data = json.loads(json_str)

        return recorder, data, comm

    def _prepare_trace_for_analyzer(self, data: dict, rank: int = 0) -> dict:
        """Prepare a FlightRecorder trace for the analyzer.

        The analyzer expects:
        - "rank" field in the trace
        - pg_config with pg_uid as key containing "desc" and "ranks" fields
        """
        # Get process group info from entries
        pg_name = "default_pg"
        pg_desc = ""
        if data.get("entries"):
            entry = data["entries"][0]
            if "process_group" in entry and len(entry["process_group"]) >= 2:
                pg_name = entry["process_group"][0]
                pg_desc = entry["process_group"][1]

        return {
            "version": data.get("version", "2.10"),
            "rank": rank,
            "pg_config": {
                pg_name: {
                    "desc": pg_desc,
                    "ranks": "[0]",  # Single rank for testing
                }
            },
            "pg_status": data.get("pg_status", {}),
            "entries": data.get("entries", []),
        }

    def test_build_db_parses_trace(self) -> None:
        """Test that build_db from flight_recorder can parse our real trace format.

        This test uses the flight_recorder's build_db function to validate
        that our trace format is compatible with the OSS trace analyzer.
        Reference: caffe2/torch/distributed/debug/_frontend.py#L483-L496
        """

        recorder, data, comm = self._create_recorder_with_entries(
            comm_name="test_comm_e2e"
        )

        # Verify the trace structure
        self.assertEqual(data["version"], "2.10")
        self.assertGreater(len(data["entries"]), 0)

        # Validate each entry
        for entry in data["entries"]:
            self._validate_entry_format(entry)

        output = [{} for _ in range(comm.get_size())]
        all_gather_object(comm, output, data)

        try:
            # Prepare traces for two "ranks" (using same data, simulating 2-rank job)
            details = {
                f"rank_{i}": self._prepare_trace_for_analyzer(datum, rank=i)
                for i, datum in enumerate(output)
            }

            # Create args namespace as expected by build_db
            args = argparse.Namespace(
                verbose=False,
                just_print_entries=False,
                allow_incomplete_ranks=True,
                mismatch_cap=10,
            )

            version = "2.10"

            # Build the database - this validates our format is compatible
            db = build_db(details, args, version)

            # Verify the database was built correctly
            self.assertIsNotNone(db)
            self.assertIsInstance(db.groups, list)
            self.assertIsInstance(db.memberships, list)
            self.assertIsInstance(db.collectives, list)

            # Verify groups were parsed
            self.assertGreater(len(db.groups), 0, "Expected at least one group")
            for group in db.groups:
                self.assertIsInstance(group, Group)

            # Verify memberships were parsed
            self.assertGreater(
                len(db.memberships), 0, "Expected at least one membership"
            )
            for membership in db.memberships:
                self.assertIsInstance(membership, Membership)

            # Verify collectives were parsed from real entries
            self.assertGreater(
                len(db.collectives), 0, "Expected at least one collective"
            )
            for collective in db.collectives:
                self.assertIsInstance(collective, Collective)
        finally:
            recorder.unregister()
            comm.finalize()


if __name__ == "__main__":
    unittest.main()
