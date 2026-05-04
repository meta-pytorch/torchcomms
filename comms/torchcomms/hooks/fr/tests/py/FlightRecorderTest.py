# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-unsafe
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import os
import pickle
import tempfile
import typing
import unittest
from datetime import timedelta

# Set dynamic filename mode for DebugInfoWriter at module level.
# This ensures the singleton is created with enable_dynamic_filename_=true
# so env var TORCHCOMM_FR_DUMP_TEMP_FILE is re-read on each write.
os.environ["TORCHCOMM_FR_DUMP_DYNAMIC_FILE_NAME"] = "1"

import torch
import torchcomms
from torch.distributed.flight_recorder.components.builder import build_db
from torch.distributed.flight_recorder.components.types import (
    Collective,
    Group,
    Membership,
)
from torchcomms.hooks import FlightRecorderHook
from torchcomms.objcol import all_gather_object
from torchcomms.tests.integration.py.TorchCommTestHelpers import (
    get_rank_and_size,
    skip_backend,
)


class TestFlightRecorderHook(unittest.TestCase):
    """Test FlightRecorderHook for tracking collective operations."""

    backend = os.environ["TEST_BACKEND"]
    device = torch.device(os.environ.get("TEST_DEVICE", "cuda"))

    def test_create_flight_recorder(self) -> None:
        """Test creating a FlightRecorderHook."""
        recorder = FlightRecorderHook(isolated=True)
        self.assertIsNotNone(recorder)
        self.assertEqual(recorder.size(), 0)
        self.assertFalse(recorder.is_enabled())

    def test_create_flight_recorder_with_custom_size(self) -> None:
        """Test creating a FlightRecorderHook with custom buffer size."""
        recorder = FlightRecorderHook(max_entries=100, isolated=True)
        self.assertIsNotNone(recorder)
        self.assertEqual(recorder.size(), 0)

    def test_dump_json_empty(self) -> None:
        """Test dumping JSON when no operations have been recorded."""
        recorder = FlightRecorderHook(isolated=True)
        json_str = recorder.dump_json()

        # Parse and validate JSON structure
        data = json.loads(json_str)
        self.assertIn("version", data)
        self.assertEqual(data["version"], "2.10")
        self.assertIn("pg_config", data)
        self.assertIn("pg_status", data)

    def test_reset(self) -> None:
        """Test resetting the flight recorder."""
        recorder = FlightRecorderHook(isolated=True)
        recorder.reset()
        self.assertEqual(recorder.size(), 0)

    def test_size_across_reset(self) -> None:
        """Test that size() correctly returns 0 after reset.

        This verifies that size() returns the number of entries in the
        current epoch, not the total entries in the underlying buffer.
        """
        comm = torchcomms.new_comm(
            backend=self.backend,
            device=self.device,
            name="test_size_reset",
            timeout=timedelta(seconds=300),
        )

        recorder = FlightRecorderHook(max_entries=100, isolated=True)
        recorder.register_with_comm(comm)

        # Initially size should be 0
        self.assertEqual(recorder.size(), 0)

        # Run a collective operation
        t = torch.rand(10, 10, device=self.device)
        comm.all_reduce(t, op=torchcomms.ReduceOp.SUM, async_op=False)

        # Size should be 1 after one operation
        self.assertEqual(recorder.size(), 1)

        # Run another operation
        comm.all_reduce(t, op=torchcomms.ReduceOp.SUM, async_op=False)

        # Size should be 2 after two operations
        self.assertEqual(recorder.size(), 2)

        # Reset the recorder
        recorder.reset()

        # Size should be 0 after reset
        self.assertEqual(recorder.size(), 0)

        # Run one more operation
        comm.all_reduce(t, op=torchcomms.ReduceOp.SUM, async_op=False)

        # Size should be 1 again after reset and one operation
        self.assertEqual(recorder.size(), 1)

        # Clean up
        recorder.unregister()
        comm.finalize()

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

        # Validate profiling_name format (uses backend: prefix as expected by FR analyzer)
        self.assertIsInstance(entry["profiling_name"], str)
        self.assertGreater(
            len(entry["profiling_name"]), 0, "profiling_name should not be empty"
        )
        self.assertTrue(
            entry["profiling_name"].startswith(f"{self.backend}:"),
            f"profiling_name should start with '{self.backend}:', got: {entry['profiling_name']}",
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
        """Create a FlightRecorderHook with real entries from a communicator.

        Returns:
            Tuple of (recorder, parsed_json_data, comm) or raises SkipTest if
            the specified backend is not available.
        """
        # Create communicator
        comm = torchcomms.new_comm(
            backend=self.backend,
            device=self.device,
            name=comm_name,
            timeout=timedelta(seconds=300),
        )

        recorder = FlightRecorderHook(max_entries=100, isolated=True)
        recorder.register_with_comm(comm)

        # Run a collective operation to generate entries
        t = torch.rand(10, 10, device=self.device)
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

    def test_fr_record_reset(self) -> None:
        """Test that reset clears the flight recorder entries.

        Verifies that after performing some operations, resetting the recorder,
        and then performing more operations, only the post-reset operations
        are recorded.
        """
        comm = torchcomms.new_comm(
            backend=self.backend,
            device=self.device,
            name="test_comm_reset",
            timeout=timedelta(seconds=300),
        )

        recorder = FlightRecorderHook(max_entries=100, isolated=True)
        recorder.register_with_comm(comm)

        # Run 5 collective operations
        t = torch.rand(10, 10, device=self.device)
        for _ in range(5):
            comm.all_reduce(t, op=torchcomms.ReduceOp.SUM, async_op=False)

        # Verify we have entries before reset
        json_str = recorder.dump_json()
        data = json.loads(json_str)
        pre_reset_count = len(data.get("entries", []))
        self.assertGreater(pre_reset_count, 0, "Should have entries before reset")

        # Reset the flight recorder
        recorder.reset()

        # Verify entries are cleared
        json_str = recorder.dump_json()
        data = json.loads(json_str)
        self.assertEqual(
            len(data.get("entries", [])), 0, "Entries should be cleared after reset"
        )

        # Run 4 more collective operations
        for _ in range(4):
            comm.all_reduce(t, op=torchcomms.ReduceOp.SUM, async_op=False)

        # Verify we only have the 4 new entries
        json_str = recorder.dump_json()
        data = json.loads(json_str)
        post_reset_count = len(data.get("entries", []))
        self.assertGreater(
            post_reset_count, 0, "Should have entries after new operations"
        )

        # Validate entry format for all entries
        for entry in data.get("entries", []):
            self._validate_entry_format(entry)

        recorder.unregister()
        comm.finalize()

    def test_fr_record_reset_circular_buffer_full(self) -> None:
        """Test reset when circular buffer is completely full.

        Verifies that when buffer is full, reset works correctly and
        subsequent operations are properly recorded.
        """
        comm = torchcomms.new_comm(
            backend=self.backend,
            device=self.device,
            name="test_comm_buffer_full",
            timeout=timedelta(seconds=300),
        )

        # Use a small buffer size
        buffer_size = 10
        recorder = FlightRecorderHook(max_entries=buffer_size, isolated=True)
        recorder.register_with_comm(comm)

        # Fill the buffer completely
        t = torch.rand(10, 10, device=self.device)
        for _ in range(buffer_size):
            comm.all_reduce(t, op=torchcomms.ReduceOp.SUM, async_op=False)

        # Verify buffer is at capacity
        json_str = recorder.dump_json()
        data = json.loads(json_str)
        self.assertEqual(
            len(data.get("entries", [])),
            buffer_size,
            f"Buffer should be full with {buffer_size} entries",
        )

        # Reset the flight recorder
        recorder.reset()

        # Fill the buffer again completely
        for _ in range(buffer_size):
            comm.all_reduce(t, op=torchcomms.ReduceOp.SUM, async_op=False)

        # Verify we have exactly buffer_size new entries
        json_str = recorder.dump_json()
        data = json.loads(json_str)
        entries = data.get("entries", [])
        self.assertEqual(
            len(entries),
            buffer_size,
            f"Should have {buffer_size} entries after refilling buffer",
        )

        # Verify all entries are valid
        for entry in entries:
            self._validate_entry_format(entry)
            self.assertEqual(entry["profiling_name"], f"{self.backend}:all_reduce")

        recorder.unregister()
        comm.finalize()

    def test_fr_record_reset_partial_overwrite(self) -> None:
        """Test reset followed by partial refill.

        Verifies that after filling buffer, resetting, and adding fewer
        entries than buffer size, only new entries are returned.
        """
        comm = torchcomms.new_comm(
            backend=self.backend,
            device=self.device,
            name="test_comm_partial",
            timeout=timedelta(seconds=300),
        )

        buffer_size = 10
        recorder = FlightRecorderHook(max_entries=buffer_size, isolated=True)
        recorder.register_with_comm(comm)

        # Fill the buffer completely
        t = torch.rand(10, 10, device=self.device)
        for _ in range(buffer_size):
            comm.all_reduce(t, op=torchcomms.ReduceOp.SUM, async_op=False)

        # Reset the flight recorder
        recorder.reset()

        # Add only 3 new entries
        partial_count = 3
        for _ in range(partial_count):
            comm.all_reduce(t, op=torchcomms.ReduceOp.SUM, async_op=False)

        # Verify we only get the 3 new entries
        json_str = recorder.dump_json()
        data = json.loads(json_str)
        entries = data.get("entries", [])
        self.assertEqual(
            len(entries),
            partial_count,
            f"Should have only {partial_count} entries after partial refill",
        )

        # Validate each entry
        for entry in entries:
            self._validate_entry_format(entry)

        recorder.unregister()
        comm.finalize()

    def test_fr_record_multiple_resets(self) -> None:
        """Test multiple consecutive resets.

        Verifies that multiple resets work correctly and each reset
        properly clears the previous entries.
        """
        comm = torchcomms.new_comm(
            backend=self.backend,
            device=self.device,
            name="test_comm_multi_reset",
            timeout=timedelta(seconds=300),
        )

        recorder = FlightRecorderHook(max_entries=100, isolated=True)
        recorder.register_with_comm(comm)

        t = torch.rand(10, 10, device=self.device)

        # First batch: 2 entries
        for _ in range(2):
            comm.all_reduce(t, op=torchcomms.ReduceOp.SUM, async_op=False)

        json_str = recorder.dump_json()
        data = json.loads(json_str)
        self.assertEqual(len(data.get("entries", [])), 2)

        # First reset
        recorder.reset()

        # Second batch: 3 entries
        for _ in range(3):
            comm.all_reduce(t, op=torchcomms.ReduceOp.SUM, async_op=False)

        json_str = recorder.dump_json()
        data = json.loads(json_str)
        self.assertEqual(len(data.get("entries", [])), 3)

        # Second reset
        recorder.reset()

        # Third batch: 4 entries
        for _ in range(4):
            comm.all_reduce(t, op=torchcomms.ReduceOp.SUM, async_op=False)

        # Should only see the last 4 entries
        json_str = recorder.dump_json()
        data = json.loads(json_str)
        entries = data.get("entries", [])
        self.assertEqual(len(entries), 4)

        # Validate all entries
        for entry in entries:
            self._validate_entry_format(entry)

        recorder.unregister()
        comm.finalize()

    def test_fr_multiple_collective_operations(self) -> None:
        """Test flight recorder with different collective operations.

        Verifies that the flight recorder correctly captures different
        types of collective operations with proper metadata.
        """
        comm = torchcomms.new_comm(
            backend=self.backend,
            device=self.device,
            name="test_comm_multi_ops",
            timeout=timedelta(seconds=300),
        )

        recorder = FlightRecorderHook(max_entries=100, isolated=True)
        recorder.register_with_comm(comm)

        # Perform different collective operations
        t = torch.rand(10, 10, device=self.device)

        # All-reduce
        comm.all_reduce(t, op=torchcomms.ReduceOp.SUM, async_op=False)

        # Broadcast
        comm.broadcast(t, root=0, async_op=False)

        # All-reduce again
        comm.all_reduce(t, op=torchcomms.ReduceOp.SUM, async_op=False)

        # Dump and verify entries
        json_str = recorder.dump_json(True)
        data = json.loads(json_str)
        entries = data.get("entries", [])

        # Verify we have entries for each operation
        self.assertGreater(
            len(entries), 0, "Should have recorded collective operations"
        )

        # Validate each entry format
        for entry in entries:
            self._validate_entry_format(entry)
            # Verify profiling name is one of the expected operations
            self.assertTrue(
                entry["profiling_name"].startswith(f"{self.backend}:"),
                f"profiling_name should start with '{self.backend}:', got: {entry['profiling_name']}",
            )

        recorder.unregister()
        comm.finalize()

    def test_fr_entry_ordering(self) -> None:
        """Test that flight recorder entries are in chronological order.

        Verifies that entries are returned in the correct order based on
        their sequence IDs and timestamps.
        """
        comm = torchcomms.new_comm(
            backend=self.backend,
            device=self.device,
            name="test_comm_ordering",
            timeout=timedelta(seconds=300),
        )

        recorder = FlightRecorderHook(max_entries=100, isolated=True)
        recorder.register_with_comm(comm)

        # Perform several operations
        t = torch.rand(10, 10, device=self.device)
        num_ops = 5
        for _ in range(num_ops):
            comm.all_reduce(t, op=torchcomms.ReduceOp.SUM, async_op=False)

        # Dump and verify entries
        json_str = recorder.dump_json(True)
        data = json.loads(json_str)
        entries = data.get("entries", [])

        self.assertEqual(len(entries), num_ops)

        # Verify record_id is monotonically increasing
        prev_record_id = -1
        prev_timestamp = 0
        for entry in entries:
            self._validate_entry_format(entry)

            # record_id should be increasing
            self.assertGreater(
                entry["record_id"],
                prev_record_id,
                "record_id should be monotonically increasing",
            )
            prev_record_id = entry["record_id"]

            # time_created_ns should be non-decreasing
            self.assertGreaterEqual(
                entry["time_created_ns"],
                prev_timestamp,
                "time_created_ns should be non-decreasing",
            )
            prev_timestamp = entry["time_created_ns"]

        recorder.unregister()
        comm.finalize()

    def test_fr_tensor_sizes_and_dtypes(self) -> None:
        """Test that flight recorder captures tensor sizes and dtypes correctly.

        Verifies that input/output sizes and dtypes are properly recorded.
        """
        comm = torchcomms.new_comm(
            backend=self.backend,
            device=self.device,
            name="test_comm_sizes",
            timeout=timedelta(seconds=300),
        )

        recorder = FlightRecorderHook(max_entries=100, isolated=True)
        recorder.register_with_comm(comm)

        # Create tensors with specific sizes and dtypes
        tensor_float = torch.rand(3, 4, device=self.device, dtype=torch.float32)
        tensor_int = torch.randint(
            0, 100, (5, 6), device=self.device, dtype=torch.int64
        )

        # All-reduce float tensor
        comm.all_reduce(tensor_float, op=torchcomms.ReduceOp.SUM, async_op=False)

        # All-reduce int tensor
        comm.all_reduce(tensor_int, op=torchcomms.ReduceOp.SUM, async_op=False)

        # Dump and verify entries
        json_str = recorder.dump_json()
        data = json.loads(json_str)
        entries = data.get("entries", [])

        self.assertEqual(len(entries), 2)

        # Verify first entry (float tensor)
        self._validate_entry_format(entries[0])
        self.assertIn([3, 4], entries[0]["input_sizes"])
        self.assertIn("Float", entries[0]["input_dtypes"])

        # Verify second entry (int tensor)
        self._validate_entry_format(entries[1])
        self.assertIn([5, 6], entries[1]["input_sizes"])
        self.assertIn("Long", entries[1]["input_dtypes"])

        recorder.unregister()
        comm.finalize()

    def test_fr_ranks_str_single_rank(self) -> None:
        """Test that ranks_str correctly formats single rank as iterable.

        Verifies that when a comm has a single rank, the ranks field in
        pg_config is formatted with brackets (e.g., "[0]") so it can be
        parsed as an iterable list by build_groups_memberships in the
        flight recorder trace analyzer.
        """
        backend = os.environ["TEST_BACKEND"]
        device = torch.device(os.environ.get("TEST_DEVICE", "cuda"))
        comm = torchcomms.new_comm(
            backend=backend,
            device=device,
            name="test_comm_single_rank",
            timeout=timedelta(seconds=300),
        )

        recorder = FlightRecorderHook(max_entries=100, isolated=True)
        recorder.register_with_comm(comm)

        # Dump and verify pg_config
        json_str = recorder.dump_json()
        data = json.loads(json_str)

        # Verify pg_config contains the comm with properly formatted ranks
        pg_config = data.get("pg_config", {})
        self.assertIn(
            "test_comm_single_rank",
            pg_config,
            "pg_config should contain the registered comm",
        )

        comm_config = pg_config["test_comm_single_rank"]
        ranks_str = comm_config.get("ranks", "")

        # Verify ranks string is wrapped in brackets for iterability
        self.assertTrue(
            ranks_str.startswith("[") and ranks_str.endswith("]"),
            f"ranks should be wrapped in brackets for iterability, got: {ranks_str}",
        )

        # Verify the ranks can be parsed as a list
        # The format should be "[0]" or "[0,1,2]" etc.
        self.assertRegex(
            ranks_str,
            r"^\[\d+(,\d+)*\]$",
            f"ranks should match pattern [N] or [N,N,...], got: {ranks_str}",
        )

        recorder.unregister()
        comm.finalize()

    def test_fr_enable_disable(self) -> None:
        """Test enabling and disabling the flight recorder.

        Verifies that operations are only recorded when the recorder
        is enabled.
        """
        comm = torchcomms.new_comm(
            backend=self.backend,
            device=self.device,
            name="test_comm_enable_disable",
            timeout=timedelta(seconds=300),
        )

        recorder = FlightRecorderHook(max_entries=100, isolated=True)

        # Initially not enabled (not registered)
        self.assertFalse(recorder.is_enabled())

        # Register with comm - should be enabled
        recorder.register_with_comm(comm)
        self.assertTrue(recorder.is_enabled())

        # Perform operation
        t = torch.rand(10, 10, device=self.device)
        comm.all_reduce(t, op=torchcomms.ReduceOp.SUM, async_op=False)

        # Should have recorded the operation
        json_str = recorder.dump_json()
        data = json.loads(json_str)
        self.assertGreater(len(data.get("entries", [])), 0)

        # Unregister - should be disabled
        recorder.unregister()
        self.assertFalse(recorder.is_enabled())

        comm.finalize()

    def test_fr_dump_to_configured_file(self) -> None:
        """Test that traces are dumped to configured files.

        Verifies that dump_file writes valid trace data to the file path
        configured via the TORCHCOMM_FR_DUMP_TEMP_FILE environment variable.
        """
        comm = torchcomms.new_comm(
            backend=self.backend,
            device=self.device,
            name="test_comm_dump_file",
            timeout=timedelta(seconds=300),
        )

        recorder = FlightRecorderHook(max_entries=100, isolated=True)
        recorder.register_with_comm(comm)

        # Perform some collective operations to generate entries
        t = torch.rand(10, 10, device=self.device)
        for _ in range(3):
            comm.all_reduce(t, op=torchcomms.ReduceOp.SUM, async_op=False)

        rank = comm.get_rank()

        # Create a temporary directory and configure the dump file path
        with tempfile.TemporaryDirectory() as tmpdir:
            # When dynamic filename is enabled, the env var value is used
            # directly as the full filename (no rank appended)
            dump_file_path = os.path.join(tmpdir, f"fr_trace_{rank}")

            # Save original env vars
            original_dump_file = os.environ.get("TORCHCOMM_FR_DUMP_TEMP_FILE")
            os.environ["TORCHCOMM_FR_DUMP_TEMP_FILE"] = dump_file_path

            try:
                # Dump the flight recorder trace to file
                recorder.dump_file(rank)

                # Verify the file was created
                self.assertTrue(
                    os.path.exists(dump_file_path),
                    f"Expected trace file {dump_file_path} was not created",
                )

                # Read and parse the file contents (pickled format)
                with open(dump_file_path, "rb") as f:
                    data = pickle.load(f)

                # Verify the trace structure
                self.assertIn("version", data)
                self.assertEqual(data["version"], "2.10")
                self.assertIn("entries", data)

                # Verify we have entries from our operations
                entries = data.get("entries", [])
                self.assertGreater(
                    len(entries), 0, "Dumped trace should contain entries"
                )

                # Validate each entry format (pickle format may use tuples instead of lists)
                for entry in entries:
                    # Basic required fields check
                    self.assertIn("record_id", entry)
                    self.assertIn("pg_id", entry)
                    self.assertIn("process_group", entry)
                    self.assertIn("profiling_name", entry)
                    self.assertIn("time_created_ns", entry)
                    self.assertIn("state", entry)

                    # Validate profiling_name is for all_reduce
                    self.assertEqual(
                        entry["profiling_name"], f"{self.backend}:all_reduce"
                    )

            finally:
                # Restore the original environment variables
                if original_dump_file is not None:
                    os.environ["TORCHCOMM_FR_DUMP_TEMP_FILE"] = original_dump_file
                elif "TORCHCOMM_FR_DUMP_TEMP_FILE" in os.environ:
                    del os.environ["TORCHCOMM_FR_DUMP_TEMP_FILE"]

        recorder.unregister()
        comm.finalize()

    def test_only_active_excludes_completed_collectives(self) -> None:
        """Test that completed collectives are excluded when include_completed=False.

        When dump_json is called with include_completed=False (onlyActive=True),
        completed/retired collectives should not appear in the output.
        When called with include_completed=True, they should appear.
        """
        backend = os.environ["TEST_BACKEND"]
        device = torch.device(os.environ.get("TEST_DEVICE", "cuda"))
        comm = torchcomms.new_comm(
            backend=backend,
            device=device,
            name="test_only_active",
            timeout=timedelta(seconds=300),
        )

        recorder = FlightRecorderHook(max_entries=100, isolated=True)
        recorder.register_with_comm(comm)

        # Run synchronous collectives - these will complete (retire) immediately
        t = torch.rand(10, 10, device=device)
        num_ops = 3
        for _ in range(num_ops):
            comm.all_reduce(t, op=torchcomms.ReduceOp.SUM, async_op=False)

        # Verify that include_completed=True returns all entries
        json_str_all = recorder.dump_json(True)
        data_all = json.loads(json_str_all)
        entries_all = data_all.get("entries", [])
        self.assertEqual(
            len(entries_all),
            num_ops,
            f"include_completed=True should return all {num_ops} entries",
        )
        for entry in entries_all:
            self._validate_entry_format(entry)

        # Verify that include_completed=False excludes completed collectives
        json_str_active = recorder.dump_json(False)
        data_active = json.loads(json_str_active)
        entries_active = data_active.get("entries", [])
        self.assertEqual(
            len(entries_active),
            0,
            "include_completed=False should return 0 entries when all collectives "
            "have completed",
        )

        recorder.unregister()
        comm.finalize()

    def test_only_active_via_dump_file_excludes_completed(self) -> None:
        """Test that dump_file with include_completed=False excludes retired collectives.

        This exercises the getCollectiveTrace code path which uses the retired_
        flag to filter completed collectives when onlyActive is true.
        """
        backend = os.environ["TEST_BACKEND"]
        device = torch.device(os.environ.get("TEST_DEVICE", "cuda"))
        comm = torchcomms.new_comm(
            backend=backend,
            device=device,
            name="test_only_active_dump_file",
            timeout=timedelta(seconds=300),
        )

        recorder = FlightRecorderHook(max_entries=100, isolated=True)
        recorder.register_with_comm(comm)

        # Run synchronous collectives - these will complete (retire) immediately
        t = torch.rand(10, 10, device=device)
        num_ops = 3
        for _ in range(num_ops):
            comm.all_reduce(t, op=torchcomms.ReduceOp.SUM, async_op=False)

        rank = comm.get_rank()

        with tempfile.TemporaryDirectory() as tmpdir:
            original_dump_file = os.environ.get("TORCHCOMM_FR_DUMP_TEMP_FILE")

            try:
                # Dump with include_completed=True - should have all entries
                all_file = os.path.join(tmpdir, f"fr_all_{rank}")
                os.environ["TORCHCOMM_FR_DUMP_TEMP_FILE"] = all_file
                recorder.dump_file(rank, True)

                self.assertTrue(os.path.exists(all_file))
                with open(all_file, "rb") as f:
                    data_all = pickle.load(f)

                entries_all = data_all.get("entries", [])
                self.assertEqual(
                    len(entries_all),
                    num_ops,
                    f"include_completed=True should return all {num_ops} entries",
                )

                # Dump with include_completed=False - completed/retired should be excluded
                active_file = os.path.join(tmpdir, f"fr_active_{rank}")
                os.environ["TORCHCOMM_FR_DUMP_TEMP_FILE"] = active_file
                recorder.dump_file(rank, False)

                self.assertTrue(os.path.exists(active_file))
                with open(active_file, "rb") as f:
                    data_active = pickle.load(f)

                entries_active = data_active.get("entries", [])
                self.assertEqual(
                    len(entries_active),
                    0,
                    "include_completed=False should return 0 entries when all "
                    "collectives have retired",
                )

            finally:
                if original_dump_file is not None:
                    os.environ["TORCHCOMM_FR_DUMP_TEMP_FILE"] = original_dump_file
                elif "TORCHCOMM_FR_DUMP_TEMP_FILE" in os.environ:
                    del os.environ["TORCHCOMM_FR_DUMP_TEMP_FILE"]

        recorder.unregister()
        comm.finalize()

    def test_only_active_with_mixed_operations(self) -> None:
        """Test onlyActive filtering with multiple collective types.

        Verifies that all types of completed collectives (all_reduce, broadcast)
        are excluded when include_completed=False.
        """
        backend = os.environ["TEST_BACKEND"]
        device = torch.device(os.environ.get("TEST_DEVICE", "cuda"))
        comm = torchcomms.new_comm(
            backend=backend,
            device=device,
            name="test_only_active_mixed",
            timeout=timedelta(seconds=300),
        )

        recorder = FlightRecorderHook(max_entries=100, isolated=True)
        recorder.register_with_comm(comm)

        t = torch.rand(10, 10, device=device)

        # Run different types of synchronous collectives
        comm.all_reduce(t, op=torchcomms.ReduceOp.SUM, async_op=False)
        comm.broadcast(t, root=0, async_op=False)
        comm.all_reduce(t, op=torchcomms.ReduceOp.SUM, async_op=False)

        # All operations completed - include_completed=True should show all
        json_str_all = recorder.dump_json(True)
        data_all = json.loads(json_str_all)
        entries_all = data_all.get("entries", [])
        self.assertGreaterEqual(
            len(entries_all),
            3,
            "include_completed=True should return all completed entries",
        )

        # include_completed=False should show none (all completed)
        json_str_active = recorder.dump_json(False)
        data_active = json.loads(json_str_active)
        entries_active = data_active.get("entries", [])
        self.assertEqual(
            len(entries_active),
            0,
            "include_completed=False should return 0 entries when all mixed "
            "collectives have completed",
        )

        recorder.unregister()
        comm.finalize()

    def test_split_hook_registers_new_comm(self) -> None:
        """Test that splitHook automatically registers the new communicator.

        Verifies that when a communicator is split, the FlightRecorderHook
        automatically registers the new communicator via splitHook, allowing
        operations on the split communicator to be tracked.
        """
        backend = os.environ["TEST_BACKEND"]
        device = torch.device(os.environ.get("TEST_DEVICE", "cuda"))
        comm = torchcomms.new_comm(
            backend=backend,
            device=device,
            name="test_split_hook",
            timeout=timedelta(seconds=300),
        )

        recorder = FlightRecorderHook(max_entries=100, isolated=True)
        recorder.register_with_comm(comm)

        # Get all ranks for the split
        all_ranks = list(range(comm.get_size()))

        # Split the communicator - this should trigger splitHook
        split_comm = comm.split(all_ranks, "split_child")

        # Perform an operation on the parent communicator
        t_parent = torch.rand(10, 10, device=device)
        comm.all_reduce(t_parent, op=torchcomms.ReduceOp.SUM, async_op=False)

        # Perform an operation on the split communicator
        # This should be tracked because splitHook registered it
        t_child = torch.rand(5, 5, device=device)
        split_comm.all_reduce(t_child, op=torchcomms.ReduceOp.SUM, async_op=False)

        # Dump and verify entries (include_completed=True to see completed ops)
        json_str = recorder.dump_json(True)
        data = json.loads(json_str)
        entries = data.get("entries", [])

        # We expect at least 3 entries:
        # 1. nccl:split (from the split operation)
        # 2. nccl:all_reduce (from parent communicator)
        # 3. nccl:all_reduce (from split communicator)
        self.assertGreaterEqual(
            len(entries), 3, "Should have entries for split and both all_reduce ops"
        )

        # Verify split operation is recorded
        split_entries = [e for e in entries if e["profiling_name"].endswith(":split")]
        self.assertGreater(
            len(split_entries), 0, "Should have recorded the split operation"
        )

        # Verify all_reduce operations are recorded (at least 2 - one from each comm)
        all_reduce_entries = [
            e for e in entries if e["profiling_name"].endswith(":all_reduce")
        ]
        self.assertGreaterEqual(
            len(all_reduce_entries),
            2,
            "Should have all_reduce entries from both parent and child communicators",
        )

        # Verify entries have different process_group names
        # Parent should be "test_split_hook" and child should include "split_child"
        pg_names = set()
        for entry in entries:
            if "process_group" in entry and len(entry["process_group"]) >= 1:
                pg_names.add(entry["process_group"][0])

        self.assertGreater(
            len(pg_names),
            1,
            "Should have entries from multiple process groups (parent and child)",
        )

        # Validate all entry formats
        for entry in entries:
            self._validate_entry_format(entry)

        # Clean up
        split_comm.finalize()
        recorder.unregister()
        comm.finalize()

    def test_split_hook_multiple_levels(self) -> None:
        """Test that splitHook works correctly with multi-level splits.

        Verifies that when a split communicator is further split, the
        FlightRecorderHook correctly registers all levels of split communicators.
        """
        backend = os.environ["TEST_BACKEND"]
        device = torch.device(os.environ.get("TEST_DEVICE", "cuda"))
        comm = torchcomms.new_comm(
            backend=backend,
            device=device,
            name="test_multi_split",
            timeout=timedelta(seconds=300),
        )

        recorder = FlightRecorderHook(max_entries=100, isolated=True)
        recorder.register_with_comm(comm)

        # Get all ranks
        all_ranks = list(range(comm.get_size()))

        # First level split
        level1_comm = comm.split(all_ranks, "level1")

        # Second level split from the first split
        level2_comm = level1_comm.split(all_ranks, "level2")

        # Perform operations at each level
        t = torch.rand(10, 10, device=device)
        comm.all_reduce(t, op=torchcomms.ReduceOp.SUM, async_op=False)
        level1_comm.all_reduce(t, op=torchcomms.ReduceOp.SUM, async_op=False)
        level2_comm.all_reduce(t, op=torchcomms.ReduceOp.SUM, async_op=False)

        # Dump and verify entries (include_completed=True to see completed ops)
        json_str = recorder.dump_json(True)
        data = json.loads(json_str)
        entries = data.get("entries", [])

        # We expect at least 5 entries:
        # 2 splits + 3 all_reduce operations
        self.assertGreaterEqual(
            len(entries), 5, "Should have entries for 2 splits and 3 all_reduce ops"
        )

        # Verify split operations are recorded
        split_entries = [e for e in entries if e["profiling_name"].endswith(":split")]
        self.assertEqual(
            len(split_entries), 2, "Should have recorded both split operations"
        )

        # Verify all_reduce operations are recorded from all three communicators
        all_reduce_entries = [
            e for e in entries if e["profiling_name"].endswith(":all_reduce")
        ]
        self.assertGreaterEqual(
            len(all_reduce_entries),
            3,
            "Should have all_reduce entries from all three communicators",
        )

        # Verify we have entries from multiple process groups
        pg_names = set()
        for entry in entries:
            if "process_group" in entry and len(entry["process_group"]) >= 1:
                pg_names.add(entry["process_group"][0])

        self.assertGreaterEqual(
            len(pg_names),
            3,
            "Should have entries from 3 different process groups (root, level1, level2)",
        )

        # Validate all entry formats
        for entry in entries:
            self._validate_entry_format(entry)

        # Clean up
        level2_comm.finalize()
        level1_comm.finalize()
        recorder.unregister()
        comm.finalize()


if __name__ == "__main__":
    unittest.main()
