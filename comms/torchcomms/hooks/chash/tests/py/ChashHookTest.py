# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-strict

import os
import re
import tempfile
import unittest
from datetime import timedelta

import torch
import torchcomms
from torchcomms.hooks.chash import chash


def _read_log_lines(path: str) -> list[str]:
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def _has_line_matching(lines: list[str], pattern: str) -> bool:
    return any(re.search(pattern, line) for line in lines)


class TestChashHook(unittest.TestCase):
    """Tests for the chash communication hash hook."""

    def setUp(self) -> None:
        self.backend: str = os.environ["TEST_BACKEND"]
        self.device: torch.device = torch.device(os.environ.get("TEST_DEVICE", "cuda"))

    def _create_comm(self, name: str) -> torchcomms.TorchComm:
        return torchcomms.new_comm(
            backend=self.backend,
            device=self.device,
            name=name,
            timeout=timedelta(seconds=300),
        )

    def test_basic_hash_logging(self) -> None:
        """Register chash, run a sync all_reduce, verify log output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "chash.log")
            hook = chash(output=log_path, ring_size=64)
            comm = self._create_comm("test_basic")
            hook.register_with_comm(comm)

            t = torch.ones(10, device=self.device)
            comm.all_reduce(t, op=torchcomms.ReduceOp.SUM, async_op=False)

            comm.finalize()

            lines = _read_log_lines(log_path)
            self.assertTrue(
                _has_line_matching(lines, r"new_comm\|comm=test_basic"),
                "Expected new_comm log line",
            )
            self.assertTrue(
                _has_line_matching(lines, r"C\d+\|sig\|all_reduce"),
                "Expected signature log line",
            )
            self.assertTrue(
                _has_line_matching(lines, r"C\d+\|S\|hash=0x"),
                "Expected start-hash log line",
            )
            self.assertTrue(
                _has_line_matching(lines, r"C\d+\|E\|hash=0x"),
                "Expected end-hash log line",
            )

    def test_hash_determinism(self) -> None:
        """Same input should produce same pre-hash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "chash_determ.log")
            hook = chash(output=log_path, ring_size=64)
            comm = self._create_comm("test_determ")
            hook.register_with_comm(comm)

            t = torch.ones(100, device=self.device)
            comm.all_reduce(t.clone(), op=torchcomms.ReduceOp.SUM, async_op=False)

            t2 = torch.ones(100, device=self.device)
            comm.all_reduce(t2.clone(), op=torchcomms.ReduceOp.SUM, async_op=False)

            comm.finalize()

            lines = _read_log_lines(log_path)
            pre_hashes = [line for line in lines if re.search(r"C\d+\|S\|hash=", line)]
            self.assertEqual(len(pre_hashes), 2)
            hash1 = pre_hashes[0].split("hash=")[1]
            hash2 = pre_hashes[1].split("hash=")[1]
            self.assertEqual(hash1, hash2, "Same input should produce same hash")

    def test_hash_changes_with_data(self) -> None:
        """Different input data should produce different hashes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "chash_diff.log")
            hook = chash(output=log_path, ring_size=64)
            comm = self._create_comm("test_diff")
            hook.register_with_comm(comm)

            t1 = torch.ones(100, device=self.device)
            comm.all_reduce(t1, op=torchcomms.ReduceOp.SUM, async_op=False)

            t2 = torch.arange(100, dtype=torch.float32, device=self.device)
            comm.all_reduce(t2, op=torchcomms.ReduceOp.SUM, async_op=False)

            comm.finalize()

            lines = _read_log_lines(log_path)
            pre_hashes = [line for line in lines if re.search(r"C\d+\|S\|hash=", line)]
            self.assertEqual(len(pre_hashes), 2)
            hash1 = pre_hashes[0].split("hash=")[1]
            hash2 = pre_hashes[1].split("hash=")[1]
            self.assertNotEqual(
                hash1, hash2, "Different data should produce different hashes"
            )

    def test_async_op_hash(self) -> None:
        """Async all_reduce should produce both pre and post hashes after wait."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "chash_async.log")
            hook = chash(output=log_path, ring_size=64)
            comm = self._create_comm("test_async")
            hook.register_with_comm(comm)

            t = torch.ones(10, device=self.device)
            work = comm.all_reduce(t, op=torchcomms.ReduceOp.SUM, async_op=True)
            work.wait()

            comm.finalize()

            lines = _read_log_lines(log_path)
            self.assertTrue(
                _has_line_matching(lines, r"C\d+\|S\|hash=0x"),
                "Expected start-hash for async op",
            )
            self.assertTrue(
                _has_line_matching(lines, r"C\d+\|E\|hash=0x"),
                "Expected end-hash after wait",
            )

    def test_multiple_collectives(self) -> None:
        """Multiple collectives get unique labels with pre+post pairs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "chash_multi.log")
            hook = chash(output=log_path, ring_size=64)
            comm = self._create_comm("test_multi")
            hook.register_with_comm(comm)

            t = torch.ones(10, device=self.device)
            comm.all_reduce(t, op=torchcomms.ReduceOp.SUM, async_op=False)
            comm.all_reduce(t, op=torchcomms.ReduceOp.SUM, async_op=False)
            comm.all_reduce(t, op=torchcomms.ReduceOp.SUM, async_op=False)

            comm.finalize()

            lines = _read_log_lines(log_path)
            sig_lines = [line for line in lines if "|sig|" in line]
            pre_lines = [line for line in lines if "|S|" in line]
            post_lines = [line for line in lines if "|E|" in line]

            self.assertEqual(len(sig_lines), 3)
            self.assertEqual(len(pre_lines), 3)
            self.assertEqual(len(post_lines), 3)

    def test_split_logging(self) -> None:
        """Split creates a log entry and child comm gets hashes too."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "chash_split.log")
            hook = chash(output=log_path, ring_size=64)
            comm = self._create_comm("test_split")
            hook.register_with_comm(comm)

            ranks = list(range(comm.get_size()))
            split_comm = comm.split(ranks, "child")

            t = torch.ones(10, device=self.device)
            split_comm.all_reduce(t, op=torchcomms.ReduceOp.SUM, async_op=False)

            split_comm.finalize()
            comm.finalize()

            lines = _read_log_lines(log_path)
            self.assertTrue(
                _has_line_matching(lines, r"split\|parent=test_split\|child=child"),
                "Expected split log line",
            )
            self.assertTrue(
                _has_line_matching(lines, r"C\d+\|sig\|all_reduce.*comm=child"),
                "Expected signature with child comm name",
            )

    def test_multiple_comms(self) -> None:
        """One chash instance shared across multiple comms."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "chash_multi_comm.log")
            hook = chash(output=log_path, ring_size=64)

            comm_a = self._create_comm("comm_a")
            comm_b = self._create_comm("comm_b")
            hook.register_with_comm(comm_a)
            hook.register_with_comm(comm_b)

            t = torch.ones(10, device=self.device)
            comm_a.all_reduce(t, op=torchcomms.ReduceOp.SUM, async_op=False)
            comm_b.all_reduce(t, op=torchcomms.ReduceOp.SUM, async_op=False)

            comm_a.finalize()
            comm_b.finalize()

            lines = _read_log_lines(log_path)
            self.assertTrue(
                _has_line_matching(lines, r"comm=comm_a"),
                "Expected log entry for comm_a",
            )
            self.assertTrue(
                _has_line_matching(lines, r"comm=comm_b"),
                "Expected log entry for comm_b",
            )


if __name__ == "__main__":
    unittest.main()
