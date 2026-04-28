# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# pyre-unsafe

import os
import tempfile
import unittest
from datetime import timedelta

import torch
import torchcomms
from torchcomms.hooks import clog


def _read_log_lines(path: str) -> list[str]:
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def _has_line_containing(lines: list[str], substr: str) -> bool:
    return any(substr in line for line in lines)


def _count_lines_containing(lines: list[str], substr: str) -> int:
    return sum(1 for line in lines if substr in line)


class TestClog(unittest.TestCase):
    """Test clog for collective operation logging."""

    backend = os.environ["TEST_BACKEND"]
    device = torch.device(os.environ.get("TEST_DEVICE", "cuda"))

    def test_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "clog_test.log")
            clog_handle = clog(output=log_path, events=["ALL"])

            # -- Version header --
            lines = _read_log_lines(log_path)
            self.assertGreater(len(lines), 0)
            self.assertTrue(lines[0].startswith("V|1|base_timestamp="))

            # -- new_comm logged on registration --
            comm = torchcomms.new_comm(
                backend=self.backend,
                device=self.device,
                name="test_clog_comm",
                timeout=timedelta(seconds=300),
            )
            clog_handle.register_with_comm(comm)

            lines = _read_log_lines(log_path)
            self.assertTrue(
                _has_line_containing(lines, "new_comm|comm=test_clog_comm"),
                "Expected new_comm log line",
            )

            # -- all_reduce signature with dedup --
            t = torch.ones(1024, device=self.device)
            comm.all_reduce(t, op=torchcomms.ReduceOp.SUM, async_op=False)
            comm.all_reduce(t, op=torchcomms.ReduceOp.SUM, async_op=False)

            lines = _read_log_lines(log_path)
            sig_count = _count_lines_containing(
                lines,
                "sig|all_reduce|in_count=1024|out_count=1024|dtype=f32|red_op=sum",
            )
            self.assertEqual(
                sig_count,
                1,
                "Signature should appear exactly once (dedup)",
            )

            # -- broadcast with root --
            t_bcast = torch.ones(256, device=self.device)
            comm.broadcast(t_bcast, root=0, async_op=False)

            lines = _read_log_lines(log_path)
            self.assertTrue(
                _has_line_containing(
                    lines,
                    "sig|broadcast|in_count=256|out_count=256|dtype=f32|root=0",
                ),
                "Expected broadcast signature",
            )

            # -- barrier --
            comm.barrier(async_op=False)

            lines = _read_log_lines(log_path)
            self.assertTrue(
                _has_line_containing(lines, "sig|barrier|async_op=f"),
                "Expected barrier signature",
            )

            # -- different dtypes --
            t_f16 = torch.ones(128, device=self.device, dtype=torch.float16)
            comm.all_reduce(t_f16, op=torchcomms.ReduceOp.SUM, async_op=False)

            lines = _read_log_lines(log_path)
            self.assertTrue(
                _has_line_containing(lines, "dtype=f16|red_op=sum"),
                "Expected f16 signature",
            )

            # -- enqueue events (Q) logged --
            q_count = _count_lines_containing(lines, "|Q|")
            self.assertGreater(
                q_count,
                0,
                "Expected at least one Q enqueue event",
            )

            # -- Cleanup --
            clog_handle.unregister()
            comm.finalize()

    def test_multiple_comms_shared_clog_handle(self) -> None:
        """One clog instance shared across multiple comms."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "clog_multi.log")
            clog_handle = clog(output=log_path, events=["ALL"])

            comm_a = torchcomms.new_comm(
                backend=self.backend,
                device=self.device,
                name="comm_a",
                timeout=timedelta(seconds=300),
            )
            comm_b = torchcomms.new_comm(
                backend=self.backend,
                device=self.device,
                name="comm_b",
                timeout=timedelta(seconds=300),
            )
            clog_handle.register_with_comm(comm_a)
            clog_handle.register_with_comm(comm_b)

            t = torch.ones(64, device=self.device)
            comm_a.all_reduce(t, op=torchcomms.ReduceOp.SUM, async_op=False)
            comm_b.all_reduce(t, op=torchcomms.ReduceOp.SUM, async_op=False)

            lines = _read_log_lines(log_path)
            self.assertTrue(
                _has_line_containing(lines, "comm=comm_a"),
                "Expected log entry for comm_a",
            )
            self.assertTrue(
                _has_line_containing(lines, "comm=comm_b"),
                "Expected log entry for comm_b",
            )

            clog_handle.unregister()
            comm_a.finalize()
            comm_b.finalize()

    def test_independent_clog_handles(self) -> None:
        """Two clog instances write to separate files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path_a = os.path.join(tmpdir, "clog_a.log")
            path_b = os.path.join(tmpdir, "clog_b.log")
            clog_a = clog(output=path_a, events=["ALL"])
            clog_b = clog(output=path_b, events=["ALL"])

            comm_a = torchcomms.new_comm(
                backend=self.backend,
                device=self.device,
                name="iso_a",
                timeout=timedelta(seconds=300),
            )
            comm_b = torchcomms.new_comm(
                backend=self.backend,
                device=self.device,
                name="iso_b",
                timeout=timedelta(seconds=300),
            )
            clog_a.register_with_comm(comm_a)
            clog_b.register_with_comm(comm_b)

            t = torch.ones(32, device=self.device)
            comm_a.all_reduce(t, op=torchcomms.ReduceOp.SUM, async_op=False)
            comm_b.all_reduce(t, op=torchcomms.ReduceOp.SUM, async_op=False)

            lines_a = _read_log_lines(path_a)
            lines_b = _read_log_lines(path_b)

            self.assertTrue(
                _has_line_containing(lines_a, "comm=iso_a"),
            )
            self.assertFalse(
                _has_line_containing(lines_a, "comm=iso_b"),
            )
            self.assertTrue(
                _has_line_containing(lines_b, "comm=iso_b"),
            )
            self.assertFalse(
                _has_line_containing(lines_b, "comm=iso_a"),
            )

            clog_a.unregister()
            clog_b.unregister()
            comm_a.finalize()
            comm_b.finalize()


@unittest.skipIf(os.getenv("TEST_BACKEND") != "ncclx", "CUDA graph test requires ncclx")
class TestClogCudaGraph(unittest.TestCase):
    """Test clog QC logging during CUDA graph capture."""

    backend = os.environ["TEST_BACKEND"]
    device = torch.device(os.environ.get("TEST_DEVICE", "cuda"))

    def test_cuda_graph_capture_logs_qc(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "clog_graph.log")
            clog_handle = clog(output=log_path, events=["ALL"])

            comm = torchcomms.new_comm(
                backend=self.backend,
                device=self.device,
                name="graph_comm",
                timeout=timedelta(seconds=300),
            )
            clog_handle.register_with_comm(comm)

            t = torch.ones(64, device=self.device)

            # Eager collective (should produce C<id>|Q|, not G<id>|C<id>|Q|)
            comm.all_reduce(t, op=torchcomms.ReduceOp.SUM, async_op=False)

            lines = _read_log_lines(log_path)
            q_before = _count_lines_containing(lines, "|Q|")
            graph_q_before = len([x for x in lines if x.startswith("G") and "|Q|" in x])
            self.assertGreater(q_before, 0, "Expected Q event for eager op")
            self.assertEqual(
                graph_q_before, 0, "No graph Q expected before graph capture"
            )

            # Capture a collective into a CUDA graph
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                comm.all_reduce(t, op=torchcomms.ReduceOp.SUM, async_op=False)

            lines = _read_log_lines(log_path)
            graph_q_after_capture = len(
                [x for x in lines if x.startswith("G") and "|Q|" in x]
            )
            self.assertGreater(
                graph_q_after_capture,
                0,
                "Expected graph Q event during graph capture",
            )

            num_replays = 3
            for _ in range(num_replays):
                g.replay()
            torch.cuda.synchronize()
            g.reset()
            comm.finalize()
            clog_handle.unregister()

            lines = _read_log_lines(log_path)
            graph_q_after_replay = len(
                [x for x in lines if x.startswith("G") and "|Q|" in x]
            )
            self.assertEqual(
                graph_q_after_replay,
                graph_q_after_capture,
                "Replay should not produce new graph Q events",
            )
            total_s = _count_lines_containing(lines, "|S|")
            total_e = _count_lines_containing(lines, "|E|")
            # Replay lines have format G<gid>|R<rid>|C<cid>|S|+<ts>
            replay_s = len([x for x in lines if "|R" in x and "|S|" in x])
            replay_e = len([x for x in lines if "|R" in x and "|E|" in x])
            self.assertEqual(
                total_s, 1 + num_replays, "Expected S from capture + each replay"
            )
            self.assertEqual(
                total_e, 1 + num_replays, "Expected E from capture + each replay"
            )
            self.assertEqual(replay_s, num_replays, "Expected one replay S per replay")
            self.assertEqual(replay_e, num_replays, "Expected one replay E per replay")


if __name__ == "__main__":
    unittest.main()
