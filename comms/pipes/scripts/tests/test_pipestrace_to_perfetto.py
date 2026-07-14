# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""Unit tests for pipestrace_to_perfetto.

Run with:
``buck2 test @fbcode//mode/opt -c hpc_comms.use_ncclx=stable //comms/pipes/scripts/tests:test_pipestrace_to_perfetto``
or directly: ``fbpython -m unittest`` from this directory.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import tempfile
import unittest

# Keep direct `fbpython -m unittest` execution working outside Buck.
_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SCRIPT_DIR))
import pipestrace_to_perfetto as ptp  # noqa: E402


def _event_line(
    rank: int, name: str, step: int, r: int, detail: int, slot: int, ns: int
) -> str:
    return (
        f"[1,{rank}]<stderr>:I0531 12:00:00.000000 1234 PipesTrace.cc:168] "
        f"Prims trace event={name} step={step} rank={r} detail={detail} "
        f"slot={slot} wall_time_ns={ns}\n"
    )


def _loss_line(rank: int, lost: int) -> str:
    return (
        f"[1,{rank}]<stderr>:I0531 12:00:00.000100 1234 PipesTrace.cc:179] "
        f"Prims trace lost {lost} entries\n"
    )


class EventRegexTest(unittest.TestCase):
    def test_matches_well_formed_event(self) -> None:
        line = _event_line(7, ptp.AG_IB_BEGIN, 3, 3, 0, 0, 1780268274964010916)
        match = ptp.EVENT_RE.fullmatch(line)
        self.assertIsNotNone(match)
        assert match is not None
        self.assertEqual("7", match["mpi_rank"])
        self.assertEqual(ptp.AG_IB_BEGIN, match["name"])
        self.assertEqual("1780268274964010916", match["ns"])

    def test_matches_loss_report(self) -> None:
        match = ptp.LOSS_RE.fullmatch(_loss_line(1, 68))
        self.assertIsNotNone(match)
        assert match is not None
        self.assertEqual("68", match["lost"])

    def test_negative_rank_field(self) -> None:
        line = _event_line(0, ptp.AG_IB_BEGIN, 0, -1, 0, 0, 100)
        event = ptp._parse_event_line(line)
        self.assertIsNotNone(event)
        assert event is not None
        self.assertEqual(0, event.step)
        self.assertEqual(-1, event.rank)


class ParseEventsTest(unittest.TestCase):
    def test_concatenated_tagged_events_are_dropped(self) -> None:
        log = _event_line(0, ptp.AG_IB_BEGIN, 0, 0, 0, 0, 1_000_000_000).rstrip(
            "\n"
        ) + _event_line(0, ptp.AG_IB_READY, 0, 0, 0, 1, 1_000_000_100)
        with tempfile.TemporaryDirectory() as d:
            log_path = os.path.join(d, "x.log")
            with open(log_path, "w") as f:
                f.write(log)
            stats = ptp.ParseStats()
            events, _ = ptp.parse_events([log_path], stats)
        self.assertEqual([], events)
        self.assertEqual(0, stats.events)
        self.assertEqual(1, stats.dropped_lines)

    def test_loss_reports_are_counted(self) -> None:
        log = _event_line(0, ptp.AG_IB_BEGIN, 0, 0, 0, 0, 1_000_000_000)
        log += _loss_line(0, lost=3)
        with tempfile.TemporaryDirectory() as d:
            log_path = os.path.join(d, "x.log")
            with open(log_path, "w") as f:
                f.write(log)
            stats = ptp.ParseStats()
            events, _ = ptp.parse_events([log_path], stats)
        self.assertEqual(1, len(events))
        self.assertEqual(1, stats.loss_reports)
        self.assertEqual(3, stats.lost_entries)


class FilterEventsTest(unittest.TestCase):
    def test_rank_and_type_and_time_filters(self) -> None:
        events = [
            ptp.Event(0, ptp.AG_IB_BEGIN, 0, 0, 0, 0, 100),
            ptp.Event(1, ptp.AG_IB_BEGIN, 0, 1, 0, 0, 200),
            ptp.Event(0, ptp.AG_NVL_WAIT_BEGIN, 0, 0, 0, 1, 300),
            ptp.Event(0, ptp.AG_IB_READY, 0, 0, 0, 2, 400),
        ]
        out = list(
            ptp.filter_events(
                events,
                rank_filter={0},
                type_filter={"ib"},
                time_start=150,
                time_end=500,
            )
        )
        self.assertCountEqual([events[3]], out)


class PairIbTest(unittest.TestCase):
    def test_self_ready_closes_slice_forwarded_are_counted(self) -> None:
        evs = [
            ptp.Event(6, ptp.AG_IB_BEGIN, 0, 3, 0, 0, 100),
            ptp.Event(6, ptp.AG_IB_READY, 0, 3, 0, 1, 200),  # self
            ptp.Event(6, ptp.AG_IB_READY, 0, 2, 0, 2, 250),  # forwarded
            ptp.Event(6, ptp.AG_IB_READY, 0, 1, 0, 3, 300),  # forwarded
        ]
        slices, unpaired_instant_count, unmatched = ptp.pair_events({6: evs})
        self.assertEqual(1, len(slices))
        self.assertEqual("IB chunk 0", slices[0].name)
        self.assertEqual(100, slices[0].begin_ns)
        self.assertEqual(200, slices[0].end_ns)
        self.assertEqual(3, slices[0].args["ib_rank"])
        self.assertEqual(2, unpaired_instant_count)
        self.assertEqual([], unmatched)

    def test_missing_begin_is_counted_not_sliced(self) -> None:
        evs = [ptp.Event(0, ptp.AG_IB_READY, 0, 0, 0, 1, 200)]
        slices, unpaired_instant_count, unmatched = ptp.pair_events({0: evs})
        self.assertEqual([], slices)
        self.assertEqual(1, unpaired_instant_count)
        self.assertEqual([], unmatched)

    def test_unmatched_begin_reported(self) -> None:
        evs = [ptp.Event(0, ptp.AG_IB_BEGIN, 0, 0, 0, 0, 100)]
        _, _, unmatched = ptp.pair_events({0: evs})
        self.assertEqual(1, len(unmatched))
        self.assertIn("ib_chunk_begin", unmatched[0][1])

    def test_ib_send_recv_forward_slices(self) -> None:
        evs = [
            ptp.Event(0, ptp.IB_SEND_BEGIN, 0, 1, 0, 0, 100),
            ptp.Event(0, ptp.IB_SEND_END, 16, 1, 0, 1, 120),
            ptp.Event(0, ptp.IB_RECV_BEGIN, 32, 2, 0, 2, 130),
            ptp.Event(0, ptp.IB_RECV_END, 48, 2, 0, 3, 160),
            ptp.Event(0, ptp.IB_FORWARD_BEGIN, 64, 3, 0, 4, 170),
            ptp.Event(0, ptp.IB_FORWARD_END, 96, 3, 0, 5, 210),
        ]
        slices, unpaired_instant_count, unmatched = ptp.pair_events({0: evs})
        self.assertEqual(0, unpaired_instant_count)
        self.assertEqual([], unmatched)
        self.assertCountEqual(
            [ptp.IB_SEND, ptp.IB_RECV, ptp.IB_FORWARD],
            [s.cat for s in slices],
        )
        send = next(s for s in slices if s.cat == ptp.IB_SEND)
        self.assertEqual((100, 120), (send.begin_ns, send.end_ns))
        self.assertEqual(16, send.args["bytes"])
        self.assertEqual(0, send.args["begin_step"])
        self.assertEqual(16, send.args["end_step"])
        self.assertNotIn("byte_begin", send.args)


class PairNvlTest(unittest.TestCase):
    def test_wait_and_bcast_slices(self) -> None:
        evs = [
            ptp.Event(0, ptp.AG_NVL_WAIT_BEGIN, 0, 3, 0, 0, 100),
            ptp.Event(0, ptp.AG_NVL_CHUNK_READY, 0, 3, 0, 1, 150),
            ptp.Event(0, ptp.AG_NVL_TASK_DONE, 0, 3, 0, 2, 200),
        ]
        slices, unpaired_instant_count, unmatched = ptp.pair_events({0: evs})
        wait = next(s for s in slices if s.cat == "nvl_wait")
        bcast = next(s for s in slices if s.cat == "nvl_bcast")
        self.assertEqual((100, 150), (wait.begin_ns, wait.end_ns))
        self.assertEqual((150, 200), (bcast.begin_ns, bcast.end_ns))
        self.assertEqual(3, wait.args["ib_src"])
        self.assertEqual(0, unpaired_instant_count)
        self.assertEqual([], unmatched)

    def test_orphan_task_done_reported(self) -> None:
        evs = [ptp.Event(0, ptp.AG_NVL_TASK_DONE, 0, 3, 0, 0, 100)]
        slices, _, unmatched = ptp.pair_events({0: evs})
        self.assertEqual([], slices)
        self.assertEqual(1, len(unmatched))
        self.assertIn("task_done", unmatched[0][1])


class BuildTraceTest(unittest.TestCase):
    def test_ts_dur_offset_and_metadata(self) -> None:
        slices = [
            ptp.Slice_(
                pid=6,
                tid=100,
                cat="ib",
                name="IB chunk 0",
                begin_ns=1_000_000_000,
                end_ns=1_000_017_296,
                args={"chunk": 0, "ib_rank": 3, "block": 0},
            )
        ]
        topo = ptp.Topology(nvl_size=2, ib_size=4, auto_detected_nvl=False)
        trace = ptp.build_trace(slices, topo, pids=[6])

        self.assertEqual("ns", trace["displayTimeUnit"])
        meta = trace["meta"]
        assert isinstance(meta, dict)
        self.assertEqual(1_000_000_000, meta["time_offset_ns"])
        self.assertEqual(2, meta["nvl_size"])

        events = trace["traceEvents"]
        assert isinstance(events, list)
        # process_name reflects ib/nvl derivation: rank 6 / nvl_size 2 → ib=3, nvl=0
        proc_name = next(
            e for e in events if e.get("ph") == "M" and e.get("name") == "process_name"
        )
        self.assertEqual("rank 6 (ib=3 nvl=0)", proc_name["args"]["name"])

        x = next(e for e in events if e.get("ph") == "X")
        self.assertEqual(0.0, x["ts"])
        self.assertAlmostEqual(17.296, x["dur"])
        self.assertFalse(any(e.get("ph") == "i" for e in events))


class EndToEndCliTest(unittest.TestCase):
    def test_writes_well_formed_json(self) -> None:
        log = (
            "INFO:root:Running mpirun -np 1 -host h:1 --tag-output app\n"
            + _event_line(0, ptp.AG_IB_BEGIN, 0, 0, 0, 0, 1_000_000_000)
            + _event_line(0, ptp.AG_IB_READY, 0, 0, 0, 1, 1_000_000_100)
            + _event_line(0, ptp.AG_NVL_WAIT_BEGIN, 0, 0, 0, 2, 1_000_000_200)
            + _event_line(0, ptp.AG_NVL_CHUNK_READY, 0, 0, 0, 3, 1_000_000_300)
            + _event_line(0, ptp.AG_NVL_TASK_DONE, 0, 0, 0, 4, 1_000_000_400)
        )
        with tempfile.TemporaryDirectory() as d:
            log_path = os.path.join(d, "x.log")
            out_path = os.path.join(d, "x.perfetto.json")
            with open(log_path, "w") as f:
                f.write(log)
            rc = ptp.main([log_path, "-o", out_path])
            self.assertEqual(0, rc)
            with open(out_path) as f:
                trace = json.load(f)
        events = trace["traceEvents"]
        x_phases = [e for e in events if e.get("ph") == "X"]
        # Three slices: IB chunk + NVL wait + NVL bcast.
        self.assertEqual(3, len(x_phases))

    def test_logs_unpaired_instant_count(self) -> None:
        log = (
            _event_line(0, ptp.AG_IB_BEGIN, 0, 0, 0, 0, 1_000_000_000)
            + _event_line(0, ptp.AG_IB_READY, 0, 0, 0, 1, 1_000_000_100)
            + _event_line(0, ptp.AG_IB_READY, 0, 1, 0, 2, 1_000_000_200)
        )
        with tempfile.TemporaryDirectory() as d:
            log_path = os.path.join(d, "x.log")
            out_path = os.path.join(d, "x.perfetto.json")
            with open(log_path, "w") as f:
                f.write(log)
            with self.assertLogs(ptp.logger, level="WARNING") as logs:
                rc = ptp.main([log_path, "-o", out_path, "--nvl-size", "1"])
            self.assertEqual(0, rc)
        self.assertTrue(
            any("1 unpaired instants suppressed" in message for message in logs.output)
        )

    def test_strict_returns_2_on_unmatched(self) -> None:
        log = _event_line(0, ptp.AG_IB_BEGIN, 0, 0, 0, 0, 100)  # no ready
        with tempfile.TemporaryDirectory() as d:
            log_path = os.path.join(d, "x.log")
            out_path = os.path.join(d, "x.perfetto.json")
            with open(log_path, "w") as f:
                f.write(log)
            rc = ptp.main([log_path, "-o", out_path, "--nvl-size", "1", "--strict"])
            self.assertEqual(2, rc)

    def test_strict_returns_2_on_dropped_trace_line(self) -> None:
        log = (
            _event_line(0, ptp.AG_IB_BEGIN, 0, 0, 0, 0, 1_000_000_000)
            + _event_line(0, ptp.AG_IB_READY, 0, 0, 0, 1, 1_000_000_100)
            + _event_line(0, ptp.AG_IB_READY, 1, 0, 0, 2, 1_000_000_200).rstrip("\n")
            + _event_line(0, ptp.AG_IB_READY, 2, 0, 0, 3, 1_000_000_300)
        )
        with tempfile.TemporaryDirectory() as d:
            log_path = os.path.join(d, "x.log")
            out_path = os.path.join(d, "x.perfetto.json")
            with open(log_path, "w") as f:
                f.write(log)
            rc = ptp.main([log_path, "-o", out_path, "--nvl-size", "1", "--strict"])
            self.assertEqual(2, rc)

    def test_no_events_returns_1(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            log_path = os.path.join(d, "x.log")
            out_path = os.path.join(d, "x.perfetto.json")
            with open(log_path, "w") as f:
                f.write("no pipes trace here\n")
            rc = ptp.main([log_path, "-o", out_path])
            self.assertEqual(1, rc)

    def test_warns_when_loss_report_records_lost_entries(self) -> None:
        log = (
            _event_line(0, ptp.AG_IB_BEGIN, 0, 0, 0, 0, 1_000_000_000)
            + _event_line(0, ptp.AG_IB_READY, 0, 0, 0, 1, 1_000_000_100)
            + _loss_line(0, lost=5)
        )
        with tempfile.TemporaryDirectory() as d:
            log_path = os.path.join(d, "x.log")
            out_path = os.path.join(d, "x.perfetto.json")
            with open(log_path, "w") as f:
                f.write(log)
            with self.assertLogs(ptp.logger, level="WARNING") as logs:
                rc = ptp.main([log_path, "-o", out_path, "--nvl-size", "1"])
            self.assertEqual(0, rc)
        self.assertTrue(
            any("reported 5 lost trace entries" in message for message in logs.output)
        )

    def test_strict_returns_2_on_lost_entries(self) -> None:
        log = (
            _event_line(0, ptp.AG_IB_BEGIN, 0, 0, 0, 0, 1_000_000_000)
            + _event_line(0, ptp.AG_IB_READY, 0, 0, 0, 1, 1_000_000_100)
            + _loss_line(0, lost=1)
        )
        with tempfile.TemporaryDirectory() as d:
            log_path = os.path.join(d, "x.log")
            out_path = os.path.join(d, "x.perfetto.json")
            with open(log_path, "w") as f:
                f.write(log)
            rc = ptp.main([log_path, "-o", out_path, "--nvl-size", "1", "--strict"])
            self.assertEqual(2, rc)
