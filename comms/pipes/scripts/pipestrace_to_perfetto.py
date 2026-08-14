#!/usr/bin/env fbpython
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""Convert supported `PipesTrace` log lines into Perfetto JSON.

Supports hierarchical allgather overlap traces and fine-grained AllReduce
transport traces, including the staged/registered path actually selected.

Run from `fbcode/comms` with either:
  buck2 run @fbcode//mode/opt \
    //comms/pipes/scripts:pipestrace_to_perfetto_cli -- LOG -o trace.json
  fbpython pipes/scripts/pipestrace_to_perfetto.py LOG -o trace.perfetto.json

Consumes one or more launcher logs, usually `mpirun --tag-output` captures,
and emits a single JSON timeline that can be opened at https://ui.perfetto.dev/.
"""

from __future__ import annotations

import argparse
import dataclasses
import gzip
import json
import logging
import os
import re
import sys
from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from typing import TextIO

logger: logging.Logger = logging.getLogger(__name__)

# Event name strings currently supported from pipesTraceEventTypeName() in
# PipesTrace.cc.
AG_IB_BEGIN = "hier_ag_ib_chunk_begin"
AG_IB_READY = "hier_ag_ib_chunk_ready"
AG_NVL_WAIT_BEGIN = "hier_ag_nvl_wait_begin"
AG_NVL_CHUNK_READY = "hier_ag_nvl_chunk_ready"
AG_NVL_TASK_DONE = "hier_ag_nvl_task_done"
IB_SEND_BEGIN = "ib_send_begin"
IB_SEND_END = "ib_send_end"
IB_RECV_BEGIN = "ib_recv_begin"
IB_RECV_END = "ib_recv_end"
IB_FORWARD_BEGIN = "ib_forward_begin"
IB_FORWARD_END = "ib_forward_end"
AR_PHASE1_BEGIN = "allreduce_phase1_begin"
AR_PHASE1_END = "allreduce_phase1_end"
AR_PHASE2_BEGIN = "allreduce_phase2_begin"
AR_PHASE2_END = "allreduce_phase2_end"
AR_PHASE3_BEGIN = "allreduce_phase3_begin"
AR_PHASE3_END = "allreduce_phase3_end"
AR_RING_RS_BEGIN = "allreduce_ring_rs_begin"
AR_RING_RS_END = "allreduce_ring_rs_end"
AR_RING_AG_BEGIN = "allreduce_ring_ag_begin"
AR_RING_AG_END = "allreduce_ring_ag_end"
AR_SEND_SYNC_BEGIN = "allreduce_send_sync_begin"
AR_SEND_SYNC_END = "allreduce_send_sync_end"
AR_WQE_SUBMIT_BEGIN = "allreduce_wqe_submit_begin"
AR_WQE_SUBMIT_END = "allreduce_wqe_submit_end"
AR_DATA_READY_WAIT_BEGIN = "allreduce_data_ready_wait_begin"
AR_DATA_READY_WAIT_END = "allreduce_data_ready_wait_end"
AR_REDUCE_COPY_BEGIN = "allreduce_reduce_copy_begin"
AR_REDUCE_COPY_END = "allreduce_reduce_copy_end"
AR_DRAIN_BEGIN = "allreduce_drain_begin"
AR_DRAIN_END = "allreduce_drain_end"
AR_BOOKKEEPING_BEGIN = "allreduce_bookkeeping_begin"
AR_BOOKKEEPING_END = "allreduce_bookkeeping_end"
AR_LOCAL_COMPLETION_WAIT_BEGIN = "allreduce_local_completion_wait_begin"
AR_LOCAL_COMPLETION_WAIT_END = "allreduce_local_completion_wait_end"
AR_REMOTE_SLOT_FREE_WAIT_BEGIN = "allreduce_remote_slot_free_wait_begin"
AR_REMOTE_SLOT_FREE_WAIT_END = "allreduce_remote_slot_free_wait_end"
AR_STAGE_COPY_BEGIN = "allreduce_stage_copy_begin"
AR_STAGE_COPY_END = "allreduce_stage_copy_end"
AR_PATH_STAGED = "allreduce_path_staged"
AR_PATH_REGISTERED_PROGRESS = "allreduce_path_registered_progress"
AR_TREE_SCHEDULER_IDLE_BEGIN = "allreduce_tree_scheduler_idle_begin"
AR_TREE_SCHEDULER_IDLE_END = "allreduce_tree_scheduler_idle_end"

IB_SEND = "ib_send"
IB_RECV = "ib_recv"
IB_FORWARD = "ib_forward"

IB_NAMES = frozenset(
    {
        AG_IB_BEGIN,
        AG_IB_READY,
        IB_SEND_BEGIN,
        IB_SEND_END,
        IB_RECV_BEGIN,
        IB_RECV_END,
        IB_FORWARD_BEGIN,
        IB_FORWARD_END,
    }
)
NVL_NAMES = frozenset({AG_NVL_WAIT_BEGIN, AG_NVL_CHUNK_READY, AG_NVL_TASK_DONE})
AR_PAIRS = {
    AR_PHASE1_END: (AR_PHASE1_BEGIN, "allreduce_phase1", "input/reduce-scatter"),
    AR_PHASE2_END: (AR_PHASE2_BEGIN, "allreduce_phase2", "inter-node"),
    AR_PHASE3_END: (AR_PHASE3_BEGIN, "allreduce_phase3", "output/all-gather"),
}
AR_BEGIN_NAMES = frozenset(v[0] for v in AR_PAIRS.values())
AR_NAMES = frozenset(AR_PAIRS) | AR_BEGIN_NAMES
FINE_AR_PAIRS = {
    AR_RING_RS_END: (AR_RING_RS_BEGIN, "ring reduce-scatter"),
    AR_RING_AG_END: (AR_RING_AG_BEGIN, "ring all-gather"),
    AR_SEND_SYNC_END: (AR_SEND_SYNC_BEGIN, "send sync"),
    AR_WQE_SUBMIT_END: (AR_WQE_SUBMIT_BEGIN, "WQE submit"),
    AR_DATA_READY_WAIT_END: (AR_DATA_READY_WAIT_BEGIN, "DATA_READY wait"),
    AR_REDUCE_COPY_END: (AR_REDUCE_COPY_BEGIN, "reduce/copy"),
    AR_DRAIN_END: (AR_DRAIN_BEGIN, "registered drain"),
    AR_BOOKKEEPING_END: (AR_BOOKKEEPING_BEGIN, "bookkeeping"),
    AR_LOCAL_COMPLETION_WAIT_END: (
        AR_LOCAL_COMPLETION_WAIT_BEGIN,
        "local completion wait",
    ),
    AR_REMOTE_SLOT_FREE_WAIT_END: (
        AR_REMOTE_SLOT_FREE_WAIT_BEGIN,
        "remote SLOT_FREE wait",
    ),
    AR_STAGE_COPY_END: (AR_STAGE_COPY_BEGIN, "staging copy"),
    AR_TREE_SCHEDULER_IDLE_END: (
        AR_TREE_SCHEDULER_IDLE_BEGIN,
        "tree scheduler idle",
    ),
}
FINE_AR_PATHS = {
    AR_PATH_STAGED: "path: staged",
    AR_PATH_REGISTERED_PROGRESS: "path: registered progress",
}
FINE_AR_NAMES = (
    frozenset(FINE_AR_PAIRS)
    | frozenset(v[0] for v in FINE_AR_PAIRS.values())
    | frozenset(FINE_AR_PATHS)
)
FINE_AR_BEGIN_NAMES = frozenset(v[0] for v in FINE_AR_PAIRS.values())
ALL_AR_NAMES = AR_NAMES | FINE_AR_NAMES
KNOWN_EVENT_NAMES = IB_NAMES | NVL_NAMES | ALL_AR_NAMES

# `detail` is uint16_t, so one full uint16 range per kind makes track IDs
# disjoint for every representable block index.
THREAD_KIND_STRIDE = 1 << 16
THREAD_OFFSET_IB = 100
THREAD_OFFSET_NVL = THREAD_OFFSET_IB + THREAD_KIND_STRIDE
THREAD_OFFSET_AR = THREAD_OFFSET_NVL + THREAD_KIND_STRIDE
THREAD_OFFSET_FINE_AR = THREAD_OFFSET_AR + THREAD_KIND_STRIDE
FINE_AR_LANE_STRIDE = 1 << 8

_PREFIX = (
    r"^(?:\[1,(?P<mpi_rank>\d+)\]<(?:stderr|stdout)>:"
    r"|\[(?P<mast_rank>\d+)\]:)"
)
_MAST_STREAM_WRAPPER = r"(?:\[(?:stderr|stdout)\]\s*)?"
_TRACE_LOG_PREFIX = r"(?:Prims|Pipes) trace"

# `[^\[]*?` forbids a stray "[" in the gap between rank prefix and the trace
# marker — that's how mpirun --tag-output splices another rank's prefix
# mid-line under load. Without this, the regex would backtrack across the
# splice and synthesize a wrong-rank event.
EVENT_RE: re.Pattern[str] = re.compile(
    _PREFIX + rf"[^\[]*?{_MAST_STREAM_WRAPPER}{_TRACE_LOG_PREFIX} event=(?P<name>\w+)"
    r" step=(?P<step>\d+)"
    r" rank=(?P<rank>-?\d+)"
    r" detail=(?P<detail>\d+)"
    r" slot=(?P<slot>\d+)"
    r" wall_time_ns=(?P<ns>\d+)\s*$"
)

FINE_EVENT_V1_RE: re.Pattern[str] = re.compile(
    _PREFIX + rf"[^\[]*?{_MAST_STREAM_WRAPPER}Prims fine trace event=(?P<name>\w+)"
    r" rank=(?P<rank>\d+)"
    r" phase=(?P<phase>\w+)"
    r" dependency_step=(?P<dependency_step>\d+)"
    r" stripe=(?P<stripe>\d+)"
    r" lane=(?P<lane>\d+)"
    r" peer=(?P<peer>\d+)"
    r" qp_lane=(?P<qp_lane>\d+)"
    r" bytes=(?P<bytes>\d+)"
    r" slot=(?P<slot>\d+)"
    r" wall_time_ns=(?P<ns>\d+)\s*$"
)

FINE_EVENT_V2_RE: re.Pattern[str] = re.compile(
    _PREFIX
    + rf"[^\[]*?{_MAST_STREAM_WRAPPER}Prims fine trace schema_version=(?P<schema_version>\d+)"
    r" trace_session=(?P<trace_session>\d+)"
    r" event=(?P<name>\w+)"
    r" rank=(?P<rank>\d+)"
    r" op_tag=(?P<op_tag>\d+)"
    r" phase=(?P<phase>\w+)"
    r" dependency_step=(?P<dependency_step>\d+)"
    r" block=(?P<block>\d+)"
    r" lane=(?P<lane>\d+)"
    r" chunk_tag=(?P<chunk>\d+)"
    r" role=(?P<role>\w+)"
    r" peer=(?P<peer>\d+)"
    r" qp_lane=(?P<qp_lane>\d+)"
    r" bytes=(?P<bytes>\d+)"
    r" slot=(?P<slot>\d+)"
    r" wall_time_ns=(?P<ns>\d+)\s*$"
)

# Kept for callers that import the original V1 pattern directly.
FINE_EVENT_RE = FINE_EVENT_V1_RE

LOSS_RE: re.Pattern[str] = re.compile(
    _PREFIX
    + rf"[^\[]*?{_MAST_STREAM_WRAPPER}{_TRACE_LOG_PREFIX} lost (?P<lost>\d+) entries\s*$"
)

LEGACY_DRAIN_RE: re.Pattern[str] = re.compile(
    _PREFIX
    + rf"[^\[]*?{_MAST_STREAM_WRAPPER}{_TRACE_LOG_PREFIX} drain entries_read=(?P<read>\d+)"
    r" entries_lost=(?P<lost>\d+)"
    r" last_read=(?P<last>\d+)\s*$"
)

# mpirun ... -host h1:p,h2:p,... — first ":p" is procs-per-node (= nvl_size).
MPIRUN_HOST_RE: re.Pattern[str] = re.compile(
    r"mpirun\s.*?-host\s+([\w\.\-]+:\d+(?:,[\w\.\-]+:\d+)*)"
)


@dataclasses.dataclass(frozen=True)
class Event:
    mpi_rank: int
    name: str
    step: int  # chunk id for AG events, producer step value for IB ops
    rank: int  # ib_rank (IB events) / ib_src (NVL events)
    group: int  # detail = group.group_id
    slot: int
    ns: int
    phase: str | None = None
    dependency_step: int = 0
    stripe: int = 0
    lane: int = 0
    peer: int = 0
    qp_lane: int = 0
    bytes: int = 0
    schema_version: int = 1
    trace_session: int = 0
    op_tag: int = 0
    block: int = 0
    chunk: int = 0
    role: str = "unknown"


@dataclasses.dataclass
class ParseStats:
    events: int = 0
    loss_reports: int = 0
    dropped_lines: int = 0
    lost_entries: int = 0


@dataclasses.dataclass(frozen=True)
class Topology:
    nvl_size: int
    ib_size: int
    auto_detected_nvl: bool


@dataclasses.dataclass
class Slice_:
    pid: int
    tid: int
    cat: str
    name: str
    begin_ns: int
    end_ns: int
    args: dict[str, object]
    instant: bool = False


@dataclasses.dataclass
class PairState:
    """Per-rank state used while walking events in (ns, slot) order."""

    mpi_rank: int
    # (group, chunk) -> (begin_ns, begin_slot, self_ib_rank)
    pending_ib: dict[tuple[int, int], tuple[int, int, int]] = dataclasses.field(
        default_factory=dict
    )
    # (group, chunk, ib_src) -> (begin_ns, begin_slot)
    pending_nvl_wait: dict[tuple[int, int, int], tuple[int, int]] = dataclasses.field(
        default_factory=dict
    )
    pending_nvl_bcast: dict[tuple[int, int, int], tuple[int, int]] = dataclasses.field(
        default_factory=dict
    )
    # (group, self_ib_rank) -> begin event
    pending_ib_send: dict[tuple[int, int], Event] = dataclasses.field(
        default_factory=dict
    )
    pending_ib_recv: dict[tuple[int, int], Event] = dataclasses.field(
        default_factory=dict
    )
    pending_ib_forward: dict[tuple[int, int], Event] = dataclasses.field(
        default_factory=dict
    )
    pending_allreduce: dict[tuple[str, int, int], Event] = dataclasses.field(
        default_factory=dict
    )
    pending_fine_allreduce: dict[tuple[object, ...], Event] = dataclasses.field(
        default_factory=dict
    )


def _open_log(path: str) -> TextIO:
    if path == "-":
        return sys.stdin
    return open(path, "r", encoding="utf-8", errors="replace")


def _detect_nvl_from_mpirun(line: str) -> int | None:
    m = MPIRUN_HOST_RE.search(line)
    if m is None:
        return None
    first = m.group(1).split(",", 1)[0]
    try:
        return int(first.split(":")[1])
    except (IndexError, ValueError):
        return None


def _parse_event_line(line: str) -> Event | None:
    m = EVENT_RE.fullmatch(line)
    if m is not None:
        name = m["name"]
        if name not in KNOWN_EVENT_NAMES:
            return None
        prefix_rank = m["mpi_rank"] or m["mast_rank"]
        return Event(
            mpi_rank=int(prefix_rank),
            name=name,
            step=int(m["step"]),
            rank=int(m["rank"]),
            group=int(m["detail"]),
            slot=int(m["slot"]),
            ns=int(m["ns"]),
        )

    fine_v2 = FINE_EVENT_V2_RE.fullmatch(line)
    if fine_v2 is not None and fine_v2["name"] in FINE_AR_NAMES:
        dependency_step = int(fine_v2["dependency_step"])
        block = int(fine_v2["block"])
        peer = int(fine_v2["peer"])
        fine_v2_prefix_rank = fine_v2["mpi_rank"] or fine_v2["mast_rank"]
        return Event(
            mpi_rank=int(fine_v2_prefix_rank),
            name=fine_v2["name"],
            step=dependency_step,
            rank=peer,
            group=block,
            slot=int(fine_v2["slot"]),
            ns=int(fine_v2["ns"]),
            phase=fine_v2["phase"],
            dependency_step=dependency_step,
            stripe=block,
            lane=int(fine_v2["lane"]),
            peer=peer,
            qp_lane=int(fine_v2["qp_lane"]),
            bytes=int(fine_v2["bytes"]),
            schema_version=int(fine_v2["schema_version"]),
            trace_session=int(fine_v2["trace_session"]),
            op_tag=int(fine_v2["op_tag"]),
            block=block,
            chunk=int(fine_v2["chunk"]),
            role=fine_v2["role"],
        )

    fine = FINE_EVENT_V1_RE.fullmatch(line)
    if fine is None or fine["name"] not in FINE_AR_NAMES:
        return None
    dependency_step = int(fine["dependency_step"])
    stripe = int(fine["stripe"])
    peer = int(fine["peer"])
    fine_prefix_rank = fine["mpi_rank"] or fine["mast_rank"]
    return Event(
        mpi_rank=int(fine_prefix_rank),
        name=fine["name"],
        step=dependency_step,
        rank=peer,
        group=stripe,
        slot=int(fine["slot"]),
        ns=int(fine["ns"]),
        phase=fine["phase"],
        dependency_step=dependency_step,
        stripe=stripe,
        lane=int(fine["lane"]),
        peer=peer,
        qp_lane=int(fine["qp_lane"]),
        bytes=int(fine["bytes"]),
    )


def _is_trace_line(line: str) -> bool:
    return "Prims trace" in line or "Prims fine trace" in line or "Pipes trace" in line


def parse_events(
    paths: Sequence[str], stats: ParseStats
) -> tuple[list[Event], int | None]:
    """Parse all log paths into Event records; also detect nvl_size from mpirun."""
    events: list[Event] = []
    nvl_hint: int | None = None
    for path in paths:
        with _open_log(path) as fh:
            for line in fh:
                if nvl_hint is None and "mpirun" in line and "-host" in line:
                    nvl_hint = _detect_nvl_from_mpirun(line)

                event = _parse_event_line(line)
                if event is not None:
                    events.append(event)
                    stats.events += 1
                    continue

                loss = LOSS_RE.fullmatch(line) or LEGACY_DRAIN_RE.fullmatch(line)
                if loss is not None:
                    stats.loss_reports += 1
                    stats.lost_entries += int(loss["lost"])
                    continue

                # Count only trace lines that failed to parse — mpirun
                # occasionally splices its rank prefix mid-line under load.
                # Other log lines are not interesting.
                if _is_trace_line(line):
                    stats.dropped_lines += 1
    return events, nvl_hint


def resolve_topology(
    events: Sequence[Event],
    nvl_size_arg: int | None,
    ib_size_arg: int | None,
    nvl_size_hint: int | None,
) -> Topology:
    auto_detected = False
    if nvl_size_arg is not None:
        nvl_size = nvl_size_arg
    elif nvl_size_hint is not None:
        nvl_size = nvl_size_hint
        auto_detected = True
    else:
        logger.warning("Could not infer nvl_size from log; defaulting to 2")
        nvl_size = 2

    if ib_size_arg is not None:
        ib_size = ib_size_arg
    else:
        max_rank = max((e.mpi_rank for e in events), default=0)
        ib_size = max_rank // max(nvl_size, 1) + 1
    return Topology(nvl_size=nvl_size, ib_size=ib_size, auto_detected_nvl=auto_detected)


def filter_events(
    events: Iterable[Event],
    rank_filter: set[int] | None,
    type_filter: set[str],
    time_start: int | None,
    time_end: int | None,
) -> Iterator[Event]:
    for e in events:
        if rank_filter is not None and e.mpi_rank not in rank_filter:
            continue
        if "ib" not in type_filter and e.name in IB_NAMES:
            continue
        if "nvl" not in type_filter and e.name in NVL_NAMES:
            continue
        if "allreduce" not in type_filter and e.name in ALL_AR_NAMES:
            continue
        if time_start is not None and e.ns < time_start:
            continue
        if time_end is not None and e.ns >= time_end:
            continue
        yield e


def group_by_rank(events: Iterable[Event]) -> dict[int, list[Event]]:
    per_rank: dict[int, list[Event]] = defaultdict(list)
    for e in events:
        per_rank[e.mpi_rank].append(e)
    for lst in per_rank.values():
        lst.sort(key=lambda e: (e.ns, e.slot))
    return per_rank


def _ib_args(
    chunk: int,
    self_r: int,
    group: int,
    begin_ns: int,
    end_ns: int,
    begin_slot: int,
    end_slot: int,
) -> dict[str, object]:
    return {
        "chunk": chunk,
        "ib_rank": self_r,
        "block": group,
        "slot_begin": begin_slot,
        "slot_end": end_slot,
        "begin_ns": begin_ns,
        "end_ns": end_ns,
        "dur_ns": end_ns - begin_ns,
    }


def _nvl_args(
    chunk: int,
    ib_src: int,
    group: int,
    begin_ns: int,
    end_ns: int,
    begin_slot: int,
    end_slot: int,
) -> dict[str, object]:
    return {
        "chunk": chunk,
        "ib_src": ib_src,
        "block": group,
        "slot_begin": begin_slot,
        "slot_end": end_slot,
        "begin_ns": begin_ns,
        "end_ns": end_ns,
        "dur_ns": end_ns - begin_ns,
    }


def _ib_op_args(begin: Event, end: Event, op_name: str) -> dict[str, object]:
    return {
        "op": op_name,
        "ib_rank": end.rank,
        "block": end.group,
        "begin_step": begin.step,
        "end_step": end.step,
        "bytes": end.step - begin.step,
        "slot_begin": begin.slot,
        "slot_end": end.slot,
        "begin_ns": begin.ns,
        "end_ns": end.ns,
        "dur_ns": end.ns - begin.ns,
    }


def _handle_ib_op_begin(
    pending: dict[tuple[int, int], Event],
    e: Event,
    unmatched: list[tuple[Event, str]],
) -> None:
    key = (e.group, e.rank)
    previous = pending.get(key)
    if previous is not None:
        unmatched.append((previous, f"{previous.name} without matching end"))
    pending[key] = e


def _handle_ib_op_end(
    state: PairState,
    pending: dict[tuple[int, int], Event],
    e: Event,
    op_name: str,
    slices: list[Slice_],
    unmatched: list[tuple[Event, str]],
) -> None:
    key = (e.group, e.rank)
    begin = pending.pop(key, None)
    if begin is None:
        unmatched.append((e, f"{e.name} without matching begin"))
        return
    slices.append(
        Slice_(
            pid=state.mpi_rank,
            tid=THREAD_OFFSET_IB + e.group,
            cat=op_name,
            name=op_name,
            begin_ns=begin.ns,
            end_ns=e.ns,
            args=_ib_op_args(begin, e, op_name),
        )
    )


def _handle_ib_ready(
    state: PairState,
    e: Event,
    slices: list[Slice_],
) -> int:
    key = (e.group, e.step)
    pending = state.pending_ib.get(key)
    if pending is not None and e.rank == pending[2]:
        begin_ns, begin_slot, self_r = pending
        slices.append(
            Slice_(
                pid=state.mpi_rank,
                tid=THREAD_OFFSET_IB + e.group,
                cat="ib",
                name=f"IB chunk {e.step}",
                begin_ns=begin_ns,
                end_ns=e.ns,
                args=_ib_args(
                    e.step,
                    self_r,
                    e.group,
                    begin_ns,
                    e.ns,
                    begin_slot,
                    e.slot,
                ),
            )
        )
        del state.pending_ib[key]
        return 0
    # Forwarded ready (rank != self) or self-ready with missing begin used to
    # become Perfetto instant events. Keep only the count now.
    return 1


def _handle_nvl_chunk_ready(
    state: PairState,
    e: Event,
    slices: list[Slice_],
    unmatched: list[tuple[Event, str]],
) -> None:
    key = (e.group, e.step, e.rank)
    wait = state.pending_nvl_wait.pop(key, None)
    if wait is not None:
        begin_ns, begin_slot = wait
        slices.append(
            Slice_(
                pid=state.mpi_rank,
                tid=THREAD_OFFSET_NVL + e.group,
                cat="nvl_wait",
                name=f"NVL wait src={e.rank} chunk={e.step}",
                begin_ns=begin_ns,
                end_ns=e.ns,
                args=_nvl_args(
                    e.step,
                    e.rank,
                    e.group,
                    begin_ns,
                    e.ns,
                    begin_slot,
                    e.slot,
                ),
            )
        )
    else:
        unmatched.append((e, "nvl_chunk_ready without matching wait_begin"))
    state.pending_nvl_bcast[key] = (e.ns, e.slot)


def _handle_nvl_task_done(
    state: PairState,
    e: Event,
    slices: list[Slice_],
    unmatched: list[tuple[Event, str]],
) -> None:
    key = (e.group, e.step, e.rank)
    bcast = state.pending_nvl_bcast.pop(key, None)
    if bcast is None:
        unmatched.append((e, "nvl_task_done without matching chunk_ready"))
        return
    begin_ns, begin_slot = bcast
    slices.append(
        Slice_(
            pid=state.mpi_rank,
            tid=THREAD_OFFSET_NVL + e.group,
            cat="nvl_bcast",
            name=f"NVL bcast src={e.rank} chunk={e.step}",
            begin_ns=begin_ns,
            end_ns=e.ns,
            args=_nvl_args(
                e.step,
                e.rank,
                e.group,
                begin_ns,
                e.ns,
                begin_slot,
                e.slot,
            ),
        )
    )


def _drain_state(state: PairState, unmatched: list[tuple[Event, str]]) -> None:
    for (group, chunk), (_, _, self_r) in state.pending_ib.items():
        unmatched.append(
            (
                Event(
                    mpi_rank=state.mpi_rank,
                    name=AG_IB_BEGIN,
                    step=chunk,
                    rank=self_r,
                    group=group,
                    slot=-1,
                    ns=-1,
                ),
                "ib_chunk_begin without matching ready",
            )
        )
    for (group, chunk, src), _ in state.pending_nvl_wait.items():
        unmatched.append(
            (
                Event(
                    mpi_rank=state.mpi_rank,
                    name=AG_NVL_WAIT_BEGIN,
                    step=chunk,
                    rank=src,
                    group=group,
                    slot=-1,
                    ns=-1,
                ),
                "nvl_wait_begin without chunk_ready",
            )
        )
    for (group, chunk, src), _ in state.pending_nvl_bcast.items():
        unmatched.append(
            (
                Event(
                    mpi_rank=state.mpi_rank,
                    name=AG_NVL_CHUNK_READY,
                    step=chunk,
                    rank=src,
                    group=group,
                    slot=-1,
                    ns=-1,
                ),
                "nvl_chunk_ready without task_done",
            )
        )
    for _, event in state.pending_ib_send.items():
        unmatched.append((event, f"{event.name} without matching end"))
    for _, event in state.pending_ib_recv.items():
        unmatched.append((event, f"{event.name} without matching end"))
    for _, event in state.pending_ib_forward.items():
        unmatched.append((event, f"{event.name} without matching end"))
    for _, event in state.pending_allreduce.items():
        unmatched.append((event, f"{event.name} without matching end"))
    for _, event in state.pending_fine_allreduce.items():
        unmatched.append((event, f"{event.name} without matching end"))


def _handle_allreduce_event(
    state: PairState,
    e: Event,
    slices: list[Slice_],
    unmatched: list[tuple[Event, str]],
) -> None:
    if e.name in AR_BEGIN_NAMES:
        key = (e.name, e.group, e.step)
        previous = state.pending_allreduce.get(key)
        if previous is not None:
            unmatched.append((previous, f"{previous.name} without matching end"))
        state.pending_allreduce[key] = e
        return

    begin_name, category, display_name = AR_PAIRS[e.name]
    begin = state.pending_allreduce.pop((begin_name, e.group, e.step), None)
    if begin is None:
        unmatched.append((e, f"{e.name} without matching begin"))
        return
    slices.append(
        Slice_(
            pid=state.mpi_rank,
            tid=THREAD_OFFSET_AR + e.group,
            cat=category,
            name=display_name,
            begin_ns=begin.ns,
            end_ns=e.ns,
            args={
                "block": e.group,
                "step": e.step,
                "slot_begin": begin.slot,
                "slot_end": e.slot,
                "begin_ns": begin.ns,
                "end_ns": e.ns,
                "dur_ns": e.ns - begin.ns,
            },
        )
    )


def _fine_allreduce_key(e: Event, begin_name: str) -> tuple[object, ...]:
    return (
        begin_name,
        e.schema_version,
        e.trace_session,
        e.op_tag,
        e.phase or "unknown",
        e.dependency_step,
        _fine_allreduce_block(e),
        e.lane,
        e.chunk,
        e.role,
        e.peer,
        e.qp_lane,
        e.bytes,
    )


def _fine_allreduce_block(e: Event) -> int:
    return e.block if e.schema_version >= 2 else e.stripe


def _fine_allreduce_tid(e: Event) -> int:
    return (
        THREAD_OFFSET_FINE_AR + _fine_allreduce_block(e) * FINE_AR_LANE_STRIDE + e.lane
    )


def _fine_allreduce_args(e: Event) -> dict[str, object]:
    return {
        "schema_version": e.schema_version,
        "trace_session": e.trace_session,
        "op_tag": e.op_tag,
        "phase": e.phase or "unknown",
        "dependency_step": e.dependency_step,
        "block": _fine_allreduce_block(e),
        "lane": e.lane,
        "chunk": e.chunk,
        "role": e.role,
        "peer": e.peer,
        "qp_lane": e.qp_lane,
        "bytes": e.bytes,
        "slot": e.slot,
        "wall_time_ns": e.ns,
    }


def _handle_fine_allreduce_event(
    state: PairState,
    e: Event,
    slices: list[Slice_],
    unmatched: list[tuple[Event, str]],
) -> None:
    path_name = FINE_AR_PATHS.get(e.name)
    if path_name is not None:
        slices.append(
            Slice_(
                pid=state.mpi_rank,
                tid=_fine_allreduce_tid(e),
                cat="allreduce_path",
                name=path_name,
                begin_ns=e.ns,
                end_ns=e.ns,
                args=_fine_allreduce_args(e),
                instant=True,
            )
        )
        return

    if e.name in FINE_AR_BEGIN_NAMES:
        key = _fine_allreduce_key(e, e.name)
        previous = state.pending_fine_allreduce.get(key)
        if previous is not None:
            unmatched.append((previous, f"{previous.name} without matching end"))
        state.pending_fine_allreduce[key] = e
        return

    begin_name, display_name = FINE_AR_PAIRS[e.name]
    begin = state.pending_fine_allreduce.pop(_fine_allreduce_key(e, begin_name), None)
    if begin is None:
        unmatched.append((e, f"{e.name} without matching begin"))
        return
    event_args = _fine_allreduce_args(e)
    event_args.update(
        {
            "slot_begin": begin.slot,
            "slot_end": e.slot,
            "begin_ns": begin.ns,
            "end_ns": e.ns,
            "dur_ns": e.ns - begin.ns,
        }
    )
    slices.append(
        Slice_(
            pid=state.mpi_rank,
            tid=_fine_allreduce_tid(e),
            cat="allreduce_fine",
            name=display_name,
            begin_ns=begin.ns,
            end_ns=e.ns,
            args=event_args,
        )
    )


def _handle_ib_event(
    state: PairState,
    e: Event,
    slices: list[Slice_],
    unmatched: list[tuple[Event, str]],
) -> int:
    if e.name == AG_IB_BEGIN:
        state.pending_ib[(e.group, e.step)] = (e.ns, e.slot, e.rank)
    elif e.name == AG_IB_READY:
        return _handle_ib_ready(state, e, slices)
    elif e.name == IB_SEND_BEGIN:
        _handle_ib_op_begin(state.pending_ib_send, e, unmatched)
    elif e.name == IB_SEND_END:
        _handle_ib_op_end(state, state.pending_ib_send, e, IB_SEND, slices, unmatched)
    elif e.name == IB_RECV_BEGIN:
        _handle_ib_op_begin(state.pending_ib_recv, e, unmatched)
    elif e.name == IB_RECV_END:
        _handle_ib_op_end(state, state.pending_ib_recv, e, IB_RECV, slices, unmatched)
    elif e.name == IB_FORWARD_BEGIN:
        _handle_ib_op_begin(state.pending_ib_forward, e, unmatched)
    elif e.name == IB_FORWARD_END:
        _handle_ib_op_end(
            state,
            state.pending_ib_forward,
            e,
            IB_FORWARD,
            slices,
            unmatched,
        )
    else:
        unmatched.append((e, f"unknown IB event name {e.name!r}"))
    return 0


def _handle_nvl_event(
    state: PairState,
    e: Event,
    slices: list[Slice_],
    unmatched: list[tuple[Event, str]],
) -> None:
    if e.name == AG_NVL_WAIT_BEGIN:
        state.pending_nvl_wait[(e.group, e.step, e.rank)] = (
            e.ns,
            e.slot,
        )
    elif e.name == AG_NVL_CHUNK_READY:
        _handle_nvl_chunk_ready(state, e, slices, unmatched)
    elif e.name == AG_NVL_TASK_DONE:
        _handle_nvl_task_done(state, e, slices, unmatched)
    else:
        unmatched.append((e, f"unknown NVL event name {e.name!r}"))


def pair_events(
    per_rank: dict[int, list[Event]],
) -> tuple[list[Slice_], int, list[tuple[Event, str]]]:
    slices: list[Slice_] = []
    unpaired_instant_count = 0
    unmatched: list[tuple[Event, str]] = []

    for mpi_rank, evs in per_rank.items():
        state = PairState(mpi_rank=mpi_rank)
        for e in evs:
            if e.name in IB_NAMES:
                unpaired_instant_count += _handle_ib_event(state, e, slices, unmatched)
            elif e.name in NVL_NAMES:
                _handle_nvl_event(state, e, slices, unmatched)
            elif e.name in AR_NAMES:
                _handle_allreduce_event(state, e, slices, unmatched)
            elif e.name in FINE_AR_NAMES:
                _handle_fine_allreduce_event(state, e, slices, unmatched)
            else:
                unmatched.append((e, f"unknown event name {e.name!r}"))
        _drain_state(state, unmatched)
    return slices, unpaired_instant_count, unmatched


def _compute_used_tids(slices: Sequence[Slice_]) -> dict[int, set[int]]:
    used: dict[int, set[int]] = defaultdict(set)
    for s in slices:
        used[s.pid].add(s.tid)
    return used


def _compute_min_ns(slices: Sequence[Slice_]) -> int:
    candidates: list[int] = [s.begin_ns for s in slices]
    return min(candidates) if candidates else 0


def _emit_metadata(
    pids: Sequence[int],
    used_tids: dict[int, set[int]],
    topology: Topology,
) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    nvl_size = max(topology.nvl_size, 1)
    for pid in sorted(set(pids)):
        ib_rank = pid // nvl_size
        nvl_rank = pid % nvl_size
        out.append(
            {
                "ph": "M",
                "name": "process_name",
                "pid": pid,
                "args": {"name": f"rank {pid} (ib={ib_rank} nvl={nvl_rank})"},
            }
        )
        out.append(
            {
                "ph": "M",
                "name": "process_sort_index",
                "pid": pid,
                "args": {"sort_index": pid},
            }
        )
        for tid in sorted(used_tids.get(pid, set())):
            if tid >= THREAD_OFFSET_FINE_AR:
                track = tid - THREAD_OFFSET_FINE_AR
                kind, block = "AllReduce fine", track // FINE_AR_LANE_STRIDE
                lane = track % FINE_AR_LANE_STRIDE
            elif tid >= THREAD_OFFSET_AR:
                kind, block = "AllReduce", tid - THREAD_OFFSET_AR
                lane = None
            elif tid >= THREAD_OFFSET_NVL:
                kind, block = "NVL", tid - THREAD_OFFSET_NVL
                lane = None
            else:
                kind, block = "IB", tid - THREAD_OFFSET_IB
                lane = None
            out.append(
                {
                    "ph": "M",
                    "name": "thread_name",
                    "pid": pid,
                    "tid": tid,
                    "args": {
                        "name": (
                            f"{kind} block {block} lane {lane}"
                            if lane is not None
                            else f"{kind} block {block}"
                        )
                    },
                }
            )
            out.append(
                {
                    "ph": "M",
                    "name": "thread_sort_index",
                    "pid": pid,
                    "tid": tid,
                    "args": {"sort_index": tid},
                }
            )
    return out


def build_trace(
    slices: Sequence[Slice_],
    topology: Topology,
    pids: Sequence[int],
) -> dict[str, object]:
    used_tids = _compute_used_tids(slices)
    min_ns = _compute_min_ns(slices)

    trace_events: list[dict[str, object]] = _emit_metadata(pids, used_tids, topology)

    for s in slices:
        if s.instant:
            trace_events.append(
                {
                    "ph": "i",
                    "s": "t",
                    "name": s.name,
                    "cat": s.cat,
                    "pid": s.pid,
                    "tid": s.tid,
                    "ts": (s.begin_ns - min_ns) / 1000.0,
                    "args": s.args,
                }
            )
            continue
        trace_events.append(
            {
                "ph": "X",
                "name": s.name,
                "cat": s.cat,
                "pid": s.pid,
                "tid": s.tid,
                "ts": (s.begin_ns - min_ns) / 1000.0,
                "dur": (s.end_ns - s.begin_ns) / 1000.0,
                "args": s.args,
            }
        )

    return {
        "displayTimeUnit": "ns",
        "traceEvents": trace_events,
        "meta": {
            "time_offset_ns": min_ns,
            "nvl_size": topology.nvl_size,
            "ib_size": topology.ib_size,
            "nvl_size_auto_detected": topology.auto_detected_nvl,
        },
    }


def write_json(trace: dict[str, object], out_path: str, gzip_: bool) -> str:
    payload = json.dumps(trace, separators=(",", ":"))
    if gzip_:
        if not out_path.endswith(".gz"):
            out_path += ".gz"
        with gzip.open(out_path, "wt", encoding="utf-8") as fh:
            fh.write(payload)
    else:
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(payload)
    return out_path


def parse_csv_ints(value: str) -> set[int]:
    return {int(x.strip()) for x in value.split(",") if x.strip()}


def parse_csv_strs(value: str) -> set[str]:
    return {x.strip() for x in value.split(",") if x.strip()}


def default_output(first_log: str) -> str:
    if first_log == "-":
        return "pipestrace.perfetto.json"
    base = os.path.basename(first_log)
    if base.endswith(".log"):
        base = base[:-4]
    return f"{base}.perfetto.json"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pipestrace_to_perfetto",
        description=(
            "Convert supported collective PipesTrace log lines "
            "into a Perfetto / Chrome Trace Event JSON timeline that can be "
            "opened at https://ui.perfetto.dev/."
        ),
    )
    p.add_argument(
        "log_file",
        nargs="+",
        help="One or more captured launcher logs. '-' reads stdin.",
    )
    p.add_argument(
        "-o",
        "--output",
        help="Output JSON path. Default: <first log basename>.perfetto.json",
    )
    p.add_argument(
        "--gzip",
        action="store_true",
        help="Write gzip-compressed JSON (.json.gz). Perfetto UI accepts gz.",
    )
    p.add_argument(
        "--nvl-size",
        type=int,
        help=(
            "Procs per node (= nvl_size). Default: auto-detect from the "
            "mpirun -host h1:p,... line; fall back to 2 with a warning."
        ),
    )
    p.add_argument(
        "--ib-size",
        type=int,
        help="Number of IB nodes. Default: max_mpi_rank // nvl_size + 1.",
    )
    p.add_argument(
        "--ranks",
        type=parse_csv_ints,
        help="Comma-list of MPI ranks to include. Default: all.",
    )
    p.add_argument(
        "--types",
        type=parse_csv_strs,
        default={"ib", "nvl", "allreduce"},
        help=("Block kinds to include: ib,nvl,allreduce. Default: ib,nvl,allreduce."),
    )
    p.add_argument(
        "--time-start",
        type=int,
        help="Drop events with wall_time_ns < NS.",
    )
    p.add_argument(
        "--time-end",
        type=int,
        help="Drop events with wall_time_ns >= NS.",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Log parse summary (counts, dropped, unmatched).",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Exit 2 on any unmatched begin/end pair or lost trace entries.",
    )
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if not args.types.intersection({"ib", "nvl", "allreduce"}):
        logger.error("--types must include at least one of ib,nvl,allreduce")
        return 1

    stats = ParseStats()
    raw_events, nvl_hint = parse_events(args.log_file, stats)
    if not raw_events:
        logger.error("No Prims/Pipes trace events parsed from %s", args.log_file)
        return 1

    topology = resolve_topology(raw_events, args.nvl_size, args.ib_size, nvl_hint)
    logger.debug(
        "Topology: nvl_size=%d ib_size=%d auto_detected=%s",
        topology.nvl_size,
        topology.ib_size,
        topology.auto_detected_nvl,
    )

    filtered = list(
        filter_events(
            raw_events,
            rank_filter=args.ranks,
            type_filter=args.types,
            time_start=args.time_start,
            time_end=args.time_end,
        )
    )
    logger.debug("Filtered events: %d (from %d raw)", len(filtered), len(raw_events))

    per_rank = group_by_rank(filtered)
    slices, unpaired_instant_count, unmatched = pair_events(per_rank)
    logger.debug(
        "Parse stats: events=%d loss_reports=%d lost_entries=%d dropped_lines=%d "
        "unpaired_instants=%d",
        stats.events,
        stats.loss_reports,
        stats.lost_entries,
        stats.dropped_lines,
        unpaired_instant_count,
    )
    if stats.lost_entries:
        logger.warning(
            "Prims/Pipes trace reported %d lost trace entries across %d loss records",
            stats.lost_entries,
            stats.loss_reports,
        )
    logger.debug(
        "Pair results: slices=%d unpaired_instants=%d unmatched=%d",
        len(slices),
        unpaired_instant_count,
        len(unmatched),
    )
    if args.verbose:
        for ev, reason in unmatched[:50]:
            logger.debug("unmatched: %s (%s)", reason, ev)
        if len(unmatched) > 50:
            logger.debug("... and %d more", len(unmatched) - 50)

    pids = sorted({e.mpi_rank for e in filtered})
    trace = build_trace(slices, topology, pids)
    out_path = args.output or default_output(args.log_file[0])
    written = write_json(trace, out_path, gzip_=args.gzip)
    logger.warning(
        "Wrote %s (%d slices, %d unpaired instants suppressed)",
        written,
        len(slices),
        unpaired_instant_count,
    )

    if args.strict and (unmatched or stats.lost_entries or stats.dropped_lines):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
