# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

"""Correctness tests for NVLink copy-based Triton send/recv."""

from __future__ import annotations

import os
import socket
import sys
from collections.abc import Callable

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from comms.pipes.triton.collectives.nvl.sendrecv_op import (
    triton_nvl_recv,
    triton_nvl_send,
    triton_nvl_sendrecv,
    triton_nvl_sendrecv_ws,
)


# ---------------------------------------------------------------------------
# Warp-specialized (WS) test gate.
#
# The WS path uses TLX `async_tasks` / `async_task` warp-specialization
# codegen. In the current `//triton:triton` (in-tree TLX) build, the WS
# kernel hits a runtime hang / barrier issue (see TLX_BARRIER_BUG.md and
# the original D106331816 commit message: "tlx.sync_threads() has an
# iteration-count-dependent barrier bug — hangs after >2 while-loop
# iterations"). The kernel logic itself is correct — the failure is in
# upstream TLX codegen.
#
# Until upstream TLX is fixed, we DO NOT run the WS test cases by default
# (they would hang the entire test binary). The cases are kept inline so
# they re-activate automatically once `TRITON_NVL_WS_TESTS_ENABLED=1` is
# set, e.g. when a developer wants to validate WS against a fixed TLX
# build, or once the upstream fix lands and we can flip the default.
# ---------------------------------------------------------------------------
_WS_TESTS_ENABLED: bool = os.environ.get("TRITON_NVL_WS_TESTS_ENABLED", "0") == "1"


_BASE_CASES: list[tuple[str, int]] = [
    ("small_64KB", 64 * 1024),
    ("medium_1MB", 1024 * 1024),
    ("large_16MB", 16 * 1024 * 1024),
]


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _numel(msg_bytes: int, dtype: torch.dtype) -> int:
    return max(msg_bytes // dtype.itemsize, 1)


def _make_pattern(
    rank: int, numel: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    if dtype == torch.int32:
        idx = torch.arange(numel, dtype=torch.int32, device=device)
        return (idx & 0x00FFFFFF) | (int(rank) << 24)
    return torch.full((numel,), float(rank + 1), dtype=dtype, device=device)


def _all_ok(local_ok: bool, device: torch.device, group: dist.ProcessGroup) -> bool:
    status = torch.tensor([1 if local_ok else 0], dtype=torch.int32, device=device)
    dist.all_reduce(status, op=dist.ReduceOp.MIN, group=group)
    return bool(status.item())


def _check_equal(
    actual: torch.Tensor,
    expected: torch.Tensor,
    label: str,
    local_rank: int,
) -> bool:
    ok = torch.equal(actual, expected)
    if not ok:
        n_wrong = (actual != expected).sum().item()
        print(
            f"  FAIL(rank {local_rank}): {label} — {n_wrong}/{actual.numel()} mismatched, "
            f"first-10 expected={expected[:10].tolist()} got={actual[:10].tolist()}"
        )
    return ok


def _run_unidirectional(
    name: str,
    msg_bytes: int,
    dtype: torch.dtype,
    src_rank: int,
    local_rank: int,
    group: dist.ProcessGroup,
    *,
    num_blocks: int = 16,
    num_warps: int = 8,
) -> bool:
    peer_rank = 1 - local_rank
    device = torch.device(f"cuda:{local_rank}")
    numel = _numel(msg_bytes, dtype)

    if local_rank == src_rank:
        send = _make_pattern(local_rank, numel, dtype, device)
        triton_nvl_send(
            send, peer_rank, group=group, num_blocks=num_blocks, num_warps=num_warps
        )
        local_ok = True
    else:
        recv = torch.zeros(numel, dtype=dtype, device=device)
        expected = _make_pattern(peer_rank, numel, dtype, device)
        triton_nvl_recv(
            recv, peer_rank, group=group, num_blocks=num_blocks, num_warps=num_warps
        )
        torch.cuda.synchronize(device)
        local_ok = _check_equal(recv, expected, name, local_rank)

    torch.cuda.synchronize(device)
    ok = _all_ok(local_ok, device, group)
    if local_rank == 0:
        print(f"  {'PASS' if ok else 'FAIL'}: {name}")
    return ok


def _run_bidirectional(
    name: str,
    msg_bytes: int,
    dtype: torch.dtype,
    local_rank: int,
    group: dist.ProcessGroup,
    *,
    num_blocks: int = 16,
    num_warps: int = 8,
    warp_specialized: bool = False,
    sender_warps: int = 4,
    receiver_warps: int = 4,
) -> bool:
    peer_rank = 1 - local_rank
    device = torch.device(f"cuda:{local_rank}")
    numel = _numel(msg_bytes, dtype)
    send = _make_pattern(local_rank, numel, dtype, device)
    recv = torch.zeros(numel, dtype=dtype, device=device)
    expected = _make_pattern(peer_rank, numel, dtype, device)

    if warp_specialized:
        triton_nvl_sendrecv_ws(
            send,
            recv,
            peer_rank,
            group=group,
            num_blocks=num_blocks,
            sender_warps=sender_warps,
            receiver_warps=receiver_warps,
        )
    else:
        triton_nvl_sendrecv(
            send,
            recv,
            peer_rank,
            group=group,
            num_blocks=num_blocks,
            num_warps=num_warps,
        )
    torch.cuda.synchronize(device)

    ok = _all_ok(_check_equal(recv, expected, name, local_rank), device, group)
    if local_rank == 0:
        print(f"  {'PASS' if ok else 'FAIL'}: {name}")
    return ok


def _run_with_signal_bytes_override(
    name: str,
    signal_bytes: int,
    local_rank: int,
    group: dist.ProcessGroup,
) -> bool:
    """Run a small bidirectional case with a forced ``signal_bytes`` override.

    Drops a temporary tuning JSON whose ``default_stable.signal_bytes`` is
    smaller than ``_BLOCK_STRIDE_BYTES``, so the host sets
    ``CHUNKS_PER_SLOT = _BLOCK_STRIDE_BYTES // signal_bytes > 1`` and the
    kernel exercises the sub-slot signaling branch (per-chunk DATA_READY,
    slot-boundary backpressure). Verifies functional equivalence vs the
    ``CHUNKS_PER_SLOT == 1`` fast path on the same message size.
    """
    import json as _json
    import tempfile as _tempfile

    from comms.pipes.triton.collectives.nvl import tuning_config as _tc

    msg_bytes = 256 * 1024
    f = _tempfile.NamedTemporaryFile(  # noqa: SIM115
        mode="w", suffix=".json", delete=False
    )
    _json.dump(
        {
            "default_stable": {
                "tile_rows": 32,
                "tile_row_bytes": 2048,
                "signal_bytes": signal_bytes,
                "num_blocks": 16,
                "num_warps": 8,
            },
            "configs": [],
        },
        f,
    )
    f.close()
    prev = os.environ.get("TRITON_NVL_TUNING_JSON")
    os.environ["TRITON_NVL_TUNING_JSON"] = f.name
    _tc._JSON_CACHE = None
    _tc._CONFIG_CACHE.clear()
    try:
        return _run_bidirectional(
            name, msg_bytes, torch.int32, local_rank, group, num_blocks=16
        )
    finally:
        if prev is None:
            os.environ.pop("TRITON_NVL_TUNING_JSON", None)
        else:
            os.environ["TRITON_NVL_TUNING_JSON"] = prev
        _tc._JSON_CACHE = None
        _tc._CONFIG_CACHE.clear()
        os.unlink(f.name)


def _run_back_to_back_across_chunks_per_slot(
    name: str,
    local_rank: int,
    group: dist.ProcessGroup,
) -> bool:
    """Three back-to-back launches with three distinct CHUNKS_PER_SLOT layouts.

    The persistent ``sender_step_ptr`` / ``recver_step_ptr`` state survives
    across launches and across CHUNKS_PER_SLOT changes. Argument is that
    the signal counter is sender-published TAIL (``send_step + 1``) opaque
    to the in-slot chunk layout, so the receiver only needs to see TAIL ≥
    its expected step. This test pins the invariant: launch A with
    ``signal_bytes = _BLOCK_STRIDE_BYTES`` (CHUNKS_PER_SLOT=1, fast path),
    launch B with ``signal_bytes = 64KB`` (CHUNKS_PER_SLOT=4, sub-slot),
    launch C back to ``signal_bytes = _BLOCK_STRIDE_BYTES``. All three
    must produce correct receiver output despite reusing the same
    persistent step counters.
    """
    import json as _json
    import tempfile as _tempfile

    from comms.pipes.triton.collectives.nvl import (
        sendrecv_op as _so,
        tuning_config as _tc,
    )

    def _write_json(signal_bytes: int) -> str:
        f = _tempfile.NamedTemporaryFile(  # noqa: SIM115
            mode="w", suffix=".json", delete=False
        )
        _json.dump(
            {
                "default_stable": {
                    "tile_rows": 32,
                    "tile_row_bytes": 2048,
                    "signal_bytes": signal_bytes,
                    "num_blocks": 16,
                    "num_warps": 8,
                },
                "configs": [],
            },
            f,
        )
        f.close()
        return f.name

    msg_bytes = 256 * 1024
    elements = msg_bytes // 4
    fast_signal = _so._BLOCK_STRIDE_BYTES  # CHUNKS_PER_SLOT == 1
    subslot_signal = 64 * 1024  # CHUNKS_PER_SLOT == _BLOCK_STRIDE_BYTES // 64KB
    layouts = [
        ("fast0", fast_signal, 1),
        ("subslot", subslot_signal, fast_signal // subslot_signal),
        ("fast1", fast_signal, 1),
    ]
    paths: list[str] = []
    prev = os.environ.get("TRITON_NVL_TUNING_JSON")
    rank = dist.get_rank(group=group)
    peer = 1 - rank
    try:
        for stage_name, signal_bytes, expected_chunks_per_slot in layouts:
            path = _write_json(signal_bytes)
            paths.append(path)
            os.environ["TRITON_NVL_TUNING_JSON"] = path
            _tc._JSON_CACHE = None
            _tc._CONFIG_CACHE.clear()

            cfg = _tc.get_sendrecv_config(
                msg_bytes=msg_bytes,
                element_size=4,
                num_peers=1,
            )
            if cfg.stable.signal_bytes != signal_bytes:
                print(
                    f"FAIL {name}/{stage_name}: JSON config inactive; "
                    f"expected signal_bytes={signal_bytes}, "
                    f"got {cfg.stable.signal_bytes}"
                )
                return False
            chunks_per_slot = _so._BLOCK_STRIDE_BYTES // cfg.stable.signal_bytes
            if chunks_per_slot != expected_chunks_per_slot:
                print(
                    f"FAIL {name}/{stage_name}: expected CHUNKS_PER_SLOT="
                    f"{expected_chunks_per_slot}, got {chunks_per_slot}"
                )
                return False

            send_buf = torch.full(
                (elements,),
                rank * 100 + len(paths),
                dtype=torch.int32,
                device=f"cuda:{local_rank}",
            )
            recv_buf = torch.empty_like(send_buf)
            triton_nvl_sendrecv(
                send_buf,
                recv_buf,
                send_peer=peer,
                recv_peer=peer,
                group=group,
                num_blocks=16,
            )
            torch.cuda.synchronize()
            expected = torch.full_like(recv_buf, peer * 100 + len(paths))
            if not torch.equal(recv_buf, expected):
                print(
                    f"FAIL {name}/{stage_name}: rank={rank} signal_bytes={signal_bytes}"
                )
                return False
        if rank == 0:
            print(f"  PASS: {name}")
        return True
    finally:
        if prev is None:
            os.environ.pop("TRITON_NVL_TUNING_JSON", None)
        else:
            os.environ["TRITON_NVL_TUNING_JSON"] = prev
        _tc._JSON_CACHE = None
        _tc._CONFIG_CACHE.clear()
        for path in paths:
            os.unlink(path)


def _run_back_to_back(
    local_rank: int,
    group: dist.ProcessGroup,
    *,
    bidirectional: bool,
    graph: bool,
) -> bool:
    peer_rank: int = 1 - local_rank
    device = torch.device(f"cuda:{local_rank}")
    configs: list[tuple[int, int]] = [
        (64 * 1024, 2),
        (16 * 1024 * 1024, 32),
        (256 * 1024, 16),
    ]
    dtype = torch.int32
    sends: list[torch.Tensor] = []
    recvs: list[torch.Tensor] = []
    expected: list[torch.Tensor] = []
    for msg_bytes, _ in configs:
        numel = _numel(msg_bytes, dtype)
        sends.append(_make_pattern(local_rank, numel, dtype, device))
        recvs.append(torch.zeros(numel, dtype=dtype, device=device))
        expected.append(_make_pattern(peer_rank, numel, dtype, device))

    def sequence() -> None:
        for i, (_, num_blocks) in enumerate(configs):
            send = sends[i]
            recv = recvs[i]
            if bidirectional:
                triton_nvl_sendrecv(
                    send, recv, peer_rank, group=group, num_blocks=num_blocks
                )
            elif local_rank == 0:
                triton_nvl_send(send, peer_rank, group=group, num_blocks=num_blocks)
            else:
                triton_nvl_recv(recv, peer_rank, group=group, num_blocks=num_blocks)

    if graph:
        sequence()
        torch.cuda.synchronize(device)
        graph_obj = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph_obj):
            sequence()
        for _ in range(5):
            graph_obj.replay()
    else:
        sequence()
    torch.cuda.synchronize(device)

    if bidirectional or local_rank == 1:
        local_ok = all(
            _check_equal(actual, exp, "back_to_back", local_rank)
            for actual, exp in zip(recvs, expected)
        )
    else:
        local_ok = True

    label = (
        f"{'bidirectional' if bidirectional else 'unidirectional'}_"
        f"back_to_back_{'graph' if graph else 'eager'}"
    )
    ok = _all_ok(local_ok, device, group)
    if local_rank == 0:
        print(f"  {'PASS' if ok else 'FAIL'}: {label}")
    return ok


def _run_graph_case(
    local_rank: int,
    group: dist.ProcessGroup,
    *,
    bidirectional: bool,
    warp_specialized: bool = False,
    sender_warps: int = 4,
    receiver_warps: int = 4,
) -> bool:
    peer_rank: int = 1 - local_rank
    device = torch.device(f"cuda:{local_rank}")
    dtype = torch.int32
    numel = _numel(256 * 1024, dtype)
    send: torch.Tensor = _make_pattern(local_rank, numel, dtype, device)
    recv: torch.Tensor = torch.zeros(numel, dtype=dtype, device=device)
    expected = _make_pattern(peer_rank, numel, dtype, device)

    def fn() -> None:
        if warp_specialized:
            triton_nvl_sendrecv_ws(
                send,
                recv,
                peer_rank,
                group=group,
                sender_warps=sender_warps,
                receiver_warps=receiver_warps,
            )
        elif bidirectional:
            triton_nvl_sendrecv(send, recv, peer_rank, group=group)
        elif local_rank == 0:
            triton_nvl_send(send, peer_rank, group=group)
        else:
            triton_nvl_recv(recv, peer_rank, group=group)

    fn()
    torch.cuda.synchronize(device)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()

    local_ok = True
    for _ in range(5):
        recv.zero_()
        g.replay()
        torch.cuda.synchronize(device)
        if bidirectional or local_rank == 1:
            local_ok = local_ok and _check_equal(
                recv, expected, "cuda_graph", local_rank
            )

    label = f"{'bidirectional_ws' if warp_specialized else 'bidirectional' if bidirectional else 'unidirectional'}_cuda_graph"
    ok = _all_ok(local_ok, device, group)
    if local_rank == 0:
        print(f"  {'PASS' if ok else 'FAIL'}: {label}")
    return ok


def _record(
    results: list[bool], fn: Callable[[], bool], group: dist.ProcessGroup
) -> None:
    results.append(fn())
    dist.barrier(group)


def _record_base_cases(
    results: list[bool], local_rank: int, group: dist.ProcessGroup
) -> None:
    for name, msg_bytes in _BASE_CASES:
        _record(
            results,
            lambda name=name, msg_bytes=msg_bytes: _run_bidirectional(
                f"bidirectional_{name}_int32", msg_bytes, torch.int32, local_rank, group
            ),
            group,
        )


def _record_dtype_smoke_cases(
    results: list[bool], local_rank: int, group: dist.ProcessGroup
) -> None:
    for dtype in (torch.bfloat16, torch.float32):
        _record(
            results,
            lambda dtype=dtype: _run_bidirectional(
                f"bidirectional_64KB_{dtype}_smoke",
                64 * 1024,
                dtype,
                local_rank,
                group,
            ),
            group,
        )


def _record_ws_cases(
    results: list[bool], local_rank: int, group: dist.ProcessGroup
) -> None:
    if not _WS_TESTS_ENABLED:
        return

    for sw, rw in [(2, 2), (4, 4), (3, 5)]:
        for msg_label, msg_bytes in [("256KB", 256 * 1024), ("1MB", 1024 * 1024)]:
            _record(
                results,
                lambda sw=sw, rw=rw, mb=msg_bytes, ml=msg_label: _run_bidirectional(
                    f"bidirectional_ws_{sw}s{rw}r_{ml}_int32",
                    mb,
                    torch.int32,
                    local_rank,
                    group,
                    num_blocks=16,
                    warp_specialized=True,
                    sender_warps=sw,
                    receiver_warps=rw,
                ),
                group,
            )


def _record_subslot_cases(
    results: list[bool], local_rank: int, group: dist.ProcessGroup
) -> None:
    # Force `signal_bytes < _BLOCK_STRIDE_BYTES` so the kernel exercises
    # multiple DATA_READY signals per slot fill.
    _record(
        results,
        lambda: _run_with_signal_bytes_override(
            "bidirectional_subslot_chunks_per_slot_4_256KB_int32",
            64 * 1024,  # signal_bytes -> chunks_per_slot = 256K/64K = 4
            local_rank,
            group,
        ),
        group,
    )

    _record(
        results,
        lambda: _run_back_to_back_across_chunks_per_slot(
            "back_to_back_across_chunks_per_slot",
            local_rank,
            group,
        ),
        group,
    )


def _record_unidirectional_cases(
    results: list[bool], local_rank: int, group: dist.ProcessGroup
) -> None:
    for src_rank in (0, 1):
        _record(
            results,
            lambda src_rank=src_rank: _run_unidirectional(
                f"unidirectional_{src_rank}_to_{1 - src_rank}_1MB",
                1024 * 1024,
                torch.int32,
                src_rank,
                local_rank,
                group,
            ),
            group,
        )


def _record_config_cases(
    results: list[bool], local_rank: int, group: dist.ProcessGroup
) -> None:
    for num_blocks in (2, 16, 32):
        for num_warps in (4, 8):
            _record(
                results,
                lambda num_blocks=num_blocks, num_warps=num_warps: _run_bidirectional(
                    f"config_blocks_{num_blocks}_warps_{num_warps}",
                    256 * 1024,
                    torch.int32,
                    local_rank,
                    group,
                    num_blocks=num_blocks,
                    num_warps=num_warps,
                ),
                group,
            )


def _record_back_to_back_cases(
    results: list[bool], local_rank: int, group: dist.ProcessGroup
) -> None:
    for bidirectional in (False, True):
        for graph in (False, True):
            _record(
                results,
                lambda bidirectional=bidirectional, graph=graph: _run_back_to_back(
                    local_rank, group, bidirectional=bidirectional, graph=graph
                ),
                group,
            )


def _record_graph_cases(
    results: list[bool], local_rank: int, group: dist.ProcessGroup
) -> None:
    for bidirectional in (False, True):
        _record(
            results,
            lambda bidirectional=bidirectional: _run_graph_case(
                local_rank, group, bidirectional=bidirectional
            ),
            group,
        )

    if _WS_TESTS_ENABLED:
        _record(
            results,
            lambda: _run_graph_case(
                local_rank, group, bidirectional=True, warp_specialized=True
            ),
            group,
        )


def _worker(local_rank: int, master_port: int) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(local_rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = "2"

    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    group = dist.group.WORLD
    assert group is not None

    results: list[bool] = []
    _record_base_cases(results, local_rank, group)
    _record_dtype_smoke_cases(results, local_rank, group)
    _record_ws_cases(results, local_rank, group)
    _record_subslot_cases(results, local_rank, group)
    _record_unidirectional_cases(results, local_rank, group)
    _record_config_cases(results, local_rank, group)
    _record_back_to_back_cases(results, local_rank, group)
    _record_graph_cases(results, local_rank, group)

    if local_rank == 0:
        n_pass = sum(1 for ok in results if ok)
        print(f"\nResults: {n_pass}/{len(results)} passed")

    dist.barrier(group)
    dist.destroy_process_group()
    if local_rank == 0 and not all(results):
        sys.exit(1)


def _run_ring_sendrecv(
    name: str,
    msg_bytes: int,
    dtype: torch.dtype,
    local_rank: int,
    world_size: int,
    group: dist.ProcessGroup,
    *,
    num_blocks: int = 16,
) -> bool:
    """Ring topology: each rank sends to (rank+1)%N, receives from (rank-1)%N."""
    send_peer = (local_rank + 1) % world_size
    recv_peer = (local_rank - 1 + world_size) % world_size
    device = torch.device(f"cuda:{local_rank}")
    numel = _numel(msg_bytes, dtype)

    send = _make_pattern(local_rank, numel, dtype, device)
    recv = torch.zeros(numel, dtype=dtype, device=device)
    expected = _make_pattern(recv_peer, numel, dtype, device)

    triton_nvl_sendrecv(
        send,
        recv,
        send_peer,
        recv_peer,
        group=group,
        num_blocks=num_blocks,
    )
    torch.cuda.synchronize(device)

    ok = _all_ok(_check_equal(recv, expected, name, local_rank), device, group)
    if local_rank == 0:
        print(f"  {'PASS' if ok else 'FAIL'}: {name}")
    return ok


def _run_nrank_unidirectional(
    name: str,
    msg_bytes: int,
    dtype: torch.dtype,
    src_rank: int,
    dst_rank: int,
    local_rank: int,
    group: dist.ProcessGroup,
) -> bool:
    """Send from src_rank to dst_rank in an N-rank group."""
    device = torch.device(f"cuda:{local_rank}")
    numel = _numel(msg_bytes, dtype)

    if local_rank == src_rank:
        send = _make_pattern(local_rank, numel, dtype, device)
        triton_nvl_send(send, dst_rank, group=group)
        local_ok = True
    elif local_rank == dst_rank:
        recv = torch.zeros(numel, dtype=dtype, device=device)
        expected = _make_pattern(src_rank, numel, dtype, device)
        triton_nvl_recv(recv, src_rank, group=group)
        torch.cuda.synchronize(device)
        local_ok = _check_equal(recv, expected, name, local_rank)
    else:
        local_ok = True

    torch.cuda.synchronize(device)
    ok = _all_ok(local_ok, device, group)
    if local_rank == 0:
        print(f"  {'PASS' if ok else 'FAIL'}: {name}")
    return ok


def _run_ring_back_to_back(
    local_rank: int,
    world_size: int,
    group: dist.ProcessGroup,
) -> bool:
    """Back-to-back ring sendrecv calls to verify step state persistence."""
    send_peer = (local_rank + 1) % world_size
    recv_peer = (local_rank - 1 + world_size) % world_size
    device = torch.device(f"cuda:{local_rank}")
    dtype = torch.int32

    configs: list[tuple[int, int]] = [
        (64 * 1024, 4),
        (1024 * 1024, 16),
        (256 * 1024, 8),
    ]
    sends: list[torch.Tensor] = []
    recvs: list[torch.Tensor] = []
    expected: list[torch.Tensor] = []
    for msg_bytes, _ in configs:
        numel = _numel(msg_bytes, dtype)
        sends.append(_make_pattern(local_rank, numel, dtype, device))
        recvs.append(torch.zeros(numel, dtype=dtype, device=device))
        expected.append(_make_pattern(recv_peer, numel, dtype, device))

    for i, (_, num_blocks) in enumerate(configs):
        triton_nvl_sendrecv(
            sends[i],
            recvs[i],
            send_peer,
            recv_peer,
            group=group,
            num_blocks=num_blocks,
        )
    torch.cuda.synchronize(device)

    local_ok = all(
        _check_equal(actual, exp, "ring_back_to_back", local_rank)
        for actual, exp in zip(recvs, expected)
    )

    ok = _all_ok(local_ok, device, group)
    if local_rank == 0:
        print(f"  {'PASS' if ok else 'FAIL'}: ring_back_to_back")
    return ok


def _run_cross_peer_back_to_back(
    local_rank: int,
    world_size: int,
    group: dist.ProcessGroup,
) -> bool:
    """Back-to-back launches that change ``send_peer``/``recv_peer`` across
    calls. Verifies that the ``(world_size, MBPP)`` step-state matrix is
    indexed by peer-row correctly — i.e. progress against peer A does not
    leak into the slot ledger for peer B.

    Each rank pairs with two distinct peers in sequence:
      * pass 1: send→(rank+1)%N, recv←(rank-1+N)%N    (forward ring)
      * pass 2: send→(rank-1+N)%N, recv←(rank+1)%N    (reverse ring)
    """
    if world_size < 3:
        return True

    fwd_send = (local_rank + 1) % world_size
    fwd_recv = (local_rank - 1 + world_size) % world_size
    rev_send = fwd_recv
    rev_recv = fwd_send

    device = torch.device(f"cuda:{local_rank}")
    dtype = torch.int32
    numel = _numel(256 * 1024, dtype)

    # Two passes share send tensors but use independent recv buffers so
    # we can verify each peer's payload landed in the right place.
    send_pass1 = _make_pattern(local_rank, numel, dtype, device)
    send_pass2 = _make_pattern(local_rank + 1000, numel, dtype, device)
    recv_pass1 = torch.zeros(numel, dtype=dtype, device=device)
    recv_pass2 = torch.zeros(numel, dtype=dtype, device=device)

    expected_pass1 = _make_pattern(fwd_recv, numel, dtype, device)
    expected_pass2 = _make_pattern(rev_recv + 1000, numel, dtype, device)

    triton_nvl_sendrecv(
        send_pass1, recv_pass1, fwd_send, fwd_recv, group=group, num_blocks=8
    )
    triton_nvl_sendrecv(
        send_pass2, recv_pass2, rev_send, rev_recv, group=group, num_blocks=8
    )
    torch.cuda.synchronize(device)

    local_ok = _check_equal(
        recv_pass1, expected_pass1, "cross_peer_pass1", local_rank
    ) and _check_equal(recv_pass2, expected_pass2, "cross_peer_pass2", local_rank)

    ok = _all_ok(local_ok, device, group)
    if local_rank == 0:
        print(f"  {'PASS' if ok else 'FAIL'}: cross_peer_back_to_back")
    return ok


def _nrank_worker(local_rank: int, world_size: int, master_port: int) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(local_rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    group = dist.group.WORLD
    assert group is not None

    results: list[bool] = []

    for name, msg_bytes in _BASE_CASES:
        _record(
            results,
            lambda name=name, msg_bytes=msg_bytes: _run_ring_sendrecv(
                f"ring_{name}_int32",
                msg_bytes,
                torch.int32,
                local_rank,
                world_size,
                group,
            ),
            group,
        )

    for dtype in (torch.bfloat16, torch.float32):
        _record(
            results,
            lambda dtype=dtype: _run_ring_sendrecv(
                f"ring_64KB_{dtype}_smoke",
                64 * 1024,
                dtype,
                local_rank,
                world_size,
                group,
            ),
            group,
        )

    _record(
        results,
        lambda: _run_nrank_unidirectional(
            "nrank_send_0_to_1",
            1024 * 1024,
            torch.int32,
            0,
            1,
            local_rank,
            group,
        ),
        group,
    )

    if world_size > 2:
        _record(
            results,
            lambda: _run_nrank_unidirectional(
                f"nrank_send_0_to_{world_size - 1}",
                1024 * 1024,
                torch.int32,
                0,
                world_size - 1,
                local_rank,
                group,
            ),
            group,
        )

    _record(
        results,
        lambda: _run_ring_back_to_back(local_rank, world_size, group),
        group,
    )

    _record(
        results,
        lambda: _run_cross_peer_back_to_back(local_rank, world_size, group),
        group,
    )

    if local_rank == 0:
        n_pass = sum(1 for ok in results if ok)
        print(f"\nN-rank Results ({world_size} GPUs): {n_pass}/{len(results)} passed")

    dist.barrier(group)
    dist.destroy_process_group()
    if local_rank == 0 and not all(results):
        sys.exit(1)


def main() -> None:
    port = _find_free_port()
    num_gpus = torch.cuda.device_count()

    print("Running Triton NVLink SendRecv Correctness Tests")
    print("=" * 52)

    print("\n--- 2-rank tests ---")
    mp.spawn(_worker, args=(port,), nprocs=2, join=True)

    if num_gpus >= 4:
        port = _find_free_port()
        nprocs = min(num_gpus, 8)
        print(f"\n--- {nprocs}-rank tests (ring topology) ---")
        mp.spawn(_nrank_worker, args=(nprocs, port), nprocs=nprocs, join=True)
    else:
        print(f"\nSkipping N-rank tests: need >= 4 GPUs, have {num_gpus}")


if __name__ == "__main__":
    main()
