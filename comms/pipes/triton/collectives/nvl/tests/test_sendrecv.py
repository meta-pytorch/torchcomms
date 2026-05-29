# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

"""Correctness tests for NVLink copy-based Triton send/recv."""

from __future__ import annotations

import os
import socket
import sys
from collections.abc import Callable
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import torch.multiprocessing as mp
from comms.pipes.triton.collectives.nvl.sendrecv_op import (
    triton_nvl_recv,
    triton_nvl_send,
    triton_nvl_sendrecv,
    triton_nvl_sendrecv_prepare,
    triton_nvl_sendrecv_ws,
)


# ---------------------------------------------------------------------------
# Warp-specialized (WS) test gate (ON by default).
#
# The WS path uses TLX `async_tasks` / `async_task` warp-specialization.
# Earlier this kernel hung after >2 while-loop iterations (blamed on an
# upstream `tlx.sync_threads()` codegen bug) and intermittently corrupted
# data. The real cause was a raw inline-asm `bar.sync 0` (`sync_threads()`)
# inside the async_tasks: it is invisible to the TLX warp-spec lowering, so
# it stayed a full-CTA barrier across the two independent tasks (deadlock at
# higher iteration counts) and its hardcoded size corrupted data at small
# warp counts. The kernel now uses `tl.debug_barrier()`, which the lowering
# rewrites into a per-task, correctly-sized named barrier; combined with an
# all-warps credit poll this makes the path deterministically correct and
# hang-free at ANY iteration count (validated to 1 GiB / ~128 iters).
#
# The test binary pins `//triton:triton` directly (see tests/BUCK) so it uses
# the real TLX runtime, not torch's bundled shim. The sweep needs multi-GPU
# NVLink hardware, like the rest of this suite. Set
# `TRITON_NVL_WS_TESTS_ENABLED=0` to skip it.
# ---------------------------------------------------------------------------
_WS_TESTS_ENABLED: bool = os.environ.get("TRITON_NVL_WS_TESTS_ENABLED", "1") == "1"


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

    # TLX warp specialization requires each partition's warp count AND their
    # sum to be powers of two, which forces sender_warps == receiver_warps
    # (drawn from {2, 4, 8}). Non-pow2 splits like (3, 5) crash the
    # TritonTLXFixup pass; the host launcher now rejects them up front.
    for sw, rw in [(2, 2), (4, 4), (8, 8)]:
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

    # ---- Regression: asymmetric per-block tiles across the iteration range ----
    # Locks in the barrier fix that the WS path needed: the kernel used a raw
    # inline-asm `bar.sync 0` (invisible to the TLX warp-spec lowering) which
    # stayed full-CTA across the two async_tasks and caused both a data race
    # and an iteration-count-dependent hang. Switching to tl.debug_barrier()
    # (compiler-rewritten to a per-task, correctly-sized named barrier) fixes
    # both, so the kernel is now correct at ANY iteration count.
    #
    # Two sizes, both with non-tile-multiple numel (partial-tile mask) and
    # num_tiles not divisible by num_blocks (asymmetric per-block tiles):
    #   * 5 MB  (int32) -> 77 tiles  -> ~2 iters/block (boundary of old guard)
    #   * 64 MB (int32) -> ~1024 tiles -> ~8 iters/block (formerly hung)
    for label, msg_bytes in [("5MB", 5 * 1000 * 1000), ("64MB", 64 * 1000 * 1000)]:
        for sw, rw in [(4, 4), (8, 8)]:
            _record(
                results,
                lambda sw=sw, rw=rw, mb=msg_bytes, lbl=label: _run_bidirectional(
                    f"bidirectional_ws_{sw}s{rw}r_{lbl}_asym_int32",
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
    num_warps: int = 8,
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
        num_warps=num_warps,
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


def _run_ring_back_to_back_pattern(
    name: str,
    configs: list[tuple[int, int]],
    local_rank: int,
    world_size: int,
    group: dist.ProcessGroup,
) -> bool:
    """Back-to-back ring sendrecv calls to verify step state persistence."""
    send_peer = (local_rank + 1) % world_size
    recv_peer = (local_rank - 1 + world_size) % world_size
    device = torch.device(f"cuda:{local_rank}")
    dtype = torch.int32

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
        _check_equal(actual, exp, name, local_rank)
        for actual, exp in zip(recvs, expected)
    )

    ok = _all_ok(local_ok, device, group)
    if local_rank == 0:
        print(f"  {'PASS' if ok else 'FAIL'}: {name}")
    return ok


def _run_ring_back_to_back(
    local_rank: int,
    world_size: int,
    group: dist.ProcessGroup,
) -> bool:
    return _run_ring_back_to_back_pattern(
        "ring_back_to_back",
        [
            (64 * 1024, 4),
            (1024 * 1024, 16),
            (256 * 1024, 8),
        ],
        local_rank,
        world_size,
        group,
    )


def _run_ring_back_to_back_large_small(
    local_rank: int,
    world_size: int,
    group: dist.ProcessGroup,
) -> bool:
    return _run_ring_back_to_back_pattern(
        "ring_back_to_back_large_small",
        [
            (16 * 1024 * 1024, 32),
            (64 * 1024, 4),
            (16 * 1024 * 1024, 32),
        ],
        local_rank,
        world_size,
        group,
    )


def _run_ring_graph_case(
    local_rank: int,
    world_size: int,
    group: dist.ProcessGroup,
) -> bool:
    """Capture and replay forward/reverse N-rank ring steps in a CUDA graph.

    The two passes use distinct ``num_blocks`` (16, 4) so the captured graph
    exercises per-(peer, block) step-state persistence under varying grids,
    not just the eager path.
    """
    if world_size < 3:
        return True

    fwd_send: int = (local_rank + 1) % world_size
    fwd_recv: int = (local_rank - 1 + world_size) % world_size
    rev_send: int = fwd_recv
    rev_recv: int = fwd_send

    device = torch.device(f"cuda:{local_rank}")
    dtype = torch.int32
    numel = _numel(256 * 1024, dtype)

    send_pass1: torch.Tensor = _make_pattern(local_rank, numel, dtype, device)
    send_pass2: torch.Tensor = _make_pattern(local_rank + 1000, numel, dtype, device)
    recv_pass1: torch.Tensor = torch.zeros(numel, dtype=dtype, device=device)
    recv_pass2: torch.Tensor = torch.zeros(numel, dtype=dtype, device=device)
    expected_pass1 = _make_pattern(fwd_recv, numel, dtype, device)
    expected_pass2 = _make_pattern(rev_recv + 1000, numel, dtype, device)

    def sequence() -> None:
        triton_nvl_sendrecv(
            send_pass1, recv_pass1, fwd_send, fwd_recv, group=group, num_blocks=16
        )
        triton_nvl_sendrecv(
            send_pass2, recv_pass2, rev_send, rev_recv, group=group, num_blocks=4
        )

    sequence()
    torch.cuda.synchronize(device)

    graph_obj = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph_obj):
        sequence()

    local_ok = True
    for _ in range(5):
        recv_pass1.zero_()
        recv_pass2.zero_()
        graph_obj.replay()
        torch.cuda.synchronize(device)
        local_ok = local_ok and _check_equal(
            recv_pass1, expected_pass1, "ring_cuda_graph_pass1", local_rank
        )
        local_ok = local_ok and _check_equal(
            recv_pass2, expected_pass2, "ring_cuda_graph_pass2", local_rank
        )

    ok = _all_ok(local_ok, device, group)
    if local_rank == 0:
        print(f"  {'PASS' if ok else 'FAIL'}: ring_cuda_graph")
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
    device = torch.device(f"cuda:{local_rank}")

    triton_nvl_sendrecv_prepare(group=group, device=device)
    dist.barrier(group)

    # Run sparse P2P before any ring sendrecv warms the launch path. The
    # explicit prepare above covers the group-wide symmetric-memory rendezvous.
    _record(
        results,
        lambda: _run_nrank_unidirectional(
            "nrank_sparse_prepared_send_0_to_1",
            1024 * 1024,
            torch.int32,
            0,
            1,
            local_rank,
            group,
        ),
        group,
    )

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

    # num_warps=4 variant: the 2-rank suite sweeps warp counts but the N-rank
    # cases otherwise all run the default 8 warps. Cover the 4-warp config so
    # warp-scheduling-dependent barrier bugs surface on the ring path too.
    _record(
        results,
        lambda: _run_ring_sendrecv(
            "ring_1MB_int32_warps4",
            1024 * 1024,
            torch.int32,
            local_rank,
            world_size,
            group,
            num_warps=4,
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
        lambda: _run_ring_back_to_back_large_small(local_rank, world_size, group),
        group,
    )

    _record(
        results,
        lambda: _run_ring_graph_case(local_rank, world_size, group),
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


def _signal_pad_failure_worker(local_rank: int, master_port: int) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(local_rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = "2"

    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    group = dist.group.WORLD
    assert group is not None

    device = torch.device(f"cuda:{local_rank}")
    local_ok = False
    try:
        with patch.object(symm_mem, "get_signal_pad_size", return_value=0):
            triton_nvl_sendrecv_prepare(group=group, device=device)
    except RuntimeError as error:
        message = str(error)
        local_ok = (
            "Signal pad too small" in message
            and "Call symm_mem.set_signal_pad_size() before first allocation" in message
        )

    ok = _all_ok(local_ok, device, group)
    if local_rank == 0:
        print(f"  {'PASS' if ok else 'FAIL'}: signal_pad_first_allocation_failure")
    dist.destroy_process_group()
    if local_rank == 0 and not ok:
        sys.exit(1)


def main() -> None:
    port = _find_free_port()
    num_gpus = torch.cuda.device_count()

    print("Running Triton NVLink SendRecv Correctness Tests")
    print("=" * 52)

    print("\n--- signal-pad first-allocation failure test ---")
    mp.spawn(_signal_pad_failure_worker, args=(port,), nprocs=2, join=True)

    port = _find_free_port()
    print("\n--- 2-rank tests ---")
    mp.spawn(_worker, args=(port,), nprocs=2, join=True)

    if num_gpus >= 3:
        port = _find_free_port()
        print("\n--- 3-rank tests (minimal asymmetric topology) ---")
        mp.spawn(_nrank_worker, args=(3, port), nprocs=3, join=True)
    else:
        print(f"\nSkipping 3-rank tests: need >= 3 GPUs, have {num_gpus}")

    if num_gpus >= 4:
        port = _find_free_port()
        nprocs = min(num_gpus, 8)
        print(f"\n--- {nprocs}-rank tests (ring topology) ---")
        mp.spawn(_nrank_worker, args=(nprocs, port), nprocs=nprocs, join=True)
    else:
        print(f"\nSkipping N-rank tests: need >= 4 GPUs, have {num_gpus}")


if __name__ == "__main__":
    main()
