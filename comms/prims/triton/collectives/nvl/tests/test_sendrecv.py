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
from comms.prims.triton.collectives.nvl.sendrecv_op import (
    triton_nvl_recv,
    triton_nvl_send,
    triton_nvl_sendrecv,
)


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
) -> bool:
    peer_rank = 1 - local_rank
    device = torch.device(f"cuda:{local_rank}")
    numel = _numel(msg_bytes, dtype)
    send = _make_pattern(local_rank, numel, dtype, device)
    recv = torch.zeros(numel, dtype=dtype, device=device)
    expected = _make_pattern(peer_rank, numel, dtype, device)

    triton_nvl_sendrecv(
        send, recv, peer_rank, group=group, num_blocks=num_blocks, num_warps=num_warps
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
) -> bool:
    peer_rank: int = 1 - local_rank
    device = torch.device(f"cuda:{local_rank}")
    dtype = torch.int32
    numel = _numel(256 * 1024, dtype)
    send: torch.Tensor = _make_pattern(local_rank, numel, dtype, device)
    recv: torch.Tensor = torch.zeros(numel, dtype=dtype, device=device)
    expected = _make_pattern(peer_rank, numel, dtype, device)

    def fn() -> None:
        if bidirectional:
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

    label = f"{'bidirectional' if bidirectional else 'unidirectional'}_cuda_graph"
    ok = _all_ok(local_ok, device, group)
    if local_rank == 0:
        print(f"  {'PASS' if ok else 'FAIL'}: {label}")
    return ok


def _record(
    results: list[bool], fn: Callable[[], bool], group: dist.ProcessGroup
) -> None:
    results.append(fn())
    dist.barrier(group)


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
    for name, msg_bytes in _BASE_CASES:
        _record(
            results,
            lambda name=name, msg_bytes=msg_bytes: _run_bidirectional(
                f"bidirectional_{name}_int32", msg_bytes, torch.int32, local_rank, group
            ),
            group,
        )

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

    for bidirectional in (False, True):
        for graph in (False, True):
            _record(
                results,
                lambda bidirectional=bidirectional, graph=graph: _run_back_to_back(
                    local_rank, group, bidirectional=bidirectional, graph=graph
                ),
                group,
            )

    for bidirectional in (False, True):
        _record(
            results,
            lambda bidirectional=bidirectional: _run_graph_case(
                local_rank, group, bidirectional=bidirectional
            ),
            group,
        )

    if local_rank == 0:
        n_pass = sum(1 for ok in results if ok)
        print(f"\nResults: {n_pass}/{len(results)} passed")

    dist.barrier(group)
    dist.destroy_process_group()
    if local_rank == 0 and not all(results):
        sys.exit(1)


def main() -> None:
    port = _find_free_port()
    print("Running Triton NVLink SendRecv Correctness Tests")
    print("=" * 52)
    mp.spawn(_worker, args=(port,), nprocs=2, join=True)


if __name__ == "__main__":
    main()
