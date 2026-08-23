# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

"""Shared multi-rank scaffolding for the distributed correctness suites.

The small helpers every distributed test repeats: a free-port picker, the bit-exact
input builder, and the cross-rank PASS/FAIL min-reduce + reporter. Kept here so the
scaffolding lives in one place.
"""

import socket

import torch
import torch.distributed as dist


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _make_input(rank: int, numel: int, device: torch.device) -> torch.Tensor:
    idx = torch.arange(numel, device=device, dtype=torch.int64)
    return ((rank + 1) * 1_000_000 + idx).to(torch.float32)


def _all_ok(local_ok: bool, device: torch.device, group: dist.ProcessGroup) -> bool:
    status = torch.tensor([1 if local_ok else 0], dtype=torch.int32, device=device)
    dist.all_reduce(status, op=dist.ReduceOp.MIN, group=group)
    return bool(status.item())


def _report(name: str, local_ok: bool, device, group, rank: int) -> bool:
    ok = _all_ok(local_ok, device, group)
    if rank == 0:
        print(f"  {'PASS' if ok else 'FAIL'}: {name}", flush=True)
    return ok
