# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-unsafe

"""Shared multi-rank scaffolding for the distributed correctness suites.

The small helpers every distributed test repeats: a free-port picker, the bit-exact
input builder, the ``dist.all_to_all_single`` golden, the cross-rank PASS/FAIL
min-reduce + reporter, and the transport rendezvous. Kept here so the scaffolding
lives in one place; a test with a special need (e.g. multi-dtype bit-exact-per-dtype
inputs) keeps its own local variant.
"""

import socket

import torch
import torch.distributed as dist

_FP32_BYTES = 4


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _make_input(rank: int, numel: int, device: torch.device) -> torch.Tensor:
    idx = torch.arange(numel, device=device, dtype=torch.int64)
    return ((rank + 1) * 1_000_000 + idx).to(torch.float32)


def _golden(group: dist.ProcessGroup, inp: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(inp)
    dist.all_to_all_single(out, inp, group=group)
    return out


def _all_ok(local_ok: bool, device: torch.device, group: dist.ProcessGroup) -> bool:
    status = torch.tensor([1 if local_ok else 0], dtype=torch.int32, device=device)
    dist.all_reduce(status, op=dist.ReduceOp.MIN, group=group)
    return bool(status.item())


def _report(name: str, local_ok: bool, device, group, rank: int) -> bool:
    ok = _all_ok(local_ok, device, group)
    if rank == 0:
        print(f"  {'PASS' if ok else 'FAIL'}: {name}", flush=True)
    return ok


def _rendezvous(group, device, max_chunk_numel: int, *, max_blocks_per_peer: int = 32):
    from comms.dsl import nvl_rendezvous

    return nvl_rendezvous(
        group,
        device,
        per_peer_bytes=max_chunk_numel * _FP32_BYTES,
        max_blocks_per_peer=max_blocks_per_peer,
    )
