# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-unsafe

"""2-rank smoke tests: the whole interface composes end-to-end on the minimal path.

Skipped unless >=2 GPUs are present, so it is a no-op in GPU-less CI. On a 2-GPU
host each case does a real NVLink transfer through the framework (transport + ops
+ hook + send/recv) and checks correctness. Each case re-rendezvous because the
minimal path is single-shot (seq=1).
"""

import os
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

_FP32_BYTES = 4


def _directional(
    group: dist.ProcessGroup,
    rank: int,
    device: torch.device,
    numel: int,
    num_blocks: int,
) -> None:
    """rank 0 sends ``arange`` to rank 1; rank 1 verifies."""
    from comms.dsl import nvl_rendezvous
    from comms.dsl.triton import recv, send

    t = nvl_rendezvous(group, device, per_peer_bytes=numel * _FP32_BYTES)
    if rank == 0:
        send_buf = torch.arange(numel, dtype=torch.float32, device=device)
        send(t, send_buf, peer=1, num_blocks=num_blocks)
    else:
        recv_buf = torch.empty(numel, dtype=torch.float32, device=device)
        recv(t, recv_buf, peer=0, num_blocks=num_blocks)
        torch.cuda.synchronize()
        expected = torch.arange(numel, dtype=torch.float32, device=device)
        assert torch.equal(recv_buf, expected), (
            f"directional mismatch numel={numel} num_blocks={num_blocks}"
        )
    dist.barrier()


def _bidirectional(
    group: dist.ProcessGroup,
    rank: int,
    device: torch.device,
    numel: int,
    num_blocks: int,
) -> None:
    """Both ranks exchange distinct data with each other simultaneously."""
    from comms.dsl import nvl_rendezvous
    from comms.dsl.triton import sendrecv

    t = nvl_rendezvous(group, device, per_peer_bytes=numel * _FP32_BYTES)
    peer = 1 - rank
    base = (rank + 1) * 1000

    def pattern(r: int) -> torch.Tensor:
        return (r + 1) * 1000 + torch.arange(numel, dtype=torch.float32, device=device)

    send_buf = pattern(rank)
    recv_buf = torch.empty(numel, dtype=torch.float32, device=device)
    sendrecv(t, send_buf, recv_buf, send_peer=peer, num_blocks=num_blocks)
    torch.cuda.synchronize()
    assert torch.equal(recv_buf, pattern(peer)), (
        f"bidirectional mismatch on rank {rank} (base={base})"
    )
    dist.barrier()


def _worker(rank: int, world_size: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    device = torch.device(f"cuda:{rank}")
    group = dist.group.WORLD
    assert group is not None

    _directional(group, rank, device, numel=8192, num_blocks=8)  # divisible
    _directional(group, rank, device, numel=8191, num_blocks=1)  # tail mask + boundary
    _bidirectional(group, rank, device, numel=4096, num_blocks=8)

    dist.barrier()
    dist.destroy_process_group()


class MinimalSendRecvTest(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.device_count() >= 2, "needs >=2 GPUs")
    def test_2rank_paths(self) -> None:
        mp.spawn(_worker, args=(2, 12355), nprocs=2, join=True)
