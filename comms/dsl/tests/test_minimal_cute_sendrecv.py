# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-unsafe

"""2-rank smoke test for the CuTe backend.

Validates the DSL-agnostic design: the CuTe ``send``/``recv`` runs over the
*same* ``nvl_rendezvous`` transport + signal protocol as the Triton backend.
Skipped unless >=2 GPUs are present (no-op in GPU-less CI). Sizes are multiples
of the CTA thread count (the minimal CuTe path has no tail handling).
"""

import os
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

_FP32_BYTES = 4


def _directional(rank: int, device: torch.device, numel: int, num_blocks: int) -> None:
    """rank 0 sends ``arange`` to rank 1 via the CuTe backend; rank 1 verifies."""
    from comms.dsl import nvl_rendezvous
    from comms.dsl.cute import recv, send

    pg = dist.group.WORLD
    assert pg is not None
    t = nvl_rendezvous(pg, device, per_peer_bytes=numel * _FP32_BYTES)
    if rank == 0:
        send_buf = torch.arange(numel, dtype=torch.float32, device=device)
        send(t, send_buf, peer=1, num_blocks=num_blocks)
    else:
        recv_buf = torch.empty(numel, dtype=torch.float32, device=device)
        recv(t, recv_buf, peer=0, num_blocks=num_blocks)
        torch.cuda.synchronize()
        expected = torch.arange(numel, dtype=torch.float32, device=device)
        assert torch.equal(recv_buf, expected), (
            f"cute directional mismatch numel={numel} num_blocks={num_blocks}"
        )
    dist.barrier()


def _bidirectional(
    rank: int, device: torch.device, numel: int, num_blocks: int
) -> None:
    """Both ranks exchange distinct data simultaneously via the CuTe backend."""
    from comms.dsl import nvl_rendezvous
    from comms.dsl.cute import sendrecv

    pg = dist.group.WORLD
    assert pg is not None
    t = nvl_rendezvous(pg, device, per_peer_bytes=numel * _FP32_BYTES)
    peer = 1 - rank

    def pattern(r: int) -> torch.Tensor:
        return (r + 1) * 1000 + torch.arange(numel, dtype=torch.float32, device=device)

    send_buf = pattern(rank)
    recv_buf = torch.empty(numel, dtype=torch.float32, device=device)
    sendrecv(t, send_buf, recv_buf, send_peer=peer, num_blocks=num_blocks)
    torch.cuda.synchronize()
    assert torch.equal(recv_buf, pattern(peer)), (
        f"cute bidirectional mismatch on rank {rank}"
    )
    dist.barrier()


def _hooks(rank: int, device: torch.device, numel: int, num_blocks: int) -> None:
    """Prove the CuTe elementwise transform hook runs during the transfer."""
    from comms.dsl import nvl_rendezvous
    from comms.dsl.cute import recv, send
    from comms.dsl.cute.hooks import addone_consume, scale2_produce

    src = torch.arange(numel, dtype=torch.float32, device=device)

    pg = dist.group.WORLD
    assert pg is not None

    # transform on send (scale by 2): recv == 2 * src
    t = nvl_rendezvous(pg, device, per_peer_bytes=numel * _FP32_BYTES)
    if rank == 0:
        send(t, src.clone(), peer=1, produce=scale2_produce, num_blocks=num_blocks)
    else:
        out = torch.empty(numel, dtype=torch.float32, device=device)
        recv(t, out, peer=0, num_blocks=num_blocks)
        torch.cuda.synchronize()
        assert torch.equal(out, src * 2), (
            "cute send-side transform (scale2) not applied"
        )
    dist.barrier()

    # transform on recv (add 1): recv == src + 1
    t2 = nvl_rendezvous(pg, device, per_peer_bytes=numel * _FP32_BYTES)
    if rank == 0:
        send(t2, src.clone(), peer=1, num_blocks=num_blocks)
    else:
        out = torch.empty(numel, dtype=torch.float32, device=device)
        recv(t2, out, peer=0, consume=addone_consume, num_blocks=num_blocks)
        torch.cuda.synchronize()
        assert torch.equal(out, src + 1), (
            "cute recv-side transform (addone) not applied"
        )
    dist.barrier()


def _worker(rank: int, world_size: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    device = torch.device(f"cuda:{rank}")

    # CuTe minimal path requires numel divisible by the CTA thread count (128).
    _directional(rank, device, numel=8192, num_blocks=8)
    _bidirectional(rank, device, numel=4096, num_blocks=8)
    _hooks(rank, device, numel=8192, num_blocks=8)

    dist.barrier()
    dist.destroy_process_group()


class MinimalCuteSendRecvTest(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.device_count() >= 2, "needs >=2 GPUs")
    def test_2rank_paths(self) -> None:
        mp.spawn(_worker, args=(2, 12356), nprocs=2, join=True)
