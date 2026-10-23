# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-unsafe

"""Distributed correctness for the pipelined CuTe send/recv credit ring.

The load-bearing case is the free-credit (HEAD) protocol. ``pipelined_sendrecv``
reuses one per-peer staging region every call (the region is ``num_slots`` slots,
rewritten each call), so a faster rank must not overwrite a slot before its peer
has finished draining the PREVIOUS call's payload from it. The monotonic TAIL
sequence counters alone do not prevent that; ``_send_slot`` gates each slot on the
peer's HEAD credit (``nvl_ops.wait_free``) and ``_recv_slot`` publishes it
(``nvl_ops.signal_free``) after draining.

``test_credit_ring_reuse`` exercises this with a deliberately rank-skewed replay.
The stress case is UNIDIRECTIONAL on purpose: in a symmetric bidir run each rank's
send rate is throttled by its own recv wait, so the ring rarely wraps and the HEAD
gate stays trivially satisfied. A send-only rank has no such throttle -- when its
recv-only peer is stalled on-device (``torch.cuda._sleep``), the sender races ahead
across the reused ring with a DIFFERENT payload every call. Without the HEAD gate it
clobbers staging the slow rank has not yet read and the slow rank's outputs come
back corrupted; with it, back-pressure keeps every call byte-exact.
``A2A_CUTE_SLOTS=4`` forces a 4-slot ring on a small buffer so the ring wraps
(``K`` calls, ``K > num_slots``) without needing a multi-MB transfer.
"""

import os
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from ._dist_harness import _find_free_port, _report

# fp32 so the packed (call, rank, idx) pattern below is bit-exact (values stay well
# under 2**24). num_blocks=1 keeps one signal slot per peer; K wraps the 4-slot ring.
_NUMEL: int = 1 << 16
_NUM_BLOCKS: int = 1
_NUM_SLOTS: int = 4
_NUM_CALLS: int = 12
_ELEM: int = 4
# ~0.15s at H100 clocks: long enough that the un-skewed sender issues all _NUM_CALLS
# sends (wrapping the 4-slot ring several times) while the receiver is still stalled.
_SKEW_CYCLES: int = 200_000_000


def _payload(rank: int, call: int, numel: int, device: torch.device) -> torch.Tensor:
    """Bit-exact per-(rank, call, index) fp32 pattern; distinct across every call."""
    idx = torch.arange(numel, device=device, dtype=torch.int64)
    return (idx + numel * (rank + 2 * call)).to(torch.float32)


def _run_bidir(local_rank, group, transport, label) -> bool:
    """Baseline: back-to-back bidir replay, no skew (both ranks keep pace)."""
    from comms.dsl.cute.send_recv import pipelined_sendrecv

    peer = 1 - local_rank
    device = torch.device(f"cuda:{local_rank}")
    sends = [_payload(local_rank, i, _NUMEL, device) for i in range(_NUM_CALLS)]
    recvs = [
        torch.zeros(_NUMEL, dtype=torch.float32, device=device)
        for _ in range(_NUM_CALLS)
    ]
    expected = [_payload(peer, i, _NUMEL, device) for i in range(_NUM_CALLS)]

    for i in range(_NUM_CALLS):
        pipelined_sendrecv(
            transport,
            sends[i],
            recvs[i],
            peer,
            peer,
            num_blocks=_NUM_BLOCKS,
            mode="bidir",
        )
    torch.cuda.synchronize(device)

    local_ok = all(torch.equal(recvs[i], expected[i]) for i in range(_NUM_CALLS))
    return _report(label, local_ok, device, group, local_rank)


def _run_unidir_skewed(local_rank, group, transport, sender, label) -> bool:
    """Free-credit stress: ``sender`` sends back-to-back while its recv-only peer is
    stalled on-device, so the sender races across the reused ring. Only the HEAD gate
    stops the racing sender from overwriting staging the slow receiver has not drained."""
    from comms.dsl.cute.send_recv import pipelined_sendrecv

    peer = 1 - local_rank
    device = torch.device(f"cuda:{local_rank}")

    if local_rank == sender:
        bufs = [_payload(local_rank, i, _NUMEL, device) for i in range(_NUM_CALLS)]
        for i in range(_NUM_CALLS):
            pipelined_sendrecv(
                transport,
                bufs[i],
                None,
                peer,
                peer,
                num_blocks=_NUM_BLOCKS,
                mode="send",
            )
        torch.cuda.synchronize(device)
        # Sender has nothing to verify; correctness shows up in the receiver's outputs.
        return _report(label, True, device, group, local_rank)

    recvs = [
        torch.zeros(_NUMEL, dtype=torch.float32, device=device)
        for _ in range(_NUM_CALLS)
    ]
    expected = [_payload(sender, i, _NUMEL, device) for i in range(_NUM_CALLS)]
    # Stall the receiver so the sender runs ahead across the reused staging ring.
    torch.cuda._sleep(_SKEW_CYCLES)
    for i in range(_NUM_CALLS):
        pipelined_sendrecv(
            transport, None, recvs[i], peer, peer, num_blocks=_NUM_BLOCKS, mode="recv"
        )
    torch.cuda.synchronize(device)

    local_ok = all(torch.equal(recvs[i], expected[i]) for i in range(_NUM_CALLS))
    return _report(label, local_ok, device, group, local_rank)


def _worker(local_rank: int, world_size: int, master_port: int) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(local_rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    # Force a multi-slot ring on the small test buffer so the ring wraps across calls
    # (a real transfer only pipelines above _MIN_PIPELINE_CHUNK_BYTES).
    os.environ["A2A_CUTE_SLOTS"] = str(_NUM_SLOTS)

    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    group = dist.group.WORLD
    assert group is not None

    from comms.dsl import nvl_rendezvous

    # One transport reused across every case (its counters/staging are designed for reuse);
    # reuse is precisely what the free-credit protocol must make race-free.
    transport = nvl_rendezvous(
        group, device=torch.device(f"cuda:{local_rank}"), per_peer_bytes=_NUMEL * _ELEM
    )

    results: list[bool] = []
    results.append(_run_bidir(local_rank, group, transport, "bidir_replay_no_skew"))
    dist.barrier(group)
    # Rank-skewed unidirectional replay in each direction: the recv-only rank stalls
    # while the send-only rank races ahead across the reused ring.
    for sender in (0, 1):
        results.append(
            _run_unidir_skewed(
                local_rank,
                group,
                transport,
                sender,
                f"unidir_skewed_{sender}_to_{1 - sender}",
            )
        )
        dist.barrier(group)

    dist.barrier(group)
    dist.destroy_process_group()
    # Assert AFTER cleanup so a failure does not leave the peer wedged in a collective.
    assert all(results), f"rank {local_rank} failures: {results}"


class PipelinedSendRecvTest(unittest.TestCase):
    @unittest.skipUnless(
        torch.cuda.is_available() and torch.cuda.device_count() >= 2,
        "pipelined send/recv credit-ring test needs >= 2 GPUs",
    )
    def test_credit_ring_reuse(self) -> None:
        port = _find_free_port()
        mp.spawn(_worker, args=(2, port), nprocs=2, join=True)


if __name__ == "__main__":
    unittest.main()
