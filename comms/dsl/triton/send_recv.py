# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-unsafe

"""Minimal device send_tiles/recv_tiles primitives (Triton).

The composable unit: each call moves one whole peer transfer. The primitive owns
the per-tile loop and the signaling boundary and calls the user's full-leg hook
(``produce`` / ``consume``) per tile. The transport ops (``put`` /
``get`` / ``signal`` / ``wait``) and the hook are passed as
``tl.constexpr`` so the *same* ``send_tiles``/``recv_tiles`` serve NVLink today, IB later,
and mixed-transport kernels (different ops per call site).

Minimal scope (correctness only; no performance):

* **No pipeline** — no slots, no double-buffering, no credit/backpressure. Each
  block copies a contiguous chunk and raises a single data-ready signal.
* **Single-shot** — uses ``seq = 1``; not safe to reuse the same buffers
  back-to-back without re-rendezvous. The pipelined, monotonic-counter version
  is the follow-up Triton stack.
* **Cross-rank contract** — both peers of an exchange must use the same
  ``num_blocks``; it sets the chunk partition and the per-block signal slot, so a
  mismatch silently misaligns chunks or hangs a receiver block. Not validated
  here (see the launcher warning).

The hook takes a single ``Ctx`` aggregate (see ``triton/ctx.py``):
``produce(ctx) -> regs`` and ``consume(ctx, regs)``. Adding a field to ``Ctx``
never changes a hook signature.
"""

import triton
import triton.language as tl

from .ctx import Ctx
from .device_utils import get_flat_tid, sync_threads


@triton.jit
def send_tiles(
    in_ptr,
    send_dst,
    signal_dst,
    numel,
    produce: tl.constexpr,
    put: tl.constexpr,
    signal: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Send this rank's buffer to a peer through ``send_dst`` + ``signal_dst``."""
    bid = tl.program_id(0)
    nblocks = tl.num_programs(0)
    chunk = tl.cdiv(numel, nblocks)
    start = bid * chunk
    end = min(start + chunk, numel)

    off = start
    while off < end:
        idx = off + tl.arange(0, BLOCK)
        mask = idx < end
        regs = produce(Ctx(in_ptr, idx, mask))
        put(send_dst, idx, regs, mask)
        off += BLOCK

    # Ordering: sync_threads() is a block-scope barrier, so every thread's
    # staging stores above complete before tid 0 proceeds. signal() then
    # issues a system-scope release (fence.acq_rel.sys) + remote store from
    # tid 0, so the peer's acquire-poll of the signal observes this block's
    # staging data.
    sync_threads()
    if get_flat_tid() == 0:
        signal(signal_dst, bid, 1)


@triton.jit
def recv_tiles(
    out_ptr,
    recv_src,
    signal_src,
    numel,
    consume: tl.constexpr,
    get: tl.constexpr,
    wait: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Receive a peer's buffer from ``recv_src`` after ``signal_src`` fires."""
    bid = tl.program_id(0)
    nblocks = tl.num_programs(0)
    chunk = tl.cdiv(numel, nblocks)
    start = bid * chunk
    end = min(start + chunk, numel)

    # Ordering: tid 0 acquire-polls the data-ready signal (pairs with the
    # sender's release in signal); sync_threads() then publishes that
    # acquire to the whole block before any thread reads staging.
    if get_flat_tid() == 0:
        wait(signal_src, bid, 1)
    sync_threads()

    off = start
    while off < end:
        idx = off + tl.arange(0, BLOCK)
        mask = idx < end
        regs = get(recv_src, idx, mask)
        consume(Ctx(out_ptr, idx, mask), regs)
        off += BLOCK
