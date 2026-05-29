# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-unsafe

"""Triton device-side helpers for NVLink copy-based collectives."""

import triton
import triton.language as tl


@triton.jit
def get_tid():
    return tl.inline_asm_elementwise(
        """
        mov.u32 $0, %tid.x;
        mov.u32 $1, %tid.y;
        mov.u32 $2, %tid.z;
        """,
        "=r,=r,=r",
        [],
        dtype=(tl.uint32, tl.uint32, tl.uint32),
        is_pure=True,
        pack=1,
    )


@triton.jit
def get_ntid():
    return tl.inline_asm_elementwise(
        """
        mov.u32 $0, %ntid.x;
        mov.u32 $1, %ntid.y;
        mov.u32 $2, %ntid.z;
        """,
        "=r,=r,=r",
        [],
        dtype=(tl.uint32, tl.uint32, tl.uint32),
        is_pure=True,
        pack=1,
    )


@triton.jit
def get_flat_tid():
    tid_x, tid_y, tid_z = get_tid()
    ntid_x, ntid_y, _ = get_ntid()
    return tid_z * ntid_y * ntid_x + tid_y * ntid_x + tid_x


@triton.jit
def sync_threads():
    """Block-scope barrier (``bar.sync 0``).

    Triton's implicit ``__syncthreads`` is not always emitted around inline
    asm sections, so we emit one explicitly when the protocol requires
    cross-warp ordering inside a block.
    """
    tl.inline_asm_elementwise(
        "bar.sync 0;", "=r", [], dtype=tl.int32, is_pure=False, pack=1
    )


@triton.jit
def wait_ge(addr, target):
    """Acquire-poll int64 until ``*addr >= target``.

    All reads are LOCAL (no NVLink load). Used by the **receiver** to poll the
    local TAIL counter — the acquire memory order pairs with the sender's
    ``fence_and_remote_store_i64`` (release) so the receiver's subsequent
    staging-buffer reads observe the sender's writes.

    The sender's HEAD poll uses the volatile-only :func:`wait_ge_volatile`
    instead, because the receiver's HEAD update does not publish payload data
    (see :func:`remote_store_i64_relaxed`).
    """
    tl.inline_asm_elementwise(
        """
        {
            .reg .u64   %tmp64_<1>;
            .reg .pred  %p<1>;

            wait_ge:
                ld.global.acquire.sys.b64 %tmp64_0, [$1];
                setp.lt.u64 %p0, %tmp64_0, $2;
                @%p0 bra wait_ge;
        }
        """,
        "=r, l, l",
        [addr, target],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def wait_ge_volatile(addr, target):
    """Volatile-poll int64 until ``*addr >= target``.

    Use for credit counters that only transfer staging-slot ownership. The
    counter does not publish payload data, so the polling load does not need
    acquire semantics.
    """
    tl.inline_asm_elementwise(
        """
        {
            .reg .u64   %tmp64_<1>;
            .reg .pred  %p<1>;

            wait_ge_volatile:
                ld.volatile.global.u64 %tmp64_0, [$1];
                setp.lt.u64 %p0, %tmp64_0, $2;
                @%p0 bra wait_ge_volatile;
        }
        """,
        "=r, l, l",
        [addr, target],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def fence_and_remote_store_i64(addr, val):
    """System fence + REMOTE int64 store for monotonic counter advancement.

    Drains all prior writes (staging buffer data) so the remote rank's
    acquire-load of the counter sees the data. The store itself is
    relaxed because the fence has already provided release ordering.
    """
    tl.inline_asm_elementwise(
        """
        {
            fence.acq_rel.sys;
            st.global.relaxed.sys.b64 [$1], $2;
            mov.u32 $0, 0;
        }
        """,
        "=r, l, l",
        [addr, val],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def remote_store_i64_relaxed(addr, val):
    """REMOTE relaxed int64 store **without** a preceding system fence.

    For credit signals that only transfer staging-slot ownership, where the
    consumer side does not read any payload published by the producer of the
    signal.

    A local execution barrier (e.g. ``sync_threads``) before this store is
    sufficient to ensure all prior reads inside the producer block have
    completed before the credit is posted. Do **not** use this for signals
    that publish payload data (e.g. sender→receiver TAIL); use
    :func:`fence_and_remote_store_i64` for those.
    """
    tl.inline_asm_elementwise(
        """
        {
            st.global.relaxed.sys.b64 [$1], $2;
            mov.u32 $0, 0;
        }
        """,
        "=r, l, l",
        [addr, val],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )
