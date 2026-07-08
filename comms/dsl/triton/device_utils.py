# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-unsafe

"""Self-contained Triton device helpers for the framework's NVLink send/recv.

Thread-id / block-barrier helpers + the cross-rank **acquire/release** signaling
PTX. This is the Triton twin of ``cute/cute_sync.py`` and emits the **same
on-wire protocol** (``ld.acquire.sys`` poll + ``fence.acq_rel.sys`` release
store), so the two DSLs interoperate.

``comms/dsl`` owns these so it does not depend on the ``comms/pipes`` prototype
(only the minimal single-shot ops are here; credit/volatile variants live with
the pipelined follow-up).
"""

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

    Triton's implicit ``__syncthreads`` is not always emitted around inline-asm
    sections, so we emit one explicitly when the protocol requires cross-warp
    ordering inside a block.
    """
    tl.inline_asm_elementwise(
        "bar.sync 0;", "=r", [], dtype=tl.int32, is_pure=False, pack=1
    )


@triton.jit
def wait_ge(addr, target):
    """Acquire-poll int64 until ``*addr >= target`` (local read).

    The acquire memory order pairs with the sender's
    ``fence_and_remote_store_i64`` (release) so the receiver's subsequent
    staging reads observe the sender's writes.
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
def fence_and_remote_store_i64(addr, val):
    """System release fence + REMOTE relaxed int64 store (publish data).

    Drains all prior writes (staging data) so the remote rank's acquire-load of
    the counter sees the data; the store is relaxed because the fence already
    provided release ordering.
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
