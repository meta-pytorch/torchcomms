# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-unsafe

"""Inline-PTX signaling primitives for the CuTe backend.

The CuTe twin of ``triton/device_utils.py``: the cross-rank acquire/release
signaling, emitting the **same on-wire protocol** so the two DSLs interoperate.
Each op is a single PTX snippet via ``@dsl_user_op`` + ``llvm.inline_asm``.

Minimal scope: only the two ops the no-pipeline send/recv needs —
``wait_ge`` (receiver acquire-poll) and ``fence_and_remote_store_i64`` (sender
release + remote store).
"""

from __future__ import annotations

from cutlass._mlir.dialects import llvm
from cutlass.cute.typing import Int64
from cutlass.cutlass_dsl import dsl_user_op


@dsl_user_op
def wait_ge(addr: Int64, target: Int64, *, loc=None, ip=None) -> None:
    """Acquire-poll int64 until ``*addr >= target`` (local read)."""
    llvm.inline_asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip), Int64(target).ir_value(loc=loc, ip=ip)],
        """{
            .reg .u64   %tmp64;
            .reg .pred  %p;
            _fw_wait_ge:
                ld.global.acquire.sys.b64 %tmp64, [$0];
                setp.lt.u64 %p, %tmp64, $1;
                @%p bra _fw_wait_ge;
        }""",
        "l,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def fence_and_remote_store_i64(addr: Int64, val: Int64, *, loc=None, ip=None) -> None:
    """System release fence + remote relaxed int64 store (publish data)."""
    llvm.inline_asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip), Int64(val).ir_value(loc=loc, ip=ip)],
        """{
            fence.acq_rel.sys;
            st.global.relaxed.sys.b64 [$0], $1;
        }""",
        "l,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
