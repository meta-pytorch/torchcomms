# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-unsafe

"""Inline-PTX signalling primitives for the CuTe backend.

Cross-rank int64 signalling over symmetric memory via ``@dsl_user_op`` and
``llvm.inline_asm``. TAIL data-ready publication uses release/acquire ordering; HEAD
slot-free credits use a relaxed store and volatile poll because they transfer ownership
without publishing payload.
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
def wait_ge_volatile(addr: Int64, target: Int64, *, loc=None, ip=None) -> None:
    """Volatile-poll until ``*addr >= target`` for a HEAD slot-free credit."""
    llvm.inline_asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip), Int64(target).ir_value(loc=loc, ip=ip)],
        """{
            .reg .u64   %tmp64;
            .reg .pred  %p;
            _fw_wait_ge_vol:
                ld.volatile.global.u64 %tmp64, [$0];
                setp.lt.u64 %p, %tmp64, $1;
                @%p bra _fw_wait_ge_vol;
        }""",
        "l,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def remote_store_release_i64(addr: Int64, val: Int64, *, loc=None, ip=None) -> None:
    """Publish prior memory accesses with one system-scope release store."""
    llvm.inline_asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip), Int64(val).ir_value(loc=loc, ip=ip)],
        """{
            st.global.release.sys.b64 [$0], $1;
        }""",
        "l,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def remote_store_i64_relaxed(addr: Int64, val: Int64, *, loc=None, ip=None) -> None:
    """Return a HEAD slot-free credit without payload-publication ordering."""
    llvm.inline_asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip), Int64(val).ir_value(loc=loc, ip=ip)],
        """{
            st.global.relaxed.sys.b64 [$0], $1;
        }""",
        "l,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
