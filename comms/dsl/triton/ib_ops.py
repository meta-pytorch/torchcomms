# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-unsafe

"""Reserved IB device transport-ops.

These document the same four-op seam (``framework/ops.py``) that the IB
(torchcomms window / RDMA) backend must implement. Bodies are reserved: a
``tl.static_assert`` fails at *compile* time if any of these is ever launched,
which only happens once the IB stack wires them. They remain importable and
inspectable (JITFunctions) so the interface is visible now.
"""

import triton
import triton.language as tl

_RESERVED = "framework IB ops are reserved; implemented in the IB stack"


@triton.jit
def put(region, idx, regs, mask):
    tl.static_assert(False, _RESERVED)


@triton.jit
def get(region, idx, mask):
    tl.static_assert(False, _RESERVED)
    return tl.load(region + idx, mask=mask)


@triton.jit
def signal(sig_ptr, sig_idx, seq):
    tl.static_assert(False, _RESERVED)


@triton.jit
def wait(sig_ptr, sig_idx, seq):
    tl.static_assert(False, _RESERVED)
