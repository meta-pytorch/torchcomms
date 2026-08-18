# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-unsafe

"""NVLink device transport-ops (real, minimal) for the framework send/recv seam.

Thin ``@triton.jit`` wrappers over the framework's own NVLink PTX helpers in
``device_utils.py`` (self-contained; no ``comms/pipes`` dependency). These are the concrete
implementation of the four-op seam documented in ``framework/ops.py`` for the
symmetric-memory transport. No slots / credit here — that is the pipelined
version (follow-up Triton stack).
"""

import triton
import triton.language as tl

from .device_utils import fence_and_remote_store_i64, wait_ge


@triton.jit
def put(region, idx, regs, mask):
    """Write a produced tile into the peer's staging region (remote NVLink store)."""
    tl.store(region + idx, regs, mask=mask)


@triton.jit
def get(region, idx, mask):
    """Read a received tile from the local staging region."""
    return tl.load(region + idx, mask=mask)


@triton.jit
def signal(sig_ptr, sig_idx, seq):
    """Publish "data ready" to the peer (fence + remote int64 store)."""
    fence_and_remote_store_i64(sig_ptr + sig_idx, seq)


@triton.jit
def wait(sig_ptr, sig_idx, seq):
    """Wait until the peer published ``seq`` (acquire-poll local int64)."""
    wait_ge(sig_ptr + sig_idx, seq)
