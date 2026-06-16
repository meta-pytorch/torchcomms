# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-unsafe

"""NVLink device transport-ops (real, minimal) for the CuTe send/recv seam.

CuTe twin of ``triton/nvl_ops.py``: the concrete four-op seam
(``framework/ops.py``) for the symmetric-memory transport, in **cute-native**
signatures. ``put``/``get`` are ``cute.copy`` over the tile's copy atom;
``signal``/``wait`` wrap the ``device_utils`` PTX. Passed as ``constexpr`` to
``send_tiles``/``recv_tiles`` so the kernel is transport-agnostic and the
transport binds per call-site (same intent as the Triton seam).
"""

from __future__ import annotations

import cutlass.cute as cute

from .device_utils import fence_and_remote_store_i64, wait_ge


def put(atom, frag, dst_part):
    """Write a produced fragment into the peer's staging partition (remote store)."""
    cute.copy(atom, frag, dst_part)


def get(atom, src_part):
    """Read a received tile from the local staging partition into a fragment."""
    frag = cute.make_fragment_like(src_part)
    cute.copy(atom, src_part, frag)
    return frag


def signal(addr, seq):
    """Publish "data ready" to the peer (fence + remote int64 store)."""
    fence_and_remote_store_i64(addr, seq)


def wait(addr, seq):
    """Wait until the peer published ``seq`` (acquire-poll local int64)."""
    wait_ge(addr, seq)
