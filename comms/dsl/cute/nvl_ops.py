# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-unsafe

"""NVLink device transport-ops for the CuTe send/recv seam.

The concrete data movement plus TAIL data-ready and HEAD slot-free operations for the
symmetric-memory transport, in cute-native signatures.
"""

from __future__ import annotations

import cutlass.cute as cute

from .device_utils import (
    remote_store_i64_relaxed,
    remote_store_release_i64,
    wait_ge,
    wait_ge_volatile,
)


def put(atom, frag, dst_part):
    """Write a produced fragment into the peer's staging partition (remote store)."""
    cute.copy(atom, frag, dst_part)


def get(atom, src_part):
    """Read a received tile from the local staging partition into a fragment."""
    frag = cute.make_fragment_like(src_part)
    cute.copy(atom, src_part, frag)
    return frag


def signal(addr, seq):
    """Publish "data ready" to the peer with a system-scope release store."""
    remote_store_release_i64(addr, seq)


def wait(addr, seq):
    """Wait until the peer published data-ready ``seq`` with an acquire poll."""
    wait_ge(addr, seq)


def signal_free(addr, seq):
    """Return HEAD slot ownership with a relaxed remote store."""
    remote_store_i64_relaxed(addr, seq)


def wait_free(addr, seq):
    """Wait for a HEAD slot-free credit with a volatile poll."""
    wait_ge_volatile(addr, seq)
