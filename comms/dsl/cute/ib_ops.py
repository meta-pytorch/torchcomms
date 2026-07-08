# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-unsafe

"""Reserved IB device transport-ops (CuTe).

CuTe twin of ``triton/ib_ops.py``: documents the same four-op seam
(``framework/ops.py``) the IB (RDMA) backend must implement. Bodies raise so any
attempt to trace/compile a kernel with them fails loudly until the IB stack wires
them; they remain importable so the interface is visible now.
"""

from __future__ import annotations

_RESERVED = "framework IB ops (CuTe) are reserved; implemented in the IB stack"


def put(atom, frag, dst_part):
    raise NotImplementedError(_RESERVED)


def get(atom, src_part):
    raise NotImplementedError(_RESERVED)


def signal(addr, seq):
    raise NotImplementedError(_RESERVED)


def wait(addr, seq):
    raise NotImplementedError(_RESERVED)
