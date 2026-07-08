# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

"""Device transport-ops seam.

These four device primitives are the *only* transport-specific operations.
``send``/``recv`` are written once against this seam and treat the per-peer
buffers as opaque, so a single ``send``/``recv``:

  * serves NVLink today and IB later, and
  * serves **mixed-transport kernels** — a schedule may use NVLink for an
    intra-domain peer and IB for an inter-domain peer by passing different ops
    at each call site (the binding is per-call-site, not per-kernel).

There is no runtime dispatch: in Triton the concrete ops are passed as
``tl.constexpr`` and selected at compile time. Concrete implementations live per
transport + DSL (e.g. ``framework/triton/nvl_ops.py``). This module documents
the contract.

Conceptual signatures (the concrete versions are ``@triton.jit`` / cute kernels):

    put(region, idx, regs, mask) -> None   # produced tile -> peer region
    get(region, idx, mask)        -> regs   # received tile  <- region
    signal(sig_ptr, sig_idx, seq)          -> None   # publish "data ready"
    wait(sig_ptr, sig_idx, seq)            -> None   # wait for peer's data
"""

from __future__ import annotations

from typing import Protocol


class TransportOps(Protocol):
    """Structural description of the four device ops a transport must provide.

    Used for documentation/typing only; the attributes are DSL kernels
    (``@triton.jit`` functions for the Triton backend), not plain callables, so
    the parameter types are intentionally left loose.
    """

    put: object
    get: object
    signal: object
    wait: object
