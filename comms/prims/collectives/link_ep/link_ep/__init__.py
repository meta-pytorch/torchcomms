# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

"""
Prims MoE expert-parallel (EP) communication API.

Public Python surface for the prims-based dispatch / combine kernels under
`comms/prims/collectives/link_ep/`. Exports `Buffer`, `EventOverlap`, `Config`, and the
`topk_idx_t` torch dtype.
"""

from __future__ import annotations

# pyre-ignore[21]: cpp_python_extension at runtime
from comms.prims.collectives.link_ep import _cpp  # @manual
from comms.prims.collectives.link_ep.link_ep.buffer import Buffer
from comms.prims.collectives.link_ep.link_ep.utils import EventOverlap

# Tuning-knob struct re-exported from the C++ extension. Ctor:
#   Config(num_sms,
#          num_max_nvl_chunked_send_tokens,
#          num_max_nvl_chunked_recv_tokens,
#          num_max_rdma_chunked_send_tokens=6,
#          num_max_rdma_chunked_recv_tokens=128)
Config = _cpp.Config  # type: ignore[misc]

# `torch.dtype` indicating the wire format of `topk_idx` tensors. Tests cast
# `topk_idx.to(comms.prims.collectives.link_ep.link_ep.topk_idx_t)`. We standardize on
# torch.int64 (TOPK_IDX_BITS=64).
topk_idx_t = _cpp.topk_idx_t  # type: ignore[misc]

__all__ = [
    "Buffer",
    "EventOverlap",
    "Config",
    "topk_idx_t",
]
