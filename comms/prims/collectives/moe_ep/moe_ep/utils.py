# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

"""
Python-side utilities for the pipes MoE expert-parallel runtime.

EventOverlap is the public RAII wrapper around `EventHandle` from the C++
extension. It's used as `previous_event=` argument and as a return value from
`Buffer.dispatch` / `Buffer.combine` / `Buffer.get_dispatch_layout` /
`Buffer.capture()`.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import torch
import torch.distributed as dist

# pyre-ignore[21]: cpp_python_extension at runtime; static analyzer doesn't see it
from comms.prims.collectives.moe_ep import _cpp  # @manual

logger: logging.Logger = logging.getLogger(__name__)

# Re-export EventHandle so callers can `from .utils import EventHandle`
EventHandle = _cpp.EventHandle  # type: ignore[misc]


class EventOverlap:
    """
    Wrapper class to manage CUDA events for kernel-overlap convenience.

    Attributes:
        event: the captured CUDA event (or None for a no-op).
        extra_tensors: tuple of tensors held alive for the event's lifetime
            (simulates `Tensor.record_stream` in a way that's CUDA-graph
            compatible).
    """

    def __init__(
        self,
        event: EventHandle | None = None,  # type: ignore[valid-type]
        extra_tensors: tuple[torch.Tensor, ...] | None = None,
    ) -> None:
        self.event: EventHandle | None = event  # type: ignore[valid-type]
        # Use extra tensors to achieve stream-record-like semantics; plain
        # `tensor.record_stream` is incompatible with CUDA graph capture.
        self.extra_tensors = extra_tensors

    def current_stream_wait(self) -> None:
        """Make `torch.cuda.current_stream()` wait for `self.event`."""
        if self.event is None:
            raise RuntimeError("EventOverlap.current_stream_wait called with no event")
        self.event.current_stream_wait()

    def __enter__(self) -> Any:
        """Context-manager entry; the corresponding `__exit__` calls
        `current_stream_wait()` so the body runs while the event is in
        flight and the next op on the current stream waits for it."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self.event is not None:
            self.event.current_stream_wait()


def check_nvlink_connections(group: dist.ProcessGroup) -> None:
    """
    Verify that every pair of GPUs in `group` is connected via NVLink.

    Some PCIe-only GPU SKUs (e.g. some A100 PCIE) only have pairwise NVLink,
    so we hard-cap the group size at 2 in that case. This is only meaningful
    on NVIDIA — AMD MI300X uses XGMI and doesn't expose NVML.
    """
    device_name = torch.cuda.get_device_name()
    if "PCIE" not in device_name:
        return

    if group.size() > 2:
        raise RuntimeError(
            "PCIe GPUs only have pairwise NVLink connections; "
            f"group size {group.size()} > 2 is not supported"
        )

    try:
        # pyre-ignore[21]: optional third-party
        import pynvml  # @manual
    except ImportError:
        logger.warning("pynvml not available; skipping NVLink connectivity check")
        return

    pynvml.nvmlInit()
    try:
        devices = (
            os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
            .strip(",")
            .split(",")
        )
        physical_device_idx = int(devices[torch.cuda.current_device()])
        physical_device_indices = [0] * group.size()
        dist.all_gather_object(physical_device_indices, physical_device_idx, group)

        handles = [
            pynvml.nvmlDeviceGetHandleByIndex(i) for i in physical_device_indices
        ]
        for i, handle in enumerate(handles):
            for j, peer_handle in enumerate(handles):
                if i >= j:
                    continue
                status = pynvml.nvmlDeviceGetP2PStatus(
                    handle, peer_handle, pynvml.NVML_P2P_CAPS_INDEX_NVLINK
                )
                if status != pynvml.NVML_P2P_STATUS_OK:
                    raise RuntimeError(
                        f"GPU {physical_device_indices[i]} and "
                        f"GPU {physical_device_indices[j]} are not connected "
                        "via NVLink"
                    )
    finally:
        pynvml.nvmlShutdown()
