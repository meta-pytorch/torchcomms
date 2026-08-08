# Copyright (c) Meta Platforms, Inc. and affiliates.
# Confidential and proprietary.
# pyre-unsafe
"""
Pipes Triton LL Protocol Module.

Provides low-latency NVLink P2P communication using the LL (Low-Latency)
protocol with dual-flag atomicity detection. Each LL line carries 8 bytes
of payload in a 16-byte line format.

High-level API (recommended):
    LlP2pOp: Host-side wrapper managing buffers, window, and kernel launch.

Low-level API (for custom kernels):
    ll_send_kernel, ll_recv_kernel, ll_forward_kernel: Triton JIT kernels.

Utilities:
    ll_num_lines, ll_buffer_size, ll_total_buffer_size, can_use_ll,
    ll_peer_index, ll_peer_buffer_offset, ll_auto_tune, ll_auto_tune_bidirectional
"""

from comms.pipes.ll.triton.auto_tune import ll_auto_tune, ll_auto_tune_bidirectional
from comms.pipes.ll.triton.ll_ops import (
    can_use_ll,
    ll_buffer_size,
    ll_forward_kernel,
    LL_LINE_SIZE,
    ll_num_lines,
    LL_PAYLOAD_PER_LINE,
    ll_peer_buffer_offset,
    ll_peer_index,
    LL_READY_TO_WRITE,
    ll_recv_kernel,
    ll_send_kernel,
    ll_total_buffer_size,
)
from comms.pipes.ll.triton.ll_p2p_op import LlP2pOp


__all__ = [
    # High-level API
    "LlP2pOp",
    # Low-level kernels
    "ll_send_kernel",
    "ll_recv_kernel",
    "ll_forward_kernel",
    # Constants
    "LL_LINE_SIZE",
    "LL_PAYLOAD_PER_LINE",
    "LL_READY_TO_WRITE",
    # Utilities
    "ll_num_lines",
    "ll_buffer_size",
    "ll_total_buffer_size",
    "can_use_ll",
    "ll_peer_index",
    "ll_peer_buffer_offset",
    # Auto-tuning
    "ll_auto_tune",
    "ll_auto_tune_bidirectional",
]
