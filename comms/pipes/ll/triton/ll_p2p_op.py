# Copyright (c) Meta Platforms, Inc. and affiliates.
# Confidential and proprietary.
# pyre-unsafe
"""
Host-side wrapper for Triton LL P2P send/recv over NVLink.

Manages LL buffer allocation, window registration, NVLink connectivity
verification, and kernel launch. Designed as a building block that
collectives (e.g., AllToAllv) can compose, but also usable standalone
for P2P testing and benchmarking.

Public API:
  - ``LlP2pOp``: Stateful op for repeated LL send/recv/forward calls.

Usage::

    op = LlP2pOp(comm, max_nbytes=65536, device=torch.device("cuda:0"))
    with op:
        op.send(peer=1, src_tensor=src, nbytes=1024)
        op.recv(peer=1, dst_tensor=dst, nbytes=1024)

CUDA Graph Support
------------------
LlP2pOp is CUDA graph compatible. The LL flag handshake protocol
handles cross-rank synchronization internally, so ``graph.replay()``
works without any special wrapper.

CUDA Graph Usage Example::

    op = LlP2pOp(comm, max_nbytes=4096, device=device)
    with op:
        # 1. Warmup (compiles Triton kernels, outside graph capture)
        op.send(peer=1, src_tensor=src, nbytes=1024)
        torch.cuda.synchronize()

        # 2. Capture graph using op's memory pool
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=op.get_graph_pool_id()):
            op.send(peer=1, src_tensor=src, nbytes=1024)

        # 3. Replay with different content per iteration
        for i in range(num_iterations):
            src.copy_(iteration_data[i])
            graph.replay()
"""

from __future__ import annotations

import torch
from comms.pipes.ll.triton.auto_tune import ll_auto_tune
from comms.pipes.ll.triton.ll_ops import (
    ll_buffer_size,
    ll_forward_kernel,
    ll_peer_buffer_offset,
    ll_recv_kernel,
    ll_send_kernel,
    ll_sendrecv_kernel,
    ll_total_buffer_size,
)


class LlP2pOp:
    """Host-side wrapper for LL P2P send/recv over NVLink.

    Manages LL buffer allocation, window registration, and kernel launch.
    Supports up to 72 NVLink ranks (GB200). CUDA graph compatible via
    GPU-resident iteration counter.

    Args:
        comm: TorchComm communicator instance.
        max_nbytes: Maximum payload size in bytes per P2P operation.
        device: CUDA device for buffer allocation.
        buffer_num_lines: LL buffer size in lines per peer for chunked mode.
            0 (default) means unbounded (buffer sized to max_nbytes).
        timeout_ns: Reserved for future timeout support (currently unused).
    """

    def __init__(
        self,
        comm,
        max_nbytes: int,
        device: torch.device,
        buffer_num_lines: int = 0,
        timeout_ns: int = 10_000_000_000,
    ) -> None:
        self._comm = comm
        self._max_nbytes = max_nbytes
        self._device = device
        self._buffer_num_lines = buffer_num_lines
        self._timeout_ns = timeout_ns

        self._rank = comm.get_rank()
        self._world_size = comm.get_size()
        self._backend = comm.get_backend()

        self._per_peer_size = ll_buffer_size(max_nbytes)
        self._total_ll_size = ll_total_buffer_size(max_nbytes, self._world_size)

        self._window = None
        self._dev_win_ptr = None
        self._ll_buf = None
        self._ll_pool = None
        self._torn_down = False

    def setup(self) -> None:
        """Allocate LL buffer, register window, verify NVLink connectivity.

        This is a collective operation -- all ranks must call setup().
        Must be called before any send/recv/forward operations.
        """
        from comms.pipes.collectives.triton.utils import alloc_comms_buffer

        # Allocate LL buffer (BEFORE GIN activation)
        self._ll_buf, self._ll_pool = alloc_comms_buffer(
            self._total_ll_size,
            torch.uint8,
            self._device,
            self._backend,
        )

        # Initialize all flags to READY_TO_WRITE (0xFF bytes)
        self._ll_buf.fill_(255)

        # Window setup (COLLECTIVE)
        self._comm.barrier(False)
        self._window = self._comm.new_window()
        self._window.tensor_register(self._ll_buf)
        self._comm.barrier(False)
        torch.cuda.synchronize()

        # Activate GIN (signal_count=1 is safe minimum; LL uses inline flags)
        self._dev_win_ptr = self._window.get_device_window(
            signal_count=1,
            counter_count=0,
            barrier_count=0,
        )

    def send(
        self, peer: int, src_tensor: torch.Tensor, nbytes: int | None = None
    ) -> None:
        """Send data to peer via LL protocol over NVLink.

        Args:
            peer: Destination peer rank.
            src_tensor: Source data tensor (must be 8-byte aligned, contiguous).
            nbytes: Payload bytes to send. If None, uses src_tensor.nbytes.
                Must be a multiple of 8.
        """
        if nbytes is None:
            nbytes = src_tensor.nbytes

        config = ll_auto_tune(nbytes)
        grid = (config["num_blocks"],)

        peer_buf_offset = ll_peer_buffer_offset(
            target_rank=self._rank,
            self_rank=peer,
            per_peer_size=self._per_peer_size,
        )

        ll_send_kernel[grid](
            src_tensor,
            self._dev_win_ptr,
            peer,
            peer_buf_offset,
            nbytes,
            self._buffer_num_lines,
            BLOCK_SIZE=config["block_size"],
            num_warps=config["block_size"] // 32,
        )

    def recv(
        self, peer: int, dst_tensor: torch.Tensor, nbytes: int | None = None
    ) -> None:
        """Receive data from peer via LL protocol.

        Args:
            peer: Source peer rank.
            dst_tensor: Destination data tensor (must be 8-byte aligned, contiguous).
            nbytes: Payload bytes to receive. If None, uses dst_tensor.nbytes.
                Must be a multiple of 8.
        """
        if nbytes is None:
            nbytes = dst_tensor.nbytes

        config = ll_auto_tune(nbytes)
        grid = (config["num_blocks"],)

        peer_buf_offset = ll_peer_buffer_offset(
            target_rank=peer,
            self_rank=self._rank,
            per_peer_size=self._per_peer_size,
        )

        ll_recv_kernel[grid](
            dst_tensor,
            self._ll_buf,
            peer_buf_offset,
            nbytes,
            self._buffer_num_lines,
            BLOCK_SIZE=config["block_size"],
            num_warps=config["block_size"] // 32,
        )

    def forward(
        self,
        predecessor: int,
        successor: int,
        dst_tensor: torch.Tensor,
        nbytes: int | None = None,
    ) -> None:
        """Forward data from predecessor to successor, copying locally.

        Args:
            predecessor: Rank that sent the data (read from local LL buffer).
            successor: Rank to forward to (write to successor's LL buffer).
            dst_tensor: Local output tensor for the forwarded data.
            nbytes: Payload bytes. If None, uses dst_tensor.nbytes.
        """
        if nbytes is None:
            nbytes = dst_tensor.nbytes

        config = ll_auto_tune(nbytes)
        grid = (config["num_blocks"],)

        predecessor_buf_offset = ll_peer_buffer_offset(
            target_rank=predecessor,
            self_rank=self._rank,
            per_peer_size=self._per_peer_size,
        )
        successor_buf_offset = ll_peer_buffer_offset(
            target_rank=self._rank,
            self_rank=successor,
            per_peer_size=self._per_peer_size,
        )

        ll_forward_kernel[grid](
            dst_tensor,
            self._ll_buf,
            predecessor_buf_offset,
            self._dev_win_ptr,
            successor,
            successor_buf_offset,
            nbytes,
            self._buffer_num_lines,
            BLOCK_SIZE=config["block_size"],
            num_warps=config["block_size"] // 32,
        )

    def sendrecv(
        self,
        peer: int,
        src_tensor: torch.Tensor,
        dst_tensor: torch.Tensor,
        nbytes: int | None = None,
    ) -> None:
        """Fused send+recv in a single kernel launch.

        Both ranks must call sendrecv simultaneously. Each rank sends
        src_tensor to peer and receives into dst_tensor from peer.

        Args:
            peer: Peer rank to exchange with.
            src_tensor: Source data to send.
            dst_tensor: Destination buffer for received data.
            nbytes: Payload bytes. If None, uses src_tensor.nbytes.
        """
        if nbytes is None:
            nbytes = src_tensor.nbytes

        config = ll_auto_tune(nbytes)
        num_blocks_per_dir = config["num_blocks"]
        grid = (num_blocks_per_dir * 2,)

        send_peer_buf_offset = ll_peer_buffer_offset(
            target_rank=self._rank,
            self_rank=peer,
            per_peer_size=self._per_peer_size,
        )
        recv_peer_buf_offset = ll_peer_buffer_offset(
            target_rank=peer,
            self_rank=self._rank,
            per_peer_size=self._per_peer_size,
        )

        ll_sendrecv_kernel[grid](
            src_tensor,
            dst_tensor,
            self._dev_win_ptr,
            peer,
            send_peer_buf_offset,
            recv_peer_buf_offset,
            self._ll_buf,
            nbytes,
            self._buffer_num_lines,
            num_blocks_per_dir,
            BLOCK_SIZE=config["block_size"],
            num_warps=config["block_size"] // 32,
        )

    def get_graph_pool_id(self) -> tuple[int, int]:
        """Return the memory pool ID for CUDA graph capture.

        Pass this to ``torch.cuda.graph(graph, pool=...)`` to ensure
        allocations during capture use the same transport-compatible pool
        as the LL buffer.

        Buffer registration (``setup()``) must occur BEFORE graph capture.

        Returns:
            tuple[int, int]: The CUDA memory pool handle.
        """
        return self._ll_pool.id  # type: ignore[return-value]

    def teardown(self) -> None:
        """Release resources. Must be called by all ranks (collective).

        Safe to call multiple times.
        """
        if self._torn_down:
            return

        if self._window is not None:
            self._window.tensor_deregister()

        self._dev_win_ptr = None
        self._window = None
        self._ll_buf = None
        self._ll_pool = None

        self._comm.barrier(False)
        self._torn_down = True

    def __enter__(self):
        self.setup()
        return self

    def __exit__(self, *exc):
        self.teardown()
        return False
