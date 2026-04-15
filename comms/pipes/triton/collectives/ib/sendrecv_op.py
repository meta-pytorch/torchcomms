# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
Host-side wrapper for the Triton pipelined sendrecv operation.

Handles window creation, buffer registration, and kernel launch for
copy-based send/recv over InfiniBand using the torchcomms window API.

Two kernel implementations:
  - Block-scope (default): Each block independently copies and RDMA-puts
    its own tile. Posts N WQEs per step — no inter-block coordination.
  - Cooperative: N blocks cooperatively copy, then the last block issues
    a single RDMA put per step. Lower WQE count but requires atomics.

Public API:
  - ``SendRecvOp``: Stateful op for repeated calls (CUDA graph compatible).
  - ``sendrecv()``: One-shot convenience function.
"""

import torch
import torchcomms

from .sendrecv import sendrecv_kernel
from .sendrecv_cooperative_kernel import (
    _fill_kernel,
    _increment_iteration_kernel,
    sendrecv_cooperative_kernel,
)


def _default_num_blocks(msg_bytes: int) -> int:
    """Pick a default block count based on message size.

    Heuristic mirrors NCCL's SM allocation: more blocks for larger
    messages to keep the NIC pipeline full without over-subscribing SMs.
    """
    if msg_bytes >= 512 * 1024 * 1024:
        return 32
    if msg_bytes >= 64 * 1024 * 1024:
        return 16
    if msg_bytes >= 1024 * 1024:
        return 8
    return 4


def sendrecv(
    comm,
    src: torch.Tensor,
    dst: torch.Tensor,
    peer_rank: int,
    *,
    pipeline_depth: int = 4,
    section_size: int = 4 * 1024 * 1024,
    num_blocks: int | None = None,
    parallel: bool = True,
) -> None:
    """One-shot bidirectional sendrecv over IB.

    Creates a ``SendRecvOp``, executes it once, and tears down. For repeated
    calls (e.g. inside a training loop), prefer creating a ``SendRecvOp``
    directly to amortize setup cost.

    Args:
        comm: TorchComm communicator instance.
        src: Source tensor to send (float32, CUDA).
        dst: Destination tensor to receive into (float32, CUDA).
        peer_rank: Rank of the peer to exchange with.
        pipeline_depth: Number of pipeline slots (ring buffer depth).
        section_size: Size of each pipeline section in bytes.
        num_blocks: Thread blocks per direction. ``None`` = auto-select.
        parallel: If True (default), use block-scope kernel.
    """
    if num_blocks is None:
        num_blocks = _default_num_blocks(src.nbytes)
    op = SendRecvOp(comm, pipeline_depth, section_size, num_blocks, parallel=parallel)
    try:
        op(src, dst, peer_rank)
    finally:
        op.teardown()


class SendRecvOp:
    """
    Pipelined GPU-to-GPU sendrecv using torchcomms window.put.

    Usage::

        op = SendRecvOp(comm, pipeline_depth=4, section_size=4*1024*1024)
        op(src_tensor, dst_tensor, peer_rank=1)
        # ... repeat as needed ...
        op.teardown()

    Or as a context manager::

        with SendRecvOp(comm, pipeline_depth=4, section_size=4*MB) as op:
            op(src, dst, peer_rank)

    Args:
        comm: TorchComm communicator instance.
        pipeline_depth: Number of pipeline slots (ring buffer depth).
        section_size: Size of each pipeline section in bytes.
        num_blocks_per_dir: Number of thread blocks per direction (send/recv).
        parallel: If True (default), use block-scope kernel (per-block RDMA puts).
    """

    def __init__(
        self,
        comm,
        pipeline_depth,
        section_size,
        num_blocks_per_dir,
        parallel=True,
    ):
        self.comm = comm
        self.pipeline_depth = pipeline_depth
        self.section_size = section_size  # bytes
        self.parallel = parallel
        # Cap blocks to avoid GPU deadlock: total blocks (send + recv) must
        # fit within SM concurrency. H100 = 132 SMs, 2 blocks/SM @ 32 warps.
        max_blocks_per_dir = 128
        self.num_blocks = min(num_blocks_per_dir, max_blocks_per_dir)
        self.rank = comm.get_rank()
        self.device = comm.get_device()
        self.backend = comm.get_backend()

        staging_bytes = pipeline_depth * section_size
        staging_elements = staging_bytes // 4  # float32 for now

        # ── Iteration counter (created BEFORE GIN activation) ──
        self._iteration = torch.zeros(1, dtype=torch.int32, device=self.device)

        # ── Allocate RDMA-compatible staging buffers ──
        allocator = torchcomms.get_mem_allocator(self.backend)
        self._pool = torch.cuda.MemPool(allocator)
        with torch.cuda.use_mem_pool(self._pool):
            self.recv_staging = torch.zeros(
                staging_elements, dtype=torch.float32, device=self.device
            )
            self.send_staging = torch.zeros(
                staging_elements, dtype=torch.float32, device=self.device
            )

        # ── Coordination buffers (only needed for non-parallel mode) ──
        if not parallel:
            self.send_coord = torch.zeros(
                pipeline_depth, dtype=torch.int32, device=self.device
            )
            self.recv_coord = torch.zeros(
                pipeline_depth, dtype=torch.int32, device=self.device
            )

        # ── Window setup (COLLECTIVE — all ranks must participate) ──
        # Matches C++ createWindowSetup: barrier → new_window → tensor_register
        # → barrier → cudaDeviceSynchronize.
        self.comm.barrier(False)
        self.window = self.comm.new_window()
        self.window.tensor_register(self.recv_staging)
        self.comm.barrier(False)
        torch.cuda.synchronize()

        # Signal/counter budget depends on mode.
        if parallel:
            # Per-block signals with monotonically increasing values.
            # 2*N signals: N DATA_READY + N SLOT_FREE
            # N counters: per-block NIC_DONE
            signal_count = 2 * self.num_blocks
            counter_count = self.num_blocks
        else:
            # 2 signals: DATA_READY + SLOT_FREE, 1 counter: NIC_DONE
            signal_count = 2
            counter_count = 1

        self.dev_win = self.window.get_device_window(
            signal_count=signal_count,
            counter_count=counter_count,
            barrier_count=0,
        )

        # ── Register send staging for RDMA source (NON-COLLECTIVE) ──
        self.send_buf_info = self.window.register_local_buffer(self.send_staging)

        self._torn_down = False

    def __call__(self, src, dst, peer_rank):
        """
        Execute pipelined sendrecv.

        Both ranks must call this simultaneously. Each rank sends src to peer
        and receives into dst from peer.
        """
        assert src.numel() == dst.numel(), (
            "src and dst must have same number of elements"
        )
        assert src.dtype == torch.float32, "Only float32 supported currently"
        assert not self._torn_down, "SendRecvOp has been torn down"

        total_elements = src.numel()
        elem_size_bytes = src.element_size()  # 4 for float32
        section_elements = self.section_size // elem_size_bytes

        if self.parallel:
            self._call_parallel(
                src,
                dst,
                peer_rank,
                total_elements,
                section_elements,
                elem_size_bytes,
            )
        else:
            self._call_original(
                src,
                dst,
                peer_rank,
                total_elements,
                section_elements,
                elem_size_bytes,
            )

        # Iteration counter is incremented inside the kernel (sender block 0)
        # after drain. No separate kernel needed.

    def _call_original(
        self, src, dst, peer_rank, total_elements, section_elements, elem_size_bytes
    ):
        """Launch the original cooperative sendrecv kernel."""
        grid = (self.num_blocks + self.num_blocks,)

        # Reset coordination counters.
        _fill_kernel[(1,)](
            self.send_coord, 0, self.pipeline_depth, BLOCK_SIZE=64, num_warps=1
        )
        _fill_kernel[(1,)](
            self.recv_coord, 0, self.pipeline_depth, BLOCK_SIZE=64, num_warps=1
        )

        sendrecv_cooperative_kernel[grid](
            self.dev_win,
            self.send_buf_info,
            src,
            dst,
            self.send_staging,
            self.recv_staging,
            self.send_coord,
            self.recv_coord,
            self._iteration,
            total_elements,
            section_elements,
            elem_size_bytes,
            self.pipeline_depth,
            self.num_blocks,
            self.num_blocks,
            peer_rank=peer_rank,
            BLOCK_SIZE=1024,
            num_warps=32,
        )

    def _call_parallel(
        self, src, dst, peer_rank, total_elements, section_elements, elem_size_bytes
    ):
        """Launch the tile-parallel sendrecv kernel."""
        grid = (self.num_blocks + self.num_blocks,)

        sendrecv_kernel[grid](
            self.dev_win,
            self.send_buf_info,
            src,
            dst,
            self.send_staging,
            self.recv_staging,
            self._iteration,
            total_elements,
            section_elements,
            elem_size_bytes,
            self.pipeline_depth,
            self.num_blocks,
            peer_rank=peer_rank,
            BLOCK_SIZE=8192,
            num_warps=8,
        )

    def teardown(self):
        """Release resources. Must be called by all ranks (tensor_deregister is collective).

        Mirrors the C++ teardownWindow pattern:
          1. deregister_local_buffer (non-collective)
          2. tensor_deregister (collective, has internal barriers)
          3. Destroy window object (triggers ncclDevCommDestroy)
          4. Release staging buffers and memory pool
          5. Barrier after full cleanup
        """
        if self._torn_down:
            return
        # 1-2: Deregister from window
        self.window.deregister_local_buffer(self.send_buf_info)
        self.window.tensor_deregister()
        # 3: Destroy window and device window (ncclDevCommDestroy, cudaFree)
        self.dev_win = None
        self.window = None
        # 4: Release staging buffers and memory pool
        self.send_staging = None
        self.recv_staging = None
        self._pool = None
        if not self.parallel:
            self.send_coord = None
            self.recv_coord = None
        # 5: Barrier after full cleanup
        self.comm.barrier(False)
        self._torn_down = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.teardown()
        return False

    def __del__(self):
        # Best-effort cleanup, but user should call teardown() explicitly
        # because tensor_deregister is collective
        pass
