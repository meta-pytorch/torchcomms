# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""
Host-side wrapper for put_block bandwidth microbenchmark kernels.

Measures raw RDMA put throughput (no src/dst copy overhead) using three
modes:
  - fireforget:  One block sends, one block receives. No flow control —
                 all steps are posted without waiting for completion.
  - pipelined:   Same grid but with a ring-buffer flow-control protocol
                 (DATA_READY / SLOT_FREE signals) to cap in-flight data.
  - multiblock:  Multiple independent send/recv block pairs, each running
                 its own pipelined stream.
"""

import torch
import torchcomms

from .put_bw_kernel import (
    put_bw_fireforget_kernel,
    put_bw_multiblock_kernel,
    put_bw_pipelined_kernel,
)
from .sendrecv_cooperative_kernel import _increment_iteration_kernel


class PutBwOp:
    """
    GPU-to-GPU put bandwidth microbenchmark using torchcomms window.put.

    Usage:
        op = PutBwOp(comm, "pipelined", pipeline_depth=8,
                     section_size=4*1024*1024)
        op(total_bytes=256*1024*1024, peer_rank=1)
        # ... repeat as needed ...
        op.teardown()

    Args:
        comm: TorchComm communicator instance.
        mode: "fireforget" | "pipelined" | "multiblock".
        pipeline_depth: Number of pipeline slots.
        section_size: Size of each put in bytes.
        num_blocks: Number of send (=recv) block pairs (multiblock only).
        total_bytes: Total transfer size in bytes. Required for fireforget
                     mode (determines staging buffer size).
    """

    VALID_MODES = ("fireforget", "pipelined", "multiblock")

    def __init__(
        self,
        comm,
        mode,
        pipeline_depth,
        section_size,
        num_blocks=1,
        total_bytes=None,
    ):
        assert mode in self.VALID_MODES, f"Unknown mode {mode!r}"
        self.comm = comm
        self.mode = mode
        self.pipeline_depth = pipeline_depth
        self.section_size = section_size  # bytes
        self.num_blocks = num_blocks
        self.rank = comm.get_rank()
        self.device = comm.get_device()
        self.backend = comm.get_backend()

        # ── Staging buffer size depends on mode ──
        if mode == "fireforget":
            assert total_bytes is not None, "total_bytes required for fireforget mode"
            staging_bytes = total_bytes
        else:
            staging_bytes = pipeline_depth * section_size

        staging_elements = staging_bytes // 4  # float32

        # ── Iteration counter (created BEFORE GIN activation) ──
        self._iteration = torch.zeros(1, dtype=torch.int32, device=self.device)

        # ── Allocate RDMA-compatible staging buffers ──
        allocator = torchcomms.get_mem_allocator(self.backend)
        self._pool = torch.cuda.MemPool(allocator)
        with torch.cuda.use_mem_pool(self._pool):
            self.recv_staging = torch.zeros(
                staging_elements, dtype=torch.float32, device=self.device
            )
            self.send_staging = torch.ones(
                staging_elements, dtype=torch.float32, device=self.device
            )

        # ── Window setup (COLLECTIVE — all ranks must participate) ──
        self.comm.barrier(False)
        self.window = self.comm.new_window()
        self.window.tensor_register(self.recv_staging)
        self.comm.barrier(False)

        # ── Signal / counter budget depends on mode ──
        if mode == "fireforget":
            signal_count = 2  # DATA_READY + ALL_DONE
            counter_count = 1  # NIC_DONE
        elif mode == "pipelined":
            signal_count = 2  # DATA_READY + SLOT_FREE
            counter_count = 1  # NIC_DONE
        else:  # multiblock
            signal_count = 2 * num_blocks
            counter_count = num_blocks

        self.dev_win = self.window.get_device_window(
            signal_count=signal_count,
            counter_count=counter_count,
            barrier_count=0,
        )

        # ── Register send staging for RDMA source (NON-COLLECTIVE) ──
        self.send_buf_info = self.window.register_local_buffer(self.send_staging)

        self._torn_down = False

    def __call__(self, total_bytes, peer_rank):
        """
        Execute put bandwidth test.

        Both ranks must call this simultaneously. Each rank puts data to
        the peer's recv_staging via RDMA. Returns nothing — just moves bytes.
        """
        assert not self._torn_down, "PutBwOp has been torn down"

        total_elements = total_bytes // 4  # float32
        elem_size_bytes = 4
        section_elements = self.section_size // elem_size_bytes

        if self.mode == "fireforget":
            self._call_fireforget(
                total_elements, section_elements, elem_size_bytes, peer_rank
            )
        elif self.mode == "pipelined":
            self._call_pipelined(
                total_elements, section_elements, elem_size_bytes, peer_rank
            )
        else:
            self._call_multiblock(
                total_elements, section_elements, elem_size_bytes, peer_rank
            )

        # Increment iteration counter for monotonic signal tracking
        _increment_iteration_kernel[(1,)](self._iteration, num_warps=1)

    def _call_fireforget(
        self, total_elements, section_elements, elem_size_bytes, peer_rank
    ):
        put_bw_fireforget_kernel[(2,)](
            self.dev_win,
            self.send_buf_info,
            self.send_staging,
            self._iteration,
            total_elements,
            section_elements,
            elem_size_bytes,
            peer_rank=peer_rank,
            BLOCK_SIZE=1024,
            num_warps=32,
        )

    def _call_pipelined(
        self, total_elements, section_elements, elem_size_bytes, peer_rank
    ):
        put_bw_pipelined_kernel[(2,)](
            self.dev_win,
            self.send_buf_info,
            self.send_staging,
            self._iteration,
            total_elements,
            section_elements,
            elem_size_bytes,
            self.pipeline_depth,
            peer_rank=peer_rank,
            BLOCK_SIZE=1024,
            num_warps=32,
        )

    def _call_multiblock(
        self, total_elements, section_elements, elem_size_bytes, peer_rank
    ):
        put_bw_multiblock_kernel[(2 * self.num_blocks,)](
            self.dev_win,
            self.send_buf_info,
            self.send_staging,
            self._iteration,
            total_elements,
            section_elements,
            elem_size_bytes,
            self.pipeline_depth,
            self.num_blocks,
            peer_rank=peer_rank,
            BLOCK_SIZE=1024,
            num_warps=32,
        )

    def teardown(self):
        """Release resources. Must be called by all ranks (tensor_deregister is collective)."""
        if self._torn_down:
            return
        self.window.deregister_local_buffer(self.send_buf_info)
        self.window.tensor_deregister()
        self._torn_down = True

    def __del__(self):
        # Best-effort cleanup, but user should call teardown() explicitly
        # because tensor_deregister is collective
        pass
