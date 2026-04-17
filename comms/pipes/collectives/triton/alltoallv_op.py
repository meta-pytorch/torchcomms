# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-unsafe
"""High-level alltoallv operation with MSL-compatible signature.

``AlltoallvOp`` provides a simplified API matching the MSL ``all_to_all_op``
signature while preserving the zero-copy one-sided-put architecture of
the underlying ``device_alltoallv_dynamic`` kernel.

Output Layout
-------------
The output is always returned in **packed uniform layout**:
    - Shape: ``(sum(output_split_sizes), D)``
    - Contiguous: ``[peer_0_data, peer_1_data, ..., peer_{W-1}_data]``
    - Zero-copy view of internal buffer (no gather kernel needed)

**IMPORTANT**: Only uniform distribution is supported. All peers must send
exactly ``max_recv_tokens_per_peer`` tokens. Non-uniform distributions will
raise a ``ValueError``.

Internally, data lands in fixed-slot positions in the receive buffer
(one slot per peer). For uniform distribution, slots are back-to-back,
so the output is directly a view of the contiguous buffer.

Internally it:

* Pre-allocates transport-compatible send/recv buffers sized to the caller's
  declared maximum token count.
* Registers those buffers once during ``setup()`` (create window, enable GIN).
* Uses **fixed-slot** recv buffer layout — one contiguous slot of
  ``max_recv_tokens_per_peer`` rows per peer — so that destination
  offsets are deterministic and computed locally (no exchange needed).
* Converts token-level split sizes to byte-level sizes/offsets per call
  with zero CPU↔GPU synchronisation and zero per-call collectives.

Two usage patterns are supported:

Copy-in mode (simple)::

    op = AlltoallvOp(comm, max_input_tokens=1024, D=4096,
                     dtype=torch.bfloat16, device="cuda:0")
    with op:
        output = op.alltoallv(input_tensor, output_split_sizes,
                              input_split_sizes)
        # output is contiguous: shape (sum(output_split_sizes), D)

Zero-copy mode (maximum performance)::

    op = AlltoallvOp(comm, max_input_tokens=1024, D=4096,
                     dtype=torch.bfloat16, device="cuda:0")
    with op:
        send_buf = op.get_send_buffer(num_tokens=512)
        # Fill send_buf directly (e.g. from a gather / routing kernel)
        send_buf[:] = my_data
        output = op.alltoallv_from_buffer(output_split_sizes,
                                          input_split_sizes,
                                          num_input_tokens=512)
        # output is contiguous: shape (sum(output_split_sizes), D)

CUDA Graph Support
------------------
AlltoallvOp is fully CUDA graph compatible. The iteration counter and
cross-rank synchronization are handled internally via device-side kernels
that get captured in the graph. Simply call ``graph.replay()`` - no special
wrapper method needed!

CUDA Graph Usage Example::

    op = AlltoallvOp(comm, max_input_tokens, D, dtype, device)
    with op:
        send_buf = op.get_send_buffer(max_input_tokens)

        # Warmup (uses iteration 0)
        send_buf[:] = warmup_data
        op.alltoallv_from_buffer(output_splits, input_splits,
                                 num_input_tokens=max_input_tokens)

        # Capture graph (uses iteration 1 during capture)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=op.get_graph_pool_id()):
            output = op.alltoallv_from_buffer(output_splits, input_splits,
                                              num_input_tokens=max_input_tokens)

        # Replay with different content per iteration
        # Just use graph.replay() - cross-rank sync is in the kernel!
        for i in range(num_iterations):
            send_buf[:] = iteration_data[i]
            graph.replay()  # Internal device-side sync + iteration advancement
            # Output is valid here

Output lifetime
---------------
The returned tensor is a **view** of the internal pre-registered receive
buffer. It remains valid until the next ``alltoallv`` / ``alltoallv_from_buffer``
call. If the caller needs the data to persist beyond that point it must
``.clone()`` the output.
"""

import os
from typing import Any, Optional, TYPE_CHECKING, Union

import torch
import triton  # @manual
import triton.language as tl  # @manual
from comms.pipes.collectives.triton.device_alltoallv_dynamic import (
    auto_tune_alltoallv_params,
    device_alltoallv_dynamic,
)
from comms.pipes.collectives.triton.utils import alloc_comms_buffer


# =============================================================================
# GIN-safe Triton kernels
# =============================================================================


@triton.jit
def _triton_copy_1d_kernel(src_ptr, dst_ptr, N, BLOCK_SIZE: tl.constexpr):
    """Copy N elements from src to dst.  GIN-safe (pure device stores)."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    data = tl.load(src_ptr + offsets, mask=mask)
    tl.store(dst_ptr + offsets, data, mask=mask)


@triton.jit
def _sum_int64_kernel(input_ptr, output_ptr, N: tl.constexpr):
    """GIN-safe kernel to compute sum of int64 tensor."""
    offsets = tl.arange(0, N)
    vals = tl.load(input_ptr + offsets)
    total = tl.sum(vals, axis=0)
    tl.store(output_ptr, total)


@triton.jit
def _prepare_alltoallv_kernel(
    input_splits_ptr,
    output_splits_ptr,
    row_bytes,
    send_sizes_ptr,
    send_offsets_ptr,
    recv_sizes_ptr,
    W,
    BLOCK_SIZE: tl.constexpr,
):
    """Token splits → byte sizes/offsets.  Single-block, GIN-safe.

    Performs two operations entirely on GPU in one kernel launch:
    1. Multiply token counts by ``row_bytes`` to get byte sizes.
    2. Exclusive prefix sum of send byte sizes → contiguous send offsets.
    """
    idxs = tl.arange(0, BLOCK_SIZE)
    mask = idxs < W

    # Token counts → byte sizes
    input_splits = tl.load(input_splits_ptr + idxs, mask=mask, other=0)
    send_sizes = input_splits * row_bytes
    tl.store(send_sizes_ptr + idxs, send_sizes, mask=mask)

    output_splits = tl.load(output_splits_ptr + idxs, mask=mask, other=0)
    recv_sizes = output_splits * row_bytes
    tl.store(recv_sizes_ptr + idxs, recv_sizes, mask=mask)

    # Exclusive prefix sum for contiguous send offsets
    cumsum = tl.cumsum(send_sizes, axis=0)
    send_offsets = cumsum - send_sizes
    tl.store(send_offsets_ptr + idxs, send_offsets, mask=mask)


if TYPE_CHECKING:
    from torchcomms import TorchComm


__all__ = [
    "AlltoallvOp",
]


class AlltoallvOp:
    """High-level alltoallv operation with an MSL-compatible signature.

    Encapsulates buffer allocation, registration, and byte-level plumbing
    behind a token-level API.
    behind a token-level API.  The caller works exclusively in token counts and
    2-D ``(T, D)`` tensors; the op handles the conversion to byte sizes /
    offsets internally.

    Buffer layout
    ~~~~~~~~~~~~~
    * **Send buffer** — flat, contiguously packed:
      ``[tokens_for_peer_0, tokens_for_peer_1, …]``.  ``send_offsets`` are
      computed as the exclusive prefix sum of per-peer byte sizes each call.
    * **Receive buffer** — fixed-slot:
      ``[slot_0, slot_1, …, slot_{W-1}]`` where each slot is exactly
      ``max_recv_tokens_per_peer * D`` elements.  Data from peer *i* always
      lands at slot *i*.  This makes ``dst_offsets`` (the offsets the *sender*
      writes to on the *receiver's* buffer) fully deterministic and
      computable locally (no exchange needed).

    Parameters
    ----------
    comm : TorchComm
        TorchComms communicator.  All ranks in this communicator participate
        in the collective.
    max_input_tokens : int
        Maximum total input tokens (rows in the input tensor) that any single
        call will ever pass.  This determines the size of the send buffer.
        **All ranks MUST use the same value.**
    D : int
        Hidden dimension (number of columns per token row).  Must be constant
        across all calls.
    dtype : torch.dtype
        Element data type (e.g. ``torch.bfloat16``).
    device : str | torch.device
        CUDA device (e.g. ``"cuda:0"``).
    max_recv_tokens_per_peer : int
        Maximum tokens that can be received from any single peer.  This
        determines the size of each slot in the receive buffer.  For uniform
        distribution workloads where each peer sends equal amounts, this
        should be set to ``max_input_tokens // world_size``.
        **All ranks MUST use the same value.**

    Notes
    -----
    * The op auto-tunes kernel parameters (``blocks_per_peer``,
      ``num_warps``, ``chunk_size``) per call based on the maximum per-peer
      message size.
    * The iteration counter is auto-incremented after every collective call.
      Do **not** mix raw ``device_alltoallv_dynamic`` calls on the same
      window — signal counters will diverge and cause deadlocks.
    """

    # Module-level cache: one AlltoallvOp per unique configuration.
    _CACHE: dict[tuple, "AlltoallvOp"] = {}

    def __init__(
        self,
        comm: "TorchComm",
        max_input_tokens: int,
        D: int,
        dtype: torch.dtype,
        device: Union[str, torch.device],
        max_recv_tokens_per_peer: int,
        sync_buffer: bool = True,
    ) -> None:
        """Initialize AlltoallvOp.

        Args:
            comm: TorchComm communicator.
            max_input_tokens: Maximum total input tokens (sum across all peers).
            D: Hidden dimension.
            dtype: Data type.
            device: CUDA device.
            max_recv_tokens_per_peer: Maximum tokens receivable from each peer.
                For uniform distribution, this should be ``max_input_tokens // world_size``.
                **All ranks MUST use the same value.** This is a required parameter
                because only uniform distribution is supported.
            sync_buffer: If True (default), enables BUFFER_READY cross-rank
                synchronization for safe buffer reuse across iterations. This is
                REQUIRED when:
                - Using CUDA graph capture with multiple iterations (graph or non-graph)
                - Replaying CUDA graphs multiple times with the same recv_buf
                - Any scenario where recv_buf is reused without explicit host sync

                Without this flag, a fast sender on rank A could overwrite the
                recv_buf on rank B before rank B has finished reading iteration N-1's
                data, causing data corruption.

                The protocol works by having receivers signal BUFFER_READY at the
                start of iteration N (indicating iteration N-1 is complete), and
                senders wait for this signal before sending iteration N's data.

                Adds ~1-3us per-peer latency overhead. Set to False only for
                microbenchmarking raw kernel throughput without synchronization.
        """
        self.comm = comm
        self.rank: int = comm.get_rank()
        self.world_size: int = comm.get_size()
        self.max_input_tokens = max_input_tokens
        self.D = D
        self.dtype = dtype
        self.device = device
        self.sync_buffer = sync_buffer
        self._setup_done = False

        # max_recv_tokens_per_peer is required - uniform distribution only.
        # Typically set to max_input_tokens // world_size.
        self.max_recv_tokens_per_peer: int = max_recv_tokens_per_peer

        self._elem_bytes: int = torch.tensor([], dtype=dtype).element_size()
        # Bytes per token row: D elements × element_size
        self._bytes_per_token: int = D * self._elem_bytes

        # Fixed slot size per peer in the receive buffer (bytes).
        # Each peer's data lands in a slot of max_recv_tokens_per_peer × D elements.
        self._bytes_per_peer_slot: int = (
            max_recv_tokens_per_peer * self._bytes_per_token
        )

        # -----------------------------------------------------------------
        # Allocate transport-compatible buffers
        # -----------------------------------------------------------------
        backend: str = comm.get_backend()

        # Send buffer: contiguously packed, holds up to max_input_tokens rows.
        max_input_elems = max_input_tokens * D
        self.send_buf, self._send_pool = alloc_comms_buffer(
            max_input_elems, dtype, device, backend
        )

        # Receive buffer: fixed-slot layout, one slot per peer.
        # Each slot holds max_recv_tokens_per_peer tokens.
        recv_total_elems = self.world_size * max_recv_tokens_per_peer * D
        self.recv_buf, self._recv_pool = alloc_comms_buffer(
            recv_total_elems, dtype, device, backend
        )

        # Compute recv_offsets: fixed slot layout where peer i's data lands at slot i.
        # Data lands in slotted positions internally; we run a local gather kernel
        # to pack it contiguously when returning to the caller.
        #
        # Naming clarification:
        #   _local_recv_slot_offsets: Receiver's view - offsets within MY local
        #       recv buffer where each peer's data lands
        #   _remote_write_offsets: Sender's view - offsets on REMOTE peers'
        #       recv buffers where I should write my data
        self._local_recv_slot_offsets = (
            torch.arange(self.world_size, dtype=torch.int64, device=device)
            * self._bytes_per_peer_slot
        )

        # Internal state tensors (byte-level sizes/offsets for the kernel).
        # These are written by _prepare_alltoallv_kernel and read by
        # device_alltoallv_dynamic. They persist across calls, enabling
        # automatic skip of the prep kernel when split sizes are static.
        self._send_sizes_bytes: Optional[torch.Tensor] = torch.empty(
            self.world_size, dtype=torch.int64, device=device
        )
        self._send_offsets_bytes: Optional[torch.Tensor] = torch.empty(
            self.world_size, dtype=torch.int64, device=device
        )
        self._recv_sizes_bytes: Optional[torch.Tensor] = torch.empty(
            self.world_size, dtype=torch.int64, device=device
        )

        # Flag to track if prep kernel has run since setup().
        # Used for auto-skipping prep on subsequent calls with static splits.
        self._prep_done: bool = False

        # Pre-allocate buffer for total tokens sum (GIN-safe operation).
        # Must be allocated before GIN is activated.
        self._total_tokens_buf = torch.empty(1, dtype=torch.int64, device=device)

        # BLOCK_SIZE for the preparation kernel (next power-of-2 of world_size).
        self._prep_block_size: int = triton.next_power_of_2(self.world_size)

        # Auto-tune: select kernel params once from the worst-case message
        # size (all max_input_tokens routed to a single peer).  This avoids
        # per-call GPU→CPU synchronisation (.item()) and keeps the hot path
        # entirely on GPU.
        worst_case_msg_bytes = max_input_tokens * self._bytes_per_token
        params = auto_tune_alltoallv_params(worst_case_msg_bytes)
        self._blocks_per_peer: int = params["blocks_per_peer"]
        self._num_warps: int = params["num_warps"]
        self._chunk_size: int = params["chunk_size"]

        # Internal comms state (populated by setup()).
        self._window: Any = None
        self._dev_win_ptr: Optional[int] = None
        self._src_info: Optional[int] = None
        self._remote_write_offsets: Optional[torch.Tensor] = None
        self._per_peer_blocks: Optional[torch.Tensor] = None
        self._peer_is_nvl: list[bool] = []

    # ------------------------------------------------------------------
    # Factory / caching
    # ------------------------------------------------------------------

    @classmethod
    def get_or_create(
        cls,
        comm: "TorchComm",
        max_input_tokens: int,
        D: int,
        dtype: torch.dtype,
        device: Union[str, torch.device],
        max_recv_tokens_per_peer: int,
        sync_buffer: bool = True,
    ) -> "AlltoallvOp":
        """Get or create a cached ``AlltoallvOp`` for the given parameters.

        The op is cached by communicator identity and buffer dimensions.
        If a matching op already exists it is returned; otherwise a new one
        is created and ``setup()`` is called automatically.

        Args:
            comm: TorchComms communicator.
            max_input_tokens: Maximum total input tokens per call.
            D: Hidden dimension.
            dtype: Tensor dtype.
            device: CUDA device.
            max_recv_tokens_per_peer: Maximum tokens receivable from each peer.
                For uniform distribution, this should be ``max_input_tokens // world_size``.
            sync_buffer: If True, enables buffer-ready synchronization.

        Returns:
            A set-up ``AlltoallvOp`` ready for ``alltoallv`` calls.
        """
        key = (
            id(comm),
            max_input_tokens,
            D,
            dtype,
            str(device),
            max_recv_tokens_per_peer,
            sync_buffer,
        )
        if key not in cls._CACHE:
            op = cls(
                comm,
                max_input_tokens,
                D,
                dtype,
                device,
                max_recv_tokens_per_peer=max_recv_tokens_per_peer,
                sync_buffer=sync_buffer,
            )
            op.setup()
            cls._CACHE[key] = op
        return cls._CACHE[key]

    @classmethod
    def clear_cache(cls) -> None:
        """Teardown and remove all cached ``AlltoallvOp`` instances."""
        for op in cls._CACHE.values():
            op.teardown()
        cls._CACHE.clear()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """Perform one-time collective comms setup.

        1. Pre-allocate completion counters (before GIN activation).
        2. Compute ``dst_offsets`` locally (no exchange needed for slotted
           layout since all ranks use identical recv_offsets).
        3. Create a comms window, register the recv buffer, obtain a device
           window handle, and register the send buffer.

        All ranks MUST call ``setup()`` collectively.

        Raises
        ------
        RuntimeError
            If ``setup()`` is called twice without an intervening
            ``teardown()``.
        """
        if self._window is not None:
            raise RuntimeError("AlltoallvOp.setup() called twice without teardown()")

        # Reset prep state for new setup cycle.
        self._prep_done = False

        # Pre-allocate completion counters BEFORE GIN activation.
        # GIN (GPU-Initiated NCCL) blocks regular CUDA allocations after
        # get_device_window() is called.
        from comms.pipes.collectives.triton import prewarm_completion_counters

        # Convert device to torch.device if string
        device_for_prewarm = (
            torch.device(self.device) if isinstance(self.device, str) else self.device
        )
        # pyre-ignore[6]: device_for_prewarm is always torch.device after the conditional
        prewarm_completion_counters(self.world_size, device_for_prewarm)

        # Compute dst_offsets locally (same for both slotted and packed).
        #
        # All ranks have identical recv_offsets because they all use the same
        # max_recv_tokens_per_peer and D. Therefore:
        #   recv_offsets = [0, slot_bytes, 2*slot_bytes, ..., (W-1)*slot_bytes]
        # dst_offsets[peer] = where MY data lands on PEER's recv buffer
        #                   = peer's recv_offsets[my_rank]
        #                   = my_rank * slot_bytes (same for all peers)
        #
        # Data always lands in slotted positions. For uniform distribution
        # (where all peers send max_recv_tokens_per_peer), slots are back-to-back,
        # so we return a zero-copy view of the contiguous buffer.
        # Non-uniform distribution is NOT supported.
        #
        # _remote_write_offsets[peer] = where MY data lands on PEER's recv buffer
        #                             = peer's _local_recv_slot_offsets[my_rank]
        #                             = my_rank * slot_bytes (same for all peers)
        self._remote_write_offsets = torch.full(
            (self.world_size,),
            self.rank * self._bytes_per_peer_slot,
            dtype=torch.int64,
            device=self.device,
        )

        self.comm.barrier(False)
        self._window = self.comm.new_window()

        # tensor_register maps recv_buf for one-sided operations (collective).
        self._window.tensor_register(self.recv_buf)
        self.comm.barrier(False)

        # Detect topology: classify peers as NVL or IB.
        # Gated behind TRITON_ALLTOALLV_ENABLE_IB=1 — when disabled (default),
        # all peers are treated as NVL and the kernel uses NVLink-only tuning.
        self._peer_is_nvl = [True] * self.world_size  # default: all NVL
        ib_enabled = os.environ.get("TRITON_ALLTOALLV_ENABLE_IB", "0") == "1"
        if ib_enabled and hasattr(self._window, "get_nvlink_address"):
            for peer in range(self.world_size):
                if peer != self.rank:
                    self._peer_is_nvl[peer] = self._window.get_nvlink_address(peer) != 0

        # Re-tune with topology info for per-peer block counts.
        worst_case_msg_bytes = self.max_input_tokens * self._bytes_per_token
        params = auto_tune_alltoallv_params(worst_case_msg_bytes, self._peer_is_nvl)
        self._blocks_per_peer = params["blocks_per_peer"]
        self._num_warps = params["num_warps"]
        self._chunk_size = params["chunk_size"]

        # Build per_peer_blocks tensor if topology is mixed.
        if params["per_peer_blocks"] is not None:
            self._per_peer_blocks = torch.tensor(
                params["per_peer_blocks"], dtype=torch.int32, device=self.device
            )
        else:
            self._per_peer_blocks = None

        # get_device_window triggers ncclDevCommCreate (enables GIN).
        # When sync_buffer is enabled, allocate 2x signal slots:
        # signal_id=0 for DATA_COMPLETE, signal_id=1 for BUFFER_READY
        signal_count = self.world_size * 2 if self.sync_buffer else self.world_size
        self._dev_win_ptr = self._window.get_device_window(signal_count=signal_count)

        # register_local_buffer maps send_buf for one-sided puts.
        self._src_info = self._window.register_local_buffer(self.send_buf)

        # Reset iteration counter to match fresh signal memory state.
        # Without this, cross-session hangs occur because the iteration counter
        # persists globally but signal memory is recreated per-window.
        from comms.pipes.collectives.triton.device_alltoallv_dynamic import (
            _reset_iteration_counter,
        )

        _reset_iteration_counter(self.world_size, device_for_prewarm)

    def teardown(self) -> None:
        """Release comms resources and all buffers.  Safe to call multiple times."""
        # Barrier ensures all ranks have completed their kernel launches
        # before any rank deregisters buffers. Without this, fast ranks
        # can deregister recv_buf (RDMA-unmap) while slow ranks' kernels
        # are still writing to it via one-sided put operations, corrupting
        # NCCL state and causing subsequent window lifecycles to hang.
        if self._window is not None:
            torch.cuda.synchronize()
            self.comm.barrier(False)
        # First deregister comms buffers
        if self._src_info is not None:
            self._window.deregister_local_buffer(self._src_info)
            self._src_info = None
        if self._window is not None:
            self._window.tensor_deregister()
            self._window = None
        self._dev_win_ptr = None
        self._per_peer_blocks = None
        self._peer_is_nvl = []

        # Release all GPU buffers to free memory immediately.
        # Without this, Python GC may defer collection and cause OOM.
        self.send_buf = None
        self.recv_buf = None
        self._local_recv_slot_offsets = None
        self._send_sizes_bytes = None
        self._send_offsets_bytes = None
        self._recv_sizes_bytes = None
        self._total_tokens_buf = None
        self._remote_write_offsets = None

        # Release MemPools AFTER buffers (buffers use the pools)
        self._send_pool = None
        self._recv_pool = None

    def __enter__(self) -> "AlltoallvOp":
        self.setup()
        return self

    def __exit__(self, *args: Any) -> None:
        self.teardown()

    # ------------------------------------------------------------------
    # CUDA Graph Support
    # ------------------------------------------------------------------

    def get_graph_pool_id(self) -> int:
        """Return the memory pool ID for CUDA graph capture.

        When capturing a CUDA graph that includes ``alltoallv`` calls, pass
        this pool ID to ``torch.cuda.graph()`` to ensure memory allocations
        during capture use the same transport-compatible pool.

        **Critical**: Buffer registration must occur BEFORE graph capture.
        Call ``setup()`` first, then warmup, then capture with this pool.

        Example::

            op = AlltoallvOp(comm, max_input_tokens=1024, D=4096,
                             dtype=torch.bfloat16, device="cuda:0")
            with op:
                # 1. Warmup (compiles kernels, outside graph capture)
                for _ in range(10):
                    _ = op.alltoallv(input_tensor, output_splits, input_splits)
                torch.cuda.synchronize()

                # 2. Capture graph using op's memory pool
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=op.get_graph_pool_id()):
                    _ = op.alltoallv(input_tensor, output_splits, input_splits,
                                     packed_output_tokens=total_tokens)

                # 3. Replay
                graph.replay()

        Returns:
            int: The CUDA memory pool ID to pass to ``torch.cuda.graph(pool=...)``.
        """
        # Use recv_pool since it's the larger allocation and will be used
        # for any internal allocations during alltoallv execution.
        return self._recv_pool.id

    # ------------------------------------------------------------------
    # Send buffer access (zero-copy path)
    # ------------------------------------------------------------------

    def get_send_buffer(self, num_tokens: int) -> torch.Tensor:
        """Return a 2-D view into the registered send buffer for zero-copy writes.

        The returned tensor has shape ``(num_tokens, D)`` and is backed by the
        pre-registered send buffer.  The caller can fill it directly
        (e.g. via ``torch.gather(..., out=send_buf)``) and then call
        :meth:`alltoallv_from_buffer` to perform the collective **without**
        copying into the send buffer.

        Args:
            num_tokens: Number of token rows to expose.  Must be
                ``≤ max_input_tokens``.

        Returns:
            A ``(num_tokens, D)`` tensor view of the registered send buffer.

        Raises:
            ValueError: If ``num_tokens > max_input_tokens``.

        Example::

            send_buf = op.get_send_buffer(512)
            torch.gather(hidden_states, 0,
                         routed_indices.unsqueeze(1).expand(-1, D),
                         out=send_buf)
            output = op.alltoallv_from_buffer(
                recv_splits, send_splits, num_input_tokens=512)
        """
        if num_tokens > self.max_input_tokens:
            raise ValueError(
                f"num_tokens ({num_tokens}) exceeds max_input_tokens "
                f"({self.max_input_tokens})"
            )
        return self.send_buf[: num_tokens * self.D].view(num_tokens, self.D)

    def fill_send_buffer(
        self,
        input_tensor: torch.Tensor,
        num_tokens: Optional[int] = None,
    ) -> torch.Tensor:
        """Fill the registered send buffer with data from input_tensor (GIN-safe).

        This method uses a Triton kernel internally, making it safe to call
        after ``setup()`` when GIN (GPU-Initiated NCCL) is active. Regular
        CUDA operations like ``send_buf[:] = input`` fail after GIN activation
        because GIN locks GPU memory pages for RDMA access.

        Use this method for back-to-back iterations where you need to update
        the send buffer contents between collective calls.

        Args:
            input_tensor: Source tensor of shape ``(N, D)`` or ``(N * D,)``.
            num_tokens: Number of tokens to copy. If ``None``, inferred from
                ``input_tensor.shape[0]`` (for 2D) or
                ``input_tensor.numel() // D`` (for 1D).

        Returns:
            A ``(num_tokens, D)`` view of the send buffer containing the
            copied data.

        Raises:
            ValueError: If ``num_tokens > max_input_tokens``.
            RuntimeError: If ``input_tensor`` has fewer elements than required.

        Example (back-to-back iterations)::

            op = AlltoallvOp(comm, max_input_tokens, D, dtype, device)
            with op:
                for iteration in range(num_iters):
                    # Compute new data each iteration (e.g., from a Triton kernel)
                    input_data = compute_tokens_to_send()

                    # Fill send buffer (GIN-safe) and call collective
                    op.fill_send_buffer(input_data)
                    output = op.alltoallv_from_buffer(
                        output_split_sizes, input_split_sizes,
                        num_input_tokens=num_tokens
                    )
        """
        # Infer num_tokens from input shape if not provided
        if num_tokens is None:
            if input_tensor.dim() == 2:
                num_tokens = input_tensor.shape[0]
            else:
                num_tokens = input_tensor.numel() // self.D

        if num_tokens > self.max_input_tokens:
            raise ValueError(
                f"num_tokens ({num_tokens}) exceeds max_input_tokens "
                f"({self.max_input_tokens})"
            )

        # Validate input tensor size
        expected_elems = num_tokens * self.D
        if input_tensor.numel() < expected_elems:
            raise RuntimeError(
                f"input_tensor has {input_tensor.numel()} elements but "
                f"num_tokens={num_tokens} requires {expected_elems} elements"
            )

        # Use GIN-safe Triton copy kernel
        n_elems = num_tokens * self.D
        grid = (triton.cdiv(n_elems, 1024),)
        _triton_copy_1d_kernel[grid](
            input_tensor.reshape(-1),
            self.send_buf,
            n_elems,
            # pyre-fixme[6]: Triton constexpr accepts int at runtime
            BLOCK_SIZE=1024,
        )

        return self.send_buf[:n_elems].view(num_tokens, self.D)

    # ------------------------------------------------------------------
    # Public collective calls
    # ------------------------------------------------------------------

    def alltoallv(
        self,
        input_tensor: torch.Tensor,
        output_split_sizes: torch.Tensor,
        input_split_sizes: torch.Tensor,
        packed_output_tokens: Optional[int] = None,
        skip_prep: Optional[bool] = None,
    ) -> torch.Tensor:
        """Perform alltoallv collective.

        Args:
            input_tensor: Token tensor ``(N, D)`` where ``N ≤ max_input_tokens``.
            output_split_sizes: int64 tensor ``[world_size]`` — per-peer
                token counts to receive.
            input_split_sizes: int64 tensor ``[world_size]`` — per-peer
                token counts to send.
            packed_output_tokens: Pre-computed total output tokens. If provided,
                avoids a GPU→CPU sync (``.item()`` call) to sum output_split_sizes.
                Required for CUDA graph capture.
            skip_prep: Controls whether to skip the prep kernel:
                 - ``None`` (default): Auto-detect. Runs prep on first call after
                   setup(), skips on subsequent calls. Best for
                   static splits (typical MoE scenario).
                 - ``True``: Always skip. User guarantees sizes/offsets are valid.
                 - ``False``: Always run. Use when split sizes change between calls.

        Returns:
            ``(sum(output_split_sizes), D)`` packed tensor. Only uniform
            distribution is supported (all peers send exactly
            ``max_recv_tokens_per_peer`` tokens).
        """
        self._ensure_setup()
        iT = input_tensor.shape[0]
        if iT > self.max_input_tokens:
            raise ValueError(
                f"input_tensor has {iT} rows but max_input_tokens is "
                f"{self.max_input_tokens}"
            )

        # Copy into the pre-registered send buffer using a Triton kernel.
        # Regular CUDA copy (tensor.copy_) fails when GIN is active
        # (cudaErrorHostMemoryAlreadyRegistered).  Triton kernels use
        # device-side loads/stores that are GIN-safe.
        n_elems = iT * self.D
        grid = (triton.cdiv(n_elems, 1024),)
        _triton_copy_1d_kernel[grid](
            input_tensor.reshape(-1),
            self.send_buf,
            n_elems,
            # pyre-fixme[6]: Triton constexpr accepts int at runtime
            BLOCK_SIZE=1024,
        )

        return self._run_alltoallv(
            output_split_sizes,
            input_split_sizes,
            packed_output_tokens,
            skip_prep,
        )

    def alltoallv_from_buffer(
        self,
        output_split_sizes: torch.Tensor,
        input_split_sizes: torch.Tensor,
        num_input_tokens: int,
        packed_output_tokens: Optional[int] = None,
        skip_prep: Optional[bool] = None,
    ) -> torch.Tensor:
        """Perform alltoallv using data already in the send buffer.

        The caller must have previously obtained the send buffer via
        :meth:`get_send_buffer` and filled it with data.  This method skips
        the ``copy_`` into the send buffer, achieving **true zero-copy**
        end-to-end.

        Args:
            output_split_sizes: int64 tensor ``[world_size]`` — per-peer
                token counts to receive.
            input_split_sizes: int64 tensor ``[world_size]`` — per-peer
                token counts to send.
            num_input_tokens: Number of valid token rows in the send buffer.
                Must match the ``num_tokens`` passed to
                :meth:`get_send_buffer`.
            packed_output_tokens: Pre-computed total output tokens. If provided,
                avoids a GPU→CPU sync (``.item()`` call) to sum output_split_sizes.
                Required for CUDA graph capture.
            skip_prep: Controls whether to skip the prep kernel:
                 - ``None`` (default): Auto-detect. Runs prep on first call after
                   setup(), skips on subsequent calls. Best for
                   static splits (typical MoE scenario).
                 - ``True``: Always skip. User guarantees sizes/offsets are valid.
                 - ``False``: Always run. Use when split sizes change between calls.

        Returns:
            ``(sum(output_split_sizes), D)`` packed tensor. Only uniform
            distribution is supported (all peers send exactly
            ``max_recv_tokens_per_peer`` tokens).
        """
        self._ensure_setup()
        if num_input_tokens > self.max_input_tokens:
            raise ValueError(
                f"num_input_tokens={num_input_tokens} exceeds "
                f"max_input_tokens={self.max_input_tokens}"
            )

        return self._run_alltoallv(
            output_split_sizes,
            input_split_sizes,
            packed_output_tokens,
            skip_prep,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_setup(self) -> None:
        """Raise if ``setup()`` has not been called."""
        if self._window is None or self._src_info is None:
            raise RuntimeError(
                "AlltoallvOp has not been set up.  "
                "Call setup() or use the context manager (with statement) first."
            )

    def _run_alltoallv(
        self,
        output_split_sizes: torch.Tensor,
        input_split_sizes: torch.Tensor,
        packed_output_tokens: Optional[int] = None,
        skip_prep: Optional[bool] = None,
    ) -> torch.Tensor:
        """Shared kernel-launch logic for both copy-in and zero-copy paths.

        Args:
            output_split_sizes: Per-peer token counts to receive.
            input_split_sizes: Per-peer token counts to send.
            packed_output_tokens: Total output tokens for packed mode.
            skip_prep: Controls whether to skip the _prepare_alltoallv_kernel:
                - None (default): Auto-detect. Runs prep on first call after
                  setup(), skips on subsequent calls. Best for static splits
                  (typical MoE scenario).
                - True: Always skip. User guarantees sizes/offsets are valid.
                - False: Always run. Use when split sizes change between calls.

        Note:
            Cached byte sizes/offsets are stored in:
            - self._send_sizes_bytes: byte sizes for sending to each peer
            - self._send_offsets_bytes: byte offsets into send buffer
            - self._recv_sizes_bytes: byte sizes for receiving from each peer
            These persist across calls and are computed by _prepare_alltoallv_kernel.
        """
        # Determine whether to run prep kernel.
        # Auto-detect: skip if prep has already run since setup().
        should_run_prep = (skip_prep is None and not self._prep_done) or (
            skip_prep is False
        )

        if should_run_prep:
            # Single Triton kernel: token splits → byte sizes and contiguous
            # offsets.  Entirely GPU-side, no CUDA runtime calls (GIN-safe).
            _prepare_alltoallv_kernel[(1,)](
                input_split_sizes,
                output_split_sizes,
                self._bytes_per_token,
                self._send_sizes_bytes,
                self._send_offsets_bytes,
                self._recv_sizes_bytes,
                self.world_size,
                # pyre-fixme[6]: Triton constexpr accepts int at runtime
                BLOCK_SIZE=self._prep_block_size,
            )
            self._prep_done = True

        assert self._dev_win_ptr is not None
        assert self._src_info is not None
        assert self._remote_write_offsets is not None
        assert self._send_sizes_bytes is not None
        assert self._send_offsets_bytes is not None
        assert self._recv_sizes_bytes is not None

        device_alltoallv_dynamic(
            self.send_buf,
            self.recv_buf,
            self._send_sizes_bytes,
            self._send_offsets_bytes,
            self._recv_sizes_bytes,
            self._local_recv_slot_offsets,
            self._remote_write_offsets,
            self._dev_win_ptr,
            self._src_info,
            self.rank,
            self.world_size,
            auto_tune=False,
            blocks_per_peer=self._blocks_per_peer,
            num_warps=self._num_warps,
            chunk_size=self._chunk_size,
            sync_buffer=self.sync_buffer,
            per_peer_blocks=self._per_peer_blocks,
        )

        # NOTE: Iteration counter is now managed internally by
        # device_alltoallv_dynamic, which auto-increments after each call.
        # Cross-rank synchronization is handled via the BUFFER_READY
        # signal/wait protocol. At each iteration N > 0, senders wait for
        # BUFFER_READY=N from receivers before sending. Receivers signal
        # BUFFER_READY=N at the start of iteration N. This naturally chains
        # iterations across ranks without requiring an explicit barrier.

        # Return a 2-D view of the receive buffer with fixed-slot layout.
        # Each slot has max_recv_tokens_per_peer rows, regardless of actual data.
        # Shape: (world_size * max_recv_tokens_per_peer, D)
        slotted_output = self.recv_buf.view(
            self.world_size * self.max_recv_tokens_per_peer, self.D
        )

        # Packed output: For uniform distribution (where each peer sends
        # exactly max_recv_tokens_per_peer), data lands contiguously because
        # slots are back-to-back with no gaps. We return a direct view.
        #
        # Non-uniform distribution is NOT supported and will raise an error.

        # Determine total tokens
        if packed_output_tokens is not None:
            total_tokens = packed_output_tokens
        else:
            # Sum using GIN-safe kernel to avoid CUDA error when GIN is active
            _sum_int64_kernel[(1,)](
                output_split_sizes,
                self._total_tokens_buf,
                # pyre-fixme[6]: Triton constexpr accepts int at runtime
                N=self.world_size,
            )
            total_tokens = int(self._total_tokens_buf.item())

        # Check if distribution is uniform (all peers send max_recv_tokens_per_peer)
        # For uniform distribution, slots are back-to-back, so slotted = packed.
        expected_uniform_total = self.world_size * self.max_recv_tokens_per_peer
        if total_tokens != expected_uniform_total:
            raise ValueError(
                f"Non-uniform distribution not supported. "
                f"Expected total tokens: {expected_uniform_total} "
                f"(world_size={self.world_size} × "
                f"max_recv_tokens_per_peer={self.max_recv_tokens_per_peer}), "
                f"but got: {total_tokens}. "
                f"All peers must send exactly max_recv_tokens_per_peer tokens."
            )

        # Uniform distribution: return direct view (zero-copy)
        return slotted_output[:total_tokens]
