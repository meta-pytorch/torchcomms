# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-unsafe
"""High-level alltoallv operation with MSL-compatible signature.

``AlltoallvOp`` provides a simplified API matching the MSL ``all_to_all_op``
signature while preserving the zero-copy one-sided-put architecture of
the underlying ``device_alltoallv_dynamic`` kernel.

Zero-Copy Buffer Ownership
--------------------------
``AlltoallvOp`` uses zero-copy buffer ownership where the user provides
both input and output tensors, which become the GIN-registered send/recv
buffers. This achieves **100% memory reduction** - no internal allocation.

**Both input_tensor and output_tensor are MANDATORY parameters.**

**IMPORTANT**: Tensors must be allocated using ``alloc_buffer()`` which
uses ncclMemAlloc (cuMem APIs). Regular ``torch.empty()`` uses cudaMalloc
which is NOT compatible with GIN registration.

Usage::

    op = AlltoallvOp(comm, max_tokens, D, dtype, device, max_recv_per_peer)

    with op:
        # Allocate GIN-compatible tensors (pool managed internally)
        input_tensor = op.alloc_buffer((max_tokens, D))
        output_tensor = op.alloc_buffer((world_size * max_recv_per_peer, D))

        # First call: op takes ownership of both tensors
        input_tensor[:] = data_for_iteration_0
        result = op.alltoallv(input_tensor, output_tensor,
                              output_splits, input_splits)

        # Subsequent calls: update input_tensor in-place, then call alltoallv
        input_tensor[:] = data_for_iteration_1
        result = op.alltoallv(input_tensor, output_tensor,
                              output_splits, input_splits)

**Key Points**:
- Use ``alloc_buffer()`` to allocate GIN-compatible tensors (REQUIRED)
- User's input_tensor IS the send buffer (no copy)
- User's output_tensor IS the recv buffer (data arrives directly)
- Update input_tensor contents in-place between calls
- Must use the same tensors on every call (same memory addresses)
- To use new tensors, call release_buffers() first

Output Layout
-------------
The output is always returned in **packed uniform layout**:
    - Shape: ``(sum(output_split_sizes), D)``
    - Contiguous: ``[peer_0_data, peer_1_data, ..., peer_{W-1}_data]``
    - Zero-copy view of output_tensor

**IMPORTANT**: Only uniform distribution is supported. All peers must send
exactly ``max_recv_tokens_per_peer`` tokens.

CUDA Graph Support
------------------
AlltoallvOp is fully CUDA graph compatible::

    op = AlltoallvOp(comm, max_tokens, D, dtype, device, max_recv_per_peer)

    with op:
        # Allocate GIN-compatible tensors
        input_tensor = op.alloc_buffer((max_tokens, D))
        output_tensor = op.alloc_buffer((total_recv_tokens, D))

        # Warmup
        input_tensor[:] = warmup_data
        op.alltoallv(input_tensor, output_tensor, output_splits, input_splits)

        # Capture graph
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = op.alltoallv(input_tensor, output_tensor,
                                  output_splits, input_splits,
                                  packed_output_tokens=total_tokens)

        # Replay: update input in-place, then replay
        for i in range(num_iterations):
            input_tensor[:] = iteration_data[i]
            graph.replay()

Output lifetime
---------------
The returned tensor is a **view** of the user-provided output_tensor.
Since the user owns output_tensor, its lifetime is controlled by the user.
The tensor remains valid as long as the user keeps it alive.
"""

from typing import Any, Optional, Sequence, TYPE_CHECKING, Union

import torch
import torchcomms
import triton  # @manual
import triton.language as tl  # @manual
from comms.pipes.collectives.triton.device_alltoallv_dynamic import (
    auto_tune_alltoallv_params,
    device_alltoallv_dynamic,
)


# =============================================================================
# Triton kernels
# =============================================================================


@triton.jit
def _sum_int64_kernel(input_ptr, output_ptr, N: tl.constexpr):
    """Kernel to compute sum of int64 tensor."""
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
    """Token splits → byte sizes/offsets.  Single-block.

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

    Zero-Copy Buffer Ownership
    ~~~~~~~~~~~~~~~~~~~~~~~~~~
    User provides both input and output tensors (MANDATORY). On the first
    alltoallv() call, these tensors are registered as GIN buffers:

    - input_tensor → GIN local registration (send buffer)
    - output_tensor → GIN remote registration (recv buffer)
    - **100% memory reduction** - no internal buffers
    - User updates input_tensor in-place between calls
    - Data arrives directly in output_tensor

    **IMPORTANT**: Tensors must be allocated using ``alloc_buffer()`` which
    uses ncclMemAlloc (cuMem APIs). Regular ``torch.empty()`` uses cudaMalloc
    which is NOT compatible with GIN registration.

    Usage
    ~~~~~
    ::

        op = AlltoallvOp(comm, max_tokens, D, dtype, device, max_recv_per_peer)

        with op:
            # Allocate GIN-compatible tensors (pool managed internally by op)
            input_tensor = op.alloc_buffer((max_tokens, D))
            output_tensor = op.alloc_buffer((world_size * max_recv_per_peer, D))

            # Use the op
            input_tensor[:] = data
            result = op.alltoallv(input_tensor, output_tensor,
                                  output_splits, input_splits)

    Buffer Layout
    ~~~~~~~~~~~~~
    * **Send buffer** (user's input_tensor):
      ``[tokens_for_peer_0, tokens_for_peer_1, …]``
    * **Receive buffer** (user's output_tensor):
      ``[slot_0, slot_1, …, slot_{W-1}]`` where each slot is
      ``max_recv_tokens_per_peer * D`` elements

    Parameters
    ----------
    comm : TorchComm
        TorchComms communicator.
    max_input_tokens : int
        Maximum total input tokens. Determines minimum size of input_tensor.
    D : int
        Hidden dimension (columns per token row).
    dtype : torch.dtype
        Element data type.
    device : str | torch.device
        CUDA device.
    max_recv_tokens_per_peer : int
        Maximum tokens receivable from each peer. Determines minimum size
        of output_tensor.

    Notes
    -----
    * Use ``alloc_buffer()`` to allocate GIN-compatible tensors after
      ``setup()`` (or inside the ``with`` block).
    * The op auto-tunes kernel parameters (``blocks_per_peer``,
      ``num_warps``, ``chunk_size``) per call based on the maximum per-peer
      message size.
    * The iteration counter is auto-incremented after every collective call.
      Do **not** mix raw ``device_alltoallv_dynamic`` calls on the same
      window — signal counters will diverge and cause deadlocks.
    * **No internal buffer allocation occurs.** User MUST provide both
      input_tensor and output_tensor on every alltoallv() call.
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
        self.max_recv_tokens_per_peer: int = max_recv_tokens_per_peer

        self._elem_bytes: int = torch.tensor([], dtype=dtype).element_size()
        # Bytes per token row: D elements × element_size
        self._bytes_per_token: int = D * self._elem_bytes

        # Fixed slot size per peer in the receive buffer (bytes).
        self._bytes_per_peer_slot: int = (
            max_recv_tokens_per_peer * self._bytes_per_token
        )

        # -----------------------------------------------------------------
        # Zero-copy buffer ownership: NO internal allocation
        # User provides both input and output tensors on first alltoallv() call.
        # These tensors become the GIN-registered send/recv buffers.
        #
        # IMPORTANT: We only store data pointers, not tensor references.
        # This allows del tensor to return memory to the pool for reuse by
        # other operations captured in the same CUDA graph. The pool-based
        # caching allocator ensures the same addresses are reused each
        # iteration during graph capture, so GIN registration remains valid.
        # -----------------------------------------------------------------
        self._backend: str = comm.get_backend()

        # Store only data pointers for validation, not tensor references.
        # This allows the caching allocator to reuse memory within the graph.
        self._send_data_ptr: Optional[int] = None
        self._recv_data_ptr: Optional[int] = None

        # Track if buffers have been registered
        self._buffers_owned: bool = False

        # Required sizes for validation
        self._required_send_elems = max_input_tokens * D
        self._required_recv_elems = self.world_size * max_recv_tokens_per_peer * D

        # Compute recv_offsets: fixed slot layout where peer i's data lands at slot i.
        self._local_recv_slot_offsets: Optional[torch.Tensor] = None

        # Internal state tensors (byte-level sizes/offsets for the kernel).
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
        self._prep_done: bool = False

        # Pre-allocate buffer for total tokens sum.
        self._total_tokens_buf = torch.empty(1, dtype=torch.int64, device=device)

        # BLOCK_SIZE for the preparation kernel (next power-of-2 of world_size).
        self._prep_block_size: int = triton.next_power_of_2(self.world_size)

        # Auto-tune: select kernel params once from the worst-case message size.
        worst_case_msg_bytes = max_input_tokens * self._bytes_per_token
        params = auto_tune_alltoallv_params(worst_case_msg_bytes)
        self._blocks_per_peer: int = params["blocks_per_peer"]
        self._num_warps: int = params["num_warps"]
        self._chunk_size: int = params["chunk_size"]

        # Internal comms state (populated by setup()).
        self._window: Any = None
        self._dev_win_ptr: Optional[int] = None
        self._src_info: Optional[tuple[int, int, int]] = None
        self._remote_write_offsets: Optional[torch.Tensor] = None
        self._buffer_pool: Optional["torch.cuda.MemPool"] = None

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
    # Buffer allocation helpers
    # ------------------------------------------------------------------

    def alloc_buffer(
        self,
        shape: Union[int, Sequence[int]],
    ) -> torch.Tensor:
        """Allocate a GIN-compatible tensor from the op's internal memory pool.

        Use this method to allocate input and output tensors that will be used
        with this op. Tensors allocated this way use the NCCL memory allocator
        (ncclMemAlloc) which is compatible with GIN registration.

        Regular tensors allocated with torch.empty() use cudaMalloc which is
        NOT compatible with GIN. Using such tensors will cause "CUDA failure 1
        'invalid argument'" errors during registration.

        The memory pool is created and owned internally by this op. You do NOT
        need to manage pool lifecycle - it is automatically created during
        setup() and destroyed during teardown().

        Args:
            shape: Tensor shape (int for 1-D, or a sequence of ints).
                For input buffers, use ``(max_input_tokens, D)``.
                For output buffers, use ``(world_size * max_recv_tokens_per_peer, D)``.

        Returns:
            A zero-initialized tensor allocated from the GIN-compatible pool.

        Example:
            op = AlltoallvOp(comm, max_tokens, D, dtype, device, max_recv_per_peer)

            with op:
                # Allocate GIN-compatible buffers (pool managed internally)
                input_tensor = op.alloc_buffer((max_tokens, D))
                output_tensor = op.alloc_buffer((world_size * max_recv_per_peer, D))

                input_tensor[:] = data
                result = op.alltoallv(input_tensor, output_tensor,
                                      output_splits, input_splits)
        """
        if self._buffer_pool is None:
            raise RuntimeError(
                "alloc_buffer() called before setup(). "
                "Use the context manager (with statement) or call setup() first."
            )

        # Normalize shape to a tuple for torch.zeros
        size: Sequence[int] = (shape,) if isinstance(shape, int) else tuple(shape)
        with torch.cuda.use_mem_pool(self._buffer_pool):
            tensor = torch.zeros(size, dtype=self.dtype, device=self.device)
        return tensor

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """Perform one-time collective comms setup.

        Creates the internal memory pool for GIN-compatible buffer allocation
        and prepares internal state. Window creation and buffer registration
        are deferred until the first alltoallv() call when user provides their
        tensors.

        All ranks MUST call ``setup()`` collectively.

        Raises
        ------
        RuntimeError
            If ``setup()`` is called twice without an intervening
            ``teardown()``.
        """
        if self._setup_done:
            raise RuntimeError("AlltoallvOp.setup() called twice without teardown()")

        # Reset prep state for new setup cycle.
        self._prep_done = False

        # Create the internal buffer pool using torchcomms memory allocator.
        # This pool uses ncclMemAlloc which allocates via cuMem APIs (cuMemCreate,
        # cuMemMap) that are compatible with GIN registration. Regular torch.empty()
        # uses cudaMalloc which is NOT GIN compatible.
        allocator = torchcomms.get_mem_allocator(self._backend)
        self._buffer_pool = torch.cuda.MemPool(allocator)

        # Pre-allocate completion counters.
        from comms.pipes.collectives.triton import prewarm_completion_counters

        device_for_prewarm = (
            torch.device(self.device) if isinstance(self.device, str) else self.device
        )
        # pyre-ignore[6]: device_for_prewarm is always torch.device after the conditional
        prewarm_completion_counters(self.world_size, device_for_prewarm)

        # Compute dst_offsets locally.
        # _remote_write_offsets[peer] = where MY data lands on PEER's recv buffer
        #                             = peer's _local_recv_slot_offsets[my_rank]
        #                             = my_rank * slot_bytes (same for all peers)
        self._remote_write_offsets = torch.full(
            (self.world_size,),
            self.rank * self._bytes_per_peer_slot,
            dtype=torch.int64,
            device=self.device,
        )

        # Local recv slot offsets for fixed slot layout
        self._local_recv_slot_offsets = (
            torch.arange(self.world_size, dtype=torch.int64, device=self.device)
            * self._bytes_per_peer_slot
        )

        # Window creation and buffer registration are DEFERRED until
        # _complete_buffer_setup() which is called on first alltoallv()
        # after user provides tensors. This ensures GIN activation happens
        # AFTER user's tensors are allocated.
        self._window = None
        self._setup_done = True

    def _capture_buffer_addresses(
        self,
        input_tensor: torch.Tensor,
        output_tensor: torch.Tensor,
    ) -> None:
        """Capture data pointers from user-provided tensors for validation.

        Records the memory addresses of user's tensors. These addresses are used
        to validate that the same memory locations are used on subsequent calls,
        which is required for GIN registration validity.

        IMPORTANT: We intentionally do NOT store tensor references here. This
        allows `del tensor` to return memory back to the caching allocator's
        pool for reuse by other operations captured in the same CUDA graph.
        The pool-based caching allocator ensures the same addresses are reused
        each iteration during graph capture, so GIN registration remains valid.

        Args:
            input_tensor: User's input tensor (will be used as send buffer).
            output_tensor: User's output tensor (will be used as receive buffer).

        Raises:
            RuntimeError: If buffers have already been captured.
        """
        if self._buffers_owned:
            raise RuntimeError(
                "Buffer addresses have already been captured. "
                "Call release_buffers() before using different tensors."
            )

        # Validate input tensor size
        if input_tensor.numel() < self._required_send_elems:
            raise ValueError(
                f"input_tensor has {input_tensor.numel()} elements but "
                f"requires at least {self._required_send_elems} elements "
                f"(max_input_tokens={self.max_input_tokens} × D={self.D})"
            )

        # Validate output tensor size
        if output_tensor.numel() < self._required_recv_elems:
            raise ValueError(
                f"output_tensor has {output_tensor.numel()} elements but "
                f"requires at least {self._required_recv_elems} elements "
                f"(world_size={self.world_size} × "
                f"max_recv_tokens_per_peer={self.max_recv_tokens_per_peer} × D={self.D})"
            )

        # Store data pointers for validation, not tensor references.
        # The caching allocator guarantees these addresses remain valid
        # within a captured CUDA graph since the pool holds the memory.
        self._send_data_ptr = input_tensor.data_ptr()
        self._recv_data_ptr = output_tensor.data_ptr()
        self._buffers_owned = True

    def _complete_buffer_setup(
        self,
        input_tensor: torch.Tensor,
        output_tensor: torch.Tensor,
    ) -> None:
        """Complete buffer setup after user's tensors are owned.

        Creates the comms window and performs GIN registration of user's tensors.
        Called automatically on first alltoallv() call.

        This is where GIN activation happens, AFTER user's tensors are allocated.

        Args:
            input_tensor: User's input tensor for GIN local registration.
            output_tensor: User's output tensor for GIN remote registration.
        """
        if not self._buffers_owned:
            raise RuntimeError(
                "_complete_buffer_setup() called before buffers are owned"
            )

        if self._dev_win_ptr is not None:
            return  # Already completed

        # Create window NOW, after user's tensors are allocated.
        # This ensures GIN activation happens after tensor allocation.
        self.comm.barrier(False)
        self._window = self.comm.new_window()

        # Register user's output_tensor for one-sided operations
        self._window.tensor_register(output_tensor.view(-1))
        self.comm.barrier(False)

        # Enable GIN
        signal_count = self.world_size * 2 if self.sync_buffer else self.world_size
        self._dev_win_ptr = self._window.get_device_window(signal_count=signal_count)

        # Register user's input_tensor for one-sided puts
        self._src_info = self._window.register_local_buffer(input_tensor.view(-1))

        # Reset iteration counter
        from comms.pipes.collectives.triton.device_alltoallv_dynamic import (
            _reset_iteration_counter,
        )

        device_for_reset = (
            torch.device(self.device) if isinstance(self.device, str) else self.device
        )
        _reset_iteration_counter(self.world_size, device_for_reset)

    def release_buffers(self) -> None:
        """Release ownership of send/recv buffers.

        Deregisters buffers from GIN. After this call, the user can safely
        deallocate their tensors. New tensors can be provided on the next
        alltoallv() call.
        """
        if not self._buffers_owned:
            return

        # Deregister from GIN
        if self._src_info is not None and self._window is not None:
            self._window.deregister_local_buffer(*self._src_info)
            self._src_info = None

        if self._window is not None:
            self._window.tensor_deregister()
            self._window = None

        self._dev_win_ptr = None
        self._send_data_ptr = None
        self._recv_data_ptr = None
        self._buffers_owned = False

    def teardown(self) -> None:
        """Release comms resources. Safe to call multiple times.

        Deregisters user's tensors from GIN. The tensors themselves are NOT
        deallocated (user owns the memory).
        """
        if self._src_info is not None:
            self._window.deregister_local_buffer(*self._src_info)
            self._src_info = None
        if self._window is not None:
            self._window.tensor_deregister()
            self._window = None
        self._dev_win_ptr = None

        # Clear data pointers (memory returns to pool, not deallocated)
        self._send_data_ptr = None
        self._recv_data_ptr = None
        self._local_recv_slot_offsets = None

        # Release the buffer pool
        self._buffer_pool = None

        self._buffers_owned = False
        self._prep_done = False
        self._setup_done = False
        self._recv_sizes_bytes = None
        self._total_tokens_buf = None
        self._remote_write_offsets = None
        self._buffers_owned = False
        self._setup_done = False

    def __enter__(self) -> "AlltoallvOp":
        self.setup()
        return self

    def __exit__(self, *args: Any) -> None:
        self.teardown()

    # ------------------------------------------------------------------
    # Public collective call
    # ------------------------------------------------------------------

    def alltoallv(
        self,
        input_tensor: torch.Tensor,
        output_tensor: torch.Tensor,
        output_split_sizes: torch.Tensor,
        input_split_sizes: torch.Tensor,
        packed_output_tokens: Optional[int] = None,
        skip_prep: Optional[bool] = None,
    ) -> torch.Tensor:
        """Perform alltoallv collective.

        On the first call, the op takes ownership of both tensors. They become
        the GIN-registered send/recv buffers. On subsequent calls, the user
        updates input_tensor contents in-place before calling this method.

        Args:
            input_tensor: Token tensor ``(N, D)`` where ``N ≤ max_input_tokens``.
                On first call, this tensor is registered as the send buffer.
                On subsequent calls, must be the same tensor (same memory address).
                User updates contents in-place before each call.
            output_tensor: Pre-allocated output tensor with at least
                ``world_size * max_recv_tokens_per_peer * D`` elements.
                On first call, this tensor is registered as the recv buffer.
                On subsequent calls, must be the same tensor (same memory address).
                Data arrives directly here - no copy needed.
            output_split_sizes: int64 tensor ``[world_size]`` — per-peer
                token counts to receive.
            input_split_sizes: int64 tensor ``[world_size]`` — per-peer
                token counts to send.
            packed_output_tokens: Pre-computed total output tokens. If provided,
                avoids a GPU→CPU sync. Required for CUDA graph capture.
            skip_prep: Controls whether to skip the prep kernel.

        Returns:
            ``(sum(output_split_sizes), D)`` packed tensor view of output_tensor.

        Raises:
            ValueError: If tensors are too small.
            RuntimeError: If different tensors are passed after ownership is taken.

        Example:
            # Allocate tensors once
            input_tensor = torch.empty(max_tokens, D, dtype=dtype, device=device)
            output_tensor = torch.empty(world_size * max_recv_per_peer, D,
                                        dtype=dtype, device=device)

            op = AlltoallvOp(comm, max_tokens, D, dtype, device, max_recv_per_peer)
            with op:
                # First call: takes ownership
                input_tensor[:] = initial_data
                result = op.alltoallv(input_tensor, output_tensor,
                                      output_splits, input_splits)

                # Subsequent calls: update input_tensor in-place, then call
                input_tensor[:] = new_data
                result = op.alltoallv(input_tensor, output_tensor,
                                      output_splits, input_splits)
        """
        self._ensure_setup()

        iT = input_tensor.shape[0]
        if iT > self.max_input_tokens:
            raise ValueError(
                f"input_tensor has {iT} rows but max_input_tokens is "
                f"{self.max_input_tokens}"
            )

        # First call: capture buffer addresses for validation
        if not self._buffers_owned:
            self._capture_buffer_addresses(input_tensor, output_tensor)
            self._complete_buffer_setup(input_tensor, output_tensor)
        else:
            # Subsequent calls: verify same tensor addresses are used.
            # We only store data pointers, not tensor references, to allow
            # del tensor to return memory to the pool for reuse by other
            # operations captured in the same CUDA graph.
            if input_tensor.data_ptr() != self._send_data_ptr:
                raise RuntimeError(
                    "input_tensor memory address changed after ownership was taken. "
                    f"Expected data_ptr={self._send_data_ptr}, "
                    f"got data_ptr={input_tensor.data_ptr()}. "
                    "Use the same tensor and update its contents in-place, "
                    "or call release_buffers() first."
                )
            if output_tensor.data_ptr() != self._recv_data_ptr:
                raise RuntimeError(
                    "output_tensor memory address changed after ownership was taken. "
                    f"Expected data_ptr={self._recv_data_ptr}, "
                    f"got data_ptr={output_tensor.data_ptr()}. "
                    "Use the same tensor on every call, or call release_buffers() first."
                )

        return self._run_alltoallv(
            input_tensor,
            output_tensor,
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
        if not self._setup_done:
            raise RuntimeError(
                "AlltoallvOp has not been set up.  "
                "Call setup() or use the context manager (with statement) first."
            )

    def _run_alltoallv(
        self,
        input_tensor: torch.Tensor,
        output_tensor: torch.Tensor,
        output_split_sizes: torch.Tensor,
        input_split_sizes: torch.Tensor,
        packed_output_tokens: Optional[int] = None,
        skip_prep: Optional[bool] = None,
    ) -> torch.Tensor:
        """Shared kernel-launch logic.

        Args:
            input_tensor: User's input tensor (send buffer).
            output_tensor: User's output tensor (receive buffer).
            output_split_sizes: Per-peer token counts to receive.
            input_split_sizes: Per-peer token counts to send.
            packed_output_tokens: Total output tokens for packed mode.
            skip_prep: Controls whether to skip the _prepare_alltoallv_kernel.
        """
        assert self._local_recv_slot_offsets is not None

        # Determine whether to run prep kernel.
        should_run_prep = (skip_prep is None and not self._prep_done) or (
            skip_prep is False
        )

        if should_run_prep:
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
        assert self._local_recv_slot_offsets is not None

        # Use tensors passed as arguments (flattened views)
        send_buf = input_tensor.view(-1)
        recv_buf = output_tensor.view(-1)

        device_alltoallv_dynamic(
            send_buf,
            recv_buf,
            # pyre-fixme[6]: Pyre doesn't narrow Optional after assert
            self._send_sizes_bytes,
            # pyre-fixme[6]: Pyre doesn't narrow Optional after assert
            self._send_offsets_bytes,
            # pyre-fixme[6]: Pyre doesn't narrow Optional after assert
            self._recv_sizes_bytes,
            # pyre-fixme[6]: Pyre doesn't narrow Optional after assert
            self._local_recv_slot_offsets,
            # pyre-fixme[6]: Pyre doesn't narrow Optional after assert
            self._remote_write_offsets,
            # pyre-fixme[6]: Pyre doesn't narrow Optional after assert
            self._dev_win_ptr,
            # pyre-fixme[6]: Pyre doesn't narrow Optional after assert
            self._src_info,
            self.rank,
            self.world_size,
            auto_tune=False,
            blocks_per_peer=self._blocks_per_peer,
            num_warps=self._num_warps,
            chunk_size=self._chunk_size,
            sync_buffer=self.sync_buffer,
        )

        # Return a 2-D view of the receive buffer with fixed-slot layout.
        slotted_output = recv_buf.view(
            self.world_size * self.max_recv_tokens_per_peer, self.D
        )

        # Determine total tokens
        if packed_output_tokens is not None:
            total_tokens = packed_output_tokens
        else:
            _sum_int64_kernel[(1,)](
                output_split_sizes,
                self._total_tokens_buf,
                # pyre-fixme[6]: Triton constexpr accepts int at runtime
                N=self.world_size,
            )
            total_tokens = int(self._total_tokens_buf.item())

        # Check if distribution is uniform
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
