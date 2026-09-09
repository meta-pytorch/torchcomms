# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-unsafe
"""Pure-Python symmem backend for torchcomms.

Implements collectives on top of ``torch._C._distributed_c10d._SymmetricMemory``.
Each communicator allocates a symmetric-memory workspace shared across all
participating ranks via CUDA IPC. Collectives copy data through the workspace
and synchronize with ``barrier``/signal pad primitives.

Single-node multi-GPU only (uses the CUDA / cudaIpc symm-mem backend).
"""

import atexit
import hashlib
import os
from datetime import timedelta
from typing import List, Optional

import torch
from torch._C._distributed_c10d import (
    _register_process_group,
    _SymmetricMemory,
    _unregister_process_group,
    ProcessGroup,
    ProcessGroupGloo,
)
from torch.distributed import PrefixStore, Store, TCPStore
from torchcomms._comms import TorchCommBackend
from torchcomms.symmem.collectives import cast_buffer, nbytes_of, reduce_op_name


# Per-rank symmetric-memory workspace. Sized for the largest payload that
# may be staged through it (e.g. an all-to-all_v where one rank's input
# can reach world_size * max_chunk bytes). Override via the
# ``symmem_workspace_bytes`` hint.
_DEFAULT_WORKSPACE_BYTES: int = 128 * 1024 * 1024
# Per-peer send slot used for send/recv at the tail of the workspace.
_SEND_SLOT_BYTES: int = 64 * 1024


# Process-global cache: workspace + symm_mem + PG keyed by rank-set tuple.
# CUDASymmetricMemory does not support cleanly re-allocating after a sub-
# group has allocated (the world allocation hangs on rendezvous), so we
# cache one workspace per unique rank-set and reuse it across communicators
# targeting the same ranks. This also avoids the per-comm allocation cost.
class _GroupResources:
    __slots__ = (
        "group_name",
        "store",
        "pg",
        "workspace_tensor",
        "symm_mem",
        "workspace_bytes",
        "_pointer_cache",
    )

    def __init__(
        self,
        group_name: str,
        store,
        pg,
        workspace_tensor,
        symm_mem,
        workspace_bytes: int,
    ) -> None:
        self.group_name = group_name
        self.store = store
        self.pg = pg
        self.workspace_tensor = workspace_tensor
        self.symm_mem = symm_mem
        self.workspace_bytes = workspace_bytes
        self._pointer_cache = None


# (device_index, sorted_global_ranks_tuple) -> _GroupResources
_GROUP_RESOURCES: dict[tuple, _GroupResources] = {}


def _shutdown_all_resources() -> None:
    """Best-effort teardown of cached symm-mem resources at process exit.

    Without this, gloo worker threads for sub-group PGs may keep the
    process alive past main(), causing torchrun to time out.
    """
    for res in list(_GROUP_RESOURCES.values()):
        try:
            if res.pg is not None:
                res.pg.shutdown()
        except Exception:
            pass
        try:
            _unregister_process_group(res.group_name)
        except Exception:
            pass
        res.workspace_tensor = None
        res.symm_mem = None
        res.pg = None
        res.store = None
    _GROUP_RESOURCES.clear()


atexit.register(_shutdown_all_resources)


# _SymmetricMemory's alloc_id must (a) be unique among all active allocations
# in the process and (b) match across all ranks in the group for the same
# allocation. A per-rank counter fails (b) because ranks not in a sub-group
# don't bump it. We instead derive the id deterministically from a hash of
# the group name + per-group sequence number — all ranks computing the
# same hash from the same string. Collisions are rare with a 31-bit space.
_GROUP_ALLOC_SEQ: dict[str, int] = {}

# Process-wide counter handing out unique signal-pad channel bases so that
# multiple communicators sharing the same symm-mem workspace (and thus the
# same signal pad) don't collide on send/recv channels.
_COMM_ID_COUNTER: int = 0


def _next_comm_id() -> int:
    global _COMM_ID_COUNTER
    _COMM_ID_COUNTER += 1
    return _COMM_ID_COUNTER


def _alloc_id_for_group(group_name: str) -> int:
    seq = _GROUP_ALLOC_SEQ.get(group_name, 0) + 1
    _GROUP_ALLOC_SEQ[group_name] = seq
    h = hashlib.sha256(f"{group_name}#{seq}".encode()).digest()
    # Take first 4 bytes; mask to positive 31-bit int to fit in a C int.
    return int.from_bytes(h[:4], "big") & 0x7FFFFFFF


def _read_env_int(*names: str) -> Optional[int]:
    for n in names:
        v = os.environ.get(n)
        if v is not None:
            return int(v)
    return None


def _query_rank_size() -> tuple[int, int]:
    rank = _read_env_int(
        "TORCHCOMM_RANK", "RANK", "OMPI_COMM_WORLD_RANK", "SLURM_PROCID", "PMI_RANK"
    )
    size = _read_env_int(
        "TORCHCOMM_SIZE",
        "WORLD_SIZE",
        "OMPI_COMM_WORLD_SIZE",
        "SLURM_NTASKS",
        "PMI_SIZE",
    )
    if rank is None or size is None:
        raise RuntimeError(
            "symmem backend: cannot determine rank/world-size from environment"
        )
    return rank, size


def _build_process_group(
    store: Store,
    rank: int,
    size: int,
    group_name: str,
    timeout: timedelta,
) -> ProcessGroup:
    """Create a ProcessGroup with a Gloo backend bound to all device types.

    The PG is used by CUDASymmetricMemory only for rendezvous-time IPC handle
    exchange — collective payloads go through the symm-mem buffer directly.
    """
    pg = ProcessGroup(store, rank, size)
    gloo = ProcessGroupGloo(store, rank, size, timeout)
    backend_type = ProcessGroup.BackendType.GLOO
    for dev in (torch.device("cpu"), torch.device("cuda")):
        pg._register_backend(dev, backend_type, gloo)
    pg._set_default_backend(backend_type)
    pg._set_group_name(group_name)
    pg._set_group_desc(group_name)
    return pg


class _Work:
    """Async work handle backed by a CUDA event."""

    def __init__(self, event: Optional[torch.cuda.Event] = None) -> None:
        self._event = event

    def wait(self) -> None:
        if self._event is not None:
            # Block until the queued ops are done so callers can safely
            # read tensors on the host.
            self._event.synchronize()
            self._event = None


class TorchCommSymmem(TorchCommBackend):
    BACKEND_NAME = "symmem"

    def __init__(self) -> None:
        super().__init__()
        self._initialized: bool = False
        self._device: torch.device = torch.device("cpu")
        self._name: str = ""
        # rank/size in THIS communicator's group (sub-comm has its own local rank).
        self._rank: int = 0
        self._size: int = 1

        # Shared group resources (workspace, PG, store, symm_mem).
        # Sub-communicators share WORLD's resources because
        # CUDASymmetricMemory cannot allocate on multiple groups within a
        # process; sub-group sync uses the signal pad instead.
        self._resources: Optional[_GroupResources] = None
        # Global rank ids of the participants (used both as cache key
        # for the resources and as the peer-index space for symm-mem APIs).
        self._global_ranks: tuple = ()
        # My rank in the WORLD group that owns the symm-mem workspace.
        self._world_rank: int = 0
        self._world_size: int = 1
        # Cached split into [scratch | send slots].
        self._scratch_bytes: int = 0
        self._send_region_offset: int = 0
        # Signal-pad channel used for sub-group barriers (unique per sub).
        self._barrier_channel: int = 0
        # Per-peer signal channel for send/recv. Each (src, dst) pair gets a
        # private channel so multiple in-flight pairs do not collide.
        self._sendrecv_channel_base: int = 1  # avoid channel 0

    # ---------------------------------------------------------------- helpers

    def _err(self, msg: str) -> RuntimeError:
        return RuntimeError(f"[symmem:{self._name}] {msg}")

    def _check_init(self) -> None:
        if not self._initialized:
            raise self._err("backend not initialized")

    def _check_device(self, tensor: torch.Tensor, name: str = "tensor") -> None:
        if tensor.device != self._device:
            raise self._err(
                f"{name} device {tensor.device} != backend device {self._device}"
            )

    @property
    def _symm_mem(self) -> Optional[_SymmetricMemory]:
        return self._resources.symm_mem if self._resources else None

    @property
    def _group_name(self) -> str:
        return self._resources.group_name if self._resources else ""

    @property
    def _workspace_tensor(self) -> Optional[torch.Tensor]:
        return self._resources.workspace_tensor if self._resources else None

    @property
    def _workspace_bytes(self) -> int:
        return self._resources.workspace_bytes if self._resources else 0

    def _ensure_workspace(self, min_scratch_bytes: int) -> _SymmetricMemory:
        """Return the cached symmetric-memory workspace.

        The workspace is pre-allocated at communicator-init time at
        ``_DEFAULT_WORKSPACE_BYTES`` (or the size requested via the
        ``symmem_workspace_bytes`` hint). Operations that exceed the
        scratch capacity raise here.

        The workspace layout is:

            [ scratch | per-peer send slots ]
              scratch_bytes      world_size * _SEND_SLOT_BYTES
        """
        needed = min_scratch_bytes + self._world_size * _SEND_SLOT_BYTES
        if self._workspace_tensor is None:
            raise self._err("workspace is not allocated")
        if needed > self._workspace_bytes:
            raise self._err(
                f"collective requires {needed} bytes of scratch but workspace "
                f"is {self._workspace_bytes} bytes. Re-create the communicator "
                f"with hints={{'symmem_workspace_bytes': '{needed}'}}."
            )
        assert self._symm_mem is not None
        return self._symm_mem

    def _scratch(self) -> torch.Tensor:
        """Return this rank's scratch region (uint8) of size scratch_bytes."""
        assert self._workspace_tensor is not None
        return self._workspace_tensor[: self._scratch_bytes]

    def _peer_global(self, peer_local: int) -> int:
        """Translate local rank in this communicator to the global rank."""
        return self._global_ranks[peer_local]

    def _kernel_reduce_slice(
        self,
        output: torch.Tensor,
        op,
        elem_offset: int,
        n_elements: int,
    ) -> None:
        """Reduce ``n_elements`` from each peer's scratch (starting at
        ``elem_offset`` in the peer's typed scratch view) into ``output``.

        Always uses a Triton kernel — no Python-side reductions.
        """
        from torchcomms.symmem import triton_kernels as _tk

        op_name = reduce_op_name(op)
        op_code = _tk.OP_CODES.get(op_name, _tk._OP_SUM)
        scale = 0.0
        if op_name == "premul_sum":
            factor = op.factor
            if isinstance(factor, torch.Tensor):
                scale = float(factor.float().item())
            else:
                scale = float(factor)
        out_flat = output.reshape(-1)
        _tk.all_reduce(
            self._peer_buffer_ptrs_tensor(),
            out_flat,
            elem_offset=elem_offset,
            n_elements=n_elements,
            world_size=self._size,
            op_code=op_code,
            scale=scale,
        )

    def _peer_buffer_ptrs_tensor(self) -> torch.Tensor:
        """Return a 1-D CUDA int64 tensor of base buffer pointers in local-rank order.

        The tensor is cached per (resources, sub-group) so the per-call
        cost is a dict lookup.
        """
        assert self._symm_mem is not None
        cache = self._resources._pointer_cache
        if cache is None:
            cache = {}
            self._resources._pointer_cache = cache
        cache_key = self._global_ranks
        cached = cache.get(cache_key)
        if cached is not None:
            return cached
        ptrs = self._symm_mem.buffer_ptrs
        tensor = torch.tensor(
            [ptrs[g] for g in self._global_ranks],
            dtype=torch.int64,
            device=self._device,
        )
        cache[cache_key] = tensor
        return tensor

    def _peer_scratch(self, peer_local: int) -> torch.Tensor:
        """View a peer's scratch region as uint8.

        ``peer_local`` is a rank inside this communicator (0..self._size).
        The underlying ``get_buffer`` call indexes by GLOBAL rank since
        the symm-mem workspace is owned by the WORLD group.
        """
        assert self._symm_mem is not None
        peer_global = self._peer_global(peer_local)
        return self._symm_mem.get_buffer(
            peer_global, (self._scratch_bytes,), torch.uint8, 0
        )

    def _group_barrier(self) -> None:
        """Synchronize all ranks in this communicator.

        For the WORLD group we delegate to the native
        ``SymmetricMemory.barrier`` (single inline atomic exchange).
        For sub-groups we use the signal pad with a dedicated channel:
        each member signals every other member and waits for the same.
        """
        assert self._symm_mem is not None
        if self._size == self._world_size:
            self._symm_mem.barrier()
            return
        chan = self._barrier_channel
        my_global = self._world_rank
        for peer_global in self._global_ranks:
            if peer_global != my_global:
                self._symm_mem.put_signal(peer_global, channel=chan)
        for peer_global in self._global_ranks:
            if peer_global != my_global:
                self._symm_mem.wait_signal(peer_global, channel=chan)

    def _send_slot_offset(self, dst_world_rank: int) -> int:
        """Offset (bytes) in a sender's buffer for the slot destined to ``dst_world_rank``."""
        return self._send_region_offset + dst_world_rank * _SEND_SLOT_BYTES

    def _send_slot(self, sender_world_rank: int, dst_world_rank: int) -> torch.Tensor:
        """View into ``sender_world_rank``'s send slot for ``dst_world_rank``."""
        assert self._symm_mem is not None
        return self._symm_mem.get_buffer(
            sender_world_rank,
            (_SEND_SLOT_BYTES,),
            torch.uint8,
            self._send_slot_offset(dst_world_rank),
        )

    def _sendrecv_channel(self, src_world: int, dst_world: int) -> int:
        return self._sendrecv_channel_base + src_world * self._world_size + dst_world

    def _maybe_async_work(self, async_op: bool) -> Optional[_Work]:
        if not async_op:
            return None
        stream = torch.cuda.current_stream(self._device)
        event = torch.cuda.Event()
        event.record(stream)
        return _Work(event)

    # Process-wide TCPStore reused by all communicators so we don't
    # contend over MASTER_PORT.
    _ROOT_STORE: Optional[Store] = None

    def _get_or_create_root_store(self, options) -> Store:
        """Return a long-lived root store shared by all comms."""
        user_store = getattr(options, "store", None)
        if user_store is not None:
            return user_store
        if TorchCommSymmem._ROOT_STORE is not None:
            return TorchCommSymmem._ROOT_STORE
        master_addr = os.environ.get("MASTER_ADDR")
        master_port = os.environ.get("MASTER_PORT")
        if not master_addr or not master_port:
            raise self._err("no store provided and MASTER_ADDR/MASTER_PORT are not set")
        timeout = getattr(options, "timeout", timedelta(seconds=600))
        if not isinstance(timeout, timedelta):
            timeout = timedelta(seconds=600)
        tcp = TCPStore(
            host_name=master_addr,
            port=int(master_port),
            is_master=(self._rank == 0),
            wait_for_workers=False,
            timeout=timeout,
        )
        TorchCommSymmem._ROOT_STORE = tcp
        return tcp

    # --------------------------------------------------------------- init/info

    def init(self, device, name, options) -> None:
        if self._initialized:
            raise self._err("already initialized")
        if not isinstance(device, torch.device):
            device = torch.device(device)
        if device.type != "cuda":
            raise self._err(f"symmem requires a CUDA device, got {device}")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(device)

        self._device = device
        self._name = name
        self._rank, self._size = _query_rank_size()
        self._global_ranks = tuple(range(self._size))
        self._world_rank = self._rank
        self._world_size = self._size
        # WORLD doesn't need a dedicated barrier channel.
        self._barrier_channel = 0
        # Per-comm signal-channel base, in slots-per-comm-window units. Each
        # comm gets a unique stride of world_size*world_size channels for
        # its (src,dst) send/recv pairs so different comms don't collide
        # on the shared symm-mem signal pad.
        self._sendrecv_channel_base = 16 + _next_comm_id() * self._size * self._size

        # Configure symm-mem to use the CUDA (cudaIpc) backend explicitly so
        # it works without NVSHMEM/NCCL.
        try:
            _SymmetricMemory.set_backend("CUDA")
        except Exception:
            # Backend may already be locked in for the process.
            pass
        # Expand the signal pad up-front so multiple communicators can use
        # disjoint channel windows for send/recv. Must be set before the
        # first symm-mem allocation; subsequent set_signal_pad_size calls
        # may be ignored.
        try:
            if _SymmetricMemory.signal_pad_size < 1 << 20:
                _SymmetricMemory.signal_pad_size = 1 << 20  # 1 MiB
        except Exception:
            pass

        # Optional workspace size override via hint.
        workspace_bytes = _DEFAULT_WORKSPACE_BYTES
        if options is not None:
            hints = getattr(options, "hints", {}) or {}
            if "symmem_workspace_bytes" in hints:
                workspace_bytes = int(hints["symmem_workspace_bytes"])

        # Look up cached resources for this rank-set + device. Multiple
        # communicators sharing the same rank-set share one workspace; see
        # the rationale on _GROUP_RESOURCES above.
        store = self._get_or_create_root_store(options)
        self._resources = self._get_or_create_resources(
            global_ranks=self._global_ranks,
            rank_in_group=self._rank,
            device=self._device,
            root_store=store,
            workspace_bytes=workspace_bytes,
        )
        self._scratch_bytes = (
            self._resources.workspace_bytes - self._world_size * _SEND_SLOT_BYTES
        )
        self._send_region_offset = self._scratch_bytes
        self._initialized = True

    @staticmethod
    def _resource_key(global_ranks: tuple, device: torch.device) -> tuple:
        return (device.type, device.index, tuple(sorted(global_ranks)))

    def _get_or_create_resources(
        self,
        global_ranks: tuple,
        rank_in_group: int,
        device: torch.device,
        root_store: Store,
        workspace_bytes: int,
    ) -> _GroupResources:
        """Look up or create the cached resources for ``global_ranks``."""
        key = self._resource_key(global_ranks, device)
        cached = _GROUP_RESOURCES.get(key)
        if cached is not None:
            if cached.workspace_bytes < workspace_bytes:
                raise self._err(
                    f"existing workspace for ranks {global_ranks} is "
                    f"{cached.workspace_bytes} bytes but {workspace_bytes} was "
                    f"requested. Symmem reallocation is not supported."
                )
            return cached

        size = len(global_ranks)
        group_name = "symmem_grp_" + "_".join(str(r) for r in sorted(global_ranks))
        prefix_store = PrefixStore(group_name + "/", root_store)
        pg_store = PrefixStore("pg/", prefix_store)
        pg = _build_process_group(
            pg_store,
            rank_in_group,
            size,
            group_name,
            timedelta(seconds=600),
        )
        _register_process_group(group_name, pg)
        _SymmetricMemory.set_group_info(
            group_name,
            rank_in_group,
            size,
            PrefixStore("sm/", prefix_store),
        )
        alloc_id = _alloc_id_for_group(group_name)
        workspace_tensor = _SymmetricMemory.empty_strided_p2p(
            (workspace_bytes,),
            (1,),
            torch.uint8,
            device,
            group_name,
            alloc_id,
        )
        symm_mem = _SymmetricMemory.rendezvous(workspace_tensor)
        symm_mem.barrier()
        torch.cuda.current_stream(device).synchronize()
        cached = _GroupResources(
            group_name=group_name,
            store=prefix_store,
            pg=pg,
            workspace_tensor=workspace_tensor,
            symm_mem=symm_mem,
            workspace_bytes=workspace_bytes,
        )
        _GROUP_RESOURCES[key] = cached
        return cached

    def finalize(self) -> None:
        if not self._initialized:
            return
        # Resources are shared across communicators on the same rank-set
        # and held in the process-global cache; this just drops our
        # reference. The cache is torn down at process exit via the
        # atexit handler — SymmetricMemory doesn't tolerate
        # de-/re-rendezvous cycles once a sub-group has been allocated,
        # so per-comm release is intentionally a no-op.
        self._resources = None
        self._initialized = False

    def get_rank(self) -> int:
        return self._rank

    def get_size(self) -> int:
        return self._size

    def get_backend_name(self) -> str:
        return self.BACKEND_NAME

    def get_comm_name(self) -> str:
        return self._name

    # ----------------------------------------------------------- collectives

    def broadcast(self, tensor: torch.Tensor, root: int, async_op: bool):
        self._check_init()
        self._check_device(tensor)
        if tensor.numel() == 0:
            return self._maybe_async_work(async_op)
        nbytes = nbytes_of(tensor)
        self._ensure_workspace(nbytes)

        if self._rank == root:
            cast_buffer(self._scratch(), tensor)[: tensor.numel()].copy_(
                tensor.reshape(-1).contiguous()
            )
        self._group_barrier()
        if self._rank != root:
            root_buf = cast_buffer(self._peer_scratch(root), tensor)[: tensor.numel()]
            tensor.copy_(root_buf.view(tensor.shape))
        self._group_barrier()
        return self._maybe_async_work(async_op)

    def all_reduce(self, tensor: torch.Tensor, op, async_op: bool):
        self._check_init()
        self._check_device(tensor)
        if tensor.numel() == 0:
            return self._maybe_async_work(async_op)
        result = self._reduce_impl(tensor, op, root=None)
        tensor.copy_(result.view(tensor.shape))
        return self._maybe_async_work(async_op)

    def reduce(self, tensor: torch.Tensor, root: int, op, async_op: bool):
        self._check_init()
        self._check_device(tensor)
        if tensor.numel() == 0:
            return self._maybe_async_work(async_op)
        result = self._reduce_impl(tensor, op, root=root)
        if self._rank == root:
            tensor.copy_(result.view(tensor.shape))
        return self._maybe_async_work(async_op)

    def _reduce_impl(
        self,
        tensor: torch.Tensor,
        op,
        root: Optional[int],
    ) -> torch.Tensor:
        """Reduce ``tensor`` across this communicator.

        Stages the input into the scratch region, syncs via group barrier,
        then dispatches to either the built-in ``one_shot_all_reduce`` op
        (for the supported fp32/bf16+SUM fast path on the WORLD group) or
        a Triton kernel that reads directly from peer buffers and applies
        the requested reduction. The output is always written into freshly
        allocated tensor memory so the workspace can be reused immediately.
        """
        from torchcomms.symmem import triton_kernels as _tk

        nbytes = nbytes_of(tensor)
        self._ensure_workspace(nbytes)

        # Push our input into the scratch region (typed view).
        local_view = cast_buffer(self._scratch(), tensor)[: tensor.numel()]
        local_view.copy_(tensor.reshape(-1))
        self._group_barrier()

        out = torch.empty_like(tensor)
        op_name = reduce_op_name(op)
        # Fast path: WORLD group, dtype supported by one_shot_all_reduce,
        # op == sum. The op reads from peer buffers via cudaIPC and writes
        # the reduced tensor to fresh memory.
        if (
            self._size == self._world_size
            and tensor.dtype in (torch.float32, torch.bfloat16)
            and op_name == "sum"
        ):
            result = torch.ops.symm_mem.one_shot_all_reduce(
                local_view.view(tensor.shape),
                "sum",
                self._group_name,
            )
            out.copy_(result)
        else:
            scale = 0.0
            op_code = _tk.OP_CODES.get(op_name, _tk._OP_SUM)
            if op_name == "premul_sum":
                factor = op.factor
                if isinstance(factor, torch.Tensor):
                    scale = float(factor.float().item())
                else:
                    scale = float(factor)
            # Run a Triton kernel that aggregates from this group's peer
            # buffers at offset 0 into the output tensor.
            out_flat = out.reshape(-1)
            _tk.all_reduce(
                self._peer_buffer_ptrs_tensor(),
                out_flat,
                elem_offset=0,
                n_elements=tensor.numel(),
                world_size=self._size,
                op_code=op_code,
                scale=scale,
            )
            # Exit barrier so the scratch can be reused.
        self._group_barrier()
        return out

    def all_gather(
        self,
        tensor_list: List[torch.Tensor],
        tensor: torch.Tensor,
        async_op: bool,
    ):
        self._check_init()
        self._check_device(tensor)
        if len(tensor_list) != self._size:
            raise self._err(
                f"all_gather: tensor_list has {len(tensor_list)} != world_size {self._size}"
            )
        if tensor.numel() == 0:
            return self._maybe_async_work(async_op)
        nbytes = nbytes_of(tensor)
        self._ensure_workspace(nbytes)

        local_view = cast_buffer(self._scratch(), tensor)[: tensor.numel()]
        local_view.copy_(tensor.reshape(-1))
        self._group_barrier()
        for peer in range(self._size):
            peer_buf = cast_buffer(self._peer_scratch(peer), tensor)[: tensor.numel()]
            tensor_list[peer].copy_(peer_buf.view(tensor_list[peer].shape))
        self._group_barrier()
        return self._maybe_async_work(async_op)

    def all_gather_v(
        self,
        tensor_list: List[torch.Tensor],
        tensor: torch.Tensor,
        async_op: bool,
    ):
        # Variable-sized inputs / outputs. Each rank may have a different
        # local size; all ranks must agree on every other rank's size, which
        # they get from the corresponding entry in tensor_list.
        self._check_init()
        self._check_device(tensor)
        if len(tensor_list) != self._size:
            raise self._err(
                f"all_gather_v: tensor_list has {len(tensor_list)} != world_size {self._size}"
            )
        nbytes = max((nbytes_of(t) for t in tensor_list), default=0)
        if nbytes == 0:
            return self._maybe_async_work(async_op)
        self._ensure_workspace(nbytes)

        if tensor.numel() > 0:
            cast_buffer(self._scratch(), tensor)[: tensor.numel()].copy_(
                tensor.reshape(-1)
            )
        self._group_barrier()
        for peer in range(self._size):
            out = tensor_list[peer]
            if out.numel() == 0:
                continue
            peer_buf = cast_buffer(self._peer_scratch(peer), out)[: out.numel()]
            out.copy_(peer_buf.view(out.shape))
        self._group_barrier()
        return self._maybe_async_work(async_op)

    def all_gather_single(
        self, output: torch.Tensor, input: torch.Tensor, async_op: bool
    ):
        self._check_init()
        self._check_device(input)
        self._check_device(output)
        if output.numel() != input.numel() * self._size:
            raise self._err(
                f"all_gather_single: output numel {output.numel()} != "
                f"input numel * size {input.numel() * self._size}"
            )
        if input.numel() == 0:
            return self._maybe_async_work(async_op)
        nbytes = nbytes_of(input)
        self._ensure_workspace(nbytes)

        cast_buffer(self._scratch(), input)[: input.numel()].copy_(input.reshape(-1))
        self._group_barrier()
        out_flat = output.reshape(-1)
        for peer in range(self._size):
            peer_buf = cast_buffer(self._peer_scratch(peer), input)[: input.numel()]
            out_flat[peer * input.numel() : (peer + 1) * input.numel()].copy_(peer_buf)
        self._group_barrier()
        return self._maybe_async_work(async_op)

    def reduce_scatter(
        self,
        output: torch.Tensor,
        input_list: List[torch.Tensor],
        op,
        async_op: bool,
    ):
        self._check_init()
        self._check_device(output)
        if len(input_list) != self._size:
            raise self._err(
                f"reduce_scatter: input_list has {len(input_list)} != world_size {self._size}"
            )
        nbytes = sum(nbytes_of(t) for t in input_list)
        # NB: do NOT early-return when output is empty: with variable-sized
        # chunks (reduce_scatter_v) some ranks may receive zero elements
        # while peers still need them to participate in the barriers.
        self._ensure_workspace(max(nbytes, 1))

        # Lay out our contributions per destination peer in our scratch.
        scratch_typed = cast_buffer(self._scratch(), output)
        offset = 0
        offsets = []
        for t in input_list:
            self._check_device(t)
            offsets.append(offset)
            if t.numel() > 0:
                scratch_typed[offset : offset + t.numel()].copy_(t.reshape(-1))
            offset += t.numel()
        self._group_barrier()

        # Pull the chunk that all peers wrote for us, then reduce.
        my_count = input_list[self._rank].numel()
        if my_count > 0:
            self._kernel_reduce_slice(output, op, offsets[self._rank], my_count)
        self._group_barrier()
        return self._maybe_async_work(async_op)

    def reduce_scatter_v(
        self,
        output: torch.Tensor,
        input_list: List[torch.Tensor],
        op,
        async_op: bool,
    ):
        # Same as reduce_scatter but inputs may have different sizes; only
        # constraint is input_list[r].numel() == output.numel() for the
        # destination rank r across all participants.
        return self.reduce_scatter(output, input_list, op, async_op)

    def reduce_scatter_single(
        self,
        output: torch.Tensor,
        input: torch.Tensor,
        op,
        async_op: bool,
    ):
        self._check_init()
        self._check_device(output)
        self._check_device(input)
        if input.numel() != output.numel() * self._size:
            raise self._err(
                f"reduce_scatter_single: input numel {input.numel()} != "
                f"output numel * size {output.numel() * self._size}"
            )
        if output.numel() == 0:
            return self._maybe_async_work(async_op)
        nbytes = nbytes_of(input)
        self._ensure_workspace(nbytes)

        cast_buffer(self._scratch(), input)[: input.numel()].copy_(input.reshape(-1))
        self._group_barrier()
        my_count = output.numel()
        start = self._rank * my_count
        self._kernel_reduce_slice(output, op, start, my_count)
        self._group_barrier()
        return self._maybe_async_work(async_op)

    def all_to_all_single(
        self, output: torch.Tensor, input: torch.Tensor, async_op: bool
    ):
        self._check_init()
        self._check_device(output)
        self._check_device(input)
        if input.numel() % self._size != 0:
            raise self._err(
                f"all_to_all_single: input numel {input.numel()} not divisible by world size"
            )
        if input.numel() == 0:
            return self._maybe_async_work(async_op)
        per_rank = input.numel() // self._size
        nbytes = nbytes_of(input)
        self._ensure_workspace(nbytes)

        cast_buffer(self._scratch(), input)[: input.numel()].copy_(input.reshape(-1))
        self._group_barrier()
        out_flat = output.reshape(-1)
        for peer in range(self._size):
            peer_buf = cast_buffer(self._peer_scratch(peer), input)
            chunk = peer_buf[self._rank * per_rank : (self._rank + 1) * per_rank]
            out_flat[peer * per_rank : (peer + 1) * per_rank].copy_(chunk)
        self._group_barrier()
        return self._maybe_async_work(async_op)

    def all_to_all_v_single(
        self,
        output: torch.Tensor,
        input: torch.Tensor,
        output_split_sizes: List[int],
        input_split_sizes: List[int],
        async_op: bool,
    ):
        self._check_init()
        self._check_device(output)
        self._check_device(input)
        if (
            len(input_split_sizes) != self._size
            or len(output_split_sizes) != self._size
        ):
            raise self._err(
                "all_to_all_v_single: split-size list length must equal world size"
            )
        if input.numel() == 0 and output.numel() == 0:
            return self._maybe_async_work(async_op)

        # Split sizes refer to the first-dim count; trailing dims act as a
        # per-element multiplier (matches NCCL/Gloo semantics).
        in_stride = (
            input.numel() // input.size(0)
            if input.dim() > 0 and input.size(0) > 0
            else 1
        )
        out_stride = (
            output.numel() // output.size(0)
            if output.dim() > 0 and output.size(0) > 0
            else 1
        )

        self._ensure_workspace(
            max(nbytes_of(input), output.element_size() * output.numel())
        )

        # Phase 1: all_gather the input_split_sizes so each rank knows
        # everyone's full split layout (in element counts, not row counts).
        all_splits = torch.zeros(
            (self._size, self._size), dtype=torch.int64, device=self._device
        )
        my_splits = torch.tensor(
            [int(s) * in_stride for s in input_split_sizes],
            dtype=torch.int64,
            device=self._device,
        )
        # Reuse all_gather_single on a small int64 buffer.
        self.all_gather_single(all_splits.reshape(-1), my_splits, async_op=False)

        # Compute per-peer offsets for OUR scratch.
        my_input_offsets = [0]
        for i in range(self._size - 1):
            my_input_offsets.append(
                my_input_offsets[-1] + int(input_split_sizes[i]) * in_stride
            )
        # Stage our input into scratch by peer.
        scratch_typed = cast_buffer(self._scratch(), input)
        if input.numel() > 0:
            in_flat = input.reshape(-1)
            for peer in range(self._size):
                n = int(input_split_sizes[peer]) * in_stride
                if n == 0:
                    continue
                src_off = my_input_offsets[peer]
                dst_off = src_off
                scratch_typed[dst_off : dst_off + n].copy_(
                    in_flat[src_off : src_off + n]
                )
        self._group_barrier()

        # Pull each peer's chunk for us. Peer's offset for us is the prefix
        # sum (in element units) of peer's input splits up to my_rank.
        out_flat = output.reshape(-1)
        out_pos = 0
        for peer in range(self._size):
            recv_count = int(output_split_sizes[peer]) * out_stride
            if recv_count == 0:
                continue
            peer_splits = all_splits[peer]
            peer_off_for_me = int(peer_splits[: self._rank].sum().item())
            peer_buf = cast_buffer(self._peer_scratch(peer), output)
            out_flat[out_pos : out_pos + recv_count].copy_(
                peer_buf[peer_off_for_me : peer_off_for_me + recv_count]
            )
            out_pos += recv_count
        self._group_barrier()
        return self._maybe_async_work(async_op)

    def all_to_all(
        self,
        output_tensor_list: List[torch.Tensor],
        input_tensor_list: List[torch.Tensor],
        async_op: bool,
    ):
        self._check_init()
        if (
            len(output_tensor_list) != self._size
            or len(input_tensor_list) != self._size
        ):
            raise self._err("all_to_all: list lengths must equal world size")
        # Stage input chunks consecutively in our scratch.
        offsets = [0]
        for t in input_tensor_list:
            self._check_device(t)
            offsets.append(offsets[-1] + t.numel())
        total = offsets[-1]
        if total == 0 and all(t.numel() == 0 for t in output_tensor_list):
            return self._maybe_async_work(async_op)

        # Use the dtype of the first non-empty input tensor as the staging
        # dtype. AllToAll requires all entries to have the same dtype.
        dtype_template = next(
            (t for t in input_tensor_list if t.numel() > 0),
            next(
                (t for t in output_tensor_list if t.numel() > 0), input_tensor_list[0]
            ),
        )
        nbytes = total * dtype_template.element_size()
        self._ensure_workspace(nbytes)

        # Exchange per-peer counts so we know where each peer wrote our
        # chunk. AllToAll uses input_tensor_list[dst] as the chunk sent
        # to dst, so each peer's offset for us is the prefix sum of the
        # lengths of that peer's input_tensor_list[0..my_rank-1]. Run
        # this BEFORE staging the payload because all_gather_single
        # reuses the same scratch buffer.
        my_counts = torch.tensor(
            [t.numel() for t in input_tensor_list],
            dtype=torch.int64,
            device=self._device,
        )
        all_counts = torch.zeros(
            (self._size, self._size), dtype=torch.int64, device=self._device
        )
        self.all_gather_single(all_counts.reshape(-1), my_counts, async_op=False)

        scratch_typed = cast_buffer(self._scratch(), dtype_template)
        for i, t in enumerate(input_tensor_list):
            if t.numel() > 0:
                scratch_typed[offsets[i] : offsets[i] + t.numel()].copy_(t.reshape(-1))
        self._group_barrier()

        for peer in range(self._size):
            out = output_tensor_list[peer]
            if out.numel() == 0:
                continue
            self._check_device(out)
            peer_off = int(all_counts[peer, : self._rank].sum().item())
            peer_buf = cast_buffer(self._peer_scratch(peer), out)
            out.copy_(peer_buf[peer_off : peer_off + out.numel()].view(out.shape))
        self._group_barrier()
        return self._maybe_async_work(async_op)

    def scatter(
        self,
        output: torch.Tensor,
        input_list: List[torch.Tensor],
        root: int,
        async_op: bool,
    ):
        self._check_init()
        self._check_device(output)
        if self._rank == root:
            if len(input_list) != self._size:
                raise self._err(
                    f"scatter: input_list has {len(input_list)} != world_size {self._size}"
                )

        if output.numel() == 0:
            return self._maybe_async_work(async_op)
        # Root stages all inputs consecutively in scratch.
        self._ensure_workspace(self._size * nbytes_of(output))
        if self._rank == root:
            scratch_typed = cast_buffer(self._scratch(), output)
            off = 0
            for t in input_list:
                self._check_device(t)
                if t.numel() == 0:
                    continue
                scratch_typed[off : off + t.numel()].copy_(t.reshape(-1))
                off += t.numel()
        self._group_barrier()
        # Each rank reads its slice from root's scratch.
        root_buf = cast_buffer(self._peer_scratch(root), output)
        start = self._rank * output.numel()
        output.copy_(root_buf[start : start + output.numel()].view(output.shape))
        self._group_barrier()
        return self._maybe_async_work(async_op)

    def gather(
        self,
        output_list: List[torch.Tensor],
        input: torch.Tensor,
        root: int,
        async_op: bool,
    ):
        self._check_init()
        self._check_device(input)
        if input.numel() == 0:
            return self._maybe_async_work(async_op)
        self._ensure_workspace(nbytes_of(input))

        # Each rank places input in its own scratch; root reads from all.
        cast_buffer(self._scratch(), input)[: input.numel()].copy_(input.reshape(-1))
        self._group_barrier()
        if self._rank == root:
            if len(output_list) != self._size:
                raise self._err(
                    f"gather: output_list has {len(output_list)} != world_size {self._size}"
                )
            for peer in range(self._size):
                out = output_list[peer]
                self._check_device(out)
                if out.numel() == 0:
                    continue
                peer_buf = cast_buffer(self._peer_scratch(peer), input)[: input.numel()]
                out.copy_(peer_buf.view(out.shape))
        self._group_barrier()
        return self._maybe_async_work(async_op)

    def barrier(self, async_op: bool):
        self._check_init()
        self._group_barrier()
        return self._maybe_async_work(async_op)

    # ------------------------------------------------------------ point-to-point

    def send(self, tensor: torch.Tensor, dst: int, async_op: bool):
        self._check_init()
        self._check_device(tensor)
        if not 0 <= dst < self._size:
            raise self._err(f"send: dst {dst} out of range")
        nbytes = nbytes_of(tensor)
        if nbytes > _SEND_SLOT_BYTES:
            raise self._err(
                f"send: tensor size {nbytes} exceeds slot capacity {_SEND_SLOT_BYTES}"
            )
        self._ensure_workspace(0)
        assert self._symm_mem is not None
        dst_world = self._peer_global(dst)

        if tensor.numel() > 0:
            slot = self._send_slot(self._world_rank, dst_world)
            slot_typed = cast_buffer(slot, tensor)[: tensor.numel()]
            slot_typed.copy_(tensor.reshape(-1).contiguous())
        self._symm_mem.put_signal(
            dst_world, channel=self._sendrecv_channel(self._world_rank, dst_world)
        )
        return self._maybe_async_work(async_op)

    def recv(self, tensor: torch.Tensor, src: int, async_op: bool):
        self._check_init()
        self._check_device(tensor)
        if not 0 <= src < self._size:
            raise self._err(f"recv: src {src} out of range")
        nbytes = nbytes_of(tensor)
        if nbytes > _SEND_SLOT_BYTES:
            raise self._err(
                f"recv: tensor size {nbytes} exceeds slot capacity {_SEND_SLOT_BYTES}"
            )
        self._ensure_workspace(0)
        assert self._symm_mem is not None
        src_world = self._peer_global(src)

        self._symm_mem.wait_signal(
            src_world, channel=self._sendrecv_channel(src_world, self._world_rank)
        )
        if tensor.numel() > 0:
            src_slot = self._send_slot(src_world, self._world_rank)
            slot_typed = cast_buffer(src_slot, tensor)[: tensor.numel()]
            tensor.copy_(slot_typed.view(tensor.shape))
        return self._maybe_async_work(async_op)

    # ----------------------------------------------------------------- split

    def split(self, ranks: List[int], name: str, options):
        # New sub-communicator over the given ranks. This is a collective —
        # ranks NOT in the list must still call split() with the same args.
        self._check_init()

        # Validate the ranks list. NOTE: split() is collective across the
        # parent communicator, so any rank can raise to abort the call;
        # the absence of the matched call on other ranks won't lead to a
        # hang because the failed rank does not enter rendezvous.
        ranks_list = list(ranks)
        if len(set(ranks_list)) != len(ranks_list):
            raise self._err("split: ranks list contains duplicates")
        for r in ranks_list:
            if not 0 <= r < self._size:
                raise self._err(f"split: rank {r} out of range [0, {self._size})")
        # If our rank is excluded, return None — this is the supported way
        # to opt out of a sub-group. The test suite also passes an empty
        # ranks list from excluded ranks, which we accept.
        if ranks_list and self._rank not in ranks_list:
            raise self._err(
                f"split: this rank ({self._rank}) must appear in the ranks list"
            )

        sub = TorchCommSymmem()
        if not isinstance(self._device, torch.device):
            device = torch.device(self._device)
        else:
            device = self._device

        # Translate parent-local ranks to GLOBAL (WORLD) ranks.
        if self._rank not in ranks_list:
            return None
        sub_global_ranks = tuple(self._global_ranks[r] for r in ranks_list)
        sub_rank = ranks_list.index(self._rank)
        sub_size = len(ranks_list)

        sub._device = device
        sub._name = name
        sub._rank = sub_rank
        sub._size = sub_size
        sub._global_ranks = sub_global_ranks
        sub._world_rank = self._world_rank
        sub._world_size = self._world_size
        # Sub-communicator shares WORLD's symm-mem workspace; sync is
        # handled via the signal pad on a unique channel.
        if self._resources is None:
            raise self._err("split: parent resources missing")
        sub._resources = self._resources
        sub._scratch_bytes = (
            sub._resources.workspace_bytes - sub._world_size * _SEND_SLOT_BYTES
        )
        sub._send_region_offset = sub._scratch_bytes
        # Pick a deterministic non-zero signal channel for this sub-group.
        # Channel space is signal_pad_size / 4 ≈ 2k slots; hash to fit.
        chan_space = max(_SymmetricMemory.signal_pad_size // 4 - 32, 64)
        h = hashlib.sha256(
            ("sub_barrier:" + "_".join(map(str, sub_global_ranks))).encode()
        ).digest()
        sub._barrier_channel = int.from_bytes(h[:4], "big") % chan_space + 16
        # Each sub-comm gets its own send/recv channel window in the shared
        # signal pad so distinct comms don't share slots.
        sub._sendrecv_channel_base = (
            16 + _next_comm_id() * sub._world_size * sub._world_size
        )
        sub._initialized = True
        return sub
