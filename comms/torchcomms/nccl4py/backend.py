# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-unsafe

"""Pure-Python TorchComm backend implemented on top of nccl4py (``nccl.core``).

The backend implements the core collective and point-to-point APIs against
NVIDIA's official nccl4py bindings. Bootstrap of the NCCL ``ncclUniqueId`` is
performed through the ``c10d::Store`` provided by torchcomms, mirroring what the
C++ NCCL backend does in ``TorchCommNCCLBootstrap``.

The backend supports:
* All ranked collective operations exposed by ``TorchCommBackend`` (all_reduce,
  broadcast, reduce, all_gather variants, reduce_scatter variants, all_to_all
  variants, barrier, scatter, gather, send/recv) using the ``nccl.Communicator``
  API.
* Sync (``async_op=False``) and async (``async_op=True``) modes. In sync mode
  the collective is enqueued on the user's current CUDA stream. In async mode
  the collective runs on a dedicated internal stream and the returned work
  handle makes the user's current stream wait on the NCCL completion event when
  ``work.wait()`` is called.
* Sub-communicator creation via ``split``.

Window, batch_send_recv, persistent all-gather and reconfigure are deliberately
left unimplemented (the abstract base class raises informative errors for those
when used).
"""

import os
from datetime import timedelta
from typing import List, Optional

import nccl.core as nccl
import torch
from torch.distributed import PrefixStore, Store, TCPStore
from torchcomms._comms import RedOpType, ReduceOp, TorchCommBackend


_TORCH_TO_NCCL_DTYPE = {
    torch.float32: nccl.FLOAT32,
    torch.float16: nccl.FLOAT16,
    torch.bfloat16: nccl.BFLOAT16,
    torch.float64: nccl.FLOAT64,
    torch.int8: nccl.INT8,
    torch.uint8: nccl.UINT8,
    torch.int32: nccl.INT32,
    torch.int64: nccl.INT64,
}


def _nccl_dtype(tensor: torch.Tensor) -> nccl.NcclDataType:
    try:
        return _TORCH_TO_NCCL_DTYPE[tensor.dtype]
    except KeyError as e:
        raise RuntimeError(
            f"nccl4py backend: unsupported torch dtype {tensor.dtype}"
        ) from e


def _torch_to_nccl_redop(op: ReduceOp, comm: "nccl.Communicator", tensor: torch.Tensor):
    """Translate torchcomms ``ReduceOp`` to a nccl4py reduce-op handle.

    PREMUL_SUM is implemented via a per-call ``CustomRedOp`` that is closed on
    ``__exit__`` to avoid leaking handles on the communicator.
    """

    t = op.type
    if t == RedOpType.SUM:
        return _StaticRedOp(nccl.SUM)
    if t == RedOpType.PRODUCT:
        return _StaticRedOp(nccl.PROD)
    if t == RedOpType.MIN:
        return _StaticRedOp(nccl.MIN)
    if t == RedOpType.MAX:
        return _StaticRedOp(nccl.MAX)
    if t == RedOpType.AVG:
        return _StaticRedOp(nccl.AVG)
    if t == RedOpType.PREMUL_SUM:
        factor = op.factor
        return _PreMulSumRedOp(comm, factor, tensor)
    raise RuntimeError(f"nccl4py backend: unsupported reduce op {t}")


class _StaticRedOp:
    def __init__(self, op):
        self.op = op

    def __enter__(self):
        return self.op

    def __exit__(self, *_):
        return False


class _PreMulSumRedOp:
    """Context manager that creates an nccl4py CustomRedOp for PREMUL_SUM."""

    def __init__(self, comm: "nccl.Communicator", factor, tensor: torch.Tensor):
        self._comm = comm
        self._tensor = tensor
        self._factor = factor
        self._handle = None

    def __enter__(self):
        import numpy as np

        dtype = _nccl_dtype(self._tensor)
        # The factor can be a Python float (host scalar) or a 1-element tensor
        # (device scalar). nccl4py's ``create_pre_mul_sum`` accepts a scalar of
        # either residence and infers based on the type, but requires that the
        # scalar's dtype matches the reduction dtype.
        if isinstance(self._factor, torch.Tensor):
            scalar = self._factor
        elif self._tensor.dtype is torch.bfloat16:
            import ml_dtypes  # required for bfloat16 numpy support

            scalar = np.array([float(self._factor)], dtype=ml_dtypes.bfloat16)
        else:
            scalar = torch.tensor(
                [float(self._factor)], dtype=self._tensor.dtype
            ).numpy()
        self._handle = self._comm.create_pre_mul_sum(scalar, dtype)
        return self._handle

    def __exit__(self, *_):
        if self._handle is not None:
            try:
                self._handle.close()
            finally:
                self._handle = None
        return False


class _NcclWork:
    """Work handle returned for ``async_op=True``.

    ``wait()`` makes the user's current CUDA stream wait on the recorded
    completion event so subsequent torch ops will see the NCCL output.
    """

    def __init__(self, end_event: torch.cuda.Event, device: torch.device):
        self._end_event = end_event
        self._device = device

    def wait(self):
        if self._end_event is None:
            return
        current = torch.cuda.current_stream(self._device)
        self._end_event.wait(current)
        self._end_event = None


def _read_env_int(*names: str) -> Optional[int]:
    for name in names:
        v = os.environ.get(name)
        if v is not None and v != "":
            return int(v)
    return None


def _query_rank_size():
    rank = _read_env_int(
        "TORCHCOMM_RANK",
        "OMPI_COMM_WORLD_RANK",
        "PMI_RANK",
        "PALS_RANKID",
        "SLURM_PROCID",
        "RANK",
    )
    size = _read_env_int(
        "TORCHCOMM_SIZE",
        "OMPI_COMM_WORLD_SIZE",
        "PMI_SIZE",
        "SLURM_NTASKS",
        "WORLD_SIZE",
    )
    if rank is None or size is None:
        raise RuntimeError(
            "nccl4py backend: could not determine rank/size from environment "
            "(set TORCHCOMM_RANK/TORCHCOMM_SIZE or RANK/WORLD_SIZE)"
        )
    return rank, size


class TorchCommNCCL4Py(TorchCommBackend):
    """Python TorchComm backend backed by NVIDIA's nccl4py bindings."""

    _UID_STORE_KEY = "nccl4py_uid"

    def __init__(self):
        super().__init__()
        self._rank: int = -1
        self._size: int = -1
        self._device: Optional[torch.device] = None
        self._name: str = ""
        self._comm: Optional[nccl.Communicator] = None
        self._internal_stream: Optional[torch.cuda.Stream] = None
        # If True, this instance owns ``_comm`` and is responsible for
        # destroying it on ``finalize``. Sub-communicators created from
        # ``split`` always own their underlying communicator.
        self._owns_comm: bool = True

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def init(self, device: torch.device, name: str, options) -> None:
        if self._comm is not None:
            raise RuntimeError("nccl4py backend already initialized")

        self._name = name
        if device.type != "cuda":
            raise RuntimeError(f"nccl4py backend requires a cuda device, got {device}")

        rank, size = _query_rank_size()
        if device.index is None or device.index < 0:
            count = torch.cuda.device_count()
            device = torch.device("cuda", rank % count)
        self._device = device
        torch.cuda.set_device(device)

        store = self._get_or_create_store(options, name)

        uid = self._exchange_unique_id(store, rank)

        self._comm = nccl.Communicator.init(nranks=size, rank=rank, unique_id=uid)
        self._rank = rank
        self._size = size

        high_priority = bool(getattr(options, "high_priority_stream", False))
        try:
            high_priority_hint = options.hints.get("highPriorityStream")
            if high_priority_hint is not None:
                high_priority = high_priority_hint in ("1", "true", "True", "ON")
        except AttributeError:
            pass
        priority = -1 if high_priority else 0
        self._internal_stream = torch.cuda.Stream(device=device, priority=priority)

    def finalize(self) -> None:
        if self._comm is not None and self._owns_comm:
            try:
                self._comm.destroy()
            except Exception:
                # Best-effort cleanup; mirror C++ backend behavior.
                pass
        self._comm = None
        self._internal_stream = None

    def get_rank(self) -> int:
        return self._rank

    def get_size(self) -> int:
        return self._size

    def get_backend_name(self) -> str:
        return "nccl4py"

    def get_comm_name(self) -> str:
        return self._name

    # ------------------------------------------------------------------
    # Bootstrap helpers
    # ------------------------------------------------------------------
    def _get_or_create_store(self, options, name: str) -> Store:
        store = getattr(options, "store", None)
        if store is not None:
            return PrefixStore(name, store)

        master_addr = os.environ.get("MASTER_ADDR")
        master_port = os.environ.get("MASTER_PORT")
        if not master_addr or not master_port:
            raise RuntimeError(
                "nccl4py backend: no store provided and MASTER_ADDR/MASTER_PORT "
                "are not set; cannot bootstrap NCCL unique id"
            )
        rank, _ = _query_rank_size()
        timeout_ms = int(
            getattr(options, "timeout", timedelta(seconds=600)).total_seconds() * 1000
        )
        tcp = TCPStore(
            host_name=master_addr,
            port=int(master_port),
            is_master=(rank == 0),
            wait_for_workers=False,
            timeout=timedelta(milliseconds=max(timeout_ms, 1000)),
        )
        return PrefixStore(name, tcp)

    def _exchange_unique_id(self, store: Store, rank: int) -> nccl.UniqueId:
        if rank == 0:
            uid = nccl.get_unique_id()
            store.set(self._UID_STORE_KEY, bytes(uid))
            return uid
        # Wait for rank 0 to publish the unique id then fetch it.
        store.wait([self._UID_STORE_KEY])
        raw = store.get(self._UID_STORE_KEY)
        return nccl.UniqueId.from_bytes(bytes(raw))

    # ------------------------------------------------------------------
    # Stream helpers
    # ------------------------------------------------------------------
    def _check_init(self):
        if self._comm is None:
            raise RuntimeError("nccl4py backend not initialized")

    def _check_tensor(self, tensor: torch.Tensor):
        if not tensor.is_contiguous():
            raise RuntimeError("Tensor must be contiguous for nccl4py operations")
        if tensor.device != self._device:
            raise RuntimeError(
                f"Expected tensor on {self._device} but found tensor on {tensor.device}"
            )

    def _check_tensors(self, tensors):
        for t in tensors:
            self._check_tensor(t)

    def _stream_for(self, async_op: bool) -> torch.cuda.Stream:
        if async_op:
            current = torch.cuda.current_stream(self._device)
            self._internal_stream.wait_stream(current)
            return self._internal_stream
        return torch.cuda.current_stream(self._device)

    def _finish(self, stream: torch.cuda.Stream, async_op: bool):
        if not async_op:
            return None
        event = torch.cuda.Event()
        event.record(stream)
        return _NcclWork(event, self._device)

    # ------------------------------------------------------------------
    # Point-to-point
    # ------------------------------------------------------------------
    def send(self, tensor: torch.Tensor, dst: int, async_op: bool):
        self._check_init()
        self._check_tensor(tensor)
        stream = self._stream_for(async_op)
        self._comm.send(tensor, dst, stream=stream.cuda_stream)
        return self._finish(stream, async_op)

    def recv(self, tensor: torch.Tensor, src: int, async_op: bool):
        self._check_init()
        self._check_tensor(tensor)
        stream = self._stream_for(async_op)
        self._comm.recv(tensor, src, stream=stream.cuda_stream)
        return self._finish(stream, async_op)

    # ------------------------------------------------------------------
    # Core collectives
    # ------------------------------------------------------------------
    def broadcast(self, tensor: torch.Tensor, root: int, async_op: bool):
        self._check_init()
        self._check_tensor(tensor)
        stream = self._stream_for(async_op)
        self._comm.broadcast(tensor, tensor, root, stream=stream.cuda_stream)
        return self._finish(stream, async_op)

    def all_reduce(self, tensor: torch.Tensor, op: ReduceOp, async_op: bool):
        self._check_init()
        self._check_tensor(tensor)
        stream = self._stream_for(async_op)
        with _torch_to_nccl_redop(op, self._comm, tensor) as nccl_op:
            self._comm.allreduce(tensor, tensor, nccl_op, stream=stream.cuda_stream)
        return self._finish(stream, async_op)

    def reduce(self, tensor: torch.Tensor, root: int, op: ReduceOp, async_op: bool):
        self._check_init()
        self._check_tensor(tensor)
        stream = self._stream_for(async_op)
        with _torch_to_nccl_redop(op, self._comm, tensor) as nccl_op:
            self._comm.reduce(
                tensor,
                tensor,
                nccl_op,
                root=root,
                stream=stream.cuda_stream,
            )
        return self._finish(stream, async_op)

    def all_gather(
        self,
        tensor_list: List[torch.Tensor],
        tensor: torch.Tensor,
        async_op: bool,
    ):
        self._check_init()
        if len(tensor_list) != self._size:
            raise RuntimeError("tensor_list size must equal comm size for all_gather")
        self._check_tensor(tensor)
        self._check_tensors(tensor_list)
        for t in tensor_list:
            if t.numel() != tensor.numel():
                raise RuntimeError(
                    "All tensors in tensor_list must have same size as input tensor"
                )
        stream = self._stream_for(async_op)
        with nccl.group():
            for i, out in enumerate(tensor_list):
                self._comm.broadcast(tensor, out, i, stream=stream.cuda_stream)
        return self._finish(stream, async_op)

    def all_gather_v(
        self,
        tensor_list: List[torch.Tensor],
        tensor: torch.Tensor,
        async_op: bool,
    ):
        self._check_init()
        if len(tensor_list) != self._size:
            raise RuntimeError("tensor_list size must equal comm size for all_gather_v")
        self._check_tensor(tensor)
        self._check_tensors(tensor_list)
        stream = self._stream_for(async_op)
        with nccl.group():
            for i, out in enumerate(tensor_list):
                inp = tensor if i == self._rank else out
                if inp.numel() != out.numel():
                    raise RuntimeError(
                        "Output tensor size must equal input tensor size for "
                        "all_gather_v"
                    )
                self._comm.broadcast(inp, out, i, stream=stream.cuda_stream)
        return self._finish(stream, async_op)

    def all_gather_single(
        self, output: torch.Tensor, input: torch.Tensor, async_op: bool
    ):
        self._check_init()
        self._check_tensor(output)
        self._check_tensor(input)
        if output.numel() != input.numel() * self._size:
            raise RuntimeError(
                "Output tensor size must be input_size * comm_size for "
                "all_gather_single"
            )
        stream = self._stream_for(async_op)
        self._comm.allgather(input, output, stream=stream.cuda_stream)
        return self._finish(stream, async_op)

    def reduce_scatter(
        self,
        output: torch.Tensor,
        input_list: List[torch.Tensor],
        op: ReduceOp,
        async_op: bool,
    ):
        self._check_init()
        if len(input_list) != self._size:
            raise RuntimeError(
                "input_list size must equal comm size for reduce_scatter"
            )
        self._check_tensor(output)
        self._check_tensors(input_list)
        for t in input_list:
            if t.numel() != output.numel():
                raise RuntimeError(
                    "All input tensors must have same size as output tensor"
                )
        stream = self._stream_for(async_op)
        with _torch_to_nccl_redop(op, self._comm, output) as nccl_op:
            with nccl.group():
                for i, inp in enumerate(input_list):
                    recv = output if i == self._rank else None
                    self._comm.reduce(
                        inp,
                        recv if recv is not None else inp,
                        nccl_op,
                        root=i,
                        stream=stream.cuda_stream,
                    )
        return self._finish(stream, async_op)

    def reduce_scatter_v(
        self,
        output: torch.Tensor,
        input_list: List[torch.Tensor],
        op: ReduceOp,
        async_op: bool,
    ):
        self._check_init()
        if len(input_list) != self._size:
            raise RuntimeError(
                "input_list size must equal comm size for reduce_scatter_v"
            )
        self._check_tensor(output)
        self._check_tensors(input_list)
        stream = self._stream_for(async_op)
        with _torch_to_nccl_redop(op, self._comm, output) as nccl_op:
            with nccl.group():
                for i, inp in enumerate(input_list):
                    if i == self._rank:
                        if inp.numel() != output.numel():
                            raise RuntimeError(
                                "Output tensor size must equal input tensor size "
                                "for reduce_scatter_v"
                            )
                        self._comm.reduce(
                            inp,
                            output,
                            nccl_op,
                            root=i,
                            stream=stream.cuda_stream,
                        )
                    else:
                        self._comm.reduce(
                            inp,
                            inp,
                            nccl_op,
                            root=i,
                            stream=stream.cuda_stream,
                        )
        return self._finish(stream, async_op)

    def reduce_scatter_single(
        self,
        output: torch.Tensor,
        input: torch.Tensor,
        op: ReduceOp,
        async_op: bool,
    ):
        self._check_init()
        self._check_tensor(output)
        self._check_tensor(input)
        if input.numel() != output.numel() * self._size:
            raise RuntimeError(
                "Input tensor size must be output_size * comm_size for "
                "reduce_scatter_single"
            )
        stream = self._stream_for(async_op)
        with _torch_to_nccl_redop(op, self._comm, output) as nccl_op:
            self._comm.reduce_scatter(
                input,
                output,
                nccl_op,
                stream=stream.cuda_stream,
            )
        return self._finish(stream, async_op)

    def all_to_all_single(
        self, output: torch.Tensor, input: torch.Tensor, async_op: bool
    ):
        self._check_init()
        self._check_tensor(output)
        self._check_tensor(input)
        if input.numel() != output.numel():
            raise RuntimeError(
                "Input and output tensors must have same size for all_to_all_single"
            )
        if input.numel() % self._size != 0:
            raise RuntimeError(
                "Tensor size must be divisible by comm_size for all_to_all_single"
            )
        stream = self._stream_for(async_op)
        chunk = input.numel() // self._size
        element_size = input.element_size()
        sptr = input.data_ptr()
        rptr = output.data_ptr()
        dtype = _nccl_dtype(input)
        with nccl.group():
            for i in range(self._size):
                send_view = (
                    sptr + i * chunk * element_size,
                    chunk,
                    int(dtype),
                )
                recv_view = (
                    rptr + i * chunk * element_size,
                    chunk,
                    int(dtype),
                )
                # nccl4py expects an NcclBufferSpec, which can be a tuple of
                # (buffer, dtype). Construct device-pointer based slices by
                # using a small wrapper tensor.
                # Simpler: use sub-tensors via narrow().
                send_sub = input.view(-1).narrow(0, i * chunk, chunk)
                recv_sub = output.view(-1).narrow(0, i * chunk, chunk)
                self._comm.send(send_sub, i, stream=stream.cuda_stream)
                self._comm.recv(recv_sub, i, stream=stream.cuda_stream)
        return self._finish(stream, async_op)

    def all_to_all_v_single(
        self,
        output: torch.Tensor,
        input: torch.Tensor,
        output_split_sizes: List[int],
        input_split_sizes: List[int],
        async_op: bool,
    ):
        self._check_init()
        self._check_tensor(output)
        self._check_tensor(input)
        if (
            len(input_split_sizes) != self._size
            or len(output_split_sizes) != self._size
        ):
            raise RuntimeError(
                "split_sizes length must equal comm_size for all_to_all_v_single"
            )

        send_per_slice = input.numel() // input.size(0) if input.numel() else 0
        recv_per_slice = output.numel() // output.size(0) if output.numel() else 0

        stream = self._stream_for(async_op)
        send_offset = 0
        recv_offset = 0
        with nccl.group():
            for i in range(self._size):
                send_count = int(input_split_sizes[i]) * send_per_slice
                recv_count = int(output_split_sizes[i]) * recv_per_slice
                if send_count > 0:
                    send_sub = input.view(-1).narrow(0, send_offset, send_count)
                    self._comm.send(send_sub, i, stream=stream.cuda_stream)
                if recv_count > 0:
                    recv_sub = output.view(-1).narrow(0, recv_offset, recv_count)
                    self._comm.recv(recv_sub, i, stream=stream.cuda_stream)
                send_offset += send_count
                recv_offset += recv_count
        return self._finish(stream, async_op)

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
            raise RuntimeError("Tensor list sizes must equal comm_size for all_to_all")
        self._check_tensors(input_tensor_list)
        self._check_tensors(output_tensor_list)
        stream = self._stream_for(async_op)
        with nccl.group():
            for i in range(self._size):
                self._comm.send(
                    input_tensor_list[i],
                    i,
                    stream=stream.cuda_stream,
                )
                self._comm.recv(
                    output_tensor_list[i],
                    i,
                    stream=stream.cuda_stream,
                )
        return self._finish(stream, async_op)

    def barrier(self, async_op: bool):
        self._check_init()
        stream = self._stream_for(async_op)
        # Use a one-element float32 all-reduce as the barrier primitive, same
        # as the C++ NCCL backend.
        scratch = torch.zeros(1, dtype=torch.float32, device=self._device)
        self._comm.allreduce(scratch, scratch, nccl.SUM, stream=stream.cuda_stream)
        return self._finish(stream, async_op)

    def scatter(
        self,
        output_tensor: torch.Tensor,
        input_tensor_list: List[torch.Tensor],
        root: int,
        async_op: bool,
    ):
        self._check_init()
        self._check_tensor(output_tensor)
        if self._rank == root:
            if len(input_tensor_list) != self._size:
                raise RuntimeError(
                    "input_tensor_list size must equal comm_size for scatter"
                )
            self._check_tensors(input_tensor_list)
            for t in input_tensor_list:
                if t.numel() != output_tensor.numel():
                    raise RuntimeError(
                        "All input tensors must have same size as output tensor"
                    )
        stream = self._stream_for(async_op)
        if self._rank == root:
            with nccl.group():
                for i, src in enumerate(input_tensor_list):
                    if i == root:
                        continue
                    self._comm.send(src, i, stream=stream.cuda_stream)
            # Root copies its own slice into the output tensor on the same
            # stream so subsequent ops on that stream see the result.
            with torch.cuda.stream(stream):
                output_tensor.copy_(input_tensor_list[root], non_blocking=True)
        else:
            self._comm.recv(output_tensor, root, stream=stream.cuda_stream)
        return self._finish(stream, async_op)

    def gather(
        self,
        output_tensor_list: List[torch.Tensor],
        input_tensor: torch.Tensor,
        root: int,
        async_op: bool,
    ):
        self._check_init()
        self._check_tensor(input_tensor)
        if self._rank == root:
            if len(output_tensor_list) != self._size:
                raise RuntimeError(
                    "output_tensor_list size must equal comm_size for gather"
                )
            self._check_tensors(output_tensor_list)
            for t in output_tensor_list:
                if t.numel() != input_tensor.numel():
                    raise RuntimeError(
                        "All output tensors must have same size as input tensor"
                    )
        stream = self._stream_for(async_op)
        if self._rank == root:
            with nccl.group():
                for i, dst in enumerate(output_tensor_list):
                    if i == root:
                        continue
                    self._comm.recv(dst, i, stream=stream.cuda_stream)
            with torch.cuda.stream(stream):
                output_tensor_list[root].copy_(input_tensor, non_blocking=True)
        else:
            self._comm.send(input_tensor, root, stream=stream.cuda_stream)
        return self._finish(stream, async_op)

    # ------------------------------------------------------------------
    # Sub-communicators
    # ------------------------------------------------------------------
    def split(
        self, ranks: List[int], name: str, options
    ) -> Optional["TorchCommNCCL4Py"]:
        self._check_init()
        seen = set()
        for r in ranks:
            if r < 0 or r >= self._size:
                raise RuntimeError(
                    f"Invalid rank {r} in ranks. Valid: 0..{self._size - 1}"
                )
            if r in seen:
                raise RuntimeError(f"Rank {r} appears multiple times in ranks")
            seen.add(r)

        if not ranks:
            color = nccl.NCCL_SPLIT_NOCOLOR
            new_rank = -1
        else:
            if self._rank not in seen:
                raise RuntimeError(
                    f"Current rank {self._rank} is not included in the provided "
                    f"ranks list"
                )
            color = min(ranks)
            new_rank = ranks.index(self._rank)

        new_nccl_comm = self._comm.split(
            color=color if color != nccl.NCCL_SPLIT_NOCOLOR else None, key=new_rank
        )

        if new_rank == -1:
            # Rank excluded from the new sub-communicator.
            return None

        child = TorchCommNCCL4Py()
        child._comm = new_nccl_comm
        child._rank = new_rank
        child._size = len(ranks)
        child._device = self._device
        child._name = name
        child._owns_comm = True
        priority = 0
        child._internal_stream = torch.cuda.Stream(
            device=self._device, priority=priority
        )
        # Propagate device / options into the C++ side of PyTorchCommBackend
        # so c10d wrappers (BackendWrapper) see the correct device type. The
        # regular ``init()`` path normally does this but ``split`` skips it.
        child._set_device_and_options(self._device, options)
        return child
