#!/usr/bin/env python3
# pyre-strict
# Copyright (c) Meta Platforms, Inc. and affiliates.

from datetime import timedelta
from enum import auto, Enum
from typing import Any, Dict, List

class RedOpType(Enum):
    SUM = auto()
    PRODUCT = auto()
    MIN = auto()
    MAX = auto()
    BAND = auto()
    BOR = auto()
    BXOR = auto()
    PREMUL_SUM = auto()
    AVG = auto()

class ReduceOp:
    SUM: ReduceOp = ...
    PRODUCT: ReduceOp = ...
    MIN: ReduceOp = ...
    MAX: ReduceOp = ...
    BAND: ReduceOp = ...
    BOR: ReduceOp = ...
    BXOR: ReduceOp = ...
    AVG: ReduceOp = ...
    @staticmethod
    def PREMUL_SUM(factor: Any) -> ReduceOp: ...
    @property
    def type(self) -> RedOpType: ...

class CommOptions:
    abort_process_on_timeout_or_error: bool
    timeout: timedelta
    store: Any
    name: str
    hints: Dict[str, str]
    def __init__(self) -> None: ...

class SendOptions:
    def __init__(self) -> None: ...
    timeout: timedelta
    hints: Dict[str, str]

class RecvOptions:
    def __init__(self) -> None: ...
    timeout: timedelta
    hints: Dict[str, str]

class BatchP2POptions:
    def __init__(self) -> None: ...
    timeout: timedelta
    hints: Dict[str, str]

class BroadcastOptions:
    def __init__(self) -> None: ...
    timeout: timedelta
    hints: Dict[str, str]

class AllReduceOptions:
    def __init__(self) -> None: ...
    timeout: timedelta
    hints: Dict[str, str]

class ReduceOptions:
    def __init__(self) -> None: ...
    timeout: timedelta
    hints: Dict[str, str]

class AllGatherOptions:
    def __init__(self) -> None: ...
    timeout: timedelta
    hints: Dict[str, str]

class AllGatherSingleOptions:
    def __init__(self) -> None: ...
    timeout: timedelta
    hints: Dict[str, str]

class ReduceScatterOptions:
    def __init__(self) -> None: ...
    timeout: timedelta
    hints: Dict[str, str]

class ReduceScatterSingleOptions:
    def __init__(self) -> None: ...
    timeout: timedelta
    hints: Dict[str, str]

class AllToAllOptions:
    def __init__(self) -> None: ...
    timeout: timedelta
    hints: Dict[str, str]

class AllToAllSingleOptions:
    def __init__(self) -> None: ...
    timeout: timedelta
    hints: Dict[str, str]

class AllToAllvSingleOptions:
    def __init__(self) -> None: ...
    timeout: timedelta
    hints: Dict[str, str]

class BarrierOptions:
    def __init__(self) -> None: ...
    timeout: timedelta
    hints: Dict[str, str]

class ScatterOptions:
    def __init__(self) -> None: ...
    timeout: timedelta
    hints: Dict[str, str]

class GatherOptions:
    def __init__(self) -> None: ...
    timeout: timedelta
    hints: Dict[str, str]

class TorchWork:
    def is_completed(self) -> bool: ...
    def wait(self) -> None: ...

class TorchCommlWinAccessType(Enum):
    WIN_ACCESS_TYPE_UNIFIED = auto()
    WIN_ACCESS_TYPE_SEPARATE = auto()

class TorchCommWindowAttr:
    def __init__(self) -> None: ...
    access_type: TorchCommlWinAccessType

class TorchCommWindow:
    def get_buf_dtype(self) -> Any: ...
    def get_buf_shape(self) -> List[int]: ...
    def get_buf_device(self) -> Any: ...
    def get_size(self) -> int: ...
    def get_device(self) -> Any: ...
    def tensor_register(
        self,
        tensor: Any,
    ) -> None: ...
    def tensor_deregister(
        self,
    ) -> None: ...
    def put(
        self,
        tensor: Any,
        dst_rank: int,
        target_offset_nelems: int,
        async_op: bool,
        hints: Dict[str, str] | None = None,
        timeout: timedelta | None = None,
    ) -> TorchWork: ...
    def signal(
        self,
        dst_rank: int,
        async_op: bool,
        hints: Dict[str, str] | None = None,
        timeout: timedelta | None = None,
    ) -> TorchWork: ...
    def wait_signal(
        self,
        peer_rank: int,
        async_op: bool,
        hints: Dict[str, str] | None = None,
        timeout: timedelta | None = None,
    ) -> TorchWork: ...
    def map_remote_tensor(
        self,
        rank: int,
    ) -> Any: ...
    def get_attr(self, peer_rank: int) -> TorchCommWindowAttr: ...

class P2POpType(Enum):
    SEND = auto()
    RECV = auto()

class P2POp:
    type: P2POpType
    tensor: Any
    peer: int
    def __init__(self, type: P2POpType, tensor: Any, peer: int) -> None: ...

class BatchSendRecv:
    ops: List[P2POp]
    def send(self, tensor: Any, dst: int) -> None: ...
    def recv(self, tensor: Any, src: int) -> None: ...
    def issue(self, async_op: bool, options: BatchP2POptions = ...) -> TorchWork: ...

class TorchCommBackend: ...

class TorchComm:
    def finalize(self) -> None: ...
    def get_rank(self) -> int: ...
    def get_size(self) -> int: ...
    def get_name(self) -> str: ...
    def get_options(self) -> CommOptions: ...
    def get_device(self) -> Any: ...
    def get_backend(self) -> str: ...
    def unsafe_get_backend(self) -> TorchCommBackend: ...
    def send(
        self,
        tensor: Any,
        dst: int,
        async_op: bool,
        hints: Dict[str, str] | None = None,
        timeout: timedelta | None = None,
    ) -> TorchWork: ...
    def recv(
        self,
        tensor: Any,
        src: int,
        async_op: bool,
        hints: Dict[str, str] | None = None,
        timeout: timedelta | None = None,
    ) -> TorchWork: ...
    def broadcast(
        self,
        tensor: Any,
        root: int,
        async_op: bool,
        hints: Dict[str, str] | None = None,
        timeout: timedelta | None = None,
    ) -> TorchWork: ...
    def all_reduce(
        self,
        tensor: Any,
        op: ReduceOp,
        async_op: bool,
        hints: Dict[str, str] | None = None,
        timeout: timedelta | None = None,
    ) -> TorchWork: ...
    def reduce(
        self,
        tensor: Any,
        root: int,
        op: ReduceOp,
        async_op: bool,
        hints: Dict[str, str] | None = None,
        timeout: timedelta | None = None,
    ) -> TorchWork: ...
    def all_gather(
        self,
        tensor_list: List[Any],
        tensor: Any,
        async_op: bool,
        hints: Dict[str, str] | None = None,
        timeout: timedelta | None = None,
    ) -> TorchWork: ...
    def all_gather_v(
        self,
        tensor_list: List[Any],
        tensor: Any,
        async_op: bool,
        hints: Dict[str, str] | None = None,
        timeout: timedelta | None = None,
    ) -> TorchWork: ...
    def all_gather_single(
        self,
        output: Any,
        input: Any,
        async_op: bool,
        hints: Dict[str, str] | None = None,
        timeout: timedelta | None = None,
    ) -> TorchWork: ...
    def reduce_scatter(
        self,
        output: Any,
        input_list: List[Any],
        op: ReduceOp,
        async_op: bool,
        hints: Dict[str, str] | None = None,
        timeout: timedelta | None = None,
    ) -> TorchWork: ...
    def reduce_scatter_v(
        self,
        output: Any,
        input_list: List[Any],
        op: ReduceOp,
        async_op: bool,
        hints: Dict[str, str] | None = None,
        timeout: timedelta | None = None,
    ) -> TorchWork: ...
    def reduce_scatter_single(
        self,
        output: Any,
        input: Any,
        op: ReduceOp,
        async_op: bool,
        hints: Dict[str, str] | None = None,
        timeout: timedelta | None = None,
    ) -> TorchWork: ...
    def all_to_all_single(
        self,
        output: Any,
        input: Any,
        async_op: bool,
        hints: Dict[str, str] | None = None,
        timeout: timedelta | None = None,
    ) -> TorchWork: ...
    def all_to_all_v_single(
        self,
        output: Any,
        input: Any,
        output_split_sizes: List[int],
        input_split_sizes: List[int],
        async_op: bool,
        hints: Dict[str, str] | None = None,
        timeout: timedelta | None = None,
    ) -> TorchWork: ...
    def all_to_all(
        self,
        output_tensor_list: List[Any],
        input_tensor_list: List[Any],
        async_op: bool,
        hints: Dict[str, str] | None = None,
        timeout: timedelta | None = None,
    ) -> TorchWork: ...
    def barrier(
        self,
        async_op: bool,
        hints: Dict[str, str] | None = None,
        timeout: timedelta | None = None,
    ) -> TorchWork: ...
    def scatter(
        self,
        output_tensor: Any,
        input_tensor_list: List[Any],
        root: int,
        async_op: bool,
        hints: Dict[str, str] | None = None,
        timeout: timedelta | None = None,
    ) -> TorchWork: ...
    def gather(
        self,
        output_tensor_list: List[Any],
        input_tensor: Any,
        root: int,
        async_op: bool,
        hints: Dict[str, str] | None = None,
        timeout: timedelta | None = None,
    ) -> TorchWork: ...
    def split(
        self,
        rank_groups: List[List[int]],
        name: str,
        hints: Dict[str, str] | None = None,
        timeout: timedelta | None = None,
    ) -> TorchComm: ...
    def batch_op_create(self) -> BatchSendRecv: ...
    def new_window(self) -> TorchCommWindow: ...
    @property
    def mem_allocator(self) -> Any: ...

def new_comm(
    backend: str,
    device: Any,
    abort_process_on_timeout_or_error: bool | None = ...,
    timeout: timedelta | None = ...,
    store: Any | None = ...,
    name: str | None = ...,
    hints: Dict[str, str] | None = ...,
) -> TorchComm: ...

class _BackendWrapper:
    def __init__(self, comm: TorchComm) -> None: ...
    def get_comm(self) -> TorchComm: ...

def _get_store(backend_name: str, name_str: str, timeout: timedelta = ...) -> Any: ...
def get_mem_allocator(backend: str) -> Any: ...
