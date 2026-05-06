# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-strict

from typing import List

import torch
from torchcomms._comms import TorchCommWindow

class TorchCommWindowNCCLXGin(TorchCommWindow):
    def get_nvlink_address(self, peer: int, offset: int = 0) -> int: ...
    def get_multimem_address(self, offset: int = 0) -> int: ...

class TorchCommWindowNCCLXPipes(TorchCommWindow): ...

class TorchWork:
    def is_completed(self) -> bool: ...
    def wait(self) -> None: ...

class TorchCommNCCLX:
    def device_alltoallv_single(
        self,
        output: torch.Tensor,
        input: torch.Tensor,
        output_split_sizes: torch.Tensor,
        input_split_sizes: torch.Tensor,
        async_op: bool,
    ) -> TorchWork: ...
    def alltoallv_dynamic_dispatch(
        self,
        output_tensor_list: List[torch.Tensor],
        output_chunk_sizes_per_rank: torch.Tensor,
        input_tensor: torch.Tensor,
        input_chunk_sizes: torch.Tensor,
        input_chunk_indices: torch.Tensor,
        input_chunk_count_per_rank: torch.Tensor,
        hidden_dim: int,
        async_op: bool,
    ) -> TorchWork: ...
    def alltoallv_dynamic_combine(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        input_chunk_sizes: torch.Tensor,
        input_chunk_indices: torch.Tensor,
        input_chunk_count_per_rank: torch.Tensor,
        hidden_dim: int,
        async_op: bool,
    ) -> TorchWork: ...
    def reduce_scatter_quantized(
        self,
        output: torch.Tensor,
        input: torch.Tensor,
        op: torch.distributed.ReduceOp,
        seed: torch.Tensor,
        async_op: bool,
    ) -> TorchWork: ...
    def comm_dump(self) -> dict[str, str]: ...

def comm_dump_all() -> dict[str, dict[str, str]]: ...
def init_caching_allocator_hook() -> None: ...
