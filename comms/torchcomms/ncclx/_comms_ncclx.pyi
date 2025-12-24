# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-strict

from typing import List

import torch

class TorchWork:
    def is_completed(self) -> bool: ...
    def wait(self) -> None: ...

class TorchCommNCCLXPersistentRequest:
    def map_remote_tensor(self) -> torch.Tensor: ...

class TorchCommNCCLX:
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
    def alltoallv_dedup_init(
        self,
        num_send_blocks: int,
        block_count: int,
        block_num_recv_buckets: int,
        num_recv_buckets: int,
        dtype: torch.dtype,
        async_op: bool,
    ) -> TorchCommNCCLXPersistentRequest: ...
    def alltoallv_dedup_exec(
        self,
        output_tensor: torch.Tensor,
        recv_block_ids: torch.Tensor,
        input_tensor: torch.Tensor,
        send_indices: torch.Tensor,
        forward_indices: torch.Tensor,
        recv_indices: torch.Tensor,
        request: TorchCommNCCLXPersistentRequest,
    ) -> TorchWork: ...
    def alltoallv_dedup_combine(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        send_indices: torch.Tensor,
        forward_indices: torch.Tensor,
        recv_indices: torch.Tensor,
        request: TorchCommNCCLXPersistentRequest,
        async_op: bool,
    ) -> TorchWork: ...
