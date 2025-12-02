# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-strict

from typing import List

import torch

class TorchWork:
    def is_completed(self) -> bool: ...
    def wait(self) -> None: ...

class TorchCommNCCLX:
    def alltoallv_dynamic_dispatch(
        self,
        output_tensor_list: List[torch.Tensor],
        output_chunk_sizes_per_rank: torch.Tensor,
        input_tensor: torch.Tensor,
        input_chunk_sizes: torch.Tensor,
        input_chunk_indices: torch.Tensor,
        input_chunk_count_per_rank: torch.Tensor,
        async_op: bool,
    ) -> TorchWork: ...
    def alltoallv_dynamic_combine(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        input_chunk_sizes: torch.Tensor,
        input_chunk_indices: torch.Tensor,
        input_chunk_count_per_rank: torch.Tensor,
        async_op: bool,
    ) -> TorchWork: ...
