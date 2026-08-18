# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-strict

import torch

class TorchWork:
    def is_completed(self) -> bool: ...
    def wait(self) -> None: ...

class TorchCommRCCLX:
    def sharded_relay_multi_group_all_reduce(
        self,
        tensors: list[torch.Tensor],
        op: torch.distributed.ReduceOp,
        all_active_ranks: list[list[int]],
        per_group_counts: list[int],
        async_op: bool = False,
    ) -> TorchWork | None:
        """
        Fused multi-group sharded relay allreduce for 2D sparse parallelism.

        This executes multiple allreduce groups in lockstep phases to eliminate
        XGMI link contention on MI300x GPUs.

        Args:
            tensors: List of tensors to allreduce (one per group, modified in-place)
            op: Reduction operation (e.g., ReduceOp.SUM)
            all_active_ranks: List of lists, where each inner list contains the
                active rank IDs for one sparse group
            per_group_counts: List of element counts (one per group). This allows
                different groups to have different tensor sizes.
            async_op: If True, returns a TorchWork handle for async operation

        Returns:
            TorchWork handle if async_op=True, else None
        """
        ...

    def sharded_relay_multi_group_reduce_scatter(
        self,
        input_tensors: list[torch.Tensor],
        output_tensors: list[torch.Tensor],
        op: torch.distributed.ReduceOp,
        all_active_ranks: list[list[int]],
        per_group_recv_counts: list[int],
        async_op: bool = False,
    ) -> TorchWork | None:
        """
        Fused multi-group sharded relay reduce-scatter for 2D sparse parallelism.

        Reduce-scatter analogue of sharded_relay_multi_group_all_reduce. Each
        active rank's input holds nActiveRanks x per_group_recv_counts[g]
        elements (block[i] is the slice destined for active index i) and its
        output holds per_group_recv_counts[g] elements.

        Args:
            input_tensors: List of send tensors (one per group). Active rank:
                nActiveRanks x per_group_recv_counts[g] elements. Helper rank:
                two-slot scratch tensor.
            output_tensors: List of receive tensors (one per group). Active
                rank: per_group_recv_counts[g] elements (pass the local block of
                the input for in-place). Helper rank: same scratch as input.
            op: Reduction operation (ReduceOp.SUM or ReduceOp.AVG)
            all_active_ranks: List of lists, where each inner list contains the
                active rank IDs for one sparse group
            per_group_recv_counts: List of OUTPUT element counts (one per group)
            async_op: If True, returns a TorchWork handle for async operation

        Returns:
            TorchWork handle if async_op=True, else None
        """
        ...
