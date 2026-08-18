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
        output_tensors: list[torch.Tensor] | None = None,
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
            output_tensors: Optional list of output segment tensors (one list
                per group), parallel to `tensors`. When None (default) the
                allreduce is in-place. When provided, the active group's
                reduced result is written out-of-place into these tensors
                while the inputs are preserved.

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
                nActiveRanks-slot scratch tensor.
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

    def sharded_relay_multi_group_all_to_all(
        self,
        input_tensors: list[torch.Tensor],
        output_tensors: list[torch.Tensor],
        all_active_ranks: list[list[int]],
        per_group_segment_counts: list[int],
        async_op: bool = False,
    ) -> TorchWork | None:
        """
        Fused multi-group sharded relay all-to-all for 2D sparse parallelism.

        All-to-all analogue of sharded_relay_multi_group_reduce_scatter. No
        reduction (pure data movement) and no reduction op. Each active rank's
        input/output hold nActiveRanks x per_group_segment_counts[g] elements
        (input = [sendSeg[0]|...|sendSeg[A-1]], output =
        [recvSeg[0]|...|recvSeg[A-1]]). nActiveRanks must be a power of two (2 or
        4); A>2 uses a flat all-to-all + 2-hop relay.

        OUT-OF-PLACE ONLY: input and output tensors for the active group must be
        distinct (matches native ncclAllToAll); aliasing raises.

        Args:
            input_tensors: List of send tensors (one per group). Active rank:
                nActiveRanks x per_group_segment_counts[g] elements. Helper rank:
                nActiveRanks-slot scratch tensor.
            output_tensors: List of receive tensors (one per group). Active rank:
                nActiveRanks x per_group_segment_counts[g] elements (distinct
                from input). Helper rank: same scratch as input.
            all_active_ranks: List of lists of active rank IDs per sparse group
            per_group_segment_counts: List of per-segment element counts
            async_op: If True, returns a TorchWork handle for async operation

        Returns:
            TorchWork handle if async_op=True, else None
        """
        ...

    def sharded_relay_multi_group_all_gather(
        self,
        input_tensors: list[torch.Tensor],
        output_tensors: list[torch.Tensor],
        all_active_ranks: list[list[int]],
        per_group_send_counts: list[int],
        async_op: bool = False,
    ) -> TorchWork | None:
        """
        Fused multi-group sharded relay all-gather for 2D sparse parallelism.

        All-gather analogue of sharded_relay_multi_group_reduce_scatter (its
        dual). No reduction (pure data movement) and no reduction op.
        nActiveRanks must be a power of two (2 or 4); A>2 uses the flat
        scatter->forward relay: a direct intra all-to-all woven with a 2-hop
        offload through the idle helper GPUs. Each active rank's input
        holds per_group_send_counts[g] elements and its output holds
        nActiveRanks x per_group_send_counts[g] elements (output[i x sendCount]
        from active index i).

        Supports both in-place and out-of-place. In-place is detected when the
        active input aliases output + myActiveIndex x sendCount.

        Args:
            input_tensors: List of send tensors (one per group). Active rank:
                per_group_send_counts[g] elements. Helper rank:
                nActiveRanks-slot scratch tensor.
            output_tensors: List of receive tensors (one per group). Active rank:
                nActiveRanks x per_group_send_counts[g] elements. Helper rank:
                same scratch as input.
            all_active_ranks: List of lists of active rank IDs per sparse group
            per_group_send_counts: List of per-rank contribution counts
            async_op: If True, returns a TorchWork handle for async operation

        Returns:
            TorchWork handle if async_op=True, else None
        """
        ...
