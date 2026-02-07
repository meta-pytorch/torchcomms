# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-strict

from typing import Any, Callable

import torch
import torch.distributed as dist

# pyre-fixme[21]: Could not find name in module (ProcessGroup backends not in stubs)
from torch.distributed import (  # noqa: F401
    get_process_group_ranks,
    get_rank,
    get_world_size,
    group,
    GroupMember,
    HashStore,
    is_available,
    is_initialized,
    ProcessGroup,
    ProcessGroupGloo,
    ProcessGroupNCCL,
    ReduceOp,
    Store,
    Work,
)
from torchcomms.distwrap.collectives import (
    _make_nccl_premul_sum,
    all_gather,
    all_gather_into_tensor,
    all_gather_object,
    all_reduce,
    all_to_all,
    all_to_all_single,
    barrier,
    batch_isend_irecv,
    broadcast,
    broadcast_object_list,
    gather,
    gather_object,
    irecv,
    isend,
    recv,
    reduce,
    reduce_scatter,
    reduce_scatter_tensor,
    scatter,
    scatter_object_list,
    send,
)
from torchcomms.distwrap.collectives_extension import (
    alltoallv_dedup_exec,
    alltoallv_dedup_init,
    alltoallv_dynamic_combine,
    alltoallv_dynamic_dispatch,
    new_window,
)
from torchcomms.distwrap.new_comm import (
    destroy_process_group,
    init_process_group,
    new_group,
    split_group,
)


# =============================================================================
# P2POp wrapper
# =============================================================================


class P2POp:
    """
    A wrapper for torch.distributed.P2POp that accepts distwrap's isend/irecv.

    This class allows users to use distwrap's isend/irecv functions when creating
    P2POp instances, which are then mapped to torch.distributed.isend/irecv
    internally for compatibility with batch_isend_irecv.
    """

    def __init__(
        self,
        op: Callable[..., Any],
        tensor: torch.Tensor,
        peer: int | None = None,
        group: dist.ProcessGroup | None = None,
        tag: int = 0,
        group_peer: int | None = None,
    ) -> None:
        # Map distwrap's isend/irecv to torch.distributed versions
        if op is isend:
            mapped_op = dist.distributed_c10d.isend
        elif op is irecv:
            mapped_op = dist.distributed_c10d.irecv
        else:
            # Assume it's already a torch.distributed function
            mapped_op = op

        # Create the underlying torch.distributed.P2POp
        self._p2p_op = dist.P2POp(
            op=mapped_op,
            tensor=tensor,
            peer=peer,
            group=group,
            tag=tag,
            group_peer=group_peer,
        )

    @property
    def op(self) -> Callable[..., Any]:
        return self._p2p_op.op

    @property
    def tensor(self) -> torch.Tensor:
        return self._p2p_op.tensor

    @property
    def group(self) -> dist.ProcessGroup:
        return self._p2p_op.group

    @property
    def peer(self) -> int:
        return self._p2p_op.peer

    @property
    def tag(self) -> int:
        return self._p2p_op.tag

    @property
    def group_peer(self) -> int:
        return self._p2p_op.group_peer

    def __repr__(self) -> str:
        return repr(self._p2p_op)


__all__ = [
    "_make_nccl_premul_sum",
    "init_process_group",
    "destroy_process_group",
    "new_group",
    "split_group",
    "all_reduce",
    "broadcast",
    "reduce",
    "all_gather",
    "all_gather_into_tensor",
    "reduce_scatter",
    "reduce_scatter_tensor",
    "all_to_all",
    "all_to_all_single",
    "scatter",
    "gather",
    "barrier",
    "send",
    "recv",
    "isend",
    "irecv",
    "batch_isend_irecv",
    "all_gather_object",
    "gather_object",
    "scatter_object_list",
    "broadcast_object_list",
    "new_window",
    "alltoallv_dedup_init",
    "alltoallv_dedup_exec",
    "alltoallv_dynamic_dispatch",
    "alltoallv_dynamic_combine",
    # Re-exported from torch.distributed
    "ReduceOp",
    "P2POp",
    "get_rank",
    "get_world_size",
    "is_initialized",
    "is_available",
    "get_process_group_ranks",
    "ProcessGroup",
    "ProcessGroupGloo",
    "ProcessGroupNCCL",
    "GroupMember",
    "group",
    "Store",
    "HashStore",
    "Work",
]
