#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Example showing a default PTD CUDA group plus a TorchComms MCCL subgroup.

The BUCK target selects the NCCLX communication stack, but the default process
group is initialized through PTD's regular NCCL backend. A second CUDA process
group uses TorchComms MCCL.
"""

import os
from typing import cast

import torch
import torch.distributed as dist
import torch.distributed.config as dist_config
import torchcomms._comms_mccl  # noqa: F401
from torchcomms._comms import _BackendWrapper


TENSOR_SIZE = 16


def main() -> None:
    rank = int(os.environ.get("RANK", os.environ.get("OMPI_COMM_WORLD_RANK", "0")))
    world_size = int(
        os.environ.get("WORLD_SIZE", os.environ.get("OMPI_COMM_WORLD_SIZE", "1"))
    )
    local_rank = int(
        os.environ.get("LOCAL_RANK", os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", rank))
    )

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    dist_config.use_torchcomms = False
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
    )
    mccl_group: dist.ProcessGroup | None = None
    try:
        dist_config.use_torchcomms = True
        mccl_group = cast(
            dist.ProcessGroup,
            dist.new_group(
                backend="cuda:mccl",
                device_id=device,
            ),
        )

        mccl_wrapper = dist.get_backend_impl(group=mccl_group, device=device)
        assert isinstance(mccl_wrapper, _BackendWrapper), type(mccl_wrapper)
        mccl_comm = mccl_wrapper.get_comm()
        assert mccl_comm.get_backend() == "mccl", mccl_comm.get_backend()
        assert type(mccl_comm.get_backend_impl()).__name__ == "TorchCommMCCL"

        tensor = torch.ones(TENSOR_SIZE, dtype=torch.float32, device=device)
        expected = torch.full_like(tensor, float(world_size))

        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize(device)
        torch.testing.assert_close(tensor, expected, rtol=0, atol=0)

        tensor.fill_(1)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=mccl_group)
        torch.cuda.synchronize(device)
        torch.testing.assert_close(tensor, expected, rtol=0, atol=0)

        if rank == 0:
            print("Verified default PTD NCCL all_reduce")
            print(
                "Verified secondary PTD group _BackendWrapper -> "
                "TorchComms mccl backend implementation TorchCommMCCL"
            )
            print(
                f"Verified default PTD NCCL and TorchComms MCCL all_reduce for {world_size} ranks"
            )
    finally:
        if mccl_group is not None:
            dist.destroy_process_group(mccl_group)
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
