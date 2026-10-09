#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Example showing PTD/c10d -> TorchComms -> MCCL for ordinary collectives.

This verifies that c10d can initialize a CUDA MCCL process group backed by
TorchComms and run an allreduce.
"""

import os

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

    dist_config.use_torchcomms = True
    dist.init_process_group(
        backend="cuda:mccl",
        rank=rank,
        world_size=world_size,
    )
    try:
        wrapper = dist.get_backend_impl(device=device)
        assert isinstance(wrapper, _BackendWrapper), type(wrapper)

        comm = wrapper.get_comm()
        assert comm.get_backend() == "mccl", comm.get_backend()
        backend_impl = comm.get_backend_impl()
        assert type(backend_impl).__name__ == "TorchCommMCCL", type(backend_impl)

        tensor = torch.full(
            (TENSOR_SIZE,),
            float(rank + 1),
            dtype=torch.float32,
            device=device,
        )
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize(device)

        expected = torch.full_like(tensor, float(world_size * (world_size + 1) // 2))
        torch.testing.assert_close(tensor, expected, rtol=0, atol=0)

        if rank == 0:
            print(
                "Verified PTD backend _BackendWrapper -> TorchComms "
                "mccl backend implementation TorchCommMCCL"
            )
            print(f"Verified PTD TorchComms MCCL all_reduce for {world_size} ranks")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
