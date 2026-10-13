#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Example showing PTD/c10d -> TorchComms -> MCCL persistent AllGatherP.

The process group is initialized through torch.distributed, so the c10d CUDA
backend implementation is a TorchComms _BackendWrapper. Custom callers can get
the underlying TorchComm through wrapper.get_comm() and call AllGatherP there.
"""

import os

import torch
import torch.distributed as dist
import torch.distributed.config as dist_config
import torchcomms
import torchcomms._comms_mccl  # noqa: F401
from torchcomms._comms import _BackendWrapper


ELEM_COUNT = 1024
NUM_REPLAYS = 3


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

    handle = None
    try:
        wrapper = dist.get_backend_impl(device=device)
        assert isinstance(wrapper, _BackendWrapper), type(wrapper)

        comm = wrapper.get_comm()
        assert comm.get_backend() == "mccl", comm.get_backend()
        backend_impl = comm.get_backend_impl()
        assert type(backend_impl).__name__ == "TorchCommMCCL", type(backend_impl)

        comm_rank = comm.get_rank()
        comm_size = comm.get_size()
        input_tensor = torch.full(
            (ELEM_COUNT,),
            float(comm_rank),
            dtype=torch.float32,
            device=device,
        )

        allocator = torchcomms.get_mem_allocator(comm.get_backend())
        pool = torch.cuda.MemPool(allocator)
        with torch.cuda.use_mem_pool(pool):
            output_tensor = torch.empty(
                ELEM_COUNT * comm_size,
                dtype=torch.float32,
                device=device,
            )

        handle = comm.all_gather_p_init(output_tensor)
        dist.barrier(device_ids=[local_rank])

        expected = torch.arange(
            comm_size,
            dtype=output_tensor.dtype,
            device=device,
        ).repeat_interleave(ELEM_COUNT)

        for _ in range(NUM_REPLAYS):
            output_tensor.zero_()
            work = comm.all_gather_p_exec(handle, input_tensor, async_op=True)
            work.wait()
            torch.cuda.synchronize(device)
            torch.testing.assert_close(output_tensor, expected, rtol=0, atol=0)

        if rank == 0:
            print(
                "Verified PTD backend _BackendWrapper -> TorchComms "
                "mccl backend implementation TorchCommMCCL"
            )
            print(f"Verified PTD TorchComms MCCL AllGatherP for {comm_size} ranks")
    finally:
        if handle is not None:
            comm.all_gather_p_free(handle)
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
