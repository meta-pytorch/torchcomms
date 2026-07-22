#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Example showing c10d -> TorchComms -> NCCLX persistent AllGatherP.

The process group is initialized through torch.distributed with
TorchComms enabled, so the c10d backend is a _BackendWrapper.  Ads-style
callers that need a custom operation can then get the underlying TorchComm
via wrapper.get_comm() and call the generic TorchComms AllGatherP methods.
"""

import os

import torch
import torch.distributed as dist
import torch.distributed.config as dist_config
import torchcomms
import torchcomms._comms_ncclx  # noqa: F401
from torchcomms._comms import _BackendWrapper


ELEM_COUNT = 1024
NUM_REPLAYS = 3


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value is not None else default


def _rank() -> int:
    return _env_int("RANK", _env_int("OMPI_COMM_WORLD_RANK", 0))


def _world_size() -> int:
    return _env_int("WORLD_SIZE", _env_int("OMPI_COMM_WORLD_SIZE", 1))


def _local_rank() -> int:
    return _env_int(
        "LOCAL_RANK",
        _env_int("OMPI_COMM_WORLD_LOCAL_RANK", _rank()),
    )


def _verify_allgather(output: torch.Tensor, elem_count: int, world_size: int) -> None:
    expected = torch.arange(
        world_size,
        dtype=output.dtype,
        device=output.device,
    ).repeat_interleave(elem_count)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


def main() -> None:
    local_rank = _local_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    dist_config.use_torchcomms = True
    dist.init_process_group(
        backend="cuda:ncclx",
        rank=_rank(),
        world_size=_world_size(),
    )

    handle = None
    try:
        wrapper = dist.get_backend_impl(device=device)
        if not isinstance(wrapper, _BackendWrapper):
            raise RuntimeError(f"Expected _BackendWrapper, got {type(wrapper)!r}")

        comm = wrapper.get_comm()
        if comm.get_backend() != "ncclx":
            raise RuntimeError(f"Expected NCCLX TorchComm, got {comm.get_backend()!r}")

        rank = dist.get_rank()
        world_size = dist.get_world_size()
        input_tensor = torch.full(
            (ELEM_COUNT,),
            rank,
            dtype=torch.float32,
            device=device,
        )

        allocator = torchcomms.get_mem_allocator(comm.get_backend())
        pool = torch.cuda.MemPool(allocator)
        with torch.cuda.use_mem_pool(pool):
            output_tensor = torch.empty(
                ELEM_COUNT * world_size,
                dtype=torch.float32,
                device=device,
            )

        handle = comm.all_gather_p_init(output_tensor)
        dist.barrier(device_ids=[local_rank])

        for _ in range(NUM_REPLAYS):
            output_tensor.zero_()
            work = comm.all_gather_p_exec(handle, input_tensor, async_op=True)
            work.wait()
            torch.cuda.synchronize(device)
            _verify_allgather(output_tensor, ELEM_COUNT, world_size)

        if rank == 0:
            print(
                "Verified c10d _BackendWrapper -> TorchComm NCCLX "
                f"AllGatherP for {world_size} ranks"
            )
    finally:
        if handle is not None:
            comm.all_gather_p_free(handle)
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
