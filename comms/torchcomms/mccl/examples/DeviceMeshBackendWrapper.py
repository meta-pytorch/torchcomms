#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Create one DeviceMesh dimension with a registered TorchComms MCCL backend."""

import os

import torch
import torch.distributed as dist
import torch.distributed.config as dist_config
from torch.distributed.device_mesh import init_device_mesh
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
        device_id=device,
    )
    try:
        assert "MCCL" not in dist.Backend._plugins
        mesh = init_device_mesh(
            "cuda",
            (1, world_size),
            mesh_dim_names=("replicate", "shard"),
            backend_override={"shard": "cuda:mccl"},
        )
        # Parsing cuda:mccl above lazily loads its torch.distributed.backends
        # entry point; the example never calls the registrar directly.
        assert "MCCL" in dist.Backend._plugins

        default_backend = dist.get_backend_impl(device=device)
        replicate_group = mesh.get_group("replicate")
        replicate_backend = dist.get_backend_impl(group=replicate_group, device=device)
        shard_group = mesh.get_group("shard")
        shard_backend = dist.get_backend_impl(group=shard_group, device=device)

        assert not dist_config.use_torchcomms
        assert not isinstance(default_backend, _BackendWrapper), type(default_backend)
        assert not isinstance(replicate_backend, _BackendWrapper), type(
            replicate_backend
        )
        assert isinstance(shard_backend, _BackendWrapper), type(shard_backend)
        assert shard_backend.get_comm().get_backend() == "mccl"
        assert dist.get_process_group_ranks(shard_group) == list(range(world_size))

        # MCCL's CTRAN algorithms register collective buffers with the selected
        # transport. Use MCCL's VMM allocator so this example remains runnable
        # on hosts whose ordinary CUDA allocations cannot be registered by IB.
        pool = torch.cuda.MemPool(shard_backend.get_mem_allocator())
        with torch.cuda.use_mem_pool(pool):
            input_tensor = torch.full(
                (TENSOR_SIZE,), float(rank + 1), dtype=torch.float32, device=device
            )
            gathered = torch.empty(
                world_size * TENSOR_SIZE, dtype=torch.float32, device=device
            )
            reduce_scatter_input = torch.full(
                (world_size * TENSOR_SIZE,),
                float(rank + 1),
                dtype=torch.float32,
                device=device,
            )
            reduce_scatter_output = torch.empty_like(input_tensor)

        dist.all_gather_into_tensor(gathered, input_tensor, group=shard_group)
        expected_gather = torch.cat(
            [
                torch.full_like(input_tensor, float(peer_rank + 1))
                for peer_rank in range(world_size)
            ]
        )
        torch.testing.assert_close(gathered, expected_gather, rtol=0, atol=0)

        dist.reduce_scatter_tensor(
            reduce_scatter_output,
            reduce_scatter_input,
            op=dist.ReduceOp.SUM,
            group=shard_group,
        )
        expected_sum = float(world_size * (world_size + 1) // 2)
        torch.testing.assert_close(
            reduce_scatter_output,
            torch.full_like(reduce_scatter_output, expected_sum),
            rtol=0,
            atol=0,
        )
        torch.cuda.synchronize(device)

        if rank == 0:
            print("Verified native default and replicate process groups")
            print("Verified DeviceMesh shard -> _BackendWrapper -> TorchCommMCCL")
            print("Verified MCCL shard all-gather and reduce-scatter")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
